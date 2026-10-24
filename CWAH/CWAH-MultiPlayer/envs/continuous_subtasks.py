"""State-driven continuous subtasks for C-WAH.

The manager owns public task progression only. It does not assign work,
inspect agent plans, or emit TRACE evidence. Active tasks reserve concrete
object instances and are exposed through the native WAH predicate format.
"""

import copy
import random
import re


SUPPORTED_RELATIONS = {"on", "inside"}
HOLD_RELATIONS = {"holds_lh", "holds_rh"}


def _parse(predicate):
    relation, object_token, target = predicate.split("_", 2)
    return relation.lower(), object_token, int(target)


def _nodes(graph):
    return {node["id"]: node for node in graph["nodes"]}


def _relation_holds(graph, relation, object_id, target_id):
    return any(
        edge["relation_type"].lower() == relation
        and edge["from_id"] == object_id
        and edge["to_id"] == target_id
        for edge in graph["edges"]
    )


def _held_object_ids(graph):
    return {
        edge["to_id"]
        for edge in graph["edges"]
        if edge["relation_type"].lower() in HOLD_RELATIONS
    }


class ContinuousSubtaskManager:
    """Maintain concrete, conflict-free subtasks until the step budget."""

    def __init__(self, task_goal, active_count=3, seed=1, targets=None,
                 max_generation_attempts=64, reverse_cooldown_steps=20):
        if type(active_count) is not int or active_count < 1:
            raise ValueError("active_count must be a positive integer")
        if type(max_generation_attempts) is not int or max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be a positive integer")
        if type(reverse_cooldown_steps) is not int or reverse_cooldown_steps < 0:
            raise ValueError("reverse_cooldown_steps must be a nonnegative integer")
        self.template_goal = copy.deepcopy(task_goal)
        self.object_classes = sorted({
            _parse(predicate)[1]
            for predicate in task_goal
            if _parse(predicate)[0] in SUPPORTED_RELATIONS
        })
        self.active_count = active_count
        self.random = random.Random(seed)
        self.targets = copy.deepcopy(targets or [])
        self.max_generation_attempts = max_generation_attempts
        self.reverse_cooldown_steps = reverse_cooldown_steps
        self.subtasks = {}
        self.next_subtask_number = 1
        self.revision = 0
        self.events = []
        self.cooldowns = {}
        self.generation_failures = 0
        self.last_generation_attempts = 0

    @staticmethod
    def _predicate(record):
        return "{}_{}_{}".format(
            record["relation"], record["object_id"], record["target_id"]
        )

    @property
    def active(self):
        return {key: value for key, value in self.subtasks.items() if value['status'] == 'active'}

    @property
    def completed_subtasks(self):
        return [copy.deepcopy(value) for value in self.subtasks.values() if value['status'] == 'completed']

    @property
    def completed_count(self):
        return len(self.completed_subtasks)

    def public_snapshot(self):
        """Detached public truth; no agent assignment or private plans."""
        return {key: {name: copy.deepcopy(value) for name, value in task.items()
                      if name != 'source_target_id'} for key, task in self.subtasks.items()}

    def task_goal(self):
        """Expose active concrete tasks in the native WAH goal dictionary."""
        return {self._predicate(record): 1 for record in self.active.values()}

    def active_subtasks(self):
        return [copy.deepcopy(record) for record in self.active.values()]

    def _validate_targets(self, graph):
        nodes = _nodes(graph)
        if not self.targets:
            raise ValueError("continuous state-based mode requires at least one target")
        seen = set()
        for target in self.targets:
            required = {"relation", "object_id", "object_class", "capacity"}
            if not required.issubset(target):
                raise ValueError("continuous target lacks required fields: {}".format(target))
            relation = target["relation"].lower()
            target_id = target["object_id"]
            if relation not in SUPPORTED_RELATIONS:
                raise ValueError("unsupported target relation: {}".format(relation))
            if type(target_id) is not int or type(target["capacity"]) is not int or target["capacity"] < 1:
                raise ValueError("target object_id and positive capacity must be integers")
            node = nodes.get(target_id)
            if node is None or node.get("class_name") != target["object_class"]:
                raise ValueError("continuous target does not match scene node: {}".format(target))
            properties = set(node.get("properties", []))
            required_property = "SURFACES" if relation == "on" else "CONTAINERS"
            if required_property not in properties:
                raise ValueError("target {} does not support {}".format(target_id, relation))
            key = (relation, target_id)
            if key in seen:
                raise ValueError("duplicate continuous target: {}".format(key))
            seen.add(key)

    def _new_record(self, object_id, object_class, relation, target_id,
                    target_class, step, source_target_id=None):
        subtask_id = "S{}".format(self.next_subtask_number)
        self.next_subtask_number += 1
        return {
            "subtask_id": subtask_id,
            "object_id": object_id,
            "object_class": object_class,
            "relation": relation,
            "target_id": target_id,
            "target_class": target_class,
            "status": "active",
            "created_step": step,
            "completed_step": None,
            "completed_by": None,
            "source_target_id": source_target_id,
        }

    @staticmethod
    def _current_support(graph, object_id):
        nodes = _nodes(graph)
        for edge in graph["edges"]:
            if (edge["from_id"] == object_id
                    and edge["relation_type"].lower() in SUPPORTED_RELATIONS
                    and nodes.get(edge["to_id"], {}).get("category") != "Rooms"):
                return edge["to_id"]
        return None

    def _initial_records(self, graph, step):
        nodes = _nodes(graph)
        held = _held_object_ids(graph)
        reserved = set()
        records = []
        for predicate, count in self.template_goal.items():
            if count <= 0:
                continue
            relation, object_class, target_id = _parse(predicate)
            if relation not in SUPPORTED_RELATIONS:
                continue
            target = nodes.get(target_id)
            choices = [
                node for node in graph["nodes"]
                if node.get("class_name") == object_class
                and "GRABBABLE" in node.get("properties", [])
                and node["id"] not in held
                and node["id"] not in reserved
                and not _relation_holds(graph, relation, node["id"], target_id)
            ]
            choices.sort(key=lambda node: node["id"])
            for node in choices[:count]:
                records.append(self._new_record(
                    node["id"], object_class, relation, target_id,
                    target.get("class_name") if target else str(target_id), step,
                    self._current_support(graph, node["id"]),
                ))
                reserved.add(node["id"])
        if len(records) > self.active_count:
            raise ValueError(
                "initial task has {} concrete subtasks, exceeding active_count {}"
                .format(len(records), self.active_count)
            )
        return records

    def _target_occupancy(self, graph, relation, target_id):
        nodes = _nodes(graph)
        return sum(
            edge["relation_type"].lower() == relation
            and edge["to_id"] == target_id
            and nodes.get(edge["from_id"], {}).get("class_name") in self.object_classes
            and "GRABBABLE" in nodes.get(edge["from_id"], {}).get("properties", [])
            for edge in graph["edges"]
        )

    def _valid_candidate(self, graph, object_node, target, step):
        object_id = object_node["id"]
        relation = target["relation"].lower()
        target_id = target["object_id"]
        if object_id == target_id or object_id in _held_object_ids(graph):
            return False, "held_or_self"
        if any(record["object_id"] == object_id for record in self.active.values()):
            return False, "object_reserved"
        if _relation_holds(graph, relation, object_id, target_id):
            return False, "already_satisfied"
        cooldown = self.cooldowns.get(object_id)
        if cooldown and step < cooldown["until_step"] and target_id == cooldown["blocked_target_id"]:
            return False, "reverse_cooldown"
        reserved_slots = sum(
            record["relation"] == relation and record["target_id"] == target_id
            for record in self.active.values()
        )
        if self._target_occupancy(graph, relation, target_id) + reserved_slots >= target["capacity"]:
            return False, "target_capacity"
        return True, None

    def _generate_one(self, graph, step):
        objects = [
            node for node in graph["nodes"]
            if node.get("class_name") in self.object_classes
            and "GRABBABLE" in node.get("properties", [])
        ]
        if not objects:
            return None, 0, {"no_eligible_objects": 1}
        rejected = {}
        for attempt in range(1, self.max_generation_attempts + 1):
            object_node = self.random.choice(objects)
            target = self.random.choice(self.targets)
            valid, reason = self._valid_candidate(graph, object_node, target, step)
            if not valid:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            record = self._new_record(
                object_node["id"], object_node["class_name"],
                target["relation"].lower(), target["object_id"],
                target["object_class"], step,
                self._current_support(graph, object_node["id"]),
            )
            return record, attempt, rejected
        return None, self.max_generation_attempts, rejected

    def _fill(self, graph, step):
        generated = []
        attempts = 0
        rejected = {}
        while len(self.active) < self.active_count:
            record, used, reasons = self._generate_one(graph, step)
            attempts += used
            for reason, count in reasons.items():
                rejected[reason] = rejected.get(reason, 0) + count
            if record is None:
                self.generation_failures += 1
                break
            self.subtasks[record["subtask_id"]] = record
            generated.append(copy.deepcopy(record))
        self.last_generation_attempts = attempts
        return generated, attempts, rejected

    def initialize(self, graph, step=0):
        self._validate_targets(graph)
        initial = self._initial_records(graph, step)
        for record in initial:
            self.subtasks[record["subtask_id"]] = record
        generated, attempts, rejected = self._fill(graph, step)
        self.revision += 1
        event = self._event(step, [], initial + generated, attempts, rejected)
        self.events.append(event)
        return copy.deepcopy(event)

    def advance(self, graph, step, actions=None):
        completed = []
        for subtask_id, record in list(self.active.items()):
            if not _relation_holds(
                graph, record["relation"], record["object_id"], record["target_id"]
            ):
                continue
            record['status'] = 'completed'
            record['completed_step'] = step
            record['completed_by'] = self._completion_actor(record, actions or {})
            completed.append(copy.deepcopy(record))
            if record["source_target_id"] is not None and self.reverse_cooldown_steps:
                self.cooldowns[record["object_id"]] = {
                    "blocked_target_id": record["source_target_id"],
                    "until_step": step + self.reverse_cooldown_steps,
                }
        generated, attempts, rejected = self._fill(graph, step)
        changed = bool(completed or generated or attempts)
        if changed:
            self.revision += 1
        event = self._event(step, completed, generated, attempts, rejected)
        if changed:
            self.events.append(event)
        return copy.deepcopy(event)

    @staticmethod
    def _completion_actor(record, actions):
        expected = 'putin' if record['relation'] == 'inside' else 'putback'
        matches = []
        for index, action in actions.items():
            if not isinstance(action, str) or not action.startswith('[' + expected + ']'):
                continue
            ids = [int(value) for value in re.findall(r'\((\d+)\)', action)]
            if len(ids) == 2 and ids == [record['object_id'], record['target_id']]:
                matches.append(index)
        return {0: 'Alice', 1: 'Bob'}.get(matches[0], 'unknown') if len(matches) == 1 else 'unknown'

    def _event(self, step, completed, generated, attempts, rejected):
        shortfall = max(0, self.active_count - len(self.active))
        return {
            "step": step,
            "revision": self.revision,
            "completed": copy.deepcopy(completed),
            "generated": copy.deepcopy(generated),
            "generation_attempts": attempts,
            "rejection_counts": copy.deepcopy(rejected),
            "active_goal": copy.deepcopy(self.task_goal()),
            "active_subtasks": self.active_subtasks(),
            "subtasks": self.public_snapshot(),
            "active_predicates": len(self.active),
            "target_active_predicates": self.active_count,
            "cumulative_completed": self.completed_count,
            "generation_shortfall": shortfall,
            "candidate_exhausted": False,
        }

    def summary(self):
        return {
            "revision": self.revision,
            "completed_subtasks": self.completed_count,
            "completed_subtask_records": copy.deepcopy(self.completed_subtasks),
            "active_goal": copy.deepcopy(self.task_goal()),
            "active_subtasks": self.active_subtasks(),
            "generation_failures": self.generation_failures,
            "candidate_exhausted": False,
            "events": copy.deepcopy(self.events),
            "subtasks": self.public_snapshot(),
        }
