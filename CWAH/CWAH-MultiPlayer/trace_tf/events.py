"""Episode-local evidence that is legal for the target (Alice) to observe."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .model import _json_copy


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    step: int
    kind: str
    payload: Dict[str, Any]

    def to_dict(self):
        return {"event_id": self.event_id, "step": self.step,
                "kind": self.kind, "payload": _json_copy(self.payload)}


class EvidenceLedger:
    """Never accepts full graphs, peer plans, profile labels, or evaluator info."""

    def __init__(self, target_id=1, peer_id=2):
        self.target_id = target_id
        self.peer_id = peer_id
        self.events: List[TraceEvent] = []
        self._observed_steps = set()
        self._prior_observation: Optional[Dict[str, Any]] = None
        self._active_tasks: List[Dict[str, Any]] = []
        self._known_tasks: Dict[str, Dict[str, Any]] = {}

    def add(self, step, kind, payload):
        if kind not in ("observation", "message", "own_action", "peer_task_progress",
                        "peer_task_outcome", "task_completion", "own_outcome"):
            raise ValueError("unsupported TRACE evidence kind")
        event = TraceEvent("e{:05d}".format(len(self.events)), int(step), kind,
                           _json_copy(payload))
        self.events.append(event)
        return event

    def set_active_tasks(self, tasks):
        """Provide only the public active-task snapshot used by both agents."""
        self._active_tasks = _json_copy(list(tasks or []))
        for task in self._active_tasks:
            if task.get('subtask_id'):
                self._known_tasks[task['subtask_id']] = self._task_payload(task)

    def subtask_created_step(self, subtask_id):
        task = self._known_tasks.get(subtask_id, {})
        created = task.get('created_step')
        if type(created) is int and created >= 0:
            return created
        for event in self.events:
            task = event.payload.get('task') or {}
            if task.get('subtask_id') == subtask_id:
                created = task.get('created_step')
                if type(created) is int and created >= 0:
                    return created
        return None

    def _task_for_object(self, object_id):
        return next(
            (task for task in self._active_tasks if task.get("object_id") == object_id),
            None,
        )

    def known_subtask_ids(self):
        """IDs visible on the public task board or its completion history."""
        result = {task.get("subtask_id") for task in self._active_tasks}
        result.update(
            (event.payload.get("task") or {}).get("subtask_id")
            for event in self.events if event.kind == "task_completion"
        )
        return {item for item in result if isinstance(item, str) and item}

    @staticmethod
    def _task_payload(task):
        if not task:
            return None
        return {
            "subtask_id": task.get("subtask_id"),
            "created_step": task.get("created_step"),
            "object_id": task.get("object_id"),
            "object_class": task.get("object_class"),
            "relation": task.get("relation"),
            "target_id": task.get("target_id"),
            "target_class": task.get("target_class"),
        }

    def observe(self, step, observation):
        """Only the target's partial observation, never env.full_graph."""
        if step in self._observed_steps:
            return
        if "nodes" not in observation or "edges" not in observation:
            raise ValueError("TRACE requires a partial graph observation")
        visible = {"nodes": observation["nodes"], "edges": observation["edges"],
                   "location": observation.get("location")}
        visible = _json_copy(visible)
        self.add(step, "observation", visible)
        self._observed_steps.add(step)
        if self._prior_observation is not None:
            self._extract_visible_changes(step, self._prior_observation, visible)
        self._prior_observation = visible

    def _extract_visible_changes(self, step, prior, current):
        before = {(e["from_id"], e["to_id"], e["relation_type"]) for e in prior["edges"]}
        after = {(e["from_id"], e["to_id"], e["relation_type"]) for e in current["edges"]}
        nodes = {n["id"]: n for n in current["nodes"]}
        prior_nodes = {n["id"]: n for n in prior["nodes"]}
        for source, target, relation in sorted(after - before):
            obj = nodes.get(target) if source == self.peer_id and relation.startswith("HOLDS") else nodes.get(source)
            if obj is None:
                continue
            if source == self.peer_id and relation.startswith("HOLDS"):
                task = self._task_for_object(target)
                if task:
                    self.add(step, "peer_task_progress", {
                        "actor": "Bob", "status": "in_progress",
                        "observed_action": "grab", "task": self._task_payload(task),
                    })
                self.add(step, "peer_task_outcome", {"action": "grab", "object": obj["class_name"],
                         "object_id": target, "success": True, "verified": True})
            elif relation in ("ON", "INSIDE") and any(
                edge[0] == self.peer_id and edge[1] == source and edge[2].startswith("HOLDS")
                for edge in before
            ) and source in prior_nodes:
                task = self._task_for_object(source)
                self.add(step, "peer_task_outcome", {"action": "place", "object": obj["class_name"],
                         "object_id": source, "relation": relation.lower(),
                         "target_id": target,
                         "target_class": nodes.get(target, prior_nodes.get(target, {})).get("class_name"),
                         "task": self._task_payload(task) if task else None,
                         "success": True, "verified": True})

    def task_completion(self, step, task, completed_by="unknown"):
        if completed_by not in ("Alice", "Bob", "unknown"):
            raise ValueError("completed_by must be Alice, Bob, or unknown")
        return self.add(step, "task_completion", {
            "task": self._task_payload(task),
            "completed_by": completed_by,
        })

    def message(self, step, sender, receiver, text, kind):
        if sender not in (0, 1) or receiver not in (0, 1) or kind not in ("active", "reply"):
            raise ValueError("invalid delivered message")
        return self.add(step, "message", {"sender": sender + 1, "receiver": receiver + 1,
                                          "kind": kind, "text": text})

    def own_action(self, step, action):
        return self.add(step, "own_action", {"action": action, "status": "selected_not_executed"})

    def since(self, cursor):
        return self.events[cursor:]

    def by_id(self):
        return {event.event_id: event for event in self.events}
