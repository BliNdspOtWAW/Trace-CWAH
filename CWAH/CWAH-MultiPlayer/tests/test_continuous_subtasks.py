"""Continuous subtask scheduling tests without Unity or model calls."""

import unittest

from envs.continuous_subtasks import ContinuousSubtaskManager


def node(node_id, class_name, properties):
    return {"id": node_id, "class_name": class_name, "properties": properties, "states": []}


TARGETS = [
    {"relation": "on", "object_id": 72, "object_class": "kitchentable", "capacity": 4},
    {"relation": "on", "object_id": 268, "object_class": "coffeetable", "capacity": 4},
    {"relation": "inside", "object_id": 103, "object_class": "fridge", "capacity": 4},
]


class ContinuousSubtaskTests(unittest.TestCase):
    def setUp(self):
        self.graph = {
            "nodes": [
                node(1, "character", []),
                node(2, "character", []),
                node(72, "kitchentable", ["SURFACES"]),
                node(268, "coffeetable", ["SURFACES"]),
                node(103, "fridge", ["CONTAINERS", "CAN_OPEN"]),
                node(10, "apple", ["GRABBABLE"]),
                node(11, "juice", ["GRABBABLE"]),
                node(12, "pudding", ["GRABBABLE"]),
            ],
            "edges": [],
        }

    def manager(self, goal, active_count=1, **kwargs):
        return ContinuousSubtaskManager(
            goal, active_count=active_count, seed=1, targets=TARGETS,
            max_generation_attempts=kwargs.pop("max_generation_attempts", 64),
            reverse_cooldown_steps=kwargs.pop("reverse_cooldown_steps", 20),
            **kwargs
        )

    def test_initial_goals_reserve_concrete_objects(self):
        manager = self.manager(
            {"on_apple_72": 1, "on_juice_72": 1}, active_count=2
        )

        event = manager.initialize(self.graph)

        self.assertEqual(set(event["active_goal"]), {"on_10_72", "on_11_72"})
        self.assertEqual({item["object_id"] for item in event["active_subtasks"]}, {10, 11})
        self.assertFalse(event["candidate_exhausted"])

    def test_completion_randomly_replenishes_from_current_state(self):
        manager = self.manager({"on_apple_72": 1}, active_count=1)
        manager.initialize(self.graph)
        self.graph["edges"].append(
            {"from_id": 10, "to_id": 72, "relation_type": "ON"}
        )

        event = manager.advance(self.graph, step=7)

        self.assertEqual(event["completed"][0]["object_id"], 10)
        self.assertEqual(len(event["generated"]), 1)
        generated = event["generated"][0]
        self.assertIn(generated["object_id"], {10, 11, 12})
        self.assertIn(generated["target_id"], {72, 103, 268})
        self.assertFalse(
            generated["object_id"] == 10 and generated["target_id"] == 72
        )
        self.assertEqual(manager.completed_count, 1)

    def test_held_and_reserved_objects_are_not_generated(self):
        self.graph["edges"].append(
            {"from_id": 2, "to_id": 11, "relation_type": "HOLDS_RH"}
        )
        manager = self.manager({"on_apple_72": 1}, active_count=2)

        event = manager.initialize(self.graph)

        ids = [item["object_id"] for item in event["active_subtasks"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn(11, ids)

    def test_failed_sampling_does_not_exhaust_or_terminate(self):
        only_table = [
            {"relation": "on", "object_id": 72, "object_class": "kitchentable", "capacity": 1}
        ]
        manager = ContinuousSubtaskManager(
            {"on_apple_72": 1}, active_count=2, seed=1, targets=only_table,
            max_generation_attempts=3,
        )

        event = manager.initialize(self.graph)

        self.assertEqual(event["active_predicates"], 1)
        self.assertEqual(event["generation_shortfall"], 1)
        self.assertEqual(event["generation_attempts"], 3)
        self.assertFalse(event["candidate_exhausted"])

        retry = manager.advance(self.graph, step=1)
        self.assertEqual(retry["generation_attempts"], 3)
        self.assertFalse(retry["candidate_exhausted"])

    def test_direct_reverse_is_temporarily_blocked(self):
        manager = self.manager(
            {"on_apple_72": 1}, active_count=1,
            reverse_cooldown_steps=20,
        )
        # Record that the apple originally came from the coffee table.
        self.graph["edges"].append(
            {"from_id": 10, "to_id": 268, "relation_type": "ON"}
        )
        manager.initialize(self.graph)
        self.graph["edges"] = [
            edge for edge in self.graph["edges"] if edge["from_id"] != 10
        ]
        self.graph["edges"].append(
            {"from_id": 10, "to_id": 72, "relation_type": "ON"}
        )

        manager.advance(self.graph, step=5)

        self.assertEqual(manager.cooldowns[10]["blocked_target_id"], 268)
        manager.active = {
            key: value for key, value in manager.active.items()
            if value["object_id"] != 10
        }
        valid, reason = manager._valid_candidate(
            self.graph, next(node for node in self.graph["nodes"] if node["id"] == 10),
            TARGETS[1], step=6,
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "reverse_cooldown")

    def test_task_manager_never_adds_trace_or_assignment_fields(self):
        manager = self.manager({"on_apple_72": 1})
        event = manager.initialize(self.graph)
        rendered = repr(event)
        self.assertNotIn("assigned", rendered)
        self.assertNotIn("intent", rendered)
        self.assertNotIn("trace", rendered.lower())


if __name__ == "__main__":
    unittest.main()
