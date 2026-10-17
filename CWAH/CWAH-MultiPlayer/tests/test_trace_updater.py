"""TRACE updater safety and state tests; no Unity or network."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from trace_tf.updater import HybridUpdater
from trace_tf.policy import render_policy_context
from trace_tf.information import DecisionInformation
from LLM.LLM import LLM


def observation(edges=()):
    return {
        "nodes": [
            {"id": 1, "class_name": "character", "category": "Characters"},
            {"id": 2, "class_name": "character", "category": "Characters"},
            {"id": 12, "class_name": "apple", "category": "Objects"},
        ],
        "edges": list(edges), "location": [0, 0, 0],
        "hidden_profile": "PRIVATE_PROFILE_NEVER_ENTER_PROMPT",
        "peer_private_plan": "PRIVATE_PLAN_NEVER_ENTER_PROMPT",
    }


def planner_output(acceptance_id=None, evidence_ids=()):
    return {
        "intent": {"action": "gograb", "target": "apple", "evidence_ids": list(evidence_ids)},
        "preference": {"description": None, "evidence_ids": []},
        "commitments": ([{
            "acceptance_event_id": acceptance_id, "action": "grab", "object": "apple",
            "evidence_ids": [acceptance_id], "proposed_status": "fulfilled",
        }] if acceptance_id else []),
        "uncertainty": {"intent": 0.0, "capability": 0.0,
                        "preference": 0.0, "reliability": 0.0},
    }


class TraceUpdaterTests(unittest.TestCase):
    def make_updater(self, answer):
        planner = SimpleNamespace(
            sampling_params={"n": 1, "max_tokens": 1024, "temperature": 0},
            chat=True, lm_id="test", last_usage_metadata={"total_tokens": 20},
            generator=Mock(return_value=([json.dumps(answer)], 0)),
        )
        return HybridUpdater(planner), planner

    def test_one_call_four_uncertainties_and_no_private_fields(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.observe(0, observation())
        state, record = updater.update()
        self.assertEqual(planner.generator.call_count, 1)
        self.assertEqual(state.dimensions["uncertainty"].value["overall"], 1.0)
        self.assertEqual(set(state.dimensions["uncertainty"].value["per_dimension"]),
                         {"intent", "capability", "preference", "reliability"})
        self.assertIn("e00000", state.dimensions["information"].value["event_ids"])
        self.assertIn("PRIVATE_PROFILE_NEVER_ENTER_PROMPT", str(observation()))
        self.assertNotIn("PRIVATE_PROFILE_NEVER_ENTER_PROMPT", record["llm_call"]["prompt"])
        self.assertNotIn("PRIVATE_PLAN_NEVER_ENTER_PROMPT", render_policy_context(state, updater.ledger))
        updater.update()
        self.assertEqual(planner.generator.call_count, 1)

    def test_self_report_does_not_fulfill_commitment_but_visible_grab_does(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.observe(0, observation())
        message = updater.ledger.message(0, 1, 0, "Alice, I will grab the apple.", "active")
        planner.generator.return_value = ([json.dumps(planner_output(message.event_id, [message.event_id]))], 0)
        state, _ = updater.update()
        reliability = state.dimensions["reliability"].value
        self.assertIsNone(reliability["fulfillment_rate"])
        self.assertEqual(reliability["commitment_events"][0]["status"], "pending")
        self.assertEqual(state.dimensions["uncertainty"].value["per_dimension"]["reliability"], 1.0)

        # Bob saying "done" is still only a claim.
        updater.ledger.message(1, 1, 0, "I grabbed it", "reply")
        state, _ = updater.update()
        self.assertIsNone(state.dimensions["reliability"].value["fulfillment_rate"])

        updater.ledger.observe(2, observation([{"from_id": 2, "to_id": 12,
                                                 "relation_type": "HOLDS_RH"}]))
        state, _ = updater.update()
        self.assertEqual(state.dimensions["capability"].value["success_rate"], 1.0)
        self.assertEqual(state.dimensions["reliability"].value["fulfillment_rate"], 1.0)
        self.assertEqual(state.dimensions["reliability"].value["commitment_events"][0]["status"], "fulfilled")
        self.assertGreaterEqual(state.dimensions["uncertainty"].value["overall"], 0.25)

    def test_silence_and_unaccepted_request_are_not_violations(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.observe(0, observation())
        updater.ledger.message(0, 0, 1, "Please search the kitchen", "active")
        state, _ = updater.update()
        self.assertEqual(planner.generator.call_count, 1)
        self.assertEqual(state.dimensions["reliability"].status, "unknown")
        self.assertEqual(state.dimensions["capability"].status, "unknown")
        self.assertEqual(state.dimensions["uncertainty"].value["overall"], 1.0)

    def test_intent_returns_to_unknown_when_new_evidence_no_longer_supports_it(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.observe(0, observation())
        message = updater.ledger.message(0, 1, 0, "I will grab the apple", "active")
        planner.generator.return_value = ([json.dumps(planner_output(None, [message.event_id]))], 0)
        first, _ = updater.update()
        self.assertEqual(first.dimensions["intent"].status, "supported")
        unknown = planner_output()
        unknown["intent"] = {"action": None, "target": None, "evidence_ids": []}
        planner.generator.return_value = ([json.dumps(unknown)], 0)
        updater.ledger.observe(1, observation())
        second, _ = updater.update()
        self.assertEqual(second.dimensions["intent"].status, "unknown")
        self.assertEqual(second.dimensions["uncertainty"].value["per_dimension"]["intent"], 1.0)

    def test_alice_request_alone_cannot_establish_bob_intent(self):
        updater, planner = self.make_updater(planner_output())
        request = updater.ledger.message(0, 0, 1, "Bob, please grab the apple", "active")
        planner.generator.return_value = ([json.dumps(planner_output(None, [request.event_id]))], 0)
        state, _ = updater.update()
        self.assertEqual(state.dimensions["intent"].status, "unknown")

    def test_planner_information_is_stored_and_sent_to_updater(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.observe(7, observation([{"from_id": 2, "to_id": 12,
                                                "relation_type": "HOLDS_RH"}]))
        information = DecisionInformation(
            planner_type="active",
            goal="put apple on table",
            personality="curious",
            current_progress="Bob is visible holding apple (12)",
            action_history=("[goexplore] <kitchen> (11)",),
            dialogue_history=("Bob: I found the apple.",),
            available_actions=("[goput] <apple> (12) <table> (20)",),
            step=7,
            source_event_ids=("e00000",),
        ).to_dict()

        state, record = updater.update(information)

        stored = state.dimensions["information"].value
        self.assertEqual(stored["goal"], "put apple on table")
        self.assertEqual(stored["planner_type"], "active")
        prompt = record["llm_call"]["prompt"]
        self.assertIn("Information shared with the planner", prompt)
        self.assertIn("put apple on table", prompt)
        self.assertIn("Bob visibly holds <apple> (12)", prompt)
        self.assertNotIn("$COMMITMENTS", prompt)

    def test_commitments_live_inside_reliability_and_capability_is_rule_scored(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.observe(0, observation())
        message = updater.ledger.message(0, 1, 0, "I will grab the apple.", "active")
        planner.generator.return_value = ([json.dumps(
            planner_output(message.event_id, [message.event_id])
        )], 0)
        updater.update({"goal": "put apple on table"})
        self.assertFalse(hasattr(updater, "commitments"))
        reliability = updater.state.dimensions["reliability"].value
        self.assertEqual(reliability["commitment_events"][0]["status"], "pending")

        updater.ledger.observe(1, observation([{"from_id": 2, "to_id": 12,
                                                "relation_type": "HOLDS_RH"}]))
        state, _ = updater.update({"goal": "put apple on table"})
        self.assertEqual(state.dimensions["capability"].uncertainty, 0.5)
        self.assertEqual(
            state.dimensions["reliability"].value["commitment_events"][0]["status"],
            "fulfilled",
        )

    def test_policy_context_is_injected_into_existing_planner_prompt(self):
        planner = LLM.__new__(LLM)
        planner.prompt_template = "$GOAL\n$PROGRESS\n$STEP\n$ACTION_HISTORY\n$ASSIGNED_PERSONALITY\n$DIALOGUE_HISTORY\n$TRACE_BELIEF\n$ACTIONS_REFINE"
        planner.goal_desc = "place apple"
        planner.personality = ""
        planner.chat = True
        planner.sampling_params = {"n": 1}
        planner.total_cost = 0
        planner.last_usage_metadata = None
        planner.debug = False
        planner.oppo_name = "Bob"
        planner.agent_name = "Alice"
        planner.get_available_plans = Mock(return_value=("", "A. [goexplore]", [], ["[goexplore]"], 1))
        planner.generator = Mock(return_value=(["<action>A. [goexplore]</action>"], 0))
        planner.parse_answer = Mock(return_value=("[goexplore] <kitchen> (11)", "", None))
        planner.progress2text = Mock(return_value="none")
        _, info = planner.run(None, {}, [], [], {}, {}, None, [], [], [], None,
                              belief_context="TRACE_TEST_CONTEXT")
        self.assertIn("TRACE_TEST_CONTEXT", info["raw_prompt"])
        self.assertEqual(info["trace_belief_context"], "TRACE_TEST_CONTEXT")
        self.assertEqual(info["shared_information"]["goal"], "place apple")


if __name__ == "__main__":
    unittest.main()
