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
            "acceptance_event_id": acceptance_id,
            "subtask_id": "subtask-apple-table",
            "action": "grab", "object": "apple",
            "evidence_ids": [acceptance_id], "proposed_status": "fulfilled",
        }] if acceptance_id else []),
        "uncertainty": {"intent": 0.0, "capability": 0.0,
                        "preference": 0.0, "reliability": 0.0},
    }


TASK = {
    "subtask_id": "subtask-apple-table", "object_id": 12,
    "created_step": 0,
    "object_class": "apple", "relation": "on", "target_id": 20,
    "target_class": "kitchentable",
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
        updater.ledger.set_active_tasks([TASK])
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
        updater.ledger.task_completion(3, TASK, "Bob")
        state, _ = updater.update()
        self.assertEqual(state.dimensions["capability"].value["success_rate"], 1.0)
        self.assertEqual(state.dimensions["reliability"].value["fulfillment_rate"], 1.0)
        self.assertEqual(state.dimensions["reliability"].value["commitment_events"][0]["status"], "fulfilled")
        self.assertGreaterEqual(state.dimensions["uncertainty"].value["overall"], 0.25)

    def test_silence_and_unaccepted_request_are_not_violations(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.observe(0, observation())
        updater.ledger.set_active_tasks([TASK])
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
        updater.ledger.set_active_tasks([TASK])
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
        updater.ledger.task_completion(2, TASK, "Bob")
        state, _ = updater.update({"goal": "put apple on table"})
        self.assertEqual(state.dimensions["capability"].uncertainty, 0.5)
        self.assertEqual(
            state.dimensions["reliability"].value["commitment_events"][0]["status"],
            "fulfilled",
        )

    def test_commitment_is_settled_only_by_the_bound_subtask(self):
        updater, planner = self.make_updater(planner_output())
        other = dict(TASK, subtask_id="other-apple-task", target_id=21,
                     target_class="coffeetable")
        updater.ledger.set_active_tasks([TASK, other])
        message = updater.ledger.message(
            0, 1, 0, "I will grab the apple for subtask-apple-table.", "active"
        )
        planner.generator.return_value = ([json.dumps(
            planner_output(message.event_id, [message.event_id])
        )], 0)
        updater.update({"goal": "[subtask-apple-table] apple to table"})

        updater.ledger.add(1, "peer_task_progress", {
            "actor": "Bob", "status": "in_progress", "observed_action": "grab",
            "task": other,
        })
        pending, _ = updater.update({"goal": "two apple tasks"})
        commitment = pending.dimensions["reliability"].value["commitment_events"][0]
        self.assertEqual(commitment["subtask_id"], TASK["subtask_id"])
        self.assertEqual(commitment["status"], "pending")

        updater.ledger.add(2, "peer_task_progress", {
            "actor": "Bob", "status": "in_progress", "observed_action": "grab",
            "task": TASK,
        })
        fulfilled, _ = updater.update({"goal": "two apple tasks"})
        commitment = fulfilled.dimensions["reliability"].value["commitment_events"][0]
        self.assertEqual(commitment["status"], "fulfilled")

    def test_capability_tracks_progress_then_verified_task_success(self):
        updater, _ = self.make_updater(planner_output())
        updater.ledger.set_active_tasks([TASK])
        updater.ledger.observe(0, observation())
        updater.update({"goal": "put apple on table"})
        updater.ledger.observe(1, observation([{
            "from_id": 2, "to_id": 12, "relation_type": "HOLDS_RH",
        }]))

        progressing, _ = updater.update({"goal": "put apple on table"})
        capability = progressing.dimensions["capability"].value
        self.assertIsNone(capability["success_rate"])
        self.assertEqual(capability["task_execution_events"][0]["status"], "in_progress")
        self.assertEqual(capability["task_execution_events"][0]["outcome"], "unknown")

        updater.ledger.task_completion(2, TASK, "Bob")
        completed, _ = updater.update({"goal": "another active task"})
        capability = completed.dimensions["capability"].value
        self.assertEqual(capability["success_rate"], 1.0)
        self.assertEqual(capability["task_execution_events"][0]["status"], "completed")
        self.assertEqual(capability["task_execution_events"][0]["outcome"], "success")

    def test_repeated_commitment_merges_and_counts_fulfillment_once(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.set_active_tasks([TASK])
        first = updater.ledger.message(0, 1, 0, 'I will grab the apple.', 'active')
        second = updater.ledger.message(1, 1, 0, 'I will grab it as agreed.', 'active')
        answer = planner_output(first.event_id, [first.event_id])
        answer['commitments'] += planner_output(second.event_id)['commitments']
        planner.generator.return_value = ([json.dumps(answer)], 0)
        updater.update()
        records = updater.state.dimensions['reliability'].value['commitment_events']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['acceptance_event_id'], first.event_id)
        self.assertEqual(records[0]['accepted_step'], 0)
        self.assertIn(second.event_id, records[0]['evidence_ids'])

        updater.ledger.add(2, 'peer_task_progress', {
            'task': TASK, 'observed_action': 'grab', 'status': 'in_progress',
        })
        state, _ = updater.update()
        self.assertEqual(state.dimensions['reliability'].value['fulfillments'], 1)
        third = updater.ledger.message(3, 1, 0, 'As promised, I grabbed it.', 'reply')
        planner.generator.return_value = ([json.dumps(planner_output(third.event_id))], 0)
        state, _ = updater.update()
        value = state.dimensions['reliability'].value
        self.assertEqual(len(value['commitment_events']), 1)
        self.assertEqual(value['fulfillments'], 1)
        self.assertEqual(value['commitment_events'][0]['status'], 'fulfilled')

    def test_old_message_cannot_commit_to_future_subtask_or_fulfill_it(self):
        updater, _ = self.make_updater(planner_output())
        task = dict(TASK, subtask_id='S24', created_step=190)
        old = updater.ledger.message(142, 1, 0, 'Cupcake for the fridge.', 'active')
        updater.ledger.set_active_tasks([task])
        proposal = dict(planner_output(old.event_id)['commitments'][0], subtask_id='S24')
        commitments = {}
        updater._accept_commitments([proposal], commitments)
        updater.ledger.add(193, 'peer_task_progress', {'task': task, 'observed_action': 'grab'})
        updater._settle_commitments(commitments)
        self.assertEqual(commitments, {})

        valid = updater.ledger.message(190, 1, 0, 'I will handle S24.', 'active')
        proposal.update(acceptance_event_id=valid.event_id, evidence_ids=[valid.event_id])
        updater._accept_commitments([proposal], commitments)
        self.assertEqual(commitments['Bob:S24:grab']['accepted_step'], 190)
        updater._settle_commitments(commitments)
        self.assertEqual(commitments['Bob:S24:grab']['status'], 'fulfilled')

        # An older re-extracted declaration must not replace the valid acceptance.
        proposal.update(acceptance_event_id=old.event_id, evidence_ids=[old.event_id])
        updater._accept_commitments([proposal], commitments)
        self.assertEqual(commitments['Bob:S24:grab']['accepted_step'], 190)

    def test_creation_time_survives_task_retirement_and_missing_time_is_rejected(self):
        updater, _ = self.make_updater(planner_output())
        task = dict(TASK, subtask_id='S24', created_step=190)
        updater.ledger.set_active_tasks([task])
        updater.ledger.task_completion(200, task, 'Bob')
        updater.ledger.set_active_tasks([])
        self.assertEqual(updater.ledger.subtask_created_step('S24'), 190)
        updater.ledger.set_active_tasks([dict(TASK, subtask_id='missing', created_step=None)])
        message = updater.ledger.message(201, 1, 0, 'I will handle it.', 'active')
        proposal = dict(planner_output(message.event_id)['commitments'][0], subtask_id='missing')
        commitments = {}
        updater._accept_commitments([proposal], commitments)
        self.assertEqual(commitments, {})

    def test_alice_completion_closes_pending_without_violation(self):
        updater, planner = self.make_updater(planner_output())
        updater.ledger.set_active_tasks([TASK])
        message = updater.ledger.message(0, 1, 0, 'I will grab and place the apple.', 'active')
        answer = planner_output(message.event_id)
        answer['commitments'].append(dict(answer['commitments'][0], action='place'))
        planner.generator.return_value = ([json.dumps(answer)], 0)
        updater.update()
        updater.ledger.task_completion(1, TASK, 'Alice')
        state, _ = updater.update()
        value = state.dimensions['reliability'].value
        self.assertEqual(len(value['commitment_events']), 2)
        self.assertTrue(all(item['status'] == 'closed' and item['outcome'] == 'unknown'
                            for item in value['commitment_events']))
        self.assertEqual(value['fulfillments'], 0)
        self.assertEqual(value['violations'], 0)
        self.assertIsNone(value['fulfillment_rate'])

    def test_alice_completion_does_not_turn_bob_progress_into_failure(self):
        updater, _ = self.make_updater(planner_output())
        updater.ledger.set_active_tasks([TASK])
        updater.ledger.observe(0, observation())
        updater.ledger.observe(1, observation([{
            "from_id": 2, "to_id": 12, "relation_type": "HOLDS_RH",
        }]))
        updater.ledger.task_completion(2, TASK, "Alice")

        state, _ = updater.update({"goal": "another active task"})

        capability = state.dimensions["capability"].value
        execution = capability["task_execution_events"][0]
        self.assertEqual(execution["status"], "abandoned")
        self.assertEqual(execution["outcome"], "unknown")
        self.assertEqual(capability["failures"], 0)
        self.assertIsNone(capability["success_rate"])

    def test_preference_every_third_call_and_null_rescores_existing_description(self):
        updater, planner = self.make_updater(planner_output())
        for index in range(1, 10):
            answer = planner_output()
            answer['preference'] = {'description': 'Frequently proposes task divisions.'
                                    if index <= 3 else None}
            answer['uncertainty']['preference'] = {3: 0.6, 6: 0.2, 9: 0.4}.get(index, 0.0)
            planner.generator.return_value = ([json.dumps(answer)], 0)
            updater.ledger.own_action(index, 'walk')
            state, record = updater.update({'planner_type': 'reactive' if index % 2 else 'active'})
            self.assertEqual(record['preference_assessment_due'], index % 3 == 0)
            if index < 3:
                self.assertEqual(state.dimensions['preference'].status, 'unknown')
            else:
                self.assertEqual(state.dimensions['preference'].value['description'],
                                 'Frequently proposes task divisions.')
                expected = 0.6 if index < 6 else 0.2 if index < 9 else 0.4
                self.assertEqual(state.dimensions['preference'].uncertainty, expected)
                self.assertEqual(state.dimensions['uncertainty'].value['per_dimension']['preference'], expected)
        self.assertEqual(planner.generator.call_count, 9)
        updater.update()
        self.assertEqual(updater.call_count, 9)

    def test_preference_assessment_keeps_all_intermediate_evidence_and_unknown_score(self):
        updater, planner = self.make_updater(planner_output())
        first = updater.ledger.message(0, 1, 0, 'UNIQUE_EARLY_BEHAVIOR', 'active')
        updater.update()
        for i in range(25):
            updater.ledger.message(1, 1, 0, 'Later message {}'.format(i), 'active')
        updater.update()
        updater.ledger.own_action(2, 'walk')
        state, record = updater.update()
        self.assertIn('UNIQUE_EARLY_BEHAVIOR', record['llm_call']['prompt'])
        self.assertIn(first.event_id, record['llm_call']['prompt'])
        self.assertEqual(state.dimensions['preference'].status, 'unknown')
        self.assertEqual(state.dimensions['uncertainty'].value['per_dimension']['preference'], 1.0)

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
