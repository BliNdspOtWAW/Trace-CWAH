"""Final rule settlement, empty action semantics and legal observation replay."""
import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from agents.LLM_agent import LLM_agent
from algos.arena_mp2 import ArenaMP
import test_trace_updater as updater_tests
import test_trace_protocol as protocol_tests
from test_trace_updater import TASK, info, obs


class FinalizationTests(unittest.TestCase):
    def accepted(self):
        fixture = updater_tests.TraceUpdaterTests()
        updater, planner = fixture.make()
        response = fixture.delegate(updater)
        fixture.answer(updater, planner, [response])
        updater.update(info())
        return updater, planner

    def test_final_completion_settles_rules_without_decision_or_model_call(self):
        for actor, capability, reliability in [('Bob', 2/3, 2/3), ('Alice', 0.5, 1/3), ('unknown', 0.5, 0.5)]:
            with self.subTest(actor=actor):
                u, p = self.accepted()
                before = u.state.to_dict()
                cursor, decision_count, model_count = u.cursor, u.call_count, p.generator.call_count
                u.ledger.task_completion(8, TASK, actor)
                archived = u.archive_fields()
                stats = archived['final_rule_statistics']
                self.assertAlmostEqual(stats['capability']['mean'], capability)
                self.assertAlmostEqual(stats['reliability']['mean'], reliability)
                self.assertEqual(stats['capability']['failures'], 0)
                self.assertEqual(archived['last_decision_belief'], before)
                self.assertEqual(archived['unconsumed_intent_events'][-1]['kind'], 'task_completion')
                self.assertEqual(archived, u.archive_fields())
                self.assertEqual(u.state.to_dict(), before)
                self.assertEqual((u.cursor, u.call_count, p.generator.call_count), (cursor, decision_count, model_count))
                self.assertEqual(u.commitments['S1']['status'], 'pending')

    def test_checkpoint_exports_both_snapshots_and_unconsumed_events(self):
        u, p = self.accepted()
        u.ledger.task_completion(8, TASK, 'Bob')
        arena = ArenaMP.__new__(ArenaMP)
        arena.env = SimpleNamespace(continuous_subtasks_enabled=False)
        arena.agents = [SimpleNamespace(trace=u)]
        with tempfile.TemporaryDirectory() as folder:
            arena.record_dir = folder
            arena._save_checkpoint({'action': {0: [None], 1: [None]}, 'termination_reason': None})
            with (Path(folder) / 'trajectory_checkpoint.pik').open('rb') as f:
                snapshot = pickle.load(f)['trace']
            self.assertEqual(snapshot['final_rule_statistics']['capability']['successes'], 1)
            self.assertEqual(snapshot['last_decision_belief'], u.state.to_dict())
            self.assertEqual(len(snapshot['unconsumed_intent_events']), 1)

    def test_capability_never_counts_primitive_or_terminal_failure_as_task_failure(self):
        fixture = updater_tests.TraceUpdaterTests()
        u, p = fixture.make()
        u.ledger.add(0, 'peer_task_outcome', {'task': TASK, 'verified': True, 'terminal': True, 'success': False})
        stats = u.rule_statistics()['capability']
        self.assertEqual((stats['successes'], stats['failures'], stats['mean']), (0, 0, 0.5))
        u.ledger.task_completion(1, TASK, 'Bob')
        u.ledger.task_completion(2, TASK, 'Bob')
        stats = u.rule_statistics()['capability']
        self.assertEqual((stats['successes'], stats['failures']), (1, 0))

    def test_unclassified_final_reply_is_explicit_and_not_guessed(self):
        fixture = updater_tests.TraceUpdaterTests()
        u, _ = fixture.make()
        response = fixture.delegate(u)
        snapshot = u.archive_fields()
        self.assertEqual(snapshot['final_rule_statistics']['reliability']['alpha'], 1)
        self.assertEqual(snapshot['final_rule_statistics']['unclassified_delegate_reply_ids'], [response['reply_event_id']])


class EmptyActionTests(unittest.TestCase):
    def test_empty_and_missing_action_cannot_send_payload(self):
        fixture = protocol_tests.TraceProtocolTests()
        for raw in ('<action></action>', '<action>  \n </action>', '<cot>No action</cot>'):
            p = fixture.planner('<message>Please complete S1.</message>' + raw)
            plan, record = fixture.run_planner(p)
            self.assertIsNone(plan)
            self.assertIsNone(record['message'])
            self.assertIsNone(record['selected_subtask_id'])
            self.assertEqual(record['action_parse_status'], 'no_action')
            self.assertNotIn('trace_communication', record)

    def test_agent_no_action_is_one_step_and_replans_on_next_step(self):
        def node(i, name, category, properties=()):
            return {'id': i, 'class_name': name, 'category': category, 'properties': list(properties), 'states': []}
        graph = {'nodes': [node(1, 'character', 'Characters'), node(2, 'character', 'Characters'),
            node(11, 'kitchen', 'Rooms'), node(72, 'kitchentable', 'Furniture', ['SURFACES']),
            node(375, 'apple', 'Objects', ['GRABBABLE']), node(370, 'pudding', 'Objects', ['GRABBABLE'])],
            'edges': [{'from_id': i, 'to_id': 11, 'relation_type': 'INSIDE'} for i in (1, 2, 72, 375, 370)]
            + [{'from_id': 1, 'to_id': i, 'relation_type': 'CLOSE'} for i in (72, 375, 370)]}
        args = SimpleNamespace(debug=False, act=False, big_5=None, teammate_profile_file=None,
            trace_mode='off', obs_type='partial', source='debug', lm_id='test',
            prompt_template_path='unused', communication=True, cot=False, api_base=None, api_key=None)
        agent = LLM_agent(1, 0, args)
        observation = dict(copy.deepcopy(graph), location=None, messages=[None, None])
        goal = {'on_375_<kitchentable> (72)': [1], 'on_370_<kitchentable> (72)': [1]}
        agent.reset(observation, [], ['apple', 'pudding'], ['kitchen'], [graph['nodes'][2]], goal, 'test')
        from test_task_control import tasks
        agent.set_public_tasks(tasks())
        agent.LLM.chat = True
        agent.LLM.sampling_params = {'n': 1}
        agent.LLM.generator = Mock(side_effect=[(['<action></action>'], 0),
            (['<message></message><action>[gograb] <apple> (375)</action><subtask_id>S3</subtask_id>'], 0)])
        action, record = agent.get_action(observation, goal)
        self.assertIsNone(action)
        self.assertIsNone(agent.plan)
        self.assertEqual(agent.steps, 1)
        self.assertTrue(record['LLM']['no_action'])
        action, _ = agent.get_action(observation, goal)
        self.assertEqual(action, '[grab] <apple> (375)')
        self.assertEqual(agent.LLM.generator.call_count, 2)




if __name__ == '__main__':
    unittest.main()
