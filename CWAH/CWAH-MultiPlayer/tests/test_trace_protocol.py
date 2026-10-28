"""Real planner parsing and Arena delivery for TRACE-only communication."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from LLM.LLM import LLM
from agents.LLM_agent import LLM_agent
from algos.arena_mp2 import ArenaMP
from trace_tf.information import DecisionInformation
from trace_tf.policy import render_policy_context
from trace_tf.updater import HybridUpdater
from trace_tf.probability import uniform, candidates

TASK = {'subtask_id': 'S1', 'status': 'active', 'created_step': 0,
        'object_id': 12, 'target_id': 20, 'relation': 'on'}


class TraceProtocolTests(unittest.TestCase):
    def test_thinking_output_without_legal_final_action_is_rejected(self):
        p = self.planner('Analysis exhausted the answer budget.')
        p.chat_enable_thinking = True
        with self.assertRaisesRegex(ValueError, 'no legal final action'):
            self.run_planner(p)

    def planner(self, output):
        p = LLM.__new__(LLM)
        p.prompt_template = '$GOAL\n$PROGRESS\n$STEP\n$ACTION_HISTORY\n$ASSIGNED_PERSONALITY\n$DIALOGUE_HISTORY\n$TRACE_BELIEF\n$ACTIONS_REFINE'
        p.goal_desc = 'place apple'
        p.personality = ''
        p.chat = True
        p.sampling_params = {'n': 1}
        p.total_cost = 0
        p.last_usage_metadata = None
        p.lm_id = 'test'
        p.debug = False
        p.oppo_name, p.agent_name = 'Bob', 'Alice'
        p.generator = Mock(return_value=([output], 0))
        return p

    def run_planner(self, p, task='S1', trace=True, actions=('[send_message]',), context=None):
        if context is None:
            context = 'TRACE-v2 teammate belief about Bob' if trace else ''
        information = DecisionInformation(planner_type='active', goal='place apple', personality='',
            current_progress='', available_actions=actions,
            available_actions_with_distance=actions, public_tasks={'S1': TASK})
        return p.run(None, {}, [], [], {}, {}, None, [], [], [], None,
                     belief_context=context, decision_information=information)

    def test_wait_is_explicit_and_has_no_subtask(self):
        actions = ('[send_message]', '[wait]')
        for trace in (False, True):
            raw = ('<cot>No useful action this step.</cot><message></message>'
                   '<action>[wait]</action><subtask_id>null</subtask_id>')
            plan, result = self.run_planner(self.planner(raw), trace=trace, actions=actions)
            self.assertEqual(plan, '[wait]')
            self.assertIsNone(result['selected_subtask_id'])
            self.assertEqual(result['action_parse_status'], 'wait')
            self.assertIn('[wait]', result['raw_prompt'])
            for invalid in ('<subtask_id>S1</subtask_id>',
                            '<subtask_id>null</subtask_id><message>hello</message>'):
                bad = '<action>[wait]</action>' + invalid
                with self.subTest(trace=trace, invalid=invalid), self.assertRaisesRegex(ValueError, 'requires null'):
                    self.run_planner(self.planner(bad), trace=trace, actions=actions)

    def test_high_intent_entropy_injects_prompt_and_records_gate(self):
        output = ('<cot>Act directly.</cot><message></message>'
                  '<action>[wait]</action><subtask_id>null</subtask_id>')
        for entropy, expected in ((0.85, False), (0.86, True)):
            context = json.dumps({'intent': {'value': {'normalized_entropy': entropy}}})
            _, result = self.run_planner(self.planner(output), actions=('[wait]',), context=context)
            self.assertEqual(result['intent_normalized_entropy'], entropy)
            self.assertEqual(result['counterfactual_probe_threshold'], 0.85)
            self.assertEqual(result['counterfactual_probe_prompt_injected'], expected)
            self.assertEqual('Counterfactual Probe Check' in result['raw_prompt'], expected)

    def output(self, kind=None, task='S1', target='auto'):
        if target == 'auto':
            target = 'intent' if kind == 'probe' else None
        return ('<cot>coordinate</cot><message>Please complete S1.</message>'
                '<action>[send_message]</action><subtask_id>{}</subtask_id>'.format(task)
                + ('<communication_type>{}</communication_type>'.format(kind) if kind else '')
                + ('<probe_target>{}</probe_target>'.format(target) if target is not None else ''))

    def test_trace_active_parser_requires_typed_communication(self):
        for kind, task in [('delegate', 'S1'), ('probe', 'null')]:
            plan, result = self.run_planner(self.planner(self.output(kind, task)))
            self.assertTrue(plan.startswith('[send_message]'))
            self.assertEqual(result['trace_communication']['type'], kind)
            if kind == 'probe':
                self.assertEqual(result['trace_communication']['probe_target'], 'intent')
        for kind, task in [(None, 'S1'), ('other', 'S1'), ('delegate', 'null'),
                           ('probe', 'S1'), ('probe_intent', 'null')]:
            with self.subTest(kind=kind, task=task), self.assertRaises(ValueError):
                self.run_planner(self.planner(self.output(kind, task)))
        for raw in (self.output('probe', 'null', None),
                    self.output('probe', 'null', 'other'),
                    self.output('delegate', 'S1', 'intent')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.run_planner(self.planner(raw))

    def test_trace_final_contract_and_retry_feedback(self):
        p = self.planner(self.output('delegate'))
        p.active_retry_error = 'TRACE-v2 send_message requires delegate or probe'
        _, result = self.run_planner(p)
        prompt = result['raw_prompt']
        self.assertIn('FINAL TRACE OUTPUT CONTRACT', prompt)
        self.assertIn('Previous output was rejected: ' + p.active_retry_error, prompt)
        self.assertIn('<communication_type>delegate</communication_type>', prompt)
        p = self.planner(self.output())
        _, result = self.run_planner(p, trace=False)
        self.assertNotIn('FINAL TRACE OUTPUT CONTRACT', result['raw_prompt'])

    def test_trace_off_and_bob_keep_old_message_contract(self):
        p = self.planner(self.output())
        p.agent_name, p.oppo_name = 'Bob', 'Alice'
        plan, result = self.run_planner(p, trace=False)
        self.assertTrue(plan.startswith('[send_message]'))
        self.assertNotIn('trace_communication', result)

    def test_arena_delivery_links_request_and_reply_without_changing_skill(self):
        p = self.planner('unused')
        u = HybridUpdater(p)
        u.ledger.set_active_tasks([TASK])
        alice = LLM_agent.__new__(LLM_agent)
        alice.trace = u
        alice.dialogue_events, alice.dialogue_history = [], []
        alice.agent_names = ['Zero', 'Alice', 'Bob']
        alice.agent_control = {'current_subtask_id': 'S1', 'current_skill': '[gograb] <apple> (12)', 'skill_subtask_id': 'S1'}
        before = alice.control_context()
        bob = SimpleNamespace(receive_dialogue_event=Mock(), reply_to_message=Mock(
            return_value=('I accept responsibility for completing S1.', {})))
        arena = ArenaMP.__new__(ArenaMP)
        arena.agents = [alice, bob]
        arena.env = SimpleNamespace(steps=4, get_observations=lambda: {0: {}, 1: {}})
        u.stage_communication({'type': 'delegate', 'subtask_id': 'S1'})
        arena.process_messages(['Please complete S1.', None])
        self.assertEqual(alice.control_context(), before)
        self.assertEqual(p.generator.call_count, 0)
        request = next(iter(u.requests.values()))
        reply = u.ledger.by_id()[request['reply_event_id']]
        self.assertEqual(reply.payload['reply_to'], request['request_event_id'])
        dist = uniform(candidates({'S1': TASK}))
        p.generator.return_value = ([json.dumps({'intent': {'task_probs': dist['task_probs'],
            'skill_given_task': dist['skill_given_task']}, 'delegation_responses': [{
            'request_event_id': request['request_event_id'], 'reply_event_id': reply.event_id,
            'response': 'accepted'}]})], 0)
        u.update({'planner_type': 'active', 'step': 5, 'public_tasks': {'S1': TASK}})
        self.assertEqual(u.commitments['S1']['status'], 'pending')
        self.assertEqual(alice.control_context(), before)


if __name__ == '__main__':
    unittest.main()
