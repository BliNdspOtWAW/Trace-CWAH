"""TRACE metadata cannot enter the delivered message or override controls."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import test_trace_protocol as protocol_tests
from LLM.active_protocol import parse_trace_active
from agents.LLM_agent import LLM_agent
from algos.arena_mp2 import ArenaMP
from trace_tf.updater import HybridUpdater


def output(message='Please complete S1.', kind='delegate', task='S1'):
    return ('<cot>Coordinate.</cot><communication_type>{}</communication_type>'
            '<message>{}</message><action>[send_message]</action>'
            '<subtask_id>{}</subtask_id>').format(kind, message, task)


class MessageBoundaryTests(unittest.TestCase):
    def test_clean_delivery_preserves_ordinary_object_labels(self):
        fixture = protocol_tests.TraceProtocolTests()
        body = 'Please put <apple> (12) on <table> (20) for S1.'
        plan, record = fixture.run_planner(fixture.planner(output(body)))
        self.assertEqual(plan, '[send_message] <' + body + '>')
        self.assertEqual(record['message'], body)
        self.assertEqual(record['trace_communication'], {'type': 'delegate', 'subtask_id': 'S1'})

    def test_logged_nested_layout_is_rejected_before_delivery(self):
        fixture = protocol_tests.TraceProtocolTests()
        raw = ('<cot>Coordinate.</cot><message><communication_type>delegate</communication_type>'
               '<subtask_id>S1</subtask_id>Please complete S1.</message>'
               '<action>[send_message]</action><subtask_id>S1</subtask_id>')
        with self.assertRaisesRegex(ValueError, 'internal protocol tags'):
            fixture.run_planner(fixture.planner(raw))

    def test_valid_root_metadata_does_not_allow_nested_controls(self):
        for tag in ('communication_type', 'subtask_id', 'action', 'new_subtask_id', 'reply'):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                parse_trace_active(output('<' + tag + '>S2</' + tag + '>Hello'), ['[send_message]'])

    def test_duplicate_root_fields_are_rejected_even_if_identical(self):
        for extra in ('<subtask_id>S1</subtask_id>', '<subtask_id>S2</subtask_id>',
                      '<communication_type>delegate</communication_type>',
                      '<action>[send_message]</action>', '<message>Other</message>'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                parse_trace_active(output() + extra, ['[send_message]'])

    def test_tags_in_cot_cannot_override_top_level_fields(self):
        raw = output().replace('<cot>Coordinate.</cot>',
            '<cot><communication_type>probe_intent</communication_type><subtask_id>S2</subtask_id>Reason</cot>')
        parsed = parse_trace_active(raw, ['[send_message]'])
        self.assertEqual(parsed['communication_type'], 'delegate')
        self.assertEqual(parsed['subtask_id'], 'S1')
        self.assertEqual(parsed['message'], 'Please complete S1.')

    def test_action_only_inside_message_is_not_executable(self):
        raw = '<message><action>[send_message]</action>Hello</message><subtask_id>S1</subtask_id>'
        parsed = parse_trace_active(raw, ['[send_message]'])
        self.assertIsNone(parsed['plan'])
        self.assertIsNone(parsed['message'])

    def test_unclosed_protocol_tags_cannot_be_delivered(self):
        for message in ('<subtask_id>S1 Hello', '<communication_type>delegate Hello'):
            with self.subTest(message=message), self.assertRaises(ValueError):
                parse_trace_active(output(message), ['[send_message]'])

    def test_empty_action_stays_a_noop_without_protocol_validation(self):
        parsed = parse_trace_active('<message><subtask_id>S1</subtask_id>Hello</message><action></action>', ['[send_message]'])
        self.assertIsNone(parsed['plan'])
        self.assertIsNone(parsed['message'])

    def test_physical_action_cannot_carry_communication_type(self):
        raw = output().replace('[send_message]', '[gograb] <apple> (12)')
        with self.assertRaises(ValueError):
            parse_trace_active(raw, ['[gograb] <apple> (12)'])

    def test_option_labels_select_actions_without_message_scanning(self):
        actions = ['[gograb] <apple> (12)', '[send_message]']
        for label in ('B', '(B)', 'B. [send_message]'):
            with self.subTest(label=label):
                parsed = parse_trace_active(output().replace('[send_message]', label), actions)
                self.assertEqual(parsed['plan'], '[send_message]')

    def test_arena_delivers_prose_but_keeps_metadata_local(self):
        fixture = protocol_tests.TraceProtocolTests()
        body = 'Please complete S1 with <apple> (12).'
        planner = fixture.planner(output(body))
        plan, record = fixture.run_planner(planner)
        updater = HybridUpdater(planner)
        updater.ledger.set_active_tasks([protocol_tests.TASK])
        alice = LLM_agent.__new__(LLM_agent)
        alice.trace = updater
        alice.dialogue_events, alice.dialogue_history = [], []
        alice.agent_names = ['Zero', 'Alice', 'Bob']
        bob = SimpleNamespace(receive_dialogue_event=Mock(), reply_to_message=Mock(return_value=('Understood.', {})))
        arena = ArenaMP.__new__(ArenaMP)
        arena.agents = [alice, bob]
        arena.env = SimpleNamespace(steps=0, get_observations=lambda: {0: {}, 1: {}})
        updater.stage_communication(record['trace_communication'])
        delivered = plan[plan.find('<')+1:plan.rfind('>')]
        events = arena.process_messages([delivered, None])
        self.assertEqual(events[0]['text'], body)
        self.assertEqual(bob.reply_to_message.call_args.args[0], body)
        request = next(iter(updater.requests.values()))
        self.assertEqual(request['type'], 'delegate')
        self.assertEqual(request['subtask_id'], 'S1')
        self.assertNotIn('communication_type', bob.reply_to_message.call_args.args[0])
        self.assertNotIn('<subtask_id>', bob.reply_to_message.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
