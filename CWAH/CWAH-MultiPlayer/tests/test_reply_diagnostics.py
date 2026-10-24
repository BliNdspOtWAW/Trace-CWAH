"""Malformed replies retain raw diagnostics without changing acceptance rules."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from LLM.communication import CommunicationModule, ReplyFormatError


class ReplyDiagnosticsTests(unittest.TestCase):
    def planner(self, output):
        return SimpleNamespace(agent_name='Alice', oppo_name='Bob', agent_id=1,
                               goal_desc='Place an apple.', personality=None,
                               chat=True, sampling_params={'max_tokens': 1024},
                               lm_id='test', last_usage_metadata=None,
                               generator=Mock(return_value=([output], 0)))

    def test_invalid_output_is_preserved(self):
        planner = self.planner('<message>Understood.</message>')
        with self.assertRaises(ReplyFormatError) as raised:
            CommunicationModule(planner).reply('Please help.', [], None, 'kitchen', {})
        call = raised.exception.call
        self.assertEqual(call['raw_output'], '<message>Understood.</message>')
        self.assertEqual(call['parse_status'], 'error')
        self.assertIn('Please help.', call['prompt'])
        self.assertTrue(call['metrics_call_id'])
        self.assertEqual(planner.generator.call_args.args[1]['max_tokens'], 256)

    def test_valid_output_keeps_original_reply(self):
        planner = self.planner('<intention>Search.</intention><reply>Understood.</reply>')
        reply, call = CommunicationModule(planner).reply('Please help.', [], None, 'kitchen', {})
        self.assertEqual(reply, 'Understood.')
        self.assertEqual(call['parse_status'], 'ok')
        self.assertIsNone(call['parse_error'])


if __name__ == '__main__':
    unittest.main()
