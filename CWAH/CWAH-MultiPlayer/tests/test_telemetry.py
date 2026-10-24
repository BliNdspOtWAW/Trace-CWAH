import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from LLM.telemetry import measured_generate

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiment_runner'))
from metric_reports import export_metrics


class TelemetryTests(unittest.TestCase):
    def planner(self):
        return SimpleNamespace(
            agent_name='Alice', agent_id=1, lm_id='test-model', source='openai',
            wire_api='responses', responses_output_budget=8192,
            last_usage_metadata={'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                                 'cached_input_tokens': 50, 'reasoning_tokens': 10,
                                 'request_count': 2, 'retry_count': 1},
            generator=Mock(return_value=(['<action>test</action>\n<message></message>'], 0)))

    def test_passthrough_sizes_usage_and_reports(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                'CWAH_METRICS_PATH': str(Path(temporary) / 'llm_calls.jsonl')}):
            planner = self.planner()
            request = [{'role': 'user', 'content': 'private-prompt'}]
            params = {'max_tokens': 256, 'n': 1}
            information = {'step': 51, 'dialogue_history': ['one', 'two'], 'action_history': ['go']}
            result = measured_generate(planner, request, params, 'active', information,
                                       {'dialogue_history': 'one\ntwo'})
            self.assertIs(result[0], planner.generator.return_value[0])
            self.assertIs(planner.generator.call_args.args[0], request)
            self.assertIs(planner.generator.call_args.args[1], params)
            text = (Path(temporary) / 'llm_calls.jsonl').read_text()
            self.assertNotIn('private-prompt', text)
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual([r['event'] for r in rows], ['call_start', 'call_end'])
            self.assertEqual(rows[-1]['effective_max_output_tokens'], 8192)
            self.assertEqual(rows[-1]['context_counts']['dialogue_history'], 2)
            self.assertEqual(rows[-1]['section_chars']['dialogue_history'], 7)
            summary = export_metrics(temporary)
            self.assertEqual(summary['overall']['metrics']['prompt_tokens']['total'], 100)
            self.assertEqual(summary['step_windows'][0]['start_step'], '50')
            self.assertEqual(summary['overall']['metrics']['retry_count']['total'], 1)
            self.assertFalse((Path(temporary) / 'analysis').exists())

    def test_error_preserved_without_stale_usage_or_secret_message(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {
                'CWAH_METRICS_PATH': str(Path(temporary) / 'llm_calls.jsonl')}):
            planner = self.planner()
            planner.generator.side_effect = RuntimeError('secret-do-not-log')
            with self.assertRaisesRegex(RuntimeError, 'secret-do-not-log'):
                measured_generate(planner, 'prompt', {}, 'updater', {'step': 5})
            self.assertIsNone(planner.last_call_metrics['prompt_tokens'])
            text = (Path(temporary) / 'llm_calls.jsonl').read_text()
            self.assertNotIn('secret-do-not-log', text)
            summary = export_metrics(temporary)
            self.assertEqual(summary['overall']['generation_errors'], 1)
            self.assertIsNone(summary['overall']['metrics']['prompt_tokens']['total'])

    def test_inflight_and_incomplete_line_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'llm_calls.jsonl'
            path.write_text(json.dumps({'event': 'call_start', 'call_id': 'pending'}) + '\n{"event":')
            summary = export_metrics(temporary)
            self.assertEqual(summary['in_flight_call_ids'], ['pending'])
            self.assertEqual(summary['malformed_lines'], 1)

    def test_absent_usage_is_not_zero_cost(self):
        planner = self.planner()
        planner.last_usage_metadata = {'prompt_tokens': 0, 'completion_tokens': 0,
                                       'token_usage_available': False}
        with patch.dict(os.environ, {}, clear=True):
            measured_generate(planner, 'prompt', {}, 'reactive')
        self.assertIsNone(planner.last_call_metrics['prompt_tokens'])
        self.assertIsNone(planner.last_call_metrics['completion_tokens'])

    def test_summary_only_does_not_create_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'calls.jsonl'
            path.write_text(json.dumps({'event': 'environment_step_start', 'step': 1}) + '\n')
            summary = export_metrics(temporary)
            self.assertEqual(summary['reported_environment_step_starts'], 1)
            self.assertEqual(set(Path(temporary).iterdir()), {path})


if __name__ == '__main__':
    unittest.main()
