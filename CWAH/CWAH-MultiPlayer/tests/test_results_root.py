import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import test_experiment_runner as runner_tests

runner = runner_tests.run_experiment


class ResultsRootTests(unittest.TestCase):
    def test_versioned_trace_and_baseline_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(runner, 'TRACE_ROOT', root), patch.object(runner, 'CWAH_ROOT', root/'CWAH'):
                resolved = runner.resolve_results_root('CWAH/experiments/tracev2')
                self.assertEqual(resolved, root/'CWAH/experiments/tracev2')
                self.assertEqual(runner.resolve_results_root('tmp_v1'),
                                 root/'CWAH/experiments/tmp_v1')
                for mode, category in [('on', 'trace'), ('off', 'baseline')]:
                    selected = {'results_root': resolved, 'trace_mode': mode,
                                'alice_model': 'qwen3.8-27b', 'bob_model': 'qwen3.8-27b',
                                'profile_id': 'peer3_Self-Directed', 'max_episode_length': 200}
                    self.assertEqual(runner.result_directory(selected, 'run'),
                                     resolved/'qwen'/'qwen3.8-27b_peer3_Self-Directed_200step'/category/'run')

    def test_default_unchanged_and_external_or_empty_roots_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(runner, 'TRACE_ROOT', root), patch.object(runner, 'CWAH_ROOT', root/'CWAH'):
                self.assertEqual(runner.resolve_results_root(), root/'CWAH/experiments')
                for value in ('', '../outside', str(root/'other'), 'CWAH/experiments/../../other'):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        runner.resolve_results_root(value)

    def test_symlink_cannot_escape_experiment_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            experiments = root/'CWAH/experiments'
            experiments.mkdir(parents=True)
            outside = root/'outside'
            outside.mkdir()
            (experiments/'escape').symlink_to(outside, target_is_directory=True)
            with patch.object(runner, 'TRACE_ROOT', root), patch.object(runner, 'CWAH_ROOT', root/'CWAH'):
                with self.assertRaises(ValueError):
                    runner.resolve_results_root('CWAH/experiments/escape')
