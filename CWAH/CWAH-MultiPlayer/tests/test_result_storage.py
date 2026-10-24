"""Result-format, interruption and repeat tests; no Unity, GPU or LLM."""
import json
import os
import pickle
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiment_runner'))
import run_experiment


FIVE = {'config.json', 'summary.json', 'calls.jsonl', 'trajectory.pik', 'console.log'}


def dump(path, value):
    with path.open('wb') as stream:
        pickle.dump(value, stream)


def base_run(root):
    (root / '.runtime').mkdir()
    (root / 'config.json').write_text('{}')
    (root / 'calls.jsonl').touch()
    (root / 'console.log').touch()


def trajectory(repeat=0, interrupted=False):
    return dict(run_index=repeat, action={0: ['grab', 'put'], 1: [None, None]},
                obs={0: [{'visible_object': repeat}], 1: []}, LLM={0: [], 1: []},
                environment_steps=2, interrupted=interrupted,
                termination_reason='interrupted' if interrupted else 'step_budget',
                trace={'evidence_events': [{'event_id': 'e001'}],
                       'belief_updates': [{'intent': 'put'}], 'final_belief': {'intent': 'put'},
                       'updater_calls': 1, 'updater_token_usage': {}},
                continuous_subtasks={'completed_subtasks': repeat + 3,
                                     'completed_subtask_records': [{'subtask_id': 'S1'}],
                                     'events': [{'completed': ['S1']}], 'active_subtasks': []})


def finished_raw(root, repeat=0, interrupted=False):
    data = trajectory(repeat, interrupted)
    dump(root / '.runtime' / f'logs_agent_44_prepare_food_{repeat}.pik', data)
    dump(root / '.runtime' / 'trajectory_checkpoint.pik', data)
    return data


def evaluation(root, count):
    dump(root / '.runtime' / 'results.pik', {44: {'S': [0] * count, 'L': [2] * count}})


class ResultStorageTests(unittest.TestCase):
    def finish(self, root, code=0, repeats=1):
        return run_experiment._finish_run(root, 44, code, repeats, time.monotonic())

    def test_normal_run_is_exactly_five_files_and_full_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_run(root)
            original = finished_raw(root)
            evaluation(root, 1)
            self.assertEqual(self.finish(root), 0)
            self.assertEqual({p.name for p in root.iterdir()}, FIVE)
            with (root / 'trajectory.pik').open('rb') as stream:
                archive = pickle.load(stream)
            self.assertEqual(archive['runs'][0]['trace'], original['trace'])
            self.assertEqual(archive['runs'][0]['continuous_subtasks']['events'], original['continuous_subtasks']['events'])
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'completed')
            self.assertEqual(summary['successes'], 0)  # Pipeline completion is not task success.
            self.assertEqual(summary['completed_subtasks'], 3)
            self.assertNotIn('events', summary['continuous_subtasks'])
            self.assertIsNone(summary['error'])

    def test_multiple_repeats_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_run(root)
            finished_raw(root, 0)
            finished_raw(root, 1)
            evaluation(root, 2)
            self.assertEqual(self.finish(root, repeats=2), 0)
            with (root / 'trajectory.pik').open('rb') as stream:
                archive = pickle.load(stream)
            self.assertEqual([r['run_index'] for r in archive['runs']], [0, 1])
            self.assertEqual(archive['runs'][0]['obs'][0], [{'visible_object': 0}])
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['completed_subtasks'], 7)
            self.assertEqual({p.name for p in root.iterdir()}, FIVE)

    def test_interrupt_keeps_checkpoint_and_partial_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_run(root)
            dump(root / '.runtime' / 'trajectory_checkpoint.pik', trajectory(interrupted=True))
            self.assertEqual(self.finish(root), 1)
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'interrupted')
            self.assertTrue((root / 'trajectory_checkpoint.pik').is_file())
            with (root / 'trajectory.pik').open('rb') as stream:
                self.assertTrue(pickle.load(stream)['runs'][0]['partial'])

    def test_failed_second_repeat_retains_both_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_run(root)
            finished_raw(root, 0)
            evaluation(root, 1)
            dump(root / '.runtime' / 'trajectory_checkpoint.pik', trajectory(1))
            self.assertEqual(self.finish(root, code=1, repeats=2), 1)
            with (root / 'trajectory.pik').open('rb') as stream:
                runs = pickle.load(stream)['runs']
            self.assertEqual([r['run_index'] for r in runs], [0, 1])
            self.assertTrue(runs[1]['partial'])
            self.assertEqual(json.loads((root / 'summary.json').read_text())['status'], 'failed')

    def test_unknown_scratch_file_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_run(root)
            finished_raw(root)
            evaluation(root, 1)
            unknown = root / '.runtime' / 'future-debug.bin'
            unknown.write_bytes(b'keep')
            self.finish(root)
            self.assertEqual(unknown.read_bytes(), b'keep')

    def test_unreadable_trajectory_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_run(root)
            bad = root / '.runtime' / 'logs_agent_44_prepare_food_0.pik'
            bad.write_bytes(b'not-pickle')
            evaluation(root, 1)
            self.assertEqual(self.finish(root), 1)
            self.assertTrue(bad.exists())
            self.assertEqual(json.loads((root / 'summary.json').read_text())['error']['type'], 'artifact_error')

    def test_failure_before_first_step_still_has_five_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_run(root)
            self.assertEqual(self.finish(root, code=1), 1)
            self.assertEqual({p.name for p in root.iterdir()}, FIVE)
            with (root / 'trajectory.pik').open('rb') as stream:
                self.assertEqual(pickle.load(stream)['runs'], [])

    def test_runner_uses_scratch_and_produces_five_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            dataset = folder / 'dataset.pik'
            dump(dataset, [{'task_id': 44, 'task_name': 'prepare_food', 'env_id': 3,
                            'init_graph': {}, 'task_goal': {}}])
            executable = folder / 'unity'
            executable.touch()
            executable.chmod(0o700)
            config = dict(dataset={'path': str(dataset), 'episode_index': 0},
                          executable_file=str(executable), bob_profile=None, trace_mode='off',
                          alice_model='test', bob_model='test', max_tokens=100,
                          max_episode_length=2, num_runs=1, temperature=0)
            selected = run_experiment.validate(config)
            def child(command, **kwargs):
                scratch = Path(command[command.index('--record_dir') + 1])
                self.assertEqual(scratch.name, '.runtime')
                self.assertEqual(Path(kwargs['env']['CWAH_METRICS_PATH']).name, 'calls.jsonl')
                root = scratch.parent
                finished_raw(root)
                dump(scratch / 'results.pik', {0: {'S': [0], 'L': [2]}})
                return SimpleNamespace(stdout=['episode: 0\n', 'Step 0, Executing script: []\n'],
                                       wait=lambda: 0, poll=lambda: 0)
            with patch.object(run_experiment, 'CWAH_ROOT', folder / 'CWAH'), \
                    patch.object(run_experiment.subprocess, 'Popen', side_effect=child), \
                    patch.object(run_experiment, '_git_revision', return_value='test'), \
                    patch.object(run_experiment, '_git_dirty', return_value=True), \
                    patch.dict(os.environ, {'CWAH_AGENT_API_BASE': 'https://example.invalid',
                                            'CWAH_HUMAN_API_BASE': 'https://example.invalid'}):
                result = run_experiment._run_validated(config, selected, selected['episode'], 0, 6390)
            self.assertEqual(result, 0)
            root = next((folder / 'CWAH/experiments/baseline').iterdir())
            self.assertEqual({p.name for p in root.iterdir()}, FIVE)
            events = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            self.assertEqual(events[0]['event'], 'environment_step_start')


if __name__ == '__main__':
    unittest.main()
