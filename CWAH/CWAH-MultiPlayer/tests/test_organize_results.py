"""Migration keeps long runs and removes only runs below the step threshold."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiment_runner'))
import organize_results


class OrganizeResultsTests(unittest.TestCase):
    def test_steps_and_trajectory_references_survive_move(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            short = root / 'tracev2' / 'old' / 'short_20261027-100000-000001'
            long = root / 'tracev2' / 'old' / 'long_20261027-100000-000002'
            config = {'alice_model': 'qwen3.8-27b', 'bob_model': 'qwen3.8-27b',
                      'bob_profile_id': 'peer3_Self-Directed', 'max_episode_length': 200,
                      'trace_mode': 'on', 'run_id': 'old-id'}
            for run, steps in ((short, 49), (long, 50)):
                run.mkdir(parents=True)
                (run / 'config.json').write_text(json.dumps(config))
                (run / 'summary.json').write_text(json.dumps({
                    'status': 'completed', 'steps': [steps],
                    'trajectories': ['trajectory.pik']}))
                (run / 'trajectory.pik').write_bytes(b'trajectory bytes')
            outside = root / 'baseline' / 'short_20261027-100000-000003'
            outside.mkdir(parents=True)
            (outside / 'config.json').write_text(json.dumps(config))
            (outside / 'summary.json').write_text(json.dumps({'status': 'completed', 'steps': [1]}))
            with patch.object(organize_results, 'EXPERIMENTS', root):
                plans = organize_results.planned_runs(50)
                self.assertEqual([p[0] for p in plans].count('delete'), 1)
                self.assertEqual([p[0] for p in plans].count('move'), 1)
                organize_results.apply_plan(plans)
            self.assertFalse(short.exists())
            self.assertTrue(outside.exists())
            kept = next((root / 'tracev2/qwen/qwen3.8-27b_peer3_Self-Directed_200step/trace').iterdir())
            self.assertTrue((kept / (kept.name + '.pik')).is_file())
            self.assertEqual(json.loads((kept / 'summary.json').read_text())['trajectories'],
                             [kept.name + '.pik'])
            self.assertEqual(json.loads((kept / 'config.json').read_text())['run_id'], 'old-id')


if __name__ == '__main__':
    unittest.main()
