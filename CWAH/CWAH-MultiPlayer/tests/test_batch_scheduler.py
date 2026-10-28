"""Batch runner checks without Unity or remote Qwen services."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


RUNNER = Path(__file__).resolve().parents[2] / 'experiment_runner'
sys.path.insert(0, str(RUNNER))
from batch_scheduler import run as scheduler


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


class BatchSchedulerTests(unittest.TestCase):
    def test_real_manifest_has_twelve_matched_jobs(self):
        folder = RUNNER / 'configs/tracev2/continuous/qwen/peer1_5_200step'
        bid, cap, attempts, jobs, _ = scheduler.load_batch(folder / 'batch.json.example')
        self.assertEqual((bid, cap, attempts, len(jobs)),
                         ('qwen_peer1_5_200step_v1', 3, 1, 12))
        for number in range(1, 6):
            own = [j for j in jobs if j['id'].startswith(f'peer{number}_')]
            self.assertEqual(len(own), 3 if number in (4, 5) else 2)
            self.assertEqual({j['config']['max_episode_length'] for j in own}, {200})
            self.assertEqual({j['config']['continuous_subtasks']['seed'] for j in own}, {1})
            self.assertEqual(sum(j['no_counterfactual'] for j in own), 1 if number in (4, 5) else 0)

    def test_ablation_switch_is_per_job_and_result_paths_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = save(root / 'config.json', {'backend': 'openai',
                'alice_model': 'qwen3.8-27b', 'bob_model': 'qwen3.8-27b',
                'trace_mode': 'on', 'num_runs': 1, 'port': 'auto'})
            batch = save(root / 'batch.json', {'version': 1, 'batch_id': 'smoke',
                'limits': {'max_experiments': 2, 'max_unity_instances': 2},
                'jobs': [{'id': 'trace', 'config': str(config)},
                         {'id': 'ablation', 'config': str(config), 'no_counterfactual': True}]})
            registry = save(root / 'replicas.json', {'version': 1, 'replicas': [
                {'id': 'ppu12', 'api_base': 'http://127.0.0.1:18044/v1',
                 'models': ['qwen3.8-27b'], 'max_experiments': 1},
                {'id': 'ppu14', 'api_base': 'http://127.0.0.1:18046/v1',
                 'models': ['qwen3.8-27b'], 'max_experiments': 1}]})
            fake = root / 'fake.py'
            fake.write_text('''import json, os, pathlib, sys
config = pathlib.Path(sys.argv[sys.argv.index('--config') + 1])
assert os.environ['CWAH_AGENT_API_BASE'] == os.environ['CWAH_HUMAN_API_BASE']
ablation = os.environ['CWAH_DISABLE_COUNTERFACTUAL_PROBE'] == '1'
prefix = os.environ['CWAH_RUN_NAME_PREFIX']
assert prefix == ('no_counterfactual_' if ablation else '')
result = config.parent / ('result_ablation' if ablation else 'result_trace')
result.mkdir()
(result / 'summary.json').write_text(json.dumps({'status': 'completed', 'ablation': ablation}))
print('results=' + str(result), flush=True)
''', encoding='utf-8')
            state_dir = root / 'experiments' / 'smoke'
            with patch.object(scheduler, 'RESULTS', root / 'experiments'), \
                 patch.object(scheduler, 'SINGLE_RUNNER', fake), \
                 patch.object(scheduler, 'validate', return_value={'port_request': 'auto'}), \
                 patch.object(scheduler, 'healthy', return_value=True), \
                 patch.dict(os.environ, {'CWAH_AGENT_API_KEY': 'test',
                                         'CWAH_HUMAN_API_KEY': 'test'}):
                self.assertEqual(scheduler.execute(batch, registry, state_dir, poll=0.01), 0)
            state = json.loads((state_dir / 'state.json').read_text())
            self.assertEqual({x['status'] for x in state['jobs'].values()}, {'completed'})
            self.assertEqual({x['attempts'][0]['replica'] for x in state['jobs'].values()},
                             {'ppu12', 'ppu14'})
            self.assertNotEqual(state['jobs']['trace']['attempts'][0]['result_dir'],
                                state['jobs']['ablation']['attempts'][0]['result_dir'])


if __name__ == '__main__':
    unittest.main()
