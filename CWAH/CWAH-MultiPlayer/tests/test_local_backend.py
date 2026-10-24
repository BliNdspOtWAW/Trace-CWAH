import unittest
import json
import pickle
import sys
import tempfile
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from LLM import local_backend
from LLM.LLM import LLM


class Inputs(dict):
    def to(self, device):
        return self


class LocalBackendTests(unittest.TestCase):
    def test_local_runner_needs_no_api_and_preserves_backend(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiment_runner'))
        import run_experiment
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'config.json').write_text('{}')
            (folder / 'model.safetensors').touch()
            dataset = folder / 'dataset.pik'
            with dataset.open('wb') as stream:
                pickle.dump([dict(task_id=44, task_name='prepare_food', env_id=3,
                                 init_graph={}, task_goal={})], stream)
            executable = folder / 'unity'
            executable.touch()
            executable.chmod(0o700)
            config = dict(backend='huggingface', local_model={'thinking': False},
                          dataset={'path': str(dataset), 'episode_index': 0},
                          executable_file=str(executable), bob_profile=None, trace_mode='on',
                          alice_model=str(folder), bob_model=str(folder), temperature=0,
                          max_tokens=256, max_episode_length=5, num_runs=1)
            config_path = folder / 'experiment.json'
            config_path.write_text(json.dumps(config))
            with patch.dict('os.environ', {}, clear=True), \
                    patch.object(run_experiment, 'allocated_port', return_value=nullcontext(6390)), \
                    patch.object(run_experiment, '_run_validated', return_value=0) as run:
                self.assertEqual(run_experiment.main(['--config', str(config_path)]), 0)
                self.assertEqual(run.call_args.args[1]['backend'], 'huggingface')
            config['local_model']['thinking'] = 'false'
            with self.assertRaisesRegex(ValueError, 'boolean'):
                run_experiment.validate(config)

    def backend(self, raw='<action>[goexplore] <kitchen> (11)</action>\n<message></message>'):
        backend = local_backend.LocalBackend.__new__(local_backend.LocalBackend)
        backend.options = {'thinking': False, 'context_limit': 128}
        backend.tokenizer = Mock(eos_token_id=2)
        backend.tokenizer.apply_chat_template.return_value = 'formatted'
        backend.tokenizer.return_value = Inputs(input_ids=torch.tensor([[1, 2, 3]]),
                                                attention_mask=torch.tensor([[1, 1, 1]]))
        backend.tokenizer.decode.return_value = raw
        backend.model = Mock(device='cpu', config=SimpleNamespace(max_position_embeddings=128))
        backend.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        return backend

    def test_full_multiline_output_and_greedy_parameters(self):
        backend = self.backend()
        output, usage = backend.generate([{'role': 'user', 'content': 'test'}],
                                         {'temperature': 0, 'max_tokens': 20, 'n': 1})
        self.assertIn('\n', output[0])
        self.assertEqual(usage['total_tokens'], 5)
        self.assertFalse(backend.model.generate.call_args.kwargs['do_sample'])
        self.assertNotIn('temperature', backend.model.generate.call_args.kwargs)
        self.assertFalse(backend.tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_context_is_not_silently_truncated(self):
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, 'no history was truncated'):
            backend.generate('test', {'max_tokens': 200})
        backend.model.generate.assert_not_called()

    def test_reasoning_is_not_passed_to_action_parser(self):
        backend = self.backend('<think>private reasoning</think>\n<reply>OK</reply>')
        output, usage = backend.generate('test', {'max_tokens': 20})
        self.assertEqual(output, ['<reply>OK</reply>'])
        self.assertIn('private reasoning', usage['raw_outputs'][0])

    def test_weights_shared_without_sharing_prompts(self):
        with patch.dict(local_backend._BACKENDS, {}, clear=True), \
                patch.object(local_backend, 'LocalBackend') as constructor:
            self.assertIs(local_backend.get_backend('/tmp/model', {}),
                          local_backend.get_backend('/tmp/model', {}))
            constructor.assert_called_once()

    def test_all_planner_calls_use_shared_chat_interface(self):
        args = SimpleNamespace(debug=False, act=False, big_5=None, api_base=None,
                               api_key=None, max_tokens=128, t=0, top_p=1, n=1)
        backend = Mock()
        backend.generate.return_value = (['<action>test</action>'], {'total_tokens': 5})
        with patch('LLM.local_backend.get_backend', return_value=backend):
            planner = LLM('huggingface', '/tmp/model', 'unused', True, True, args, 1)
            self.assertTrue(planner.chat)
            planner.generator([{'role': 'user', 'content': 'test'}], planner.sampling_params)
            self.assertEqual(planner.last_usage_metadata['total_tokens'], 5)


if __name__ == '__main__':
    unittest.main()
