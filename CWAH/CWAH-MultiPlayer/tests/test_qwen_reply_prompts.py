import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiment_runner'))
from run_experiment import _reply_prompt_path


class QwenReplyPromptTests(unittest.TestCase):
    def test_local_qwen_uses_role_specific_formatting_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.json').write_text(json.dumps({'model_type': 'qwen3'}))
            for role in ('alice', 'bob'):
                path = _reply_prompt_path('huggingface', directory, role)
                self.assertEqual(path.name, f'prompt_Reply_Qwen_{role.title()}.txt')
                original = path.with_name(f'prompt_Reply_{role.title()}.txt').read_text()
                self.assertEqual(path.read_text().split('\nFormatting requirements:')[0], original.rstrip() + '\n')
                self.assertIn('literal closing tag </reply>', path.read_text())

    def test_api_models_keep_original_prompts(self):
        self.assertEqual(_reply_prompt_path('openai', 'gpt-5.6-sol', 'alice').name,
                         'prompt_Reply_Alice.txt')
        self.assertEqual(_reply_prompt_path('openai', 'deepseek-ai/DeepSeek-V3.1', 'bob').name,
                         'prompt_Reply_Bob.txt')


if __name__ == '__main__':
    unittest.main()
