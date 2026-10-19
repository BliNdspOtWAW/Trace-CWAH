import unittest
from unittest.mock import Mock, patch

from LLM.responses_client import generate_responses


class ResponsesTests(unittest.TestCase):
    @patch('LLM.responses_client.requests.post')
    def test_responses_mapping_and_usage(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {
            'status': 'completed',
            'output': [
                {'type': 'reasoning', 'summary': []},
                {'type': 'message', 'role': 'assistant',
                 'content': [{'type': 'output_text', 'text': '<action>test</action>'}]},
            ],
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30},
        }
        result, usage = generate_responses('https://example.invalid/v1', 'test-key',
            'gpt-5.6-sol', [{'role': 'user', 'content': 'test'}],
            {'n': 1, 'temperature': 0, 'top_p': 1, 'max_tokens': 256})
        self.assertEqual(result, ['<action>test</action>'])
        self.assertEqual(usage['completion_tokens'], 20)
        self.assertEqual(post.call_args.args[0], 'https://example.invalid/v1/responses')
        body = post.call_args.kwargs['json']
        self.assertEqual(body['model'], 'gpt-5.6-sol')
        self.assertEqual(body['reasoning']['effort'], 'high')
        self.assertNotIn('temperature', body)
        self.assertNotIn('max_tokens', body)

    @patch('LLM.responses_client.requests.post')
    def test_missing_key_and_http400_do_not_retry(self, post):
        with self.assertRaises(ValueError):
            generate_responses('https://example.invalid', '', 'test', [], {})
        post.assert_not_called()
        post.return_value = Mock(status_code=400)
        with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
            generate_responses('https://example.invalid', 'test-key', 'test', [], {})
        self.assertEqual(post.call_count, 1)
