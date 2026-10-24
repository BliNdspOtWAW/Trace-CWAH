import unittest
from unittest.mock import Mock, patch

from LLM.responses_client import generate_responses


class ResponsesTests(unittest.TestCase):
    @patch('LLM.responses_client.time.sleep')
    @patch('LLM.responses_client.requests.post')
    def test_detailed_usage_and_retry_metrics(self, post, sleep):
        success = Mock(status_code=200)
        success.json.return_value = {
            'status': 'completed',
            'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': 'OK'}]}],
            'usage': {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30,
                      'input_tokens_details': {'cached_tokens': 3},
                      'output_tokens_details': {'reasoning_tokens': 7}},
        }
        post.side_effect = [Mock(status_code=429), success]
        result, usage = generate_responses('https://example.invalid', 'test', 'test', [], {}, max_retries=1)
        self.assertEqual(usage['retry_count'], 1)
        self.assertEqual(usage['request_count'], 2)
        self.assertEqual(usage['cached_input_tokens'], 3)
        self.assertEqual(usage['reasoning_tokens'], 7)
        self.assertTrue(usage['token_usage_available'])
        self.assertEqual(result, ['OK'])

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
