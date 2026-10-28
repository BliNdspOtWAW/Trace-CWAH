"""The two CWAH players must read their own uppercase environment settings."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from LLM.LLM import LLM, resolve_profile_key


class LLMRoleConfigTests(unittest.TestCase):
    def test_qwen_can_enable_alice_active_without_updater_or_reactive(self):
        args = SimpleNamespace(debug=False, act=False, big_5=None,
                               api_base='http://127.0.0.1:18038/v1', api_key='unused',
                               max_tokens=2048, t=0, top_p=1, n=1)
        values = {'CWAH_AGENT_CHAT_ENABLE_THINKING': '0',
                  'CWAH_AGENT_CHAT_THINKING_ACTIVE': '1',
                  'CWAH_AGENT_CHAT_THINKING_UPDATER': '0',
                  'CWAH_AGENT_CHAT_THINKING_REACTIVE': '0'}
        responses = [
            {'choices': [{'finish_reason': 'stop', 'message': {
                'content': '<think>brief</think><action>[wait]</action>'}}]},
            {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"intent":{}}'}}]},
            {'choices': [{'finish_reason': 'stop', 'message': {'content': '<reply>OK</reply>'}}]},
        ]
        with patch.dict(os.environ, values, clear=True), \
                patch('openai.ChatCompletion.create', side_effect=responses) as chat_api:
            planner = LLM('openai', 'qwen3.8-27b', 'unused', True, False, args, 1)
            request = [{'role': 'user', 'content': 'test'}]
            self.assertEqual(planner.generator(request, planner.sampling_params)[0],
                             ['<action>[wait]</action>'])
            self.assertEqual(planner.generator(
                request, dict(planner.sampling_params, _trace_call_type='updater'))[0],
                ['{"intent":{}}'])
            self.assertEqual(planner.generator(
                request, dict(planner.sampling_params, _trace_call_type='reactive'))[0],
                ['<reply>OK</reply>'])
        self.assertEqual(chat_api.call_args_list[0].kwargs['thinking_token_budget'], 512)
        self.assertNotIn('thinking_token_budget', chat_api.call_args_list[1].kwargs)
        self.assertNotIn('thinking_token_budget', chat_api.call_args_list[2].kwargs)

    def test_qwen_thinking_uses_chat_template_kwargs_and_removes_thought(self):
        args = SimpleNamespace(debug=False, act=False, big_5=None,
                               api_base='http://127.0.0.1:18038/v1', api_key='unused',
                               max_tokens=128, t=0, top_p=1, n=1)
        response = {'choices': [{'message': {'content': '<think>analysis</think><action>[wait]</action>'}}],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        with patch.dict(os.environ, {'CWAH_AGENT_CHAT_ENABLE_THINKING': '1'}, clear=True), \
                patch('openai.ChatCompletion.create', return_value=response) as chat_api:
            planner = LLM('openai', 'qwen3.8-27b', 'unused', True, False, args, 1)
            output, _ = planner.generator([{'role': 'user', 'content': 'test'}], planner.sampling_params)
        self.assertEqual(output, ['<action>[wait]</action>'])
        self.assertEqual(planner.last_raw_chat_outputs,
                         ['<think>analysis</think><action>[wait]</action>'])
        self.assertEqual(chat_api.call_args.kwargs['chat_template_kwargs'], {'enable_thinking': True})
        self.assertEqual(chat_api.call_args.kwargs['thinking_token_budget'], 512)
        self.assertEqual(chat_api.call_args.kwargs['reasoning_effort'], 'low')

    def test_qwen_updater_budget_and_length_rejection(self):
        args = SimpleNamespace(debug=False, act=False, big_5=None,
                               api_base='http://127.0.0.1:18038/v1', api_key='unused',
                               max_tokens=2048, t=0, top_p=1, n=1)
        values = {'CWAH_AGENT_CHAT_ENABLE_THINKING': '1',
                  'CWAH_AGENT_CHAT_UPDATER_THINKING_TOKEN_BUDGET': '256'}
        response = {'choices': [{'finish_reason': 'stop', 'message': {
            'content': '<think>analysis</think>{"intent":{}}'}}]}
        with patch.dict(os.environ, values, clear=True), \
                patch('openai.ChatCompletion.create', return_value=response) as chat_api:
            planner = LLM('openai', 'qwen3.8-27b', 'unused', True, False, args, 1)
            params = dict(planner.sampling_params, _trace_call_type='updater')
            output, _ = planner.generator([{'role': 'user', 'content': 'test'}], params)
            self.assertEqual(output, ['{"intent":{}}'])
            self.assertNotIn('_trace_call_type', chat_api.call_args.kwargs)
            self.assertEqual(chat_api.call_args.kwargs['thinking_token_budget'], 256)
            response['choices'][0]['finish_reason'] = 'length'
            with self.assertRaisesRegex(ValueError, 'exhausted max_tokens'):
                planner.generator([{'role': 'user', 'content': 'test'}], params)
            response['choices'][0]['finish_reason'] = 'stop'
            response['choices'][0]['message']['content'] = '<action>[wait]</action>'
            with self.assertRaisesRegex(ValueError, 'missing </think>'):
                planner.generator([{'role': 'user', 'content': 'test'}], params)

    def test_openai_backend_passes_arbitrary_model_names_to_chat_api(self):
        args = SimpleNamespace(
            debug=False, act=False, big_5=None, api_base='https://example.invalid/v1',
            api_key='test-key', max_tokens=128, t=0, top_p=1, n=1,
        )
        messages = [{'role': 'user', 'content': 'test'}]
        response = {'choices': [{'message': {'content': 'ok'}}],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
        for model in ('gpt-5', 'custom-provider/model-alias', 'deepseek-ai/DeepSeek-V3.1'):
            with self.subTest(model=model), patch.dict(os.environ, {}, clear=True), \
                    patch('openai.ChatCompletion.create', return_value=response) as chat_api, \
                    patch('openai.Completion.create') as legacy_api:
                planner = LLM('openai', model, 'unused', True, False, args, 1)
                output, _ = planner.generator(messages, planner.sampling_params)
                self.assertTrue(planner.chat)
                self.assertEqual(output, ['ok'])
                self.assertEqual(chat_api.call_args.kwargs['model'], model)
                self.assertEqual(chat_api.call_args.kwargs['messages'], messages)
                legacy_api.assert_not_called()

    def test_profile_override_uses_real_task_and_source_cluster(self):
        values = {
            "CWAH_PROFILE_SOURCE_TASK": "26_prepare_food",
            "CWAH_PROFILE_EXPECTED_TASK": "50_prepare_food",
        }
        with patch.dict(os.environ, values):
            self.assertEqual(resolve_profile_key("50_prepare_food_17"), "26_prepare_food_17")
            with self.assertRaises(ValueError):
                resolve_profile_key("49_prepare_food_17")
            with self.assertRaises(ValueError):
                resolve_profile_key("50_prepare_food_99999")

    def test_profile_lookup_is_unchanged_without_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_profile_key("26_prepare_food_17"), "26_prepare_food_17")

    def test_agent_and_human_use_separate_settings(self):
        args = SimpleNamespace(
            debug=False,
            act=False,
            big_5=None,
            api_base="fallback-base",
            api_key="fallback-key",
        )
        values = {
            "CWAH_AGENT_API_BASE": "https://agent.example/v1",
            "CWAH_AGENT_LM_ID": "agent-model",
            "CWAH_AGENT_API_KEY": "agent-test-key",
            "CWAH_HUMAN_API_BASE": "https://human.example/v1",
            "CWAH_HUMAN_LM_ID": "human-model",
            "CWAH_HUMAN_API_KEY": "human-test-key",
        }
        with patch.dict(os.environ, values):
            agent = LLM("debug", "fallback-model", "LLM/archived_prompts/prompt_Agent.txt", True, False, args, 1)
            human = LLM("debug", "fallback-model", "LLM/archived_prompts/prompt_Agent.txt", True, False, args, 2)

        self.assertEqual((agent.lm_id, agent.api_base, agent.api_key),
                         ("agent-model", "https://agent.example/v1", "agent-test-key"))
        self.assertEqual((human.lm_id, human.api_base, human.api_key),
                         ("human-model", "https://human.example/v1", "human-test-key"))


if __name__ == "__main__":
    unittest.main()
