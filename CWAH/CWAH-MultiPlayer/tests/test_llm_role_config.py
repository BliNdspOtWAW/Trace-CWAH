"""The two CWAH players must read their own uppercase environment settings."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from LLM.LLM import LLM


class LLMRoleConfigTests(unittest.TestCase):
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
            agent = LLM("debug", "fallback-model", "LLM/prompt_Agent.txt", True, False, args, 1)
            human = LLM("debug", "fallback-model", "LLM/prompt_Agent.txt", True, False, args, 2)

        self.assertEqual((agent.lm_id, agent.api_base, agent.api_key),
                         ("agent-model", "https://agent.example/v1", "agent-test-key"))
        self.assertEqual((human.lm_id, human.api_base, human.api_key),
                         ("human-model", "https://human.example/v1", "human-test-key"))


if __name__ == "__main__":
    unittest.main()
