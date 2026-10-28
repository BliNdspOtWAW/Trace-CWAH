import re
import unittest
from unittest.mock import patch
from pathlib import Path
from LLM.prompt_composer import (compose_active, compose_reply,
                                 counterfactual_probe_prompt)
from LLM.active_protocol import parse_trace_active

ROOT = Path(__file__).resolve().parents[1] / 'LLM'

class PromptCompositionTests(unittest.TestCase):
    def template(self):
        return (ROOT / 'prompt_Alice.txt').read_text()

    def test_baseline_contains_no_trace_instructions(self):
        prompt = compose_active(self.template(), '')
        self.assertNotIn('TRACE', prompt)
        self.assertNotIn('communication_type', prompt)
        self.assertNotIn('$TRACE_BELIEF', prompt)
        self.assertIn('**Communication:**', prompt)
        self.assertNotIn('$OUTPUT_CONTRACT', prompt)

    def test_trace_has_one_contract_and_no_baseline_example(self):
        prompt = compose_active(self.template(), '{"intent": {"status": "estimated"}}')
        self.assertEqual(prompt.count('FINAL TRACE OUTPUT CONTRACT'), 1)
        self.assertNotIn('**Communication:**', prompt)
        self.assertNotIn('Communication without TRACE', prompt)
        self.assertIn('$TRACE_BELIEF', prompt)
        self.assertEqual(prompt.count('TRACE Teammate Belief'), 1)
        self.assertLess(prompt.index('TRACE beliefs are estimates'), prompt.index('$TRACE_BELIEF'))
        self.assertNotIn('**Additional teammate information:**', prompt)
        self.assertNotIn('$TEAMMATE_CONTEXT', prompt)
        self.assertNotIn('$OUTPUT_CONTRACT', prompt)

    def test_counterfactual_probe_gate_uses_only_joint_intent_entropy(self):
        for entropy, expected in ((0.85, False), (0.8501, True), (0.2, False)):
            belief = ('{"intent":{"value":{"normalized_entropy":' + str(entropy)
                      + '}},"uncertainty":{"uncertainty":1.0}}')
            value, fragment = counterfactual_probe_prompt(belief, 'Alice')
            self.assertEqual(value, entropy)
            self.assertEqual(bool(fragment), expected)
            if expected:
                prompt = compose_active(self.template(), belief, fragment)
                self.assertIn('Counterfactual Probe Check', prompt)
                self.assertLess(prompt.index('TRACE Teammate Belief'),
                                prompt.index('Counterfactual Probe Check'))
                self.assertLess(prompt.index('Counterfactual Probe Check'),
                                prompt.index('## Action Selection'))
                self.assertEqual(prompt.count('Counterfactual Probe Check'), 1)
            self.assertEqual(counterfactual_probe_prompt(belief, 'Bob'), (None, ''))
        self.assertEqual(counterfactual_probe_prompt('{}', 'Alice'), (None, ''))

    def test_counterfactual_probe_can_be_disabled_for_ablation(self):
        belief = '{"intent":{"value":{"normalized_entropy":0.95}}}'
        with patch.dict('os.environ', {'CWAH_DISABLE_COUNTERFACTUAL_PROBE': '1'}):
            self.assertEqual(counterfactual_probe_prompt(belief, 'Alice'), (0.95, ''))

    def test_trace_examples_match_real_parser(self):
        prompt = compose_active(self.template(), 'TRACE-v2 teammate belief about Bob')
        examples = re.findall(r'<cot>.*?</subtask_id>', prompt, re.S)
        messages = [e for e in examples if '<action>[send_message]</action>' in e]
        self.assertGreaterEqual(len(messages), 2)
        for text in messages:
            parsed = parse_trace_active(text, ['[send_message]'])
            self.assertIn(parsed['communication_type'], ('delegate', 'probe'))
            if parsed['communication_type'] == 'probe':
                self.assertIsNone(parsed['subtask_id'])
                self.assertIn(parsed['probe_target'],
                              ('intent', 'capability', 'reliability', 'information'))
            else:
                self.assertIsNone(parsed['probe_target'])

    def test_reply_off_omits_trace_section(self):
        template = (ROOT / 'prompt_Reply_Alice.txt').read_text()
        self.assertNotIn('TRACE teammate belief', compose_reply(template, ''))
        self.assertNotIn('TRACE teammate belief', compose_reply(
            template, 'TRACE is disabled. No teammate belief is available.'))
        self.assertIn('$TRACE_BELIEF', compose_reply(template, 'TRACE-v2 teammate belief about Bob'))

    def test_trace_reply_renders_one_belief_section(self):
        template = (ROOT / 'prompt_Reply_Alice.txt').read_text()
        context = (ROOT / 'prompts' / 'trace' / 'reply_context.txt').read_text().replace(
            '$TRACE_BELIEF', '{"intent": {}}')
        prompt = compose_reply(template, context).replace('$TRACE_BELIEF', context)
        self.assertEqual(prompt.count('## TRACE Teammate Belief'), 1)
        self.assertEqual(prompt.count('## Response'), 1)
        self.assertNotIn('$TRACE_BELIEF', prompt)

    def test_bob_prompts_do_not_claim_trace_access(self):
        active = compose_active((ROOT / 'prompt_Bob.txt').read_text(), '')
        reply = compose_reply((ROOT / 'prompt_Reply_Bob.txt').read_text(), '')
        self.assertNotIn('TRACE teammate belief', active + reply)
        self.assertNotIn('$TRACE_BELIEF', active + reply)
        self.assertIn('An explicit acceptance establishes a commitment', reply)
        self.assertIn('This snapshot determines which tasks are active', active)
