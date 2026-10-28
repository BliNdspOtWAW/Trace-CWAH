import re
import unittest
from pathlib import Path
from LLM.prompt_composer import compose_active, compose_reply
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
