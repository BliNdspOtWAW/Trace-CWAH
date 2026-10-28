"""Compose one mode-specific contract without changing execution semantics."""
import json
import math
import os
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / 'prompts'
COUNTERFACTUAL_PROBE_ENTROPY_THRESHOLD = 0.85


def intent_normalized_entropy(belief_context):
    """Read joint intent entropy without mixing in other uncertainty dimensions."""
    try:
        value = json.loads(belief_context)['intent']['value']['normalized_entropy']
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value) if 0 <= value <= 1 else None


def counterfactual_probe_prompt(belief_context, agent_name):
    entropy = intent_normalized_entropy(belief_context) if agent_name == 'Alice' else None
    if (os.environ.get('CWAH_DISABLE_COUNTERFACTUAL_PROBE') == '1' or
            entropy is None or entropy <= COUNTERFACTUAL_PROBE_ENTROPY_THRESHOLD):
        return entropy, ''
    fragment = (PROMPTS / 'trace' / 'counterfactual_probe.txt').read_text(encoding='utf-8')
    return entropy, fragment.replace('$INTENT_ENTROPY', f'{entropy:.3f}')

def trace_enabled(belief_context):
    # Active planning supplies belief values only when TRACE is enabled.
    return bool(belief_context)

def compose_active(template, belief_context, counterfactual_prompt=''):
    enabled = trace_enabled(belief_context)
    mode = 'trace' if enabled else 'baseline'
    context = ''
    if enabled:
        context = ((PROMPTS / 'trace' / 'planner_context.txt').read_text(encoding='utf-8')
                   + '\n' + (PROMPTS / 'trace' / 'communication.txt').read_text(encoding='utf-8'))
        if counterfactual_prompt:
            context += '\n' + counterfactual_prompt
    modular = '$OUTPUT_CONTRACT' in template
    result = template.replace('$TEAMMATE_CONTEXT', context)
    if enabled and counterfactual_prompt and '$TEAMMATE_CONTEXT' not in template:
        # Custom templates may omit the normal insertion point.
        result += '\n' + counterfactual_prompt
    if modular:
        contract = (PROMPTS / mode / 'active_output.txt').read_text(encoding='utf-8')
        if enabled:
            contract = 'FINAL TRACE OUTPUT CONTRACT:\n' + contract
        return result.replace('$OUTPUT_CONTRACT', contract)
    # Preserve external/custom legacy templates; retain their former TRACE override.
    if enabled:
        result += '\nFINAL TRACE OUTPUT CONTRACT (replaces generic message examples):\n' + (PROMPTS / 'trace' / 'active_output.txt').read_text(encoding='utf-8')
    return result

def compose_reply(template, belief_context):
    if not belief_context or belief_context.startswith('TRACE is disabled.'):
        template = template.replace('TRACE teammate belief:\n$TRACE_BELIEF', '')
        template = template.replace('$TRACE_BELIEF', '')
    return template
