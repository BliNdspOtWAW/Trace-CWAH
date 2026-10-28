"""Compose one mode-specific contract without changing execution semantics."""
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / 'prompts'

def trace_enabled(belief_context):
    # Active planning supplies belief values only when TRACE is enabled.
    return bool(belief_context)

def compose_active(template, belief_context):
    enabled = trace_enabled(belief_context)
    mode = 'trace' if enabled else 'baseline'
    context = ''
    if enabled:
        context = ((PROMPTS / 'trace' / 'planner_context.txt').read_text(encoding='utf-8')
                   + '\n' + (PROMPTS / 'trace' / 'communication.txt').read_text(encoding='utf-8'))
    modular = '$OUTPUT_CONTRACT' in template
    result = template.replace('$TEAMMATE_CONTEXT', context)
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
