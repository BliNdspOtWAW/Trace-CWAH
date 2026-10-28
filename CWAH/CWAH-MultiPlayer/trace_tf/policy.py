"""TRACE belief values rendered with file-backed prompt fragments."""
import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent.parent / 'LLM' / 'prompts' / 'trace'
def render_policy_context(state, ledger):
    values = {}
    for name in ('intent', 'capability', 'reliability', 'uncertainty'):
        item = state.dimensions[name]
        values[name] = {'status': item.status, 'value': item.value, 'uncertainty': item.uncertainty}
    return json.dumps(values, ensure_ascii=False)

def render_reply_context(state, ledger):
    template = (PROMPTS / 'reply_context.txt').read_text(encoding='utf-8')
    return template.replace('$TRACE_BELIEF', render_policy_context(state, ledger))
