"""TRACE-only Active output boundary: local controls versus delivered prose."""
import re

FLAGS = re.I | re.S
RESERVED = r'cot|message|action|subtask_id|communication_type|intention|new_subtask_id|reply'


def _take(text, tag):
    pattern = re.compile(r'<' + tag + r'>\s*(.*?)\s*</' + tag + r'>', FLAGS)
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError('duplicate TRACE Active field: ' + tag)
    return (matches[0].group(1).strip() if matches else None, pattern.sub('', text))


def parse_trace_active(raw, available_actions):
    # Remove prose envelopes BEFORE reading any control fields. Tags quoted in
    # cot/message cannot select an action or task or establish communication type.
    think, remaining = _take(raw, 'cot')
    message, remaining = _take(remaining, 'message')
    action, remaining = _take(remaining, 'action')
    no_action = {'plan': None, 'think': think, 'message': None,
                 'subtask_id': None, 'communication_type': None}
    if not action:
        return no_action
    if not re.fullmatch(r'\(?[A-Z]\)?', action):
        action = re.sub(r'^\s*(?:\(?[A-Z]\)?)[\s.\:)]\s*', '', action).strip()
    if not action:
        return no_action
    if re.search(r'</?\s*(?:' + RESERVED + r')\b', action, FLAGS):
        raise ValueError('protocol fields cannot be nested in action')
    if action in available_actions:
        selected = action
    elif re.fullmatch(r'\(?[A-Z]\)?', action):
        index = ord(action.strip('()')) - ord('A')
        selected = available_actions[index] if index < len(available_actions) else None
    else:
        selected = None
    if selected is None:
        return no_action
    if message and re.search(r'</?\s*(?:' + RESERVED + r')\b', message, FLAGS):
        raise ValueError('message contains internal protocol tags; retry with plain prose')
    task, remaining = _take(remaining, 'subtask_id')
    kind, remaining = _take(remaining, 'communication_type')
    if re.search(r'</?\s*(?:' + RESERVED + r')\b', remaining, FLAGS):
        raise ValueError('malformed or misplaced TRACE Active protocol field')
    if not selected.startswith('[send_message]') and kind is not None:
        raise ValueError('physical action cannot have a communication_type')
    return {'plan': selected, 'think': think, 'message': message,
            'subtask_id': None if task in (None, '', 'null', 'None') else task,
            'communication_type': kind, 'has_subtask_field': task is not None}
