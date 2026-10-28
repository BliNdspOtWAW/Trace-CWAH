"""Recover actual online fact sections without replaying all past raw graphs."""
import json

SECTIONS = (
    ('information', 'Information shared with the planner:'),
    ('prior', 'Prior aligned to current candidates:'),
    ('support', 'Public support (task -> exact allowed skill keys):'),
    ('events', 'New legal events since the PREVIOUS DECISION (possibly multiple environment steps):'),
    ('requests', 'Requests with linked reply IDs:'),
)
REQUEST_FACT_FIELDS = (
    'request_id', 'type', 'subtask_id', 'created_step', 'requested_step',
    'request_event_id', 'reply_event_id', 'request_event_text', 'reply_event_text',
)


def extract_online_input(update, raw_events):
    """Strictly decode saved JSON sections; no guessed template or silent fallback."""
    prompt = (update.get('llm_call') or {}).get('prompt')
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('saved online updater prompt is required for exact reconstruction')
    decoder = json.JSONDecoder()
    first = prompt.find(SECTIONS[0][1])
    if first < 0:
        raise ValueError('unsupported online prompt format: missing information section')
    tail = prompt[first:]
    sections = {}
    for key, marker in SECTIONS:
        tail = tail.lstrip()
        if not tail.startswith(marker):
            raise ValueError('unsupported online prompt section order: ' + key)
        body = tail[len(marker):].lstrip()
        value, end = decoder.raw_decode(body)
        sections[key] = value
        tail = body[end:]
    if sections['information'] != update['shared_information']:
        raise ValueError('saved prompt information differs from decision snapshot')
    if sections['support'] != update['support']:
        raise ValueError('saved prompt support differs from decision support')
    if 'aligned_prior' in update and sections['prior'] != update['aligned_prior']:
        raise ValueError('saved prompt prior differs from aligned online prior')
    formatted = sections['events']
    if not isinstance(formatted, list) or len(formatted) != len(raw_events):
        raise ValueError('formatted/raw event counts differ')
    for item, raw in zip(formatted, raw_events):
        if any(item.get(key) != raw[key] for key in ('event_id', 'step', 'kind')):
            raise ValueError('formatted/raw event identity or order mismatch')
        if raw['kind'] != 'observation':
            if item != raw:
                raise ValueError('non-observation event differs from online evidence')
        else:
            payload = item.get('payload', {})
            if set(payload) != {'peer_observation', 'semantic_peer_facts'}:
                raise ValueError('unsupported online observation format')
            if payload['peer_observation'].get('step') != raw['step']:
                raise ValueError('formatted observation step mismatch')
    if not isinstance(sections['requests'], list):
        raise ValueError('online requests must be a list')
    # Remove statuses assigned by the belief updater; keep actual request/reply text.
    facts = []
    for request in sections['requests']:
        if not isinstance(request, dict):
            raise ValueError('invalid online request')
        facts.append({k: request[k] for k in REQUEST_FACT_FIELDS if k in request})
    return {
        'information': sections['information'],
        'events': formatted,
        'request_facts': facts,
        'prompt': prompt,
    }
