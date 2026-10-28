"""Freeze shared legal inputs using ordered ledger cursors, never global graphs."""
import hashlib
import json
import pickle
from pathlib import Path
from .metrics import identity, validate_prior
from .storage import data_directory
from .online_inputs import extract_online_input

VERSION = 'retrospective-intent-agreement-v3-online-facts'
INFO_FIELDS = ('planner_type', 'goal', 'current_progress', 'action_history',
               'dialogue_history', 'available_actions', 'step', 'current_skill',
               'incoming_message', 'public_tasks', 'agent_control',
               'peer_observation', 'peer_memory')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def find_traces(value, path='root'):
    if isinstance(value, dict):
        if isinstance(value.get('belief_updates'), list) and isinstance(value.get('evidence_events'), list):
            yield path, value
            return
        for key, child in value.items():
            yield from find_traces(child, path + '/' + str(key))
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from find_traces(child, path + '/' + str(i))


def build_intervals(trace, trace_id):
    events, updates = trace['evidence_events'], trace['belief_updates']
    if len({e['event_id'] for e in events}) != len(events):
        raise ValueError('duplicate ledger event IDs')
    last_cursor, last_id = 0, 0
    recovered = {}
    for update in updates:
        start, end = update['interval_start_cursor'], update['interval_end_cursor']
        if not (type(start) is int and type(end) is int and
                start == last_cursor and start < end <= len(events)):
            raise ValueError('noncontiguous or invalid decision cursors')
        boundary = events[end - 1]
        if (boundary['kind'] != 'decision' or boundary['payload']['decision_id'] != update['decision_id']
                or update['decision_id'] <= last_id):
            raise ValueError('decision boundary mismatch')
        expected = [e['event_id'] for e in events[start:end - 1]]
        if expected != update['new_event_ids']:
            raise ValueError('decision evidence cursor mismatch')
        raw_events = events[start:end - 1]
        if update.get('legal_interval_events') != raw_events:
            raise ValueError('saved legal interval differs from raw ledger slice')
        recovered[update['decision_id']] = extract_online_input(update, raw_events)
        last_cursor, last_id = end, update['decision_id']
    for current, following in zip(updates, updates[1:]):
        end = current['interval_end_cursor']
        next_end = following['interval_end_cursor'] - 1
        candidates = [{'subtask_id': t, 'skill': skill}
                      for t, skills in current['support'].items() for skill in skills]
        keys = [identity(c) for c in candidates]
        if len(set(keys)) != len(keys) or not keys:
            raise ValueError('invalid start candidate support')
        saved = current['posterior']['joint']
        if len(saved) != len(keys) or {identity(c) for c in saved} != set(keys):
            raise ValueError('saved prior support mismatch')
        values = {identity(c): c['probability'] for c in saved}
        prior = validate_prior([values[k] for k in keys])
        start_input = recovered[current['decision_id']]
        next_input = recovered[following['decision_id']]
        start_facts = {k: start_input['information'][k] for k in INFO_FIELDS
                       if k in start_input['information']}
        end_facts = {k: next_input['information'][k] for k in INFO_FIELDS
                     if k in next_input['information']}
        # Facts newly available at the next boundary (e.g. public task creation)
        # can be in Information rather than the event list. Include changed factual
        # fields, but never the next posterior, belief snapshot or instructions.
        changed_facts = {k: value for k, value in end_facts.items()
                         if k not in start_facts or start_facts[k] != value}
        evidence = next_input['events']
        common = {
            'target': 'Bob Intent at START decision; not end decision',
            'start_decision_id': current['decision_id'],
            'end_decision_id': following['decision_id'],
            'candidates': candidates,
            'history': {
                'start_information': start_facts,
                'start_formatted_events': start_input['events'],
                'start_request_facts': start_input['request_facts'],
            },
            'new_evidence': evidence,
            'new_information': changed_facts,
        }
        row = {'interval_id': f"{trace_id}:{current['decision_id']}:{following['decision_id']}",
               'common': common, 'common_hash': digest(common), 'prior': prior,
               'start_step': current['environment_step'], 'end_step': following['environment_step'],
               'start_planner': current['planner_type'], 'end_planner': following['planner_type'],
               'event_cursor_start': end, 'event_cursor_stop': next_end,
               'new_event_count': len(evidence),
               'audit': {
                   'start_raw_event_ids': current['new_event_ids'],
                   'new_raw_event_ids': following['new_event_ids'],
                   'start_online_prompt_sha256': hashlib.sha256(start_input['prompt'].encode()).hexdigest(),
                   'end_online_prompt_sha256': hashlib.sha256(next_input['prompt'].encode()).hexdigest(),
                   'start_formatted_events_hash': digest(start_input['events']),
                   'new_formatted_events_hash': digest(evidence),
                   'common_input_chars': len(json.dumps(common, ensure_ascii=False)),
               },
               'intent_update_status': current.get('intent_update_status')}
        row['input_hash'] = digest(row)
        yield row


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output)
    # Pickles must originate from trusted CWAH experiments.
    raw = source.read_bytes()
    data = json.loads(raw) if source.suffix == '.json' else pickle.loads(raw)
    traces = list(find_traces(data))
    if not traces:
        raise ValueError('no TRACE v2 archives found')
    rows, tails = [], []
    for trace_id, trace in traces:
        rows.extend(build_intervals(trace, trace_id))
        updates = trace['belief_updates']
        cursor = updates[-1]['interval_end_cursor'] if updates else 0
        tails.append({'trace_id': trace_id, 'unpaired_terminal_events':
                      len([e for e in trace['evidence_events'][cursor:] if e['kind'] != 'decision'])})
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('prepare requires an empty output directory')
    details = data_directory(output)
    (details / 'intervals.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                                                for r in rows), encoding='utf-8')
    manifest = {'protocol': VERSION, 'source': str(source), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'interval_count': len(rows), 'terminal_tails': tails,
                'history_policy': 'saved online factual input at start + next online formatted events + changed factual fields; no full-ledger expansion',
                'raw_evidence_policy': 'event IDs/cursors and source SHA256 reference the original trajectory',
                'common_input_chars': {'min': min((r['audit']['common_input_chars'] for r in rows), default=0),
                                       'max': max((r['audit']['common_input_chars'] for r in rows), default=0)},
                'ground_truth': 'not required; JSD measures agreement, not accuracy'}
    (details / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
