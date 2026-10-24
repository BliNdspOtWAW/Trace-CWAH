"""Freeze shared legal inputs using ordered ledger cursors, never global graphs."""
import hashlib
import json
import pickle
from pathlib import Path
from .metrics import identity, validate_prior
from .storage import data_directory

VERSION = 'retrospective-intent-agreement-v2-scores'
INFO_FIELDS = ('planner_type', 'goal', 'current_progress', 'action_history',
               'dialogue_history', 'available_actions', 'step', 'current_skill',
               'incoming_message', 'public_tasks', 'agent_control',
               'peer_observation', 'peer_memory')
EVENT_FIELDS = {
    'observation': ('nodes', 'edges', 'location'),
    'message': ('sender', 'receiver', 'kind', 'text', 'communication', 'reply_to'),
    'own_action': ('action', 'status'),
    'own_outcome': ('action', 'success', 'status', 'error'),
    'task_completion': ('task', 'completed_by'),
    'peer_task_progress': ('actor', 'status', 'observed_action', 'task'),
    'peer_task_outcome': ('action', 'object', 'object_id', 'success', 'verified',
                          'relation', 'target_id', 'target_class', 'task'),
}


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


def legal_event(event):
    kind = event['kind']
    if kind == 'decision':
        return None
    if kind not in EVENT_FIELDS:
        raise ValueError('unknown evidence kind: ' + kind)
    payload = event['payload']
    if kind == 'observation':
        # Geometry required for visible movement, excluding unnecessary scene metadata.
        payload = dict(payload)
        payload['nodes'] = [{k: n[k] for k in ('id', 'class_name', 'states', 'obj_transform') if k in n}
                            for n in payload['nodes']]
        payload['edges'] = [{k: e[k] for k in ('from_id', 'to_id', 'relation_type')}
                            for e in payload['edges']]
    return {'event_id': event['event_id'], 'step': event['step'], 'kind': kind,
            'payload': {k: payload[k] for k in EVENT_FIELDS[kind] if k in payload}}


def build_intervals(trace, trace_id):
    events, updates = trace['evidence_events'], trace['belief_updates']
    if len({e['event_id'] for e in events}) != len(events):
        raise ValueError('duplicate ledger event IDs')
    last_cursor, last_id = 0, 0
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
        history = [legal_event(e) for e in events[:end]]
        evidence = [legal_event(e) for e in events[end:next_end]]
        evidence = [e for e in evidence if e is not None]
        common = {
            'target': 'Bob Intent at START decision; not end decision',
            'start_decision_id': current['decision_id'],
            'end_decision_id': following['decision_id'],
            'candidates': candidates,
            'history': {
                'start_information': {k: current['shared_information'][k] for k in INFO_FIELDS
                                      if k in current['shared_information']},
                'legal_events': [e for e in history if e is not None],
            },
            'new_evidence': evidence,
        }
        row = {'interval_id': f"{trace_id}:{current['decision_id']}:{following['decision_id']}",
               'common': common, 'common_hash': digest(common), 'prior': prior,
               'start_step': current['environment_step'], 'end_step': following['environment_step'],
               'start_planner': current['planner_type'], 'end_planner': following['planner_type'],
               'event_cursor_start': end, 'event_cursor_stop': next_end,
               'new_event_count': len(evidence),
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
                'history_policy': 'full ordered legal ledger plus whitelisted start snapshot; no truncation',
                'ground_truth': 'not required; JSD measures agreement, not accuracy'}
    (details / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
