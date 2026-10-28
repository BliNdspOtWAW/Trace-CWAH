"""Incremental generation metrics; never changes the request or model output."""
import json
import os
import time
import uuid
import warnings
from datetime import datetime, timezone


def _chars(value):
    if value is None:
        return 0
    return len(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))


def _field(information, name, default=None):
    return information.get(name, default) if isinstance(information, dict) else getattr(information, name, default)


def _emit(record):
    path = os.environ.get('CWAH_METRICS_PATH')
    if not path:
        return
    try:
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    except (OSError, TypeError, ValueError) as exc:
        warnings.warn('LLM metrics write failed: ' + type(exc).__name__, RuntimeWarning)


def record_decision(planner, module, valid, reason=None):
    """Record protocol validity separately from the completed API request."""
    call = getattr(planner, 'last_call_metrics', None) or {}
    if call.get('module') != module or not call.get('call_id'):
        return
    _emit({'schema_version': 1, 'event': 'decision_end', 'call_id': call['call_id'],
           'module': module, 'status': 'ok' if valid else 'error',
           'reason': None if valid else str(reason or 'invalid_output'),
           'ended_at': datetime.now(timezone.utc).isoformat()})


def measured_generate(planner, request, params, module, information=None, sections=None, counts=None):
    """Return exactly generator(request, params); log completed and failed calls.

    Success here means generation returned, not that output parsing succeeded.
    Character sizes are not token estimates and overlapping sections are not additive.
    """
    if isinstance(request, str):
        prompt_chars = len(request)
    else:
        prompt_chars = sum(_chars(item.get('content')) for item in request)
    identity = lambda value: value if isinstance(value, (str, int)) else None
    param_values = params if isinstance(params, dict) else vars(params)
    record = {
        'schema_version': 1, 'call_id': uuid.uuid4().hex,
        'run_id': os.environ.get('CWAH_METRICS_RUN_ID'),
        'run_index': (int(os.environ['CWAH_METRICS_REPEAT'])
                      if os.environ.get('CWAH_METRICS_REPEAT', '').isdigit() else None),
        'episode_index': (int(os.environ['CWAH_METRICS_EPISODE'])
                          if os.environ.get('CWAH_METRICS_EPISODE', '').isdigit() else None),
        'module': module, 'agent': identity(getattr(planner, 'agent_name', None)),
        'agent_id': identity(getattr(planner, 'agent_id', None)),
        'model': identity(getattr(planner, 'lm_id', None)),
        'backend': identity(getattr(planner, 'source', None)),
        'step': _field(information, 'step'),
        'prompt_chars': prompt_chars,
        'section_chars': {name: _chars(value) for name, value in (sections or {}).items()},
        'context_counts': dict(counts or {}),
        'requested_max_tokens': param_values.get('max_tokens', param_values.get('max_new_tokens')),
    }
    for name in ('action_history', 'dialogue_history', 'available_actions', 'source_event_ids'):
        value = _field(information, name)
        if isinstance(value, (tuple, list)):
            record['context_counts'][name] = len(value)
    if record['backend'] == 'openai' and getattr(planner, 'wire_api', None) == 'responses':
        record['effective_max_output_tokens'] = max(
            getattr(planner, 'responses_output_budget', 8192), record['requested_max_tokens'] or 1024)
    else:
        record['effective_max_output_tokens'] = record['requested_max_tokens']
    record['started_at'] = datetime.now(timezone.utc).isoformat()
    _emit(dict(record, event='call_start'))
    started = time.monotonic()
    error = None
    outputs = None
    planner.last_api_response_status = None
    try:
        outputs, cost = planner.generator(request, params)
        return outputs, cost
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        record.update(event='call_end', ended_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - started,
                      status='error' if error else 'ok', error_type=error,
                      api_status=(getattr(planner, 'last_api_response_status', None)
                                  or ('error' if error else 'ok')))
        usage = getattr(planner, 'last_usage_metadata', None)
        usage = usage if isinstance(usage, dict) and not error else {}
        known = usage.get('token_usage_available', True)
        for name in ('prompt_tokens', 'completion_tokens', 'total_tokens',
                     'cached_input_tokens', 'reasoning_tokens', 'request_count', 'retry_count'):
            value = usage.get(name)
            record[name] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
            if not known and name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                record[name] = None
        record['output_chars'] = (sum(len(text) for text in outputs if isinstance(text, str))
                                  if isinstance(outputs, (list, tuple)) else None)
        if record['completion_tokens'] is not None:
            record['output_tokens_per_second'] = record['completion_tokens'] / max(record['elapsed_seconds'], 1e-9)
        else:
            record['output_tokens_per_second'] = None
        planner.last_call_metrics = record
        _emit(record)
