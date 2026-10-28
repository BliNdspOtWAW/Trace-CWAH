"""Aggregate incremental LLM metrics in memory without creating reports."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict


NUMBERS = ('prompt_tokens', 'completion_tokens', 'total_tokens', 'cached_input_tokens',
           'reasoning_tokens', 'request_count', 'retry_count', 'elapsed_seconds', 'prompt_chars')


def _stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return dict(known_calls=0, total=None, mean=None, p50=None, p95=None, max=None)
    def percentile(q):
        pos = (len(values) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return dict(known_calls=len(values), total=sum(values), mean=sum(values) / len(values),
                p50=percentile(.5), p95=percentile(.95), max=values[-1])


def _aggregate(rows):
    return dict(calls=len(rows), generation_errors=sum(r.get('status') == 'error' for r in rows),
                api_success_calls=sum(r.get('api_status', r.get('status')) == 'ok' for r in rows),
                valid_decision_calls=sum(r.get('decision_status') == 'ok' for r in rows),
                invalid_decision_calls=sum(r.get('decision_status') == 'error' for r in rows),
                unreported_decision_calls=sum(r.get('decision_status') is None for r in rows),
                missing_token_usage_calls=sum(r.get('prompt_tokens') is None for r in rows),
                metrics={name: _stats([r.get(name) for r in rows]) for name in NUMBERS})


def export_metrics(run_dir):
    run_dir = Path(run_dir)
    source = run_dir / 'calls.jsonl'
    if not source.is_file():
        source = run_dir / 'llm_calls.jsonl'  # Read-only compatibility with older runs.
    if not source.is_file():
        return {'available': False}
    starts, ends, decisions, invalid, reported_steps = {}, {}, {}, 0, 0
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1  # A live final line or abrupt interruption is tolerated.
                continue
            if row.get('event') == 'call_start':
                starts[row['call_id']] = row
            elif row.get('event') == 'call_end':
                ends[row['call_id']] = row
            elif row.get('event') == 'decision_end':
                decisions[row['call_id']] = row
            elif row.get('event') == 'environment_step_start':
                reported_steps += 1
    rows = []
    for call_id, call in ends.items():
        row = dict(call)
        decision = decisions.get(call_id)
        if decision:
            row['decision_status'] = decision.get('status')
            row['decision_reason'] = decision.get('reason')
        rows.append(row)
    in_flight = []
    for call_id in sorted(set(starts) - set(ends)):
        item = dict(starts[call_id])
        try:
            item['elapsed_so_far_seconds'] = (datetime.now(timezone.utc) - datetime.fromisoformat(item['started_at'])).total_seconds()
        except (KeyError, TypeError, ValueError):
            item['elapsed_so_far_seconds'] = None
        in_flight.append(item)
    groups, windows = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[(row.get('agent'), row.get('module'))].append(row)
        step = row.get('step')
        window = str((step // 50) * 50) if isinstance(step, int) else 'unknown'
        windows[(window, row.get('agent'), row.get('module'))].append(row)
    summary = dict(available=True, attempted_calls=len(starts), completed_calls=len(ends),
                   reported_environment_step_starts=reported_steps,
                   in_flight_call_ids=sorted(set(starts) - set(ends)),
                   in_flight_calls=in_flight,
                   malformed_lines=invalid, overall=_aggregate(rows),
                   groups=[dict(agent=a, module=m, **_aggregate(v)) for (a, m), v in groups.items()],
                   step_windows=[dict(start_step=w, agent=a, module=m, **_aggregate(v))
                                 for (w, a, m), v in windows.items()])
    return summary
