"""Render a trusted CWAH trajectory archive for human inspection.

Pickle can execute code while loading. Only pass archives produced locally by a
trusted runner; never use this command on a downloaded or untrusted file.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _at(mapping, agent, step):
    if not isinstance(mapping, dict):
        return None
    values = mapping.get(agent, mapping.get(str(agent), []))
    return values[step] if isinstance(values, list) and step < len(values) else None


def _steps(run):
    value = run.get('environment_steps')
    if isinstance(value, int) and value >= 0:
        return value
    actions = run.get('action', {})
    return max((len(v) for v in actions.values() if isinstance(v, list)), default=0)


def render_run(run, *, include_prompts=False, full_trace=False):
    """Return a readable, step ordered view without changing the source data."""
    lines = []

    def add(label, value=None):
        lines.append(label if value is None else label + ': ' + str(value))

    add('RUN', run.get('run_index', 0))
    add('Episode', run.get('episode_index', run.get('task_id')))
    add('Environment steps', _steps(run))
    add('Termination', run.get('termination_reason'))
    add('Continuous subtasks completed', (run.get('continuous_subtasks') or {}).get('completed_subtasks'))
    add('Note: raw_output is the parsed final answer. New archives also retain raw_chat_output, including <think> text when returned by the API. Older archives cannot recover stripped text.')
    add('Note: an Updater decision can share an environment step with Active or Reactive.')

    updates = {}
    trace = run.get('trace') or {}
    for item in trace.get('belief_updates', []):
        updates.setdefault(item.get('environment_step'), []).append(item)
    planners = {}
    for agent, name in ((0, 'Alice'), (1, 'Bob')):
        for item in (run.get('LLM') or {}).get(agent, []):
            planners.setdefault(item.get('step_now', item.get('steps')), []).append((name, item))
    communications = {}
    for item in run.get('communication_events', []):
        communications.setdefault(item.get('step'), []).append(item)

    for step in range(_steps(run)):
        add('\n' + '=' * 80)
        add('STEP', step)
        for item in updates.get(step, []):
            add('\nTRACE UPDATER')
            add('Decision ID', item.get('decision_id'))
            add('Decision type', item.get('planner_type'))
            add('Update status', item.get('intent_update_status'))
            events = item.get('legal_interval_events') or []
            add('Evidence', _json([{'id': e.get('event_id'), 'step': e.get('step'),
                                    'kind': e.get('kind')} for e in events]))
            before = item.get('belief_before') or {}
            after = item.get('belief_after') or {}
            for label, belief in (('Prior', before), ('Posterior', after)):
                intent = ((belief.get('dimensions') or {}).get('intent') or {}).get('value') or {}
                add(label + ' task probabilities', _json(intent.get('task_probs')))
                if full_trace:
                    add(label + ' belief', _json(belief))
            call = item.get('llm_call') or {}
            add('Visible updater output', call.get('raw_output', call.get('output')))
            if call.get('raw_chat_output') is not None:
                add('Full API response content', call['raw_chat_output'])
            if include_prompts:
                add('Updater prompt', call.get('prompt'))
                add('Updater information', _json(item.get('shared_information')))
                add('Full legal evidence', _json(events))

        for name, item in planners.get(step, []):
            add('\n' + name.upper() + ' ACTIVE')
            add('Selected subtask', item.get('selected_subtask_id'))
            add('Selected skill', item.get('plan'))
            add('Visible model output', item.get('raw_output'))
            if item.get('raw_chat_output') is not None:
                add('Full API response content', item['raw_chat_output'])
            add('Token usage', _json(item.get('token_usage')))
            if include_prompts:
                add('Planner prompt', item.get('raw_prompt'))
                add('Decision information', _json(item.get('shared_information')))

        for agent, name in ((0, 'Alice'), (1, 'Bob')):
            add('\n' + name + ' EXECUTION')
            add('Skill', _at(run.get('plan'), agent, step))
            add('Physical action', _at(run.get('action'), agent, step))
            add('Control', _json(_at(run.get('agent_control'), agent, step)))

        task_events = run.get('continuous_subtask_events') or []
        if step < len(task_events):
            event = task_events[step]
            if event.get('completed') or event.get('generated'):
                add('\nTASK EVENTS', _json({'completed': event.get('completed'),
                                               'generated': event.get('generated')}))

        for item in communications.get(step, []):
            add('\nCOMMUNICATION', _json({k: v for k, v in item.items() if k != 'call'}))
            call = item.get('call')
            if isinstance(call, dict):
                add('Visible reactive output', call.get('raw_output'))
                if call.get('raw_chat_output') is not None:
                    add('Full API response content', call['raw_chat_output'])
                add('Parsed reply and control', _json({k: call.get(k) for k in (
                    'intention', 'reply', 'new_subtask_id', 'parse_status',
                    'macro_changed', 'agent_control_after')}))
                if include_prompts:
                    add('Reactive prompt', call.get('prompt'))

    if trace:
        add('\n' + '=' * 80)
        add('FINAL TRACE BELIEF', _json(trace.get('final_belief')))
        add('FINAL RULE STATISTICS', _json(trace.get('final_rule_statistics')))
    continuous = run.get('continuous_subtasks') or {}
    add('COMPLETED SUBTASK RECORDS', _json(continuous.get('completed_subtask_records')))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path, help='Trusted local trajectory.pik')
    parser.add_argument('--run-index', type=int, help='Render one repeat (default: all)')
    parser.add_argument('--include-prompts', action='store_true',
                        help='Include full prompts, decision information, and raw evidence')
    parser.add_argument('--full-trace', action='store_true',
                        help='Include full prior and posterior belief objects')
    parser.add_argument('--output', type=Path, help='Write UTF-8 text to this path (default: stdout)')
    args = parser.parse_args(argv)
    with args.trajectory.open('rb') as stream:
        archive = pickle.load(stream)
    if not isinstance(archive, dict) or not isinstance(archive.get('runs'), list):
        parser.error('Expected a versioned trajectory archive with a runs list')
    runs = archive['runs']
    if args.run_index is not None:
        runs = [run for run in runs if run.get('run_index') == args.run_index]
        if not runs:
            parser.error('No run with the requested run_index')
    report = '\n'.join(render_run(run, include_prompts=args.include_prompts,
                                  full_trace=args.full_trace) for run in runs)
    if args.output:
        args.output.write_text(report, encoding='utf-8')
    else:
        sys.stdout.write(report)


if __name__ == '__main__':
    main()
