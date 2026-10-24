"""Consolidate only a newly created run's private scratch files, losslessly."""
import json
import os
import pickle
import shutil
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _load(path):
    with path.open('rb') as stream:
        return pickle.load(stream)


def compact_continuous(value):
    result = {k: v for k, v in value.items()
              if k not in ('completed_subtask_records', 'events', 'active_subtasks', 'subtasks')}
    result['active_subtask_count'] = len(value.get('active_subtasks', []))
    return result


def finalize_artifacts(run_dir, episode_index, summary, exit_code, expected_runs):
    """Keep old evaluator output private until a verified final archive exists.

    Never call this on an old experiment. Only `.runtime` created by the runner
    is eligible for cleanup; unknown or unreadable data is retained.
    """
    run_dir = Path(run_dir)
    scratch = run_dir / '.runtime'
    if scratch.is_symlink() or not scratch.is_dir():
        raise ValueError('Expected a real, run-local .runtime directory')
    records, errors = {}, []
    for path in sorted(scratch.glob('logs_agent_*_*.pik')):
        try:
            data = _load(path)
            if not isinstance(data, dict):
                raise ValueError('Trajectory must be an object')
            repeat = int(path.stem.rsplit('_', 1)[-1])
            data = dict(data, run_index=repeat, episode_index=episode_index)
            data.setdefault('environment_steps', max((len(v) for v in data.get('action', {}).values()), default=0))
            records[repeat] = data
        except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            errors.append({'file': path.name, 'type': type(exc).__name__})
    checkpoint = scratch / 'trajectory_checkpoint.pik'
    checkpoint_data = None
    if checkpoint.exists():
        try:
            checkpoint_data = _load(checkpoint)
            if not isinstance(checkpoint_data, dict):
                raise ValueError('Checkpoint must be an object')
            repeat = int(checkpoint_data.get('run_index', 0))
            if repeat in records and isinstance(checkpoint_data.get('trace'), dict):
                records[repeat]['trace'] = dict(checkpoint_data['trace'], **records[repeat].get('trace', {}))
            elif repeat not in records:
                records[repeat] = dict(checkpoint_data, run_index=repeat,
                                       episode_index=episode_index, partial=True)
        except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            errors.append({'file': checkpoint.name, 'type': type(exc).__name__})
    for repeat, record in records.items():
        trace = record.get('trace')
        if trace is not None and (not isinstance(trace, dict) or not all(k in trace for k in ('evidence_events', 'belief_updates', 'final_belief'))):
            errors.append({'run_index': repeat, 'type': 'missing_trace_details'})
    interrupted = summary.get('status') == 'interrupted' and (
        exit_code == 0 or any(r.get('interrupted') for r in records.values()))
    completed = (exit_code == 0 and summary.get('status') == 'completed'
                 and summary.get('runs') == expected_runs
                 and len(records) == expected_runs and not errors
                 and all(not r.get('partial') and not r.get('interrupted') for r in records.values()))
    status = 'completed' if completed else 'interrupted' if interrupted else 'failed'
    archive = {'format_version': 1, 'episode_index': episode_index,
               'runs': [records[k] for k in sorted(records)]}
    target = run_dir / 'trajectory.pik'
    temporary = target.with_name(target.name + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(archive, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    # Read back before considering any redundant source file removable.
    verified = _load(temporary)
    if verified['format_version'] != 1 or len(verified['runs']) != len(records):
        raise ValueError('Final trajectory verification failed')
    del verified
    temporary.replace(target)
    summary.update(status=status, runner_exit_code=exit_code,
                   trajectories=['trajectory.pik'], error=None)
    summary['recorded_runs'] = len(records)
    summary['steps_by_run'] = [{'run_index': k, 'steps': r.get('environment_steps', 0),
                              'partial': bool(r.get('partial') or r.get('interrupted'))}
                             for k, r in sorted(records.items())]
    if status == 'failed':
        summary['error'] = {'type': 'artifact_error' if errors else 'incomplete_or_failed_run',
                            'exit_code': exit_code, 'details': errors,
                            'console': 'console.log'}
    continuous = [dict(run_index=k, **compact_continuous(r['continuous_subtasks']))
                  for k, r in sorted(records.items()) if isinstance(r.get('continuous_subtasks'), dict)]
    if continuous:
        summary['continuous_subtasks_by_run'] = continuous
        total = sum(r.get('completed_subtasks', 0) for r in continuous)
        summary['completed_subtasks'] = total
        summary['mean_completed_subtasks'] = total / len(continuous)
        summary['continuous_subtasks'] = ({k: v for k, v in continuous[0].items() if k != 'run_index'}
                                          if len(continuous) == 1 else
                                          {'completed_subtasks': total, 'runs': len(continuous)})
    if not completed and checkpoint.exists():
        shutil.copyfile(checkpoint, run_dir / checkpoint.name)
        summary['checkpoint'] = checkpoint.name
    known = lambda p: (p.is_file() and not p.is_symlink() and
                      (p.name in ('results.pik', 'trajectory_checkpoint.pik')
                       or (p.name.startswith('logs_agent_') and p.suffix == '.pik')))
    unknown = [p.name for p in scratch.iterdir() if not known(p)]
    if errors or unknown:
        summary['retained_runtime_directory'] = '.runtime'
        summary['retained_runtime_extra_files'] = unknown
    else:
        # This directory was created for this run alone, never the run root.
        shutil.rmtree(scratch)
    return summary
