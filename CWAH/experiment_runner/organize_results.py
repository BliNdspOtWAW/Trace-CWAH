"""Preview or migrate only existing tracev2 runs into the current layout."""

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from result_paths import run_directory
from result_storage import write_json


EXPERIMENTS = Path(__file__).resolve().parents[1] / 'experiments'
STAMP = re.compile(r'(\d{8}-\d{6}-\d{6})$')


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def observed_steps(run_dir, summary):
    values = summary.get('steps_by_run') or summary.get('steps') or []
    if not isinstance(values, list):
        values = []
    steps = max((item.get('steps', 0) if isinstance(item, dict) else item
                 for item in values if isinstance(item, (dict, int)) and not isinstance(item, bool)),
                default=0)
    calls = run_dir / 'calls.jsonl'
    if calls.is_file():
        for line in calls.open(encoding='utf-8', errors='replace'):
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get('event') == 'environment_step_start' and type(entry.get('step')) is int:
                steps = max(steps, entry['step'])
    return steps


def collection_root(run_dir):
    relative = run_dir.relative_to(EXPERIMENTS)
    return (EXPERIMENTS / relative.parts[0]
            if relative.parts[0] in ('tracev1', 'tracev2') else EXPERIMENTS)


def planned_runs(min_steps):
    plans = []
    tracev2_root = EXPERIMENTS / 'tracev2'
    for summary_path in sorted(tracev2_root.rglob('summary.json')):
        src = summary_path.parent
        if src.is_symlink() or not src.is_dir() or (src / 'config.json').is_symlink():
            raise ValueError(f'Unsafe run directory: {src}')
        summary = read_json(summary_path)
        config = read_json(src / 'config.json')
        if not config or not summary:
            raise ValueError(f'Run lacks readable config or summary: {src}')
        if summary.get('status') == 'running':
            plans.append(('skip-running', src, None, None))
            continue
        steps = observed_steps(src, summary)
        if steps < min_steps:
            plans.append(('delete', src, None, steps))
            continue
        model = config.get('alice_model')
        peer_model = config.get('bob_model') or model
        profile = config.get('bob_profile_id') or 'none'
        max_steps = config.get('max_episode_length')
        mode = config.get('trace_mode')
        if not all((model, peer_model, max_steps, mode)):
            raise ValueError(f'Run lacks layout metadata: {src}')
        match = STAMP.search(src.name)
        stamp = match.group(1) if match else datetime.fromtimestamp(src.stat().st_mtime).strftime('%Y%m%d-%H%M%S-%f')
        dest = run_directory(tracev2_root, model, peer_model, profile, max_steps, mode, stamp)
        plans.append(('keep' if src == dest else 'move', src, dest, steps))
    destinations = [dest for action, _, dest, _ in plans if action == 'move']
    if len(destinations) != len(set(destinations)):
        raise ValueError('Two runs would receive the same destination')
    for action, src, dest, _ in plans:
        if action == 'move' and dest.exists():
            raise ValueError(f'Destination already exists: {dest}')
    return plans


def rename_trajectory_files(run_dir):
    renamed = {}
    old = run_dir / 'trajectory.pik'
    if old.is_file():
        target = run_dir / (run_dir.name + '.pik')
        old.rename(target)
        renamed[old.name] = target.name
    for old in sorted(run_dir.glob('logs_agent_*_*.pik')):
        match = re.search(r'_(\d+)\.pik$', old.name)
        suffix = match.group(1) if match else old.stem
        target = run_dir / f'{run_dir.name}_run{suffix}.pik'
        old.rename(target)
        renamed[old.name] = target.name
    old = run_dir / 'trajectory_checkpoint.pik'
    if old.is_file():
        target = run_dir / f'{run_dir.name}_checkpoint.pik'
        old.rename(target)
        renamed[old.name] = target.name
    return renamed


def apply_plan(plans):
    # Move all preserved runs first; delete short runs only after every move succeeds.
    for action, src, dest, steps in plans:
        if action != 'move':
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        renamed = rename_trajectory_files(dest)
        summary = read_json(dest / 'summary.json')
        summary['storage_run_id'] = dest.name
        if isinstance(summary.get('trajectories'), list):
            summary['trajectories'] = [renamed.get(name, name) for name in summary['trajectories']]
        if summary.get('checkpoint') in renamed:
            summary['checkpoint'] = renamed[summary['checkpoint']]
        write_json(dest / 'summary.json', summary)
        config = read_json(dest / 'config.json')
        config['storage_run_id'] = dest.name
        config['storage_results_root'] = str(collection_root(dest))
        config['storage_format_version'] = 3
        write_json(dest / 'config.json', config)
        print(f'MOVED {steps} {src} -> {dest}', flush=True)
    for action, src, _, steps in plans:
        if action == 'delete':
            shutil.rmtree(src)
            print(f'DELETED {steps} {src}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-steps', type=int, default=50)
    parser.add_argument('--apply', action='store_true', help='perform moves and delete short runs')
    args = parser.parse_args()
    if args.min_steps < 1:
        parser.error('--min-steps must be positive')
    plans = planned_runs(args.min_steps)
    for action, src, dest, steps in plans:
        print(f'{action.upper()} steps={steps} {src}' + (f' -> {dest}' if dest else ''))
    print(f'TOTAL {len(plans)}; DELETE {sum(a == "delete" for a, *_ in plans)}; '
          f'MOVE {sum(a == "move" for a, *_ in plans)}')
    if args.apply:
        apply_plan(plans)


if __name__ == '__main__':
    main()
