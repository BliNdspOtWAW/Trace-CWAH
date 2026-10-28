"""Bind each CWAH experiment to one existing Qwen replica and run a batch."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

RUNNER_DIR = Path(__file__).resolve().parents[1]
ROOT = RUNNER_DIR.parents[1]
RESULTS = ROOT / 'CWAH' / 'experiments'
SINGLE_RUNNER = RUNNER_DIR / 'run_experiment.py'
sys.path.insert(0, str(RUNNER_DIR))
from run_experiment import validate  # noqa: E402


def read_json(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path)


def label(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Invalid batch/job/replica identifier: {value!r}')
    return value


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry(path):
    data = read_json(path)
    if data.get('version') != 1 or not isinstance(data.get('replicas'), list) or not data['replicas']:
        raise ValueError('replica registry requires version=1 and nonempty replicas')
    result, ids, urls = [], set(), set()
    for row in data['replicas']:
        rid = label(row['id'])
        url = row['api_base'].rstrip('/')
        parsed = urlsplit(url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                or not parsed.port or parsed.path != '/v1' or parsed.query or parsed.fragment):
            raise ValueError(f'{rid}: endpoint must be a local HTTP /v1 tunnel')
        if rid in ids or url in urls:
            raise ValueError(f'duplicate replica: {rid}')
        models = row.get('models')
        if not isinstance(models, list) or not models or any(not isinstance(m, str) or not m for m in models):
            raise ValueError(f'{rid}: models must be nonempty model IDs')
        result.append({'id': rid, 'url': url, 'models': set(models),
                       'capacity': positive(row.get('max_experiments', 1), 'max_experiments')})
        ids.add(rid)
        urls.add(url)
    return result


def load_batch(path):
    data = read_json(path)
    if data.get('version') != 1:
        raise ValueError('batch version must be 1')
    bid = label(data['batch_id'])
    limits = data['limits']
    cap = min(positive(limits['max_experiments'], 'max_experiments'),
              positive(limits['max_unity_instances'], 'max_unity_instances'))
    max_attempts = positive(data.get('max_attempts', 1), 'max_attempts')
    if not isinstance(data.get('jobs'), list) or not data['jobs']:
        raise ValueError('jobs must be nonempty')
    jobs, ids, hashes = [], set(), {}
    for row in data['jobs']:
        jid = label(row['id'])
        if jid in ids:
            raise ValueError(f'duplicate job id: {jid}')
        ids.add(jid)
        source = Path(row['config'])
        source = (source if source.is_absolute() else ROOT / source).resolve()
        config = read_json(source)
        selected = validate(config)
        if config.get('backend') != 'openai' or not all('qwen' in config[r + '_model'].lower()
                                                        for r in ('alice', 'bob')):
            raise ValueError(f'{jid}: expected OpenAI-compatible Qwen models')
        if config['num_runs'] != 1 or selected['port_request'] != 'auto':
            raise ValueError(f'{jid}: num_runs must be 1 and port must be auto')
        no_probe = row.get('no_counterfactual', False)
        if type(no_probe) is not bool or (no_probe and config['trace_mode'] != 'on'):
            raise ValueError(f'{jid}: no_counterfactual requires TRACE on')
        jobs.append({'id': jid, 'config': config, 'models': {config['alice_model'], config['bob_model']},
                     'no_counterfactual': no_probe})
        hashes[jid] = digest(source)
    return bid, cap, max_attempts, jobs, hashes


def healthy(replica, models):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(replica['url'] + '/models', timeout=6) as response:
        found = {item['id'] for item in json.load(response)['data']}
    return models.issubset(found)


def run_dir(log):
    if not log.exists():
        return None
    found = None
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        if line.startswith('results='):
            found = line[len('results='):].strip()
    return found


def settle(entry, attempt, exit_code):
    attempt['exit_code'] = exit_code
    attempt['result_dir'] = run_dir(Path(attempt['log']))
    summary = Path(attempt['result_dir']) / 'summary.json' if attempt['result_dir'] else None
    try:
        status = read_json(summary).get('status') if summary and summary.is_file() else None
    except (OSError, ValueError, json.JSONDecodeError):
        status = None
    attempt['result_status'] = status
    attempt['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    entry['status'] = 'completed' if exit_code in (0, None) and status == 'completed' else 'failed'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


def execute(batch, registry, state_dir, check=False, resume=False, poll=2):
    bid, cap, max_attempts, jobs, hashes = load_batch(batch)
    replicas = load_registry(registry)
    for job in jobs:
        if not any(job['models'].issubset(r['models']) for r in replicas):
            raise ValueError(f"No replica serves {job['id']}")
    if not all(os.environ.get(k) for k in ('CWAH_AGENT_API_KEY', 'CWAH_HUMAN_API_KEY')):
        raise ValueError('source CWAH/experiment_runner/env.sh qwen first')
    ready = {}
    for replica in replicas:
        try:
            ready[replica['id']] = healthy(replica, replica['models'])
        except (OSError, ValueError, KeyError):
            ready[replica['id']] = False
        print(f"{replica['id']}: {'ready' if ready[replica['id']] else 'unreachable'}", flush=True)
    if not any(ready.values()):
        raise ValueError('No reachable Qwen replica; check SSH tunnels')
    if check:
        if any(not any(ready[r['id']] and job['models'].issubset(r['models']) for r in replicas)
               for job in jobs):
            raise ValueError('At least one job lacks a reachable matching replica')
        print(f'Validated {len(jobs)} jobs; concurrency limit {cap}; no Unity started.')
        return 0
    state_dir = state_dir.resolve()
    try:
        state_dir.relative_to(RESULTS.resolve())
    except ValueError as exc:
        raise ValueError('state directory must be inside CWAH/experiments') from exc
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This batch is already being scheduled') from exc
        state_file = state_dir / 'state.json'
        fingerprints = {'batch': digest(batch), 'registry': digest(registry), 'configs': hashes}
        if resume:
            state = read_json(state_file)
            if state.get('fingerprints') != fingerprints:
                raise ValueError('batch, registry or source config changed since first attempt')
            for entry in state['jobs'].values():
                if entry['status'] == 'running':
                    attempt = entry['attempts'][-1]
                    if alive(attempt['pid']):
                        raise ValueError(f"Worker {attempt['pid']} is still running; will not duplicate it")
                    settle(entry, attempt, None)
        else:
            if state_file.exists():
                raise ValueError(f'Batch state exists; use --resume or another batch_id: {state_file}')
            state = {'version': 1, 'batch_id': bid, 'fingerprints': fingerprints,
                     'jobs': {job['id']: {'status': 'pending', 'attempts': []} for job in jobs}}
        write_json(state_file, state)
        active, assignments = {}, {r['id']: 0 for r in replicas}
        for entry in state['jobs'].values():
            for attempt in entry['attempts']:
                assignments[attempt['replica']] += 1
        interrupted = False
        idle_since = None
        last_wait_notice = 0
        old_term = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, interrupt)
        try:
            while True:
                for jid, (proc, log, rid) in list(active.items()):
                    code = proc.poll()
                    if code is not None:
                        log.close()
                        entry = state['jobs'][jid]
                        settle(entry, entry['attempts'][-1], code)
                        del active[jid]
                        write_json(state_file, state)
                        print(f"FINISHED {jid} exit={code} status={entry['status']}", flush=True)
                if all(state['jobs'][j['id']]['status'] == 'completed' or
                       len(state['jobs'][j['id']]['attempts']) >= max_attempts and
                       state['jobs'][j['id']]['status'] == 'failed' for j in jobs):
                    break
                for job in jobs:
                    if len(active) >= cap:
                        break
                    jid = job['id']
                    entry = state['jobs'][jid]
                    if entry['status'] not in ('pending', 'failed') or len(entry['attempts']) >= max_attempts:
                        continue
                    candidates = [r for r in replicas if ready[r['id']] and job['models'].issubset(r['models'])
                                  and sum(x[2] == r['id'] for x in active.values()) < r['capacity']]
                    if not candidates:
                        continue
                    # Preserve registry order for ties, so its preferred replicas run first.
                    candidates.sort(key=lambda r: (sum(x[2] == r['id'] for x in active.values()),
                                                   assignments[r['id']]))
                    replica = candidates[0]
                    try:
                        ready[replica['id']] = healthy(replica, job['models'])
                    except (OSError, ValueError, KeyError):
                        ready[replica['id']] = False
                    if not ready[replica['id']]:
                        continue
                    number = len(entry['attempts']) + 1
                    folder = state_dir / 'jobs' / jid
                    folder.mkdir(parents=True, exist_ok=True)
                    config_path = folder / f'attempt_{number}.json'
                    log_path = folder / f'attempt_{number}.log'
                    write_json(config_path, job['config'])
                    env = os.environ.copy()
                    env.update(CWAH_AGENT_API_BASE=replica['url'], CWAH_HUMAN_API_BASE=replica['url'],
                               QWEN_API_BASE=replica['url'],
                               CWAH_DISABLE_COUNTERFACTUAL_PROBE='1' if job['no_counterfactual'] else '0',
                               CWAH_RUN_NAME_PREFIX='no_counterfactual_' if job['no_counterfactual'] else '')
                    log = log_path.open('w', encoding='utf-8')
                    try:
                        proc = subprocess.Popen([sys.executable, '-u', str(SINGLE_RUNNER),
                                                 '--config', str(config_path)], env=env, stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                    except OSError:
                        log.close()
                        raise
                    entry['status'] = 'running'
                    entry['attempts'].append({'number': number, 'replica': replica['id'], 'pid': proc.pid,
                                              'log': str(log_path), 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')})
                    active[jid] = (proc, log, replica['id'])
                    assignments[replica['id']] += 1
                    write_json(state_file, state)
                    print(f"STARTED {jid} attempt={number} replica={replica['id']} pid={proc.pid}", flush=True)
                    idle_since = None
                if not active:
                    now = time.monotonic()
                    idle_since = now if idle_since is None else idle_since
                    if now - last_wait_notice >= 30:
                        print('No eligible replica; waiting for endpoint recovery.', flush=True)
                        last_wait_notice = now
                    if now - idle_since >= 300:
                        print('No job schedulable for five minutes; state saved for --resume.', flush=True)
                        return 3
                time.sleep(max(poll, 0.2))
                if not active:
                    for replica in replicas:
                        if not ready[replica['id']]:
                            try:
                                ready[replica['id']] = healthy(replica, replica['models'])
                            except (OSError, ValueError, KeyError):
                                pass
        except KeyboardInterrupt:
            interrupted = True
            print('Stopping only workers started by this batch.', flush=True)
            for proc, _, _ in active.values():
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGINT)
            for jid, (proc, log, _) in active.items():
                proc.wait()
                log.close()
                entry = state['jobs'][jid]
                settle(entry, entry['attempts'][-1], proc.returncode)
            write_json(state_file, state)
        finally:
            signal.signal(signal.SIGTERM, old_term)
        completed = sum(e['status'] == 'completed' for e in state['jobs'].values())
        print(f'Batch {bid}: {completed}/{len(jobs)} completed; state={state_file}', flush=True)
        return 130 if interrupted else (0 if completed == len(jobs) else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--replicas', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    batch, registry = args.batch.resolve(), args.replicas.resolve()
    bid = label(read_json(batch)['batch_id'])
    state_dir = RESULTS / 'tracev2' / 'batches' / bid
    try:
        return execute(batch, registry, state_dir, check=args.check, resume=args.resume)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f'Batch error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
