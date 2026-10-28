"""Run independent Qwen experiments across separate local API tunnels.

Source env.sh qwen first. Each --endpoint is one independent model replica;
the scheduler permits one experiment per endpoint and reuses it when free.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from run_many import preflight


RUNNER = Path(__file__).resolve().parent / 'run_experiment.py'


def check_endpoint(base, model_ids):
    parsed = urlsplit(base)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
            or not parsed.port or parsed.path.rstrip('/') != '/v1'):
        raise ValueError(f'Endpoint must be a localhost HTTP /v1 URL: {base}')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base.rstrip('/') + '/models', timeout=8) as response:
        import json
        available = {item['id'] for item in json.load(response)['data']}
    missing = model_ids - available
    if missing:
        raise ValueError(f'{base} does not serve: {sorted(missing)}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--endpoint', action='append', required=True,
                        help='local SSH tunnel URL, repeat for each replica')
    parser.add_argument('--check', action='store_true', help='validate without starting experiments')
    args = parser.parse_args(argv)
    if len(args.endpoint) != len(set(args.endpoint)):
        parser.error('endpoints must be distinct')
    jobs = preflight(args.config_dir.resolve())
    if any(config['backend'] != 'openai' for _, config, _ in jobs):
        parser.error('all configs must use the OpenAI-compatible backend')
    if not all(os.environ.get(name) for name in ('CWAH_AGENT_API_KEY', 'CWAH_HUMAN_API_KEY')):
        parser.error('source env.sh qwen first')
    models = {config[role + '_model'] for _, config, _ in jobs for role in ('alice', 'bob')}
    for endpoint in args.endpoint:
        check_endpoint(endpoint, models)
        print(f'READY {endpoint}: {sorted(models)}', flush=True)
    print(f'Validated {len(jobs)} experiments across {len(args.endpoint)} replicas.', flush=True)
    if args.check:
        return 0

    pending = list(jobs)
    free = list(args.endpoint)
    active = {}
    failed = []
    try:
        while pending or active:
            while pending and free:
                endpoint = free.pop(0)
                path, _, _ = pending.pop(0)
                env = os.environ.copy()
                env['CWAH_AGENT_API_BASE'] = endpoint
                env['CWAH_HUMAN_API_BASE'] = endpoint
                env['QWEN_API_BASE'] = endpoint
                process = subprocess.Popen([sys.executable, '-u', str(RUNNER),
                                            '--config', str(path)], env=env,
                                           start_new_session=True)
                active[process.pid] = (process, path, endpoint)
                print(f'STARTED pid={process.pid} endpoint={endpoint} config={path.name}',
                      flush=True)
            for pid, (process, path, endpoint) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                del active[pid]
                free.append(endpoint)
                if code:
                    failed.append(path.name)
                print(f'FINISHED exit={code} endpoint={endpoint} config={path.name}', flush=True)
            if active:
                time.sleep(1)
    except KeyboardInterrupt:
        for process, _, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, _, _ in active.values():
            process.wait()
        raise
    print(f'Completed {len(jobs) - len(failed)}/{len(jobs)} experiments.', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
