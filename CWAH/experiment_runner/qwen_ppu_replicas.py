"""Deploy independent Qwen API replicas on selected physical PPUs.

Run this script on the PPU host. It clones the verified container's read-only
model mount and startup files, while giving each replica its own runtime cache,
device, container, and localhost port. The default action is a read-only check.
"""

import argparse
import json
import shutil
import socket
import subprocess
from pathlib import Path


TEMPLATE = 'qwen38-api-ppu6'
MIN_FREE_MIB = 86016
FILES = ('entrypoint.sh', 'serve_api.sh', 'healthcheck.py')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def inspect(name):
    result = subprocess.run(['docker', 'inspect', name], capture_output=True, text=True)
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def gpu_status():
    rows = command('nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                   '--format=csv,noheader,nounits')
    return {int(values[0]): (int(values[1]), int(values[2]))
            for row in rows.splitlines()
            for values in [[part.strip() for part in row.split(',')]]}


def template_paths(container):
    mounts = {item['Destination']: Path(item['Source']) for item in container['Mounts']}
    return mounts['/model'], mounts['/runtime']


def port_available(port):
    with socket.socket() as sock:
        try:
            sock.bind(('127.0.0.1', port))
        except OSError:
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gpus', nargs='+', type=int, help='physical PPU indices')
    parser.add_argument('--start', action='store_true', help='create and start replicas')
    parser.add_argument('--resource-approved', action='store_true',
                        help='confirm the resource owner approved these PPUs')
    parser.add_argument('--allow-busy', action='store_true',
                        help='allow another process to use compute on the same PPU')
    parser.add_argument('--min-free-mib', type=int, default=MIN_FREE_MIB)
    args = parser.parse_args(argv)
    if args.start and not args.resource_approved:
        parser.error('--start requires --resource-approved')
    if len(args.gpus) != len(set(args.gpus)) or any(gpu < 0 for gpu in args.gpus):
        parser.error('PPU indices must be distinct nonnegative integers')

    template = inspect(TEMPLATE)
    if not template or not template['State']['Running']:
        parser.error(f'verified template container {TEMPLATE} must be running')
    model_dir, template_runtime = template_paths(template)
    for path in (model_dir, *(template_runtime / name for name in FILES)):
        if not path.exists():
            parser.error(f'missing template dependency: {path}')
    status = gpu_status()
    plans = []
    for gpu in args.gpus:
        if gpu not in status:
            parser.error(f'PPU {gpu} does not exist')
        name = f'qwen38-api-ppu{gpu}'
        port = 18032 + gpu
        free, utilization = status[gpu]
        existing = inspect(name)
        if existing:
            if not existing['State']['Running']:
                parser.error(f'{name} exists but is stopped; inspect it manually')
            print(f'EXISTS gpu={gpu} name={name} port={port}')
            continue
        if free < args.min_free_mib:
            parser.error(f'PPU {gpu}: {free} MiB free; need at least {args.min_free_mib} MiB')
        if utilization > 5 and not args.allow_busy:
            parser.error(f'PPU {gpu}: {utilization}% compute use; pass --allow-busy if approved')
        if not args.allow_busy and 'No running processes found' not in command('ppu-smi', '-i', str(gpu)):
            parser.error(f'PPU {gpu}: another process is present; pass --allow-busy if approved')
        if not Path(f'/dev/alixpu_ppu{gpu}').exists():
            parser.error(f'PPU {gpu}: device node is missing')
        if not port_available(port):
            parser.error(f'localhost port {port} is occupied')
        runtime = template_runtime.parent / f'{template_runtime.name}-ppu{gpu}'
        plans.append((gpu, name, port, free, utilization, runtime))
        print(f'READY gpu={gpu} free_mib={free} util={utilization}% '
              f'port={port} runtime={runtime}')
    if not args.start:
        return 0

    for gpu, name, port, _, _, runtime in plans:
        runtime.mkdir(parents=True, exist_ok=True)
        for filename in FILES:
            source = template_runtime / filename
            destination = runtime / filename
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                parser.error(f'existing runtime file differs: {destination}')
            shutil.copy2(source, destination)
        options = [
            'docker', 'run', '-d', '--name', name,
            '--label', f'codex.qwen.physical_ppu={gpu}', '--restart=no',
            '--network', 'bridge', '--shm-size', '8g', '--cpus', '8', '--memory', '128g',
            '--device=/dev/alixpu', '--device=/dev/alixpu_ctl', '--device=/dev/alixpu_sep',
            f'--device=/dev/alixpu_ppu{gpu}',
            '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'OMP_NUM_THREADS=4',
            '-e', 'VLLM_WORKER_MULTIPROC_METHOD=spawn',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
            '-e', 'TRITON_CACHE_DIR=/runtime/cache/triton',
            '-e', 'VLLM_CACHE_ROOT=/runtime/cache/vllm',
            '-e', 'CUDA_CACHE_PATH=/runtime/cache/cuda',
            '-v', f'{model_dir}:/model:ro', '-v', f'{runtime}:/runtime',
            '-p', f'127.0.0.1:{port}:8000',
            '--log-driver=json-file', '--log-opt', 'max-size=20m',
            '--log-opt', 'max-file=3',
            '--health-cmd', 'python /runtime/healthcheck.py',
            '--health-interval=60s', '--health-timeout=20s',
            '--health-retries=3', '--health-start-period=600s',
            '--entrypoint', '/bin/bash', template['Config']['Image'],
            '/runtime/entrypoint.sh', 'bash', '/runtime/serve_api.sh',
        ]
        subprocess.run(options, check=True, stdout=subprocess.DEVNULL)
        print(f'STARTED gpu={gpu} name={name} port={port}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
