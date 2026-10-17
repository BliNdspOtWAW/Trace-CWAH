"""Validate and run every JSON in a config directory with bounded parallelism."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_experiment import validate


RUNNER_DIR = Path(__file__).resolve().parent


def preflight(config_dir):
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise ValueError(f"No *.json configs found in {config_dir}")
    ports = {}
    jobs = []
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        selected = validate(config)
        port = selected["port_request"]
        if port != "auto" and port in ports:
            raise ValueError(f"Duplicate port {port}: {ports[port]} and {path}")
        if port != "auto":
            ports[port] = path
        jobs.append((path, config, selected))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=RUNNER_DIR / "configs")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--check", action="store_true", help="validate all configs without starting experiments")
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be at least 1")
    config_dir = args.config_dir.resolve()
    jobs = preflight(config_dir)
    for path, config, selected in jobs:
        episode = selected["episode"]
        continuous = selected.get("continuous_subtasks", {"enabled": False})
        print(f"{path.name}: episode={selected['episode_index']} "
              f"({episode['task_id']}_{episode['task_name']}), profile={selected['profile_id']}, "
              f"TRACE={selected['trace_mode']}, "
              f"continuous={continuous['enabled']}, "
              f"port={selected['port_request']}", flush=True)
    if args.check:
        print(f"Validated {len(jobs)} configs. No Unity or LLM started.")
        return 0

    pending = list(jobs)
    active = []
    finished = []
    last_activity = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < args.max_parallel:
                path, config, _ = pending.pop(0)
                command = [sys.executable, str(RUNNER_DIR / "run_experiment.py"), "--config", str(path)]
                process = subprocess.Popen(command, env=os.environ.copy(), start_new_session=True)
                active.append((path, process))
                print(f"Started {path.name} (pid={process.pid})", flush=True)
                last_activity = time.monotonic()
            for item in active[:]:
                path, process = item
                exit_code = process.poll()
                if exit_code is None:
                    continue
                active.remove(item)
                finished.append({"config": str(path), "exit_code": exit_code})
                print(f"Finished {path.name}: exit_code={exit_code}", flush=True)
                last_activity = time.monotonic()
            if active:
                if time.monotonic() - last_activity >= 30:
                    names = ", ".join(path.stem for path, _ in active)
                    print(f"[waiting] active experiments: {names}", flush=True)
                    last_activity = time.monotonic()
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("Interrupt received; stopping active experiments.", file=sys.stderr, flush=True)
        for _, process in active:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        raise

    succeeded = sum(item["exit_code"] == 0 for item in finished)
    print(f"Batch finished: {succeeded}/{len(finished)} experiments succeeded.", flush=True)
    return 0 if all(item["exit_code"] == 0 for item in finished) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Batch configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
