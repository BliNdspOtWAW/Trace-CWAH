"""Run one WAH episode with an independent Bob profile and optional TRACE."""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
import pickle
import re
import socket
import subprocess
import sys
import signal
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from metric_reports import export_metrics
from result_storage import finalize_artifacts, write_json


TRACE_ROOT = Path(__file__).resolve().parents[2]
CWAH_ROOT = TRACE_ROOT / "CWAH"
GAME_ROOT = CWAH_ROOT / "CWAH-MultiPlayer"
LOCK_ROOT = Path(__file__).resolve().parent / ".locks"
AUTO_PORTS = range(6390, 6491)


class PortUnavailable(ValueError):
    """The requested Unity port is reserved or occupied."""


def _path(value):
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else TRACE_ROOT / candidate).resolve()


def _reply_prompt_path(backend, model, role):
    """Specialize formatting only for locally configured Qwen model families."""
    qwen = False
    if backend == 'huggingface':
        model_config = _path(model) / 'config.json'
        if model_config.is_file():
            kind = json.loads(model_config.read_text(encoding='utf-8')).get('model_type', '')
            qwen = isinstance(kind, str) and kind.startswith('qwen')
    variant = 'Qwen_' if qwen else ''
    path = GAME_ROOT / 'LLM' / f'prompt_Reply_{variant}{role.title()}.txt'
    if not path.is_file():
        raise ValueError(f'Missing reply prompt: {path}')
    return path


def _label(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("bob_profile.id must contain only letters, numbers, _ or -")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(config):
    """Validate an explicit local dataset/configuration; never infer a task mapping."""
    dataset_cfg = config["dataset"]
    backend = config.get('backend', 'openai')
    if backend not in ('openai', 'huggingface'):
        raise ValueError('backend must be openai or huggingface')
    local_options = config.get('local_model', {})
    if not isinstance(local_options, dict):
        raise ValueError('local_model must be an object')
    if backend == 'huggingface':
        if set(local_options) - {'dtype', 'thinking', 'device_map', 'context_limit'}:
            raise ValueError('Unsupported local_model option')
        if local_options.get('dtype', 'bfloat16') not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('Unsupported local model dtype')
        if type(local_options.get('thinking', False)) is not bool:
            raise ValueError('thinking must be boolean')
        if type(local_options.get('context_limit', 32768)) is not int or local_options.get('context_limit', 32768) < 1:
            raise ValueError('context_limit must be a positive integer')
        device = local_options.get('device_map', 'cuda:0')
        if not isinstance(device, str) or not re.fullmatch(r'cuda:\d+|cpu|auto', device):
            raise ValueError('device_map must be cuda:N, cpu or auto')
        for role in ('alice', 'bob'):
            path = _path(config[role + '_model'])
            if not (path / 'config.json').is_file():
                raise ValueError(f'Local model config missing: {path}')
            index_file = path / 'model.safetensors.index.json'
            if index_file.is_file():
                shards = set(json.loads(index_file.read_text()).get('weight_map', {}).values())
                if not shards or any(not (path / shard).is_file() for shard in shards):
                    raise ValueError(f'Local model weights incomplete: {path}')
            elif not (path / 'model.safetensors').is_file():
                raise ValueError(f'Local model weights incomplete: {path}')
    dataset = _path(dataset_cfg["path"])
    executable = _path(config["executable_file"])
    for name, path in (("dataset", dataset), ("executable", executable)):
        if not path.is_file():
            raise ValueError(f"{name} file not found: {path}")
    if not os.access(executable, os.X_OK):
        raise ValueError(f"executable is not executable: {executable}")
    index = dataset_cfg["episode_index"]
    if type(index) is not int or index < 0:
        raise ValueError("episode_index must be a nonnegative integer")
    # Only load a locally trusted WAH pickle. This is the file the evaluator loads.
    with dataset.open("rb") as stream:
        episodes = pickle.load(stream)
    if not isinstance(episodes, (list, tuple)) or index >= len(episodes):
        raise ValueError(f"episode_index {index} is outside this dataset")
    episode = episodes[index]
    if not isinstance(episode, dict) or not all(
        key in episode for key in ("task_id", "task_name", "env_id", "init_graph", "task_goal")
    ):
        raise ValueError("selected episode lacks required initialization fields")
    mode = config["trace_mode"]
    if mode not in ("off", "on"):
        raise ValueError("trace_mode must be off or on")
    continuous = config.get("continuous_subtasks", {"enabled": False})
    if not isinstance(continuous, dict) or type(continuous.get("enabled", False)) is not bool:
        raise ValueError("continuous_subtasks must be an object with a boolean enabled field")
    continuous = {
        "enabled": continuous.get("enabled", False),
        "active_count": continuous.get("active_count", 3),
        "seed": continuous.get("seed", 1),
        "generation_mode": continuous.get("generation_mode", "state_based"),
        "max_generation_attempts": continuous.get("max_generation_attempts", 64),
        "reverse_cooldown_steps": continuous.get("reverse_cooldown_steps", 20),
        "targets": continuous.get("targets", []),
    }
    if type(continuous["active_count"]) is not int or continuous["active_count"] < 1:
        raise ValueError("continuous_subtasks.active_count must be a positive integer")
    if type(continuous["seed"]) is not int:
        raise ValueError("continuous_subtasks.seed must be an integer")
    if continuous["generation_mode"] != "state_based":
        raise ValueError("continuous_subtasks.generation_mode must be state_based")
    if type(continuous["max_generation_attempts"]) is not int or continuous["max_generation_attempts"] < 1:
        raise ValueError("continuous_subtasks.max_generation_attempts must be a positive integer")
    if type(continuous["reverse_cooldown_steps"]) is not int or continuous["reverse_cooldown_steps"] < 0:
        raise ValueError("continuous_subtasks.reverse_cooldown_steps must be a nonnegative integer")
    if continuous["enabled"]:
        initial_goal = episode["task_goal"].get(0, {})
        active_predicates = sum(value > 0 for value in initial_goal.values())
        if active_predicates > continuous["active_count"]:
            raise ValueError(
                "selected episode starts with {} active predicates, exceeding "
                "continuous_subtasks.active_count {}".format(
                    active_predicates, continuous["active_count"]
                )
            )
        nodes = {node["id"]: node for node in episode["init_graph"].get("nodes", [])}
        if not continuous["targets"]:
            raise ValueError("continuous_subtasks.targets must contain at least one target")
        seen_targets = set()
        for target in continuous["targets"]:
            required = {"relation", "object_id", "object_class", "capacity"}
            if not isinstance(target, dict) or not required.issubset(target):
                raise ValueError("each continuous target requires relation, object_id, object_class, capacity")
            relation = target["relation"]
            target_id = target["object_id"]
            capacity = target["capacity"]
            if relation not in ("on", "inside"):
                raise ValueError("continuous target relation must be on or inside")
            if type(target_id) is not int or type(capacity) is not int or capacity < 1:
                raise ValueError("continuous target object_id and positive capacity must be integers")
            node = nodes.get(target_id)
            if node is None or node.get("class_name") != target["object_class"]:
                raise ValueError(f"continuous target {target_id} does not match the episode graph")
            required_property = "SURFACES" if relation == "on" else "CONTAINERS"
            if required_property not in node.get("properties", []):
                raise ValueError(f"continuous target {target_id} does not support {relation}")
            key = (relation, target_id)
            if key in seen_targets:
                raise ValueError(f"duplicate continuous target {key}")
            seen_targets.add(key)
    prompt_files = {}
    for role in ("alice", "bob"):
        raw = config.get(f"{role}_prompt_file")
        if raw is not None and (not isinstance(raw, str) or not raw.strip()):
            raise ValueError(f"{role}_prompt_file must be a nonempty path")
        prompt_path = _path(raw) if raw else GAME_ROOT / "LLM" / f"prompt_{role.title()}.txt"
        if not prompt_path.is_file() or not prompt_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"{role} prompt missing or empty: {prompt_path}")
        required = ('$PUBLIC_TASKS', '$AGENT_CONTROL', '<subtask_id>')
        if not all(value in prompt_path.read_text(encoding='utf-8') for value in required):
            raise ValueError(f'{role} prompt does not support task-control protocol v2: {prompt_path}')
        prompt_files[role] = prompt_path
    if mode == "on" and "$TRACE_BELIEF" not in prompt_files["alice"].read_text(encoding="utf-8"):
        raise ValueError("TRACE-on Alice prompt must contain $TRACE_BELIEF")
    for role in ('alice', 'bob'):
        reply_path = _reply_prompt_path(backend, config[role + '_model'], role)
        if not reply_path.is_file() or not reply_path.read_text(encoding='utf-8').strip():
            raise ValueError(f'{role} reply prompt missing or empty: {reply_path}')
        if not all(value in reply_path.read_text(encoding='utf-8')
                   for value in ('$PUBLIC_TASKS', '$AGENT_CONTROL', '<new_subtask_id>')):
            raise ValueError(f'{role} reply prompt does not support task-control protocol v2: {reply_path}')
        if role == 'alice' and mode == 'on' and '$TRACE_BELIEF' not in reply_path.read_text(encoding='utf-8'):
            raise ValueError('TRACE-on Alice reply prompt must contain $TRACE_BELIEF')
    experiment_label = config.get("experiment_label")
    if experiment_label is not None:
        experiment_label = _label(experiment_label)
    profile_cfg = config.get("bob_profile")
    profile_id, profile_path = "none", None
    if profile_cfg is not None:
        profile_id = _label(profile_cfg["id"])
        profile_path = _path(profile_cfg["text_file"])
        if not profile_path.is_file() or not profile_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Bob profile missing or empty: {profile_path}")
    for name in ("alice_model", "bob_model"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"{name} must be a nonempty model ID")
    for name in ("max_tokens", "max_episode_length", "num_runs"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["temperature"]) not in (int, float) or config["temperature"] < 0:
        raise ValueError("temperature must be nonnegative")
    requested_port = config.get("port", "auto")
    if requested_port is None:
        requested_port = "auto"
    if requested_port != "auto" and (type(requested_port) is not int or not 1 <= requested_port <= 65535):
        raise ValueError("port must be 'auto' or an integer between 1 and 65535")
    return {
        "backend": backend, "local_model": local_options,
        "dataset": dataset,
        "dataset_size": len(episodes),
        "episode_index": index,
        "episode": episode,
        "executable": executable,
        "profile_id": profile_id,
        "profile_path": profile_path,
        "trace_mode": mode,
        "continuous_subtasks": continuous,
        "prompt_files": prompt_files,
        "experiment_label": experiment_label,
        "port_request": requested_port,
    }


def _git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRACE_ROOT,
                            text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty():
    result = subprocess.run(["git", "status", "--porcelain"], cwd=TRACE_ROOT,
                            text=True, capture_output=True, check=False)
    return bool(result.stdout) if result.returncode == 0 else None


@contextmanager
def port_guard(port):
    """Prevent two local runners from launching Unity on the same port."""
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOCK_ROOT / f"port_{port}.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PortUnavailable(f"port {port} is reserved by another CWAH experiment") from exc
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError as exc:
                raise PortUnavailable(f"port {port} is occupied") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def allocated_port(requested_port):
    """Choose a free port while holding its lock until the experiment exits."""
    candidates = AUTO_PORTS if requested_port == "auto" else (requested_port,)
    with ExitStack() as stack:
        for port in candidates:
            try:
                stack.enter_context(port_guard(port))
            except PortUnavailable:
                if requested_port != "auto":
                    raise
                continue
            yield port
            return
    raise ValueError(f"No free Unity port in {AUTO_PORTS.start}-{AUTO_PORTS.stop - 1}")


def _summary(record_dir, index):
    path = record_dir / "results.pik"
    if not path.exists():
        checkpoint = record_dir / 'trajectory_checkpoint.pik'
        if checkpoint.exists():
            with checkpoint.open('rb') as stream:
                partial = pickle.load(stream)
            return {
                'status': 'interrupted', 'episode_index': index,
                'steps': [partial.get('environment_steps', 0)],
                'trajectories': [checkpoint.name],
                'termination_reason': partial.get('termination_reason') or 'incomplete',
            }
        return {"status": "no_results", "episode_index": index}
    try:
        with path.open("rb") as stream:
            result = pickle.load(stream)
    except (pickle.UnpicklingError, EOFError):
        return {"status": "unreadable_results", "episode_index": index}
    values = result.get(index, {})
    successes = values.get("S", [])
    steps = values.get("L", [])
    summary = {"status": "completed" if successes else "no_episode_results", "episode_index": index,
               "runs": len(successes), "successes": sum(successes),
               "success_rate": sum(successes) / len(successes) if successes else None,
               "success": successes, "steps": steps,
               "mean_steps": sum(steps) / len(steps) if steps else None}
    trajectory_files = sorted(record_dir.glob("logs_agent_*_*.pik"))
    summary["trajectories"] = [path.name for path in trajectory_files]
    summary["planner_calls"] = {"alice": 0, "bob": 0}
    summary["communication_reply_calls"] = 0
    summary["trace_updater_calls"] = 0
    summary["trace_updater_token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    summary["continuous_subtasks"] = None
    for trajectory_file in trajectory_files:
        try:
            with trajectory_file.open("rb") as stream:
                trajectory = pickle.load(stream)
        except (pickle.UnpicklingError, EOFError):
            summary.setdefault("unreadable_trajectories", []).append(trajectory_file.name)
            continue
        llm_calls = trajectory.get("LLM", {})
        if trajectory.get('interrupted'):
            summary['status'] = 'interrupted'
            summary['termination_reason'] = 'interrupted'
        summary["planner_calls"]["alice"] += len(llm_calls.get(0, []))
        summary["planner_calls"]["bob"] += len(llm_calls.get(1, []))
        summary["communication_reply_calls"] += trajectory.get("communication_calls", 0)
        trace = trajectory.get("trace") or {}
        summary["trace_updater_calls"] += trace.get("updater_calls", 0)
        for key in summary["trace_updater_token_usage"]:
            summary["trace_updater_token_usage"][key] += trace.get("updater_token_usage", {}).get(key, 0)
        if trajectory.get("continuous_subtasks") is not None:
            summary["continuous_subtasks"] = trajectory["continuous_subtasks"]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.local.json"))
    parser.add_argument("--check", action="store_true", help="validate only; do not start Unity or call a model")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = validate(config)
    episode = selected["episode"]
    index = selected["episode_index"]
    print(f"WAH episode index={index}, task={episode['task_id']}_{episode['task_name']}, "
          f"env={episode['env_id']}, Bob profile={selected['profile_id']}, "
          f"TRACE={selected['trace_mode']}, continuous={selected['continuous_subtasks']['enabled']}, "
          f"port={selected['port_request']}")
    print(f"dataset={selected['dataset']}")
    print("planner_prompts=" + ", ".join(f"{role}:{path.name}" for role, path in selected["prompt_files"].items()))
    if args.check:
        print("Configuration valid; Unity and LLM were not started.")
        return 0

    for name in (("CWAH_AGENT_API_BASE", "CWAH_HUMAN_API_BASE") if selected['backend'] == 'openai' else ()):
        if not os.environ.get(name):
            raise ValueError(f"Set {name} in the shell before running")
    with allocated_port(selected["port_request"]) as port:
        print(f"assigned_port={port}", flush=True)
        return _run_validated(config, selected, episode, index, port)


def _run_validated(config, selected, episode, index, port):
    reply_files = {role: _reply_prompt_path(selected['backend'], config[role + '_model'], role)
                   for role in ('alice', 'bob')}
    run_started = time.monotonic()
    profile_id = selected["profile_id"]
    task_name = _label(str(episode["task_name"]))
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    label = f"_{selected['experiment_label']}" if selected["experiment_label"] else ""
    continuous_label = "_continuous" if selected["continuous_subtasks"]["enabled"] else ""
    run_id = (f"{task_name}_ep{index}_profile{profile_id}_trace-{selected['trace_mode']}"
              f"{continuous_label}{label}_{timestamp}")
    category = "trace" if selected["trace_mode"] == "on" else "baseline"
    record_dir = CWAH_ROOT / "experiments" / category / run_id
    record_dir.mkdir(parents=True, exist_ok=False)
    runtime_dir = record_dir / '.runtime'
    runtime_dir.mkdir()
    (record_dir / 'calls.jsonl').touch()
    write_json(record_dir / 'summary.json', {'status': 'running', 'error': None,
                                           'episode_index': index})
    child_env = os.environ.copy()
    child_env['CWAH_AGENT_REPLY_PROMPT_FILE'] = str(reply_files['alice'])
    child_env['CWAH_HUMAN_REPLY_PROMPT_FILE'] = str(reply_files['bob'])
    child_env["CWAH_AGENT_LM_ID"] = config["alice_model"]
    child_env["CWAH_HUMAN_LM_ID"] = config["bob_model"]
    if selected['backend'] == 'huggingface':
        child_env['CWAH_AGENT_LM_ID'] = str(_path(config['alice_model']))
        child_env['CWAH_HUMAN_LM_ID'] = str(_path(config['bob_model']))
        child_env['CWAH_LOCAL_MODEL_OPTIONS'] = json.dumps(selected['local_model'])
        for role in ('CWAH_AGENT', 'CWAH_HUMAN'):
            for suffix in ('API_BASE', 'API_KEY', 'WIRE_API', 'REASONING_EFFORT'):
                child_env.pop(role + '_' + suffix, None)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env['CWAH_METRICS_PATH'] = str(record_dir / 'calls.jsonl')
    child_env['CWAH_METRICS_RUN_ID'] = run_id
    child_env.pop("CWAH_PROFILE_SOURCE_TASK", None)
    child_env.pop("CWAH_PROFILE_EXPECTED_TASK", None)

    metadata = {
        'result_format_version': 2, 'trajectory_format_version': 1,
        'decision_protocol_version': 2,
        'decision_timing': 'shared_step_start_snapshot; active_barrier; reactive; physical_step',
        'macro_switch_timing': 'after_current_skill',
        'llm_metrics_schema_version': 1,
        'llm_metrics_code_sha256': {
            'telemetry.py': _sha256(GAME_ROOT / 'LLM' / 'telemetry.py'),
            'metric_reports.py': _sha256(Path(__file__).with_name('metric_reports.py')),
        },
        'llm_metrics_files': ['calls.jsonl'],
        "backend": selected['backend'], "local_model": selected['local_model'],
        "local_model_provenance": {
            role: {
                'resolved_path': str(_path(config[role + '_model'])),
                'files_sha256': {
                    filename: _sha256(_path(config[role + '_model']) / filename)
                    for filename in ('config.json', 'tokenizer_config.json',
                                     'model.safetensors.index.json', 'trace_download_manifest.json')
                    if (_path(config[role + '_model']) / filename).is_file()
                },
            } for role in ('alice', 'bob')
        } if selected['backend'] == 'huggingface' else None,
        "run_id": run_id, "started_at": datetime.now().astimezone().isoformat(),
        "experiment_kind": "WAH episode with independent Bob profile; not a CollabBench Task 26 reproduction",
        "dataset_path": str(selected["dataset"]), "dataset_sha256": _sha256(selected["dataset"]),
        "episode_index": index, "task_id": episode["task_id"], "task_name": episode["task_name"],
        "environment_id": episode["env_id"], "task_goal": episode["task_goal"],
        "bob_profile_id": profile_id,
        "bob_profile_path": str(selected["profile_path"]) if selected["profile_path"] else None,
        "bob_profile_sha256": _sha256(selected["profile_path"]) if selected["profile_path"] else None,
        "trace_mode": selected["trace_mode"], "alice_model": config["alice_model"],
        "api_protocol": {
            role: child_env.get(prefix + '_WIRE_API', 'chat') if selected['backend'] == 'openai' else None
            for role, prefix in [('alice', 'CWAH_AGENT'), ('bob', 'CWAH_HUMAN')]
        },
        "reasoning_effort": {
            role: child_env.get(prefix + '_REASONING_EFFORT', 'high')
            if child_env.get(prefix + '_WIRE_API', 'chat') == 'responses' else None
            for role, prefix in [('alice', 'CWAH_AGENT'), ('bob', 'CWAH_HUMAN')]
        },
        "continuous_subtasks": selected["continuous_subtasks"],
        "experiment_label": selected["experiment_label"],
        "planner_prompt_files": {role: str(path) for role, path in selected["prompt_files"].items()},
        "planner_prompt_sha256": {role: _sha256(path) for role, path in selected["prompt_files"].items()},
        "reply_prompt_files": {
            role: str(reply_files[role])
            for role in ('alice', 'bob')
        },
        "reply_prompt_sha256": {
            role: _sha256(reply_files[role])
            for role in ('alice', 'bob')
        },
        "bob_model": config["bob_model"], "temperature": config["temperature"],
        "max_tokens": config["max_tokens"], "max_episode_length": config["max_episode_length"],
        "num_runs": config["num_runs"], "port_request": selected["port_request"], "port": port,
        "executable_file": str(selected["executable"]),
        "api_endpoint_hosts": {
            "alice": urlsplit(child_env.get("CWAH_AGENT_API_BASE", '')).hostname,
            "bob": urlsplit(child_env.get("CWAH_HUMAN_API_BASE", '')).hostname,
        },
        "code_commit": _git_revision(), "code_has_uncommitted_changes": _git_dirty(),
        "prompt_sha256": {
            name: _sha256(path) for name, path in {
                "alice": selected['prompt_files']['alice'],
                "bob": selected['prompt_files']['bob'],
                "reply_alice": reply_files['alice'],
                "reply_bob": reply_files['bob'],
                "updater": GAME_ROOT / "trace_tf" / "prompt_Updater.txt",
                "trace_policy": GAME_ROOT / "trace_tf" / "policy.py",
            }.items()
        },
        "observation_type": "partial", "communication": True,
    }
    (record_dir / "config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n",
                                             encoding="utf-8")
    command = [sys.executable, "-u", "testing_agents/test_symbolic_LLMs.py",
               "--communication", "--act", "--cot", "--source", selected['backend'],
               "--mode", run_id, "--record_dir", str(runtime_dir),
               "--dataset_path", str(selected["dataset"]),
               "--executable_file", str(selected["executable"]),
               "--base-port", str(port), "--lm_id", config["alice_model"],
               "--api_base", child_env.get("CWAH_AGENT_API_BASE", ''),
               "--t", str(config["temperature"]), "--max_tokens", str(config["max_tokens"]),
               "--max-episode-length", str(config["max_episode_length"]),
               "--num_runs", str(config["num_runs"]), "--num-per-task", "1",
               "--test_task", str(index), "--trace-mode", selected["trace_mode"],
               "--alice-prompt-file", str(selected["prompt_files"]["alice"]),
               "--bob-prompt-file", str(selected["prompt_files"]["bob"])]
    if selected["continuous_subtasks"]["enabled"]:
        continuous_runtime = {
            key: selected["continuous_subtasks"][key]
            for key in (
                "generation_mode", "max_generation_attempts",
                "reverse_cooldown_steps", "targets",
            )
        }
        command += [
            "--continuous-subtasks",
            "--continuous-active-count", str(selected["continuous_subtasks"]["active_count"]),
            "--continuous-seed", str(selected["continuous_subtasks"]["seed"]),
            "--continuous-config-json", json.dumps(continuous_runtime, separators=(",", ":")),
        ]
    if selected["profile_path"]:
        command += ["--teammate-profile-file", str(selected["profile_path"])]

    print(f"results={record_dir}", flush=True)
    with (record_dir / "console.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(command, cwd=GAME_ROOT, env=child_env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
        except OSError as exc:
            log.write('Subprocess startup failed: ' + type(exc).__name__ + '\n')
            return _finish_run(record_dir, index, -1, config['num_runs'], run_started,
                               startup_error=type(exc).__name__)
        reported_run_index = -1
        previous_handlers = {}
        def forward_stop(signum, frame):
            if process.poll() is None:
                process.send_signal(signum)
                print('[stop] Waiting for the current step and trajectory save.', flush=True)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, forward_stop)
        try:
            for line in process.stdout:
                log.write(line)
                step_match = re.match(r"^Step (\d+), Executing script:", line)
                if step_match:
                    step = int(step_match.group(1)) + 1
                    with (record_dir / 'calls.jsonl').open('a', encoding='utf-8') as progress:
                        progress.write(json.dumps({'event': 'environment_step_start', 'step': step,
                                                   'run_index': max(reported_run_index, 0),
                                                   'elapsed_seconds': time.monotonic() - run_started}) + '\n')
                    print(f"[progress] {run_id} step={step}/{config['max_episode_length']}", flush=True)
                    log.flush()
                elif line.startswith(("Init Env", "episode:", "success", "failure", "steps:", "[stop]")):
                    if line.startswith('episode:'):
                        reported_run_index += 1
                    print(line, end="", flush=True)
            exit_code = process.wait()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    return _finish_run(record_dir, index, exit_code, config['num_runs'], run_started)


def _finish_run(record_dir, index, exit_code, expected_runs, run_started, startup_error=None):
    try:
        summary = _summary(record_dir / '.runtime', index)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        summary = {'status': 'failed', 'episode_index': index,
                   'error': {'type': type(exc).__name__}}
    try:
        summary = finalize_artifacts(record_dir, index, summary, exit_code, expected_runs)
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        summary.update(status='failed', error={'type': type(exc).__name__,
                       'details': 'Artifact consolidation failed; original .runtime data retained.'})
    if startup_error:
        summary['error'] = {'type': startup_error, 'details': 'Subprocess startup failed; see console.log.'}
    summary["runner_exit_code"] = exit_code
    summary['run_wall_seconds'] = time.monotonic() - run_started
    continuous_summary = summary.get('continuous_subtasks')
    if isinstance(continuous_summary, dict):
        summary['completed_subtasks_per_minute'] = (
            continuous_summary.get('completed_subtasks', 0) * 60 / max(summary['run_wall_seconds'], 1e-9))
    try:
        metrics = export_metrics(record_dir)
        summary['llm_metrics'] = {k: v for k, v in metrics.items() if k not in ('step_windows', 'in_flight_calls')}
    except (OSError, ValueError) as exc:
        summary['llm_metrics'] = {'available': False, 'export_error_type': type(exc).__name__}
    write_json(record_dir / 'summary.json', summary)
    outcome = summary['status']
    print(f"{outcome}: {record_dir}")
    return 0 if outcome == "completed" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, ValueError, OSError) as exc:
        print(f"Experiment configuration error: {exc}", file=sys.stderr)
        sys.exit(2)
