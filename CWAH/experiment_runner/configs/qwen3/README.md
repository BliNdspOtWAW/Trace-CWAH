# Qwen3-8B local backend

Copy each `.json.example` to `.json` before running. These templates preserve
Episode 44, Peer17 and existing Continuous/TRACE semantics; only the backend
changes. No API key or `env.sh` is needed. Prompts are not modified.

The ignored `CWAH/models/Qwen3-8B` link points to downloaded original weights.
Model downloads record official source, file revisions and SHA256 checksums.
Run `--check` first; it validates configuration but does not load weights or
prove sufficient GPU memory. No history is silently truncated.

```bash
conda activate CWAH
python CWAH/experiment_runner/run_many.py --config-dir CWAH/experiment_runner/configs/qwen3 --check
# Only after permission to use a GPU; substitute its physical index:
CUDA_VISIBLE_DEVICES=4 python CWAH/experiment_runner/run_many.py --config-dir CWAH/experiment_runner/configs/qwen3 --max-parallel 1
```

`device_map: cuda:0` means the first GPU visible to this process, not necessarily
physical GPU 0. Do not assume GPU 4 is available. Each experiment loads its own
weights, shared within that process by Alice/Bob/Updater, never their contexts.
Thinking is disabled; decoding preserves multiline outputs. Token counts include
the model chat template. The 100-step templates are for later experiments, not
evidence that a local inference or TRACE experiment has already passed.
