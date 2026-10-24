# Qwen3 local backend

The Hugging Face backend replaces generation, not the TRACE algorithm or task
rules. API experiments continue to use the OpenAI-compatible backend.

Local configurations select `backend: huggingface` and `local_model` options.
The tokenizer chat template disables thinking by default. Weights are shared
within one experiment process, but dialogue state is not. Inference is
sequential; parallel experiment processes each load their own model copy.
Context overflow raises an error rather than silently truncating history.

Official sources:
- https://huggingface.co/Qwen/Qwen3-8B
- https://modelscope.cn/models/Qwen/Qwen3-8B

The resumable downloader records file revisions and SHA256 checksums in
`trace_download_manifest.json`. `MODEL_READY` indicates checksum verification
has completed. Store weights outside Git; the repository-local model directory
`CWAH/models/` is ignored.

From the repository root, with the CWAH environment activated:

```bash
python CWAH/experiment_runner/download_qwen3.py --destination <MODEL_DIRECTORY>
```

Templates are under `CWAH/experiment_runner/configs/qwen3/`. Configure model paths
in private local JSON files. Configuration `--check` does not load weights.
Coordinate an explicit GPU before loading; device availability and memory must
be checked for each run. Parallel configurations do not automatically share one
GPU-resident model.

Qwen-specific Reactive prompts reinforce closed output tags. They improve
formatting but do not guarantee correct coordination or eliminate repetitive
communication. Task control protocol and verification details are documented in
`CWAH_TASK_CONTROL.md` and the commit records.

The inherited budget-as-success issue remains: use Continuous completed-subtask
counts rather than the legacy binary success flag for fixed-budget comparisons.
