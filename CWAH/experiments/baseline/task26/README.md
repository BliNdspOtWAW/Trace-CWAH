# CWAH baseline: task 26

This experiment targets the `26_prepare_food` task with teammate-profile
clusters `17`, `24`, `8`, and `23`. TRACE must be disabled for this baseline.

Keep each cluster's raw outputs separate. Use a timestamp for every batch so
the CWAH runner cannot silently reuse a previous `logs_agent_*.pik` file:

```text
CWAH/experiments/baseline/task26/
├── config.example.json                 # versioned, no credentials
├── config_<YYYYMMDD-HHMMSS>.json        # local batch manifest, no API key
├── summary_<YYYYMMDD-HHMMSS>.csv        # local cross-cluster summary
├── cluster_17/<YYYYMMDD-HHMMSS>/
│   ├── console.log
│   ├── results.pik
│   └── logs_agent_26_prepare_food_*.pik
├── cluster_24/<YYYYMMDD-HHMMSS>/
├── cluster_8/<YYYYMMDD-HHMMSS>/
└── cluster_23/<YYYYMMDD-HHMMSS>/
```

Pass each timestamped cluster directory as `--record_dir` (or `RECORD_DIR` in
`scripts/run_one_llm_llm.sh`). Keep the same timestamp across the four clusters
in one batch. Do not place credentials in the manifest or console log. The
generated manifests, summaries, and raw outputs are ignored by Git.

The manifest should record the effective model IDs for both players, a
non-secret endpoint label, temperature, max tokens, number of runs, prompt
versions, code commit, dataset path/hash, dataset index, simulator version,
cluster IDs, and start time. The summary should include one row per
cluster/repetition with success (`S`), episode steps (`L`), elapsed time,
message counts, planner-call counts, and any error. Preserve the native
`results.pik` and `logs_agent_*.pik` for detailed analysis; `raw_usage` in the
current LLM implementation is not a DeepSeek token count.

**Dataset prerequisite:** the current `CWAH-MultiPlayer/dataset/test_env_set_help.pik`
contains only one `0_setup_table` episode. The CWAH README does not provide a
download link for the full task dataset. Obtain the original dataset containing
`26_prepare_food` before starting this experiment, verify its SHA-256 and the
index of task ID 26, and pass its path via `DATASET_PATH`. In this runner,
`--test_task`/`TASK` is a dataset *index*, not the task ID. Generating a new
single-task dataset would not reproduce the original task 26 or its profile
lookup key.

The upstream Watch-And-Help repository publishes a
[`test_env_set_help.pik`](https://github.com/xavierpuigf/watch_and_help/blob/main/dataset/test_env_set_help.pik),
but it has not been verified as the exact CollabBench task-26 dataset. Do not
silently substitute it for the requested benchmark input.
