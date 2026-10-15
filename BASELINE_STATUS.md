# TRACE baseline verification status

Checked on 2026-10-15 for the initial `main` baseline. This baseline records
a working research environment and experiment pipeline, not a claim that
either benchmark's full task set has passed.

## Checks completed

- CWAH conda environment: Python imports and `pip check` pass.
- VirtualHome executable is present and executable; the `wah` dependency is
  at commit `6773be207680d47985091b90568e2a12fd3d321f`.
- CWAH one-task LLM baseline: `setup_table`, success `1/1`, `22` steps. Local
  result: `CWAH/runs/demo_task0_p0/results.pik` (ignored by Git).
- EntCollabBench local `.env` has nonempty API key, API URL, and model fields;
  both Compose files are present. Values were not copied into this document.
- Docker is available. The seven Arena MCP services are healthy, all eleven
  Agent `/healthz` endpoints returned HTTP 200, and a short HR Agent request
  returned `OK` through the configured model API. The EntCollabBench conda
  environment also passes `pip check`.
- A one-batch EntCollabBench run of `task_12` completed the execution and
  judging pipeline with no framework errors (2026-10-15). With all four
  required services (`hr`, `calendar`, `email`, `itsm`), seed and cleanup were
  successful for each service; runtime status was `ok`. The judged task did
  **not** pass: one of four evaluated Agents passed, and the overall
  `task_passed` value was `0`. The trace reports a delegation timeout and
  incomplete cross-agent actions. This verifies the basic experiment pipeline,
  not benchmark task success. The ignored local result is
  `EntCollabBench/scripts/result/trace_retry_all_servers_2026-10-15.jsonl`.
- Git ignore rules exclude local credentials, downloaded VirtualHome files,
  and generated benchmark results. The original EntCollabBench nested Git
  metadata remains in `.git-nested-backups/2026-10-14/`.

## Known limitations and follow-up checks

- The tested EntCollabBench task did not pass its judge. Its incomplete
  delegation and cross-agent actions are a task-execution or model/coordination
  issue in this run, not evidence that Docker, MCP seeding, or judging failed.
  The precise contribution of model behavior versus agent timeout settings
  has not been isolated. This run did not exercise Gitea isolation, so that
  earlier blocker remains unverified.
- CWAH has one successful task; multi-task behavior has not been verified.
- The earlier migration audit file (`docs/git-migration-2026-10-14.md`) was
  absent from the working tree at commit preparation time and was not restored.
