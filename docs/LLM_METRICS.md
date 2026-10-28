# LLM context and latency metrics (8004)

New experiments launched by `run_experiment.py` or `run_many.py` automatically
record metrics; no JSON option is required. Both API and local backends use the
same instrumentation. Model requests, prompts, histories and algorithms are not
changed. This does not add LLM calls. Old experiments are not retroactively
instrumented.

## Files in each experiment result directory

| File | Contents |
|---|---|
| `config.json` | Effective configuration, model, task, profile, TRACE mode, prompt hashes and Git revision; no API key |
| `summary.json` | Status/error, steps, completed subtasks, wall time and aggregate LLM metrics |
| `calls.jsonl` | Incremental call_start/call_end records for Active, Reactive and Updater, plus environment_step_start events |
| `trajectory.pik` | Versioned archive of all repeats: observations, actions, dialogue, prompts, TRACE details and task events |
| `console.log` | Evaluator output and errors |

The JSONL is flushed by closing the file after each record. CSV exports,
separate metrics reports, and the report-generation CLI have been removed.
The runner aggregates metrics in memory into `summary.json`; detailed call
records remain in `calls.jsonl` for subsequent analysis.
Legacy experiments using `llm_calls.jsonl` remain readable.
During execution, `.runtime/` holds evaluator output and the atomic per-step
checkpoint. Once the final archive has been read back successfully, redundant
scratch files are removed. An interrupted/failed run additionally retains the
checkpoint when available; unreadable or unknown scratch files are never removed.
SIGKILL/power loss cannot run finalization: the last saved checkpoint may remain
inside `.runtime/`. Old experiments are not converted or deleted.

The archive schema is `{"format_version": 1, "episode_index": 44, "runs": [...]}`.
Each run retains original trajectory fields and adds `run_index`, `episode_index`
and `environment_steps`; repeated runs are not overwritten. Load pickle only
from trusted sources. `status=completed` describes pipeline completion, not task
success; use the task and continuous-subtask metrics separately.
Abrupt termination may leave only a call_start: it is reported as in-flight,
not a completed zero-token call. An incomplete last JSONL line is tolerated.
No prompt text, reply text, API key or request headers are stored in these
metrics files. Existing original trajectories continue to hold prompts.
Decision steps use the existing Information step value; window start 0 means
steps 0–49, start 50 means 50–99, and so on. The result configuration records
the metrics schema version and implementation hashes.

## Diagnosing late-step slowdown

Compare the SAME Agent/module in early versus late 50-step windows:

- `prompt_tokens`: actual provider-reported input tokens (local tokenization
  for Hugging Face). Missing usage is null/blank, NOT zero. Summary totals are
  based on known calls; inspect `known_calls` and missing coverage.
- `prompt_chars`, `chars_dialogue_history`, `chars_action_history`,
  `chars_progress`, `chars_goal`, `chars_trace_belief`: input size and its sources.
  Updater rows also have `chars_information`, `chars_prior_belief`,
  `chars_events`, `chars_preference_assessment` and ledger/event counts.
  Character counts are NOT token estimates; nested sections overlap and cannot
  be added together to reconstruct the full prompt.
- `completion_tokens`, `reasoning_tokens`, `output_chars`: distinguish longer
  generation/reasoning from longer input. Reasoning tokens are a subset of
  completion tokens; do not add them twice.
- `elapsed_seconds`, `retry_count`, `request_count`, `cached_input_tokens`:
  distinguish growth from retries/cache changes. Retry counts are available
  for successful Responses calls, not all backends or failed requests. Missing
  cached/reasoning/retry fields mean unknown. Timing includes transport and
  retries, not just model compute. `output_tokens_per_second` is a whole-call
  ratio, NOT pure decoding speed or time-to-first-token.
- Call frequency, errors and `environment_step_start` events in `calls.jsonl`: repeated replanning or
  non-LLM overhead can slow a run even when per-call prompt sizes are stable.
  Step intervals include planning/execution; they are not isolated Unity time.

`call_end.status=ok` means generation returned; `call_end.api_status=ok` means the
Chat API returned successfully, even if the answer was truncated or malformed.
`decision_end.status=ok` requires a valid action, Reactive reply, or TRACE intent
distribution. The summary reports `api_success_calls`, `valid_decision_calls`,
`invalid_decision_calls`, and `unreported_decision_calls` separately. A missing
decision record in older runs is unknown, not an invalid decision. `decision_end.reason`
records protocol or probability failures without storing prompt or reply text.
No inference
about context being the cause should rely on latency alone: compare input,
output/reasoning tokens, retries and cache behavior together.

Active input currently retains the latest 10 high-level actions; completed
subtask descriptions retain the latest 20. Dialogue is complete. Thus not all
history fields grow without bounds. The instrumentation does not change these
existing policies. TRACE-off should produce no updater calls.

For JD Responses, `max_tokens=1024` in an experiment is NOT necessarily the
effective cap: the existing transport takes the maximum of that value and
the role's Responses budget (default 8192). Inspect `effective_max_output_tokens`.
This records the current behavior; it does not change the budget.
