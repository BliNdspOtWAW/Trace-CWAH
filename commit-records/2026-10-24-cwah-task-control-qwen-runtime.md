# CWAH task control, local Qwen and experiment instrumentation

Branch: `CWAH-TF`. This record is included in the implementation commit; obtain
its SHA with `git log --all -- commit-records/2026-10-24-cwah-task-control-qwen-runtime.md`.

## Changes

- Continuous Manager owns the public subtask table, including completion time
  and attributable actor. Agent task views are derived from this table.
- Agent control separates the macro subtask from the running skill and its
  originating subtask. Active output selects a subtask and skill; Reactive may
  revise the macro target without replacing the running skill or the already
  selected physical action.
- Active and Reactive share the step-start observation. Messages are delivered
  before frozen physical actions execute; task settlement follows execution.
- Decision context, prompt fields, parsing, preflight validation and trajectory
  instrumentation follow this protocol. TRACE still updates at decision entry
  points, not automatically on every generated subtask.
- Add a shared-per-process Hugging Face generation backend, Qwen3 downloader,
  configuration templates and server dependencies. Preserve API backends.
- Add Qwen-specific Reply formatting and structured reply parse diagnostics.
- Record per-call tokens, latency, retry counts and context measurements.
  Consolidate normal results into config.json, summary.json, calls.jsonl,
  trajectory.pik and console.log; retain recoverable interrupted trajectories.
- Include existing role prompt/profile edits and repository cleanup deletions,
  including removed root README/AGENTS, old deployment notes and Continuous
  templates. No secret files, model weights, datasets or runtime results added.
- Refresh local-backend documentation to remove obsolete download status and
  machine-specific paths.

## Verification

- 100 unit/regression tests passed; `git diff --check` passed.
- Mocked Agent and scripted Unity checks covered Reactive target switching while
  preserving the running skill and current physical action.
- Real GPT-5.6 TRACE-on Episode 44 / Peer17 completed a 20-step smoke test:
  one completed subtask, 23,969 reported tokens, no recorded errors. The actual
  Reactive response retained its target, so real-model target switching was
  not exercised. Temporary smoke outputs were removed after verification.

## Limitations

This is an exploratory checkpoint, not evidence that repetitive communication,
semantic Reliability attribution or long-context performance is solved. Latest
protocol changes have not undergone a new full-length Qwen experiment. Legacy
budget-as-success semantics remain; use completed-subtask metrics for Continuous
experiments. Existing Qwen on/off configurations may have unequal step budgets.
