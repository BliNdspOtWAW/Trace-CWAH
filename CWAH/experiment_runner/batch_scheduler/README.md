# Qwen experiment batch scheduler

`run.py` schedules independent CWAH experiments across existing, approved
Qwen API replicas. It does not create or stop remote containers, change
Alice/Bob execution order, or route individual HTTP requests. Each experiment
binds both agents to one endpoint; Unity port selection remains in
`run_experiment.py`.

Inputs are a batch JSON (`version`, `batch_id`, `limits`, `jobs`) and a replica
registry JSON (`version`, `replicas`). Each job points to one existing single
experiment config. Optional `no_counterfactual: true` is allowed only for
TRACE-on jobs; it sets `CWAH_DISABLE_COUNTERFACTUAL_PROBE=1` and adds the
`no_counterfactual_` run-name prefix for that job. Other jobs explicitly set
the switch to zero. The registry lists localhost SSH tunnels, model IDs and
`max_experiments` per replica. This is a per-experiment capacity, not an HTTP
inflight request limit.

The scheduler validates every experiment config using the single runner,
checks `/v1/models`, caps global experiments and Unity instances, then writes
queue state and attempt logs under
`CWAH/experiments/tracev2/batches/<batch_id>/`. Results stay in each job's
`results_root` and retain the standard model/profile/step/trace/time layout.
Attempts are independent and never overwrite previous results. The default
`max_attempts` is 1. A failed or interrupted batch may use `--resume` with
unchanged input files; a still-running worker blocks resume to prevent
duplicate work. After five minutes without a schedulable job, it exits with
a resumable state.

For the concrete profiles 1–5 batch and commands, see
[`../configs/tracev2/continuous/qwen/peer1_5_200step/README.md`](../configs/tracev2/continuous/qwen/peer1_5_200step/README.md).
