# CWAH TRACE task evidence and capability lifecycle

Date: 2026-10-17
Branch: `CWAH-TF`

## Purpose

Stabilize Continuous C-WAH action generation and make TRACE task evidence,
Capability, Intent, and Reliability updates correspond to complete public
subtasks rather than isolated physical actions.

## Main changes

- Fixed stale object-location candidates by reconciling direct observations
  with room memory, pruning objects that are no longer relevant to active
  goals, and forgetting invalid grab targets before replanning.
- Clarified the Peer Planner output contract: one environment step contains
  either one physical action or `[send_message]`, with separate valid output
  examples for both cases.
- Added legal `task_completion` evidence containing the concrete subtask and
  `completed_by` attribution. In the two-agent Continuous runtime, attribution
  uses the public completion event and Alice's own action; Bob's private action
  is not exposed to TRACE.
- Added task-level Capability execution records. Visible Bob activity creates
  `in_progress / unknown`; an attributable Bob completion becomes
  `completed / success`; Alice completing the task leaves Bob at
  `abandoned / unknown` rather than recording a failure.
- Connected completion evidence to Intent updates so a completed task is not
  retained as Bob's current intent without newer evidence.
- Allowed verified Bob task completion to fulfill a matching explicit
  Reliability commitment. Concrete grab/place commitments are bound to the
  public `subtask_id`, so progress on another subtask with the same object
  class cannot fulfill them. Pending or unverifiable commitments remain
  outside the fulfillment-rate denominator.
- Kept active and completed subtasks separate. Agents retain the complete
  completion history, while planners receive the latest 20 entries clearly
  marked as historical and non-actionable unless issued again as active tasks.
- Extended TRACE policy context, structured logs, documentation, and tests for
  progress/completion evidence and task-level belief settlement.

## Validation

Executed in the `CWAH` Conda environment:

```bash
conda run -n CWAH python -m unittest discover -s tests -p 'test_*.py'
```

Result: 50 tests passed.

Also passed Python compilation checks for the modified runtime and TRACE
modules, and `git diff --check`.

No Unity or paid LLM episode was run for this commit. A fresh 100-step
TRACE-on Continuous C-WAH run remains the next end-to-end validation.
