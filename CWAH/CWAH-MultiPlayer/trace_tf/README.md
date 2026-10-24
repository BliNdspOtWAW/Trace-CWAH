# TRACE v2 teammate belief

See [the complete v2 protocol](../../../docs/TRACE_V2.md) for definitions, timing,
communication semantics and limitations. This branch replaces
the v0 single intent/LLM uncertainty/commitment scheme.

- Intent updates immediately before each Active or Reactive planner call, not
  each environment step. Independent decision IDs preserve same-step ordering.
- The LLM emits dynamic `P(task)` and `P(skill|task)` tables. Code validates and
  normalizes the joint distribution and computes normalized entropy.
- Capability and Reliability use Beta(1+successes,1+failures). Uncertainty is
  `12*posterior_variance`, in [0,1]. Preference is disabled.
- Only TRACE-equipped Alice has typed delegate/probe_intent output. Bob and
  TRACE-off behavior remain on the existing natural-language contract.
- Accepted task delegations succeed on Bob's public completion and fail on
  Alice's completion; unresolved/unknown outcomes are excluded. No deadlines.
- Probe asks intent only and never creates a commitment. Communication never
  interrupts a skill or modifies an already selected environment action.
- Legal partial observations, messages and public completions are logged. No
  hidden Bob action, task control or profile is supplied to TRACE.
- Independent offline A/B starting-Intent agreement analysis is implemented in
  CWAH/offline_intent; see docs/OFFLINE_INTENT_AGREEMENT.md. No online updates
  or next-state posterior are used as offline answers.

From `CWAH/CWAH-MultiPlayer/`:

```bash
python -m unittest discover -s tests -p 'test_trace_*.py'
```

Scripted tests verify protocol behavior, not real-model calibration or benchmark
improvement. The generic default schema remains readable for old artifacts;
runtime v2 explicitly selects `trace-v2-joint-intent-beta`.


Refinements: Capability now counts public Bob completions only (failures remain
zero). Exports distinguish last-decision belief, complete rule-only statistics,
and Intent-unconsumed events. Empty actions are one-step no-ops, not persistent
wait skills. Legal partial views and event ordering remain archived. See the 2026-10-25 section of the protocol document.
