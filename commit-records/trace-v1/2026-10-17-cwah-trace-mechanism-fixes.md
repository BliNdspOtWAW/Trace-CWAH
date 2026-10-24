# CWAH TRACE mechanism fixes

Date: 2026-10-17
Branch: `CWAH-TF`

## Purpose

Repair the TRACE-TF decision pipeline after the first 100-step Continuous
C-WAH experiment showed that the updater received a fragmented context,
Intent often remained unknown, commitments were maintained outside the six
belief dimensions, and planner-facing Information was not aligned with the
updater input.

## Main changes

- Added one JSON-serializable `DecisionInformation` context shared by TRACE
  and the active/reactive planners. It contains the legal goal, personality,
  current observation/progress, action history, dialogue history, available
  actions, step, and source evidence references.
- TRACE-off sends the same Information directly to the planner. TRACE-on first
  combines it with the prior belief, updates TRACE, and appends the resulting
  teammate belief to the planner context.
- Reworked the updater prompt around shared Information, compact prior belief,
  and attributable Bob evidence. Intent can be a tentative high-level action
  inferred from visible behavior and messages; it does not require a completed
  action or explicit commitment.
- Converted the latest visible peer graph state into semantic facts for the
  updater, while retaining the complete legal evidence ledger for audit.
- Removed the separate runtime commitment store. Explicit commitments now
  persist under `Reliability.commitment_events`; rule-based settlement uses
  visible verified outcomes.
- Capability and Reliability uncertainty are now calibrated by rule-based
  evidence counts. Overall uncertainty remains the mean of Intent, Capability,
  Preference, and Reliability uncertainty.
- Reduced policy injection to inferred teammate beliefs because the active and
  reactive planners already receive the shared Information directly.
- Added `shared_information` to planner and communication call records for
  input-alignment checks.
- Clarified the active planner communication contract: `[send_message]` and a
  physical action are mutually exclusive, messages are payloads of the
  `[send_message]` action, and physical actions must leave the message empty.
  The prompt also treats personality as part of how the task is completed and
  explains the relative coordination cost of communication.

## Validation

- `python -m unittest discover -s tests`: 42 tests passed in the `CWAH` Conda
  environment after the TRACE implementation changes.
- Active/reactive communication and experiment-runner regression subset:
  19 tests passed after the final peer-prompt wording changes.
- Python compilation/import checks and `git diff --check` passed.
- A Step 31 diagnostic replay showed that the final prompt prevents an emitted
  message from being paired with a physical action. Both agents still selected
  `gograb` rather than `send_message`, so proactive communication frequency
  remains an experimental policy question rather than a resolved TRACE bug.

## Known limitations

- The repaired updater has not yet been evaluated in a fresh complete
  100-step Unity episode. The current evidence establishes code-path and
  interface correctness, not improved benchmark performance.
- Preference requires repeated attributable evidence and may remain Unknown in
  short trajectories.
- Ambiguous or unverifiable commitments may remain pending/unverifiable.
- Temporary replay artifacts under `/tmp` and generated experiment results are
  intentionally not committed.
