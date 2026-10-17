# TRACE-TF teammate-belief model

`BeliefSchema` defines replaceable dimensions. `BeliefState` stores one
teammate's current estimates as JSON-serializable snapshots. The initial
`DEFAULT_SCHEMA` has six candidate dimensions; it is not a fixed planner or
environment interface. Every dimension starts `unknown`. A supported estimate
requires new evidence and a confidence in `[0, 1]`; per-dimension uncertainty
is optional and independent of confidence. The candidate `uncertainty`
dimension can hold a broader uncertainty description; it does not replace
the per-dimension uncertainty field.

`EvidenceSource` deliberately has no hidden-profile or private-planning
source. This data model does not collect observations itself: a future adapter
must supply only information actually visible to the target agent (for example,
the target's partial observation or a received teammate message). A declared
source alone cannot prove that its contents are non-privileged, so the adapter
must be reviewed and tested separately.

`information.py` builds one legal decision context shared by TRACE and the
planner: goal/current subtasks, personality, current observation and progress,
action history, dialogue history, and currently available high-level actions.
Information is this current planner context, not an inference about what Bob
knows. `events.py` separately keeps the complete auditable ledger of Alice's
legal observations, messages, own actions, directly visible peer progress, and
public subtask completions. A completion records the concrete subtask and its
actor. In the two-agent Continuous C-WAH runtime, attribution uses the public
completion plus Alice's own action: an exact matching Alice placement is hers;
otherwise completion is attributed to Bob by exclusion. Bob's private runtime
action is never copied into Alice's TRACE evidence.

`updater.py` receives the shared Information plus the prior teammate belief and
makes at most one joint LLM call when new evidence is consumed before Alice's
next active or message-triggered reactive planning call. The LLM extracts
tentative Intent, Preference, explicit commitment candidates, and four
uncertainty proposals. Intent does not require a commitment or proof that an
action finished, and a completed referenced task must not remain a current
Intent without newer evidence. Capability keeps task-execution records with
`in_progress`, `completed`, or `abandoned` status and a separate `success`,
`failure`, or `unknown` outcome. Visible task-related peer activity starts an
in-progress record; only an attributable complete subtask counts as success.
Alice completing a task does not imply Bob failed. Commitment fulfillment is
also settled by rules using verifiable evidence. Every concrete grab/place
commitment is bound to the public `subtask_id`, so a different task involving
the same object class cannot fulfill it. Commitments are stored only under Reliability beside
its verified fulfillment rate; they are not a seventh belief dimension or a
separate plan state. Pending or unverified commitments do not enter that rate.
Repeated declarations of the same `(Bob, subtask_id, action)` merge into one
commitment with the earliest acceptance and all confirmation evidence. Grab
and place remain distinct responsibilities. A pending commitment whose task
is completed by Alice becomes `closed / unknown`, outside both success and
violation counts. Settled commitments are never settled or counted again.
Search commitments without a concrete subtask retain message-based identity.
Concrete commitments require a known public task creation time and a Bob
acceptance message at or after that time. An old declaration cannot be rebound
to a future subtask or move a valid acceptance back before task creation.
Each uncertainty is in `[0,1]`, where 1 means unknown; the overall score is the
arithmetic mean of Intent, Capability, Preference, and Reliability uncertainty.

Preference is assessed only on calls 3, 6, 9, etc. of the shared Active/Reactive
updater. Each assessment receives the prior description/uncertainty and legal
events accumulated since the previous assessment, even beyond the normal
20-event recent window. It returns only a concise `description` (or null to
retain it) and a new Preference uncertainty. Description acceptance has no
model-supplied evidence-ID requirement; input provenance is retained for audit.
A null description can still rescore an existing belief. With no established
description, null remains unknown with uncertainty 1. Other calls inherit both
values. Supporting behavior may reduce uncertainty, conflicting behavior may
raise it; elapsed calls alone are not evidence. Failed assessments retain the
event backlog for the next scheduled assessment.

`policy.py` adds only the inferred teammate-belief portion to the Information
already rendered by Alice's active or reactive planner. With TRACE off, the
same Information goes directly to the planner; with TRACE on, TRACE first sees
that Information and its prior belief, then the updated belief is appended.
A newly received Bob message is entered in the legal ledger and consumed by
TRACE before Alice replies. The
reactive planner emits a logged intention and a mandatory reply, but does not
replace the current runtime skill or create a separate persistent plan. Bob's
profile and private plan are not input to TRACE. Full legal events and
before/after snapshots are saved under `saved_info['trace']` in the episode
log. Planner and updater call records also save `shared_information`, making it
possible to verify that both received the same decision context.

Continuous agents retain a full completed-subtask history separately from the
active task list. The planner sees the most recent 20 records, explicitly marked
as historical and already satisfied, while the complete history remains
available for logging and analysis. A historical task is actionable only if a
new Continuous subtask activates that object/relation/target again.
The implementation is a training-free research prototype, not a calibrated
probabilistic estimator. It must not be treated as a fully verified benchmark
result without an episode-level run.

Run the model-only checks from `CWAH-MultiPlayer/` with:

```bash
python -m unittest discover -s tests -p 'test_trace_*.py'
```
