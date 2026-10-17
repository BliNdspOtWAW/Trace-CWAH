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
legal observations, messages, own actions, and directly visible peer outcomes.

`updater.py` receives the shared Information plus the prior teammate belief and
makes at most one joint LLM call when new evidence is consumed before Alice's
next active or message-triggered reactive planning call. The LLM extracts
tentative Intent, Preference, explicit commitment candidates, and four
uncertainty proposals. Intent does not require a commitment or proof that an
action finished. Capability and commitment fulfillment are settled by rules
using verifiable evidence. Commitments are stored only under Reliability beside
its verified fulfillment rate; they are not a seventh belief dimension or a
separate plan state. Pending or unverified commitments do not enter that rate.
Each uncertainty is in `[0,1]`, where 1 means unknown; the overall score is the
arithmetic mean of Intent, Capability, Preference, and Reliability uncertainty.

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
The implementation is a training-free research prototype, not a calibrated
probabilistic estimator. It must not be treated as a fully verified benchmark
result without an episode-level run.

Run the model-only checks from `CWAH-MultiPlayer/` with:

```bash
python -m unittest discover -s tests -p 'test_trace_*.py'
```
