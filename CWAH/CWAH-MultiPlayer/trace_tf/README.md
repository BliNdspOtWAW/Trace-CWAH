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

`events.py` stores Alice's legal episode-local observations, messages, own
actions, and directly visible peer task outcomes. Information is this evidence
history only; it does not infer what Bob knows. `updater.py` makes at most one
joint LLM call when new evidence is consumed before Alice's next active or
message-triggered reactive planning call, extracting Intent, Preference,
commitment candidates, and four uncertainty proposals. Capability and
commitment fulfillment are settled by rules using verifiable evidence.
Reliability stores commitment events beside its verified fulfillment rate;
pending or unverified commitments do not enter that rate. Each uncertainty is
in `[0,1]`, where 1 means unknown; the overall score is the arithmetic mean
of Intent, Capability, Preference, and Reliability uncertainty.

`policy.py` renders evidence-aware contexts for Alice's active planner and a
shorter context for her reactive reply planner. A newly received Bob message is
entered in the legal ledger and consumed by TRACE before Alice replies. The
reactive planner emits a logged intention and a mandatory reply, but does not
replace the current runtime skill or create a separate persistent plan. Bob's
profile and private plan are not input to TRACE. Full legal events and
before/after snapshots are saved under `saved_info['trace']` in the episode
log. Alice's own prompts receive compact relevant evidence views; the complete
dialogue remains in their separate dialogue-history context.
The implementation is a training-free research prototype, not a calibrated
probabilistic estimator. It must not be treated as a fully verified benchmark
result without an episode-level run.

Run the model-only checks from `CWAH-MultiPlayer/` with:

```bash
python -m unittest discover -s tests -p 'test_trace_*.py'
```
