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

This package does not call an LLM, alter CoELA's planner, or change baseline
behavior. Later Updaters can consume legal evidence and return a new
`BeliefState` via `with_evidence`; selectors can read `to_dict()` without
depending on CWAH internals.

Run the model-only checks from `CWAH-MultiPlayer/` with:

```bash
python -m unittest discover -s tests -p 'test_trace_*.py'
```
