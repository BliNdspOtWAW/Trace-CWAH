# TRACE v2: joint Intent, Beta statistics and offline posterior agreement

Branch: CWAH-TF-v2. Parent implementation: 004f048 (CWAH-TF).
This record is committed with the implementation; identify its commit using:
git log -1 --format=%h -- commit-records/trace-v2/2026-10-25-joint-intent-beta-offline-agreement.md

## Online changes

- Replace the legacy Intent representation with full dynamic joint
  P(subtask)*P(skill|subtask); validate support and compute normalized entropy.
  Three active tasks yield 22 cells including task-independent none skills.
- Update Intent at each actual Active/Reactive decision entry, including retries,
  with ordered decision IDs/event cursors. Environment observations only log evidence.
- Capability counts attributable public Bob task completions as successes;
  other outcomes do not count as task failures. Reliability counts accepted
  delegations: Bob completion succeeds, Alice completion fails; unknown/pending
  are excluded. Both use Beta(1+S,1+F), mean and 12*variance uncertainty.
- Distinguish current peer visibility/state from timestamped last-seen memory
  in shared Active/Reactive/TRACE information; never treat memory as current sight.
- TRACE-equipped Active communication has top-level delegate/probe_intent
  metadata and plain delivered prose. Probe targets Intent only; requests/replies
  are linked without cancelling current skills or frozen physical actions.
  Reinforce the final output contract and include parse-error feedback on retry.
- Empty/missing/unmatched actions execute nothing for one step and can replan.
- Final/checkpoint exports perform detached rule settlement, preserve the last
  decision belief, and archive Intent-unconsumed events without another LLM call.
- Support bounded results_root under CWAH/experiments, keeping trace and baseline
  separate. Preserve profile reorganization into profiles/tracev1 and tracev2,
  repair config references, and add sanitized tracev2 on/off templates.

## Independent offline analysis

- Remove the previous likelihood/plot entry points. New entry:
  CWAH/experiment_runner/compare_intent_posteriors.py prepare / infer / report.
- Freeze legal starting history, every logged interval observation, messages,
  own selected actions and public events by ordered ledger boundary, not step number.
- Both branches infer the SAME starting Intent, allowing within-interval switches.
  A sees no numeric prior and generates relative likelihood scores; code computes
  normalize(prior*score). B sees the prior and generates posterior scores; code
  normalizes scores directly, without multiplying by prior again.
- Use complete starting support and candidate identity checks, never Top-5
  computation. Reject missing/duplicate/negative/nonfinite/all-zero scores and
  zero prior-likelihood overlap. Preserve raw scores and normalization semantics.
- Base-2 JSD measures mechanism agreement, not accuracy. No training, required
  ground-truth labels, calibrated-likelihood claim or online policy feedback.
- Stateless Chat-compatible calls use the same model/settings for A and B;
  raw responses, hashes, usage and successful-branch resume records are retained.
  Changed protocol/settings require a new analysis batch.
- One visible output posterior_agreement.png includes JSD and summary counts;
  details/ contains records. Existing flat batches migrate without overwrites.
  Remove duplicate PDF/CSV exports. No automatic external model calls.

## Verification and cleanup

- 145 unit/integration regression tests passed before final commit validation;
  covers same-step boundaries, future/private-field exclusion, full support,
  B score-only normalization, JSD edge cases, failures, resume and file migration.
- Real archive extraction and mocked inference/reporting passed on 22-state
  intervals; no external trajectory transfer was performed for that check.
- Actual gpt-5.4-mini calls on one invented 22-state interval: A and B each
  succeeded on the first call (4.44s / 12.17s), posterior sums both 1, JSD
  0.01673503182075721 bits. B raw score sum 1.03 was normalized by code.
  This verifies the interface only, not accuracy or benchmark improvement.
- Remove this round's synthetic analyses, interrupted Qwen validation data,
  short smoke configs, downloaded test plots and temporary validation logs.
  Preserve formal tests, historical tracev1 runs, external dependencies and weights.
- Move nine historical records into commit-records/trace-v1; this record belongs
  to trace-v2. Remove redundant CWAH_STRUCTURE and obsolete TRACE_V2_SMOKE docs;
  retain maintained protocol, offline analysis, task-control, Qwen and metrics guides.
- No credentials, concrete private configs, model weights or generated runs
  belong to this commit. No push or experiment launch is performed.

## Known limitations

- Qwen3-8B TRACE-on validation was manually stopped after 41 steps. At diagnosis,
  39/40 Intent updates retained the prior and 39 calls hit 1536 output tokens;
  updater schema compliance/truncation remains unresolved. The run is not valid
  performance evidence and the offline normalization change does not fix it.
- Offline inference currently supports Chat-compatible endpoints, not direct HF
  loading or Responses transport. Full history is not silently truncated;
  context overflow fails explicitly. Relative likelihood is not calibrated and
  transitions are assessed implicitly by the LLM, not an explicit transition model.
- No reliable real Intent ground truth is available. If introduced, predictive
  quality needs separate NLL/Brier evaluation; low JSD does not establish truth.

- Existing tracev2 peer3_Self-Directed and peer4_Unreliable-Committer profile
  files are empty placeholders supplied in the current workspace; their text
  was not invented. Populate them before selecting them for behavioral studies.
