# Offline retrospective Intent agreement

This is independent of online TRACE. No training or real Intent labels are
required. Both posteriors target Bob's hidden `(subtask, skill)` at the START of
each adjacent TRACE decision interval, not the end. Bob may switch states inside
the interval. The likelihood evaluates compatibility of the whole later evidence
with a starting state, implicitly allowing transitions; it is not calibrated.

- A sees legal starting history, ordered new evidence and FULL starting support,
  but no numeric prior. Code normalizes `prior * relative_likelihood`.
- B sees exactly the same common input plus the original prior and directly
  estimates FULL starting-state joint posterior scores; code normalizes scores
  directly without multiplying by the prior again.
- Base-2 JSD measures agreement, NOT accuracy. If reliable ground truth becomes
  available, add separate predictive-quality metrics such as NLL/Brier score.
  The current implementation does not claim or infer ground truth.

## Commands (from repository root)

```bash
PYTHON=/home/xiaoyuan/miniconda3/envs/CWAH/bin/python
TRAJECTORY=/absolute/path/to/trusted/trajectory.pik
ANALYSIS=CWAH/experiments/tracev2/analysis/source-run/analysis-001

"$PYTHON" CWAH/experiment_runner/compare_intent_posteriors.py prepare \
  --trajectory "$TRAJECTORY" --output "$ANALYSIS"

# Uses the existing Chat API environment; not necessary for prepare/report.
export CWAH_DS_ENV_FILE="$PWD/CWAH/experiment_runner/.env.self"
source CWAH/experiment_runner/env.sh ds
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy CWAH_LOADER_PROXY

"$PYTHON" CWAH/experiment_runner/compare_intent_posteriors.py infer \
  --analysis "$ANALYSIS" --model "$CWAH_AGENT_LM_ID" --limit 2

# Omit --limit to process all unfinished intervals; rerun automatically resumes.
"$PYTHON" CWAH/experiment_runner/compare_intent_posteriors.py report \
  --analysis "$ANALYSIS"
```

`infer` supports Chat Completions endpoints (including compatible local servers).
Use `CWAH_OFFLINE_API_BASE`, `CWAH_OFFLINE_API_KEY`, `CWAH_OFFLINE_MODEL` to keep
the offline service independent of online settings. No in-process Hugging Face
backend or Responses transport is implemented here. Both branches use the same
model, temperature and token budget in separate stateless calls. API calls are
explicit: prepare/report never call a model, and no analysis runs automatically.

## Boundaries and evidence

Each start prior is `belief_updates[k].posterior.joint`, NOT the next decision's
aligned prior or online posterior. `support` at k remains fixed even if tasks
complete inside the interval. All cells participate, including zero-prior cells.
There is no Top-k computation mode.

The ordered event ledger and `interval_end_cursor` determine boundaries. Evidence
after decision k and before the decision k+1 marker is E_k; the next decision's
LLM outputs are excluded. Same-environment-step Active/Reactive/retry decisions
remain separate, including empty-evidence intervals. Environment step is only
display metadata. Cursors, IDs, prior mass and full support are validated strictly.
Unpaired terminal evidence is counted in the manifest and excluded from metrics.

History is the complete starting legal ledger plus whitelisted start information.
No automatic history truncation or independent summarization is performed. This
can exceed a model context window on long runs; such calls fail explicitly.
Every logged partial observation frame is retained (geometry/state metadata is
projected deterministically), not just endpoints. Historical/current visibility
and message claims remain distinct. No global graphs, Bob private actions,
profiles, raw online model prompts, or numerical belief snapshots enter common
inputs. Derived outcomes and observations are not independent confirmations.
Alice controls/actions are conditioning information, not automatic execution proof.

The analysis trusts the provenance of TRACE's legal ledger; it cannot repair
incorrect event attribution in a source trajectory. Load pickle only from trusted
experiments. Source hash, common-input hash and full-input hash identify the data.

## Files and validation

All outputs live in the explicitly chosen analysis directory, separate from the
source run. The top level contains only posterior_agreement.png and details/.
The figure includes valid/failed/pending counts, mean and median JSD; usually
this is the only file you need to open. Default PDF and CSV exports are removed.
All records listed below are under details/. Existing flat batches migrate
automatically on infer/report; original records are preserved without overwrites. Prepare requires an empty directory and never modifies the trajectory.

| File | Purpose |
|---|---|
| manifest.json | Protocol, source SHA256, interval count, terminal tails |
| intervals.jsonl | Frozen shared inputs, prior, candidate IDs and boundaries |
| inference_config.json | Fixed model, endpoint, temperature, budget, prompt hash |
| calls.jsonl | Exact prompts, raw responses, usage, status and bounded retries |
| branch_cache.jsonl | Successful branch outputs for interruption/failure resume |
| results.jsonl | Append-only attempt results, both posteriors and JSD |
| summary.json | Latest-result statistics, failures and pending counts |
| ../posterior_agreement.png | Only visible output: ordered JSD and summary; failed/pending are gaps |

Rows match candidates by identity, not array position. Missing/duplicate/extra
identities, negative/nonfinite values, all-zero likelihoods and zero prior-score
overlap fail rather than silently becoming uniform. Both branches emit full nonnegative score tables; neither must sum to one.
B uses JSON posterior_scores rows with score (not probability). Code computes
A=normalize(prior*likelihood) and B=normalize(posterior_scores). Original
sums/values and normalization semantics are retained. All-zero B scores fail.
Protocol v2-scores requires a new prepared analysis batch; older response schemas
and cached results are not silently reused.
No epsilon is added: A cannot recover a zero-prior state, while B's mass assigned
there is explicitly reported. Branch failures have null JSD, never zero JSD.
Only pairs with both branches valid enter JSD summary statistics. Failure rate
uses attempted intervals and pending count is separate. Successful branches are
cached; reruns retry failed branches. Configuration changes require a new batch.

The main curve is JSD in bits (0 to 1), with decision interval index on x.
Full distributions, posterior entropy, change from prior, zero-prior B mass, and
largest-disagreement IDs remain available for inspecting mechanisms. There is
no next-decision posterior comparison or accuracy curve. Low JSD may reflect
shared model biases; it does not establish correctness or probability calibration.
