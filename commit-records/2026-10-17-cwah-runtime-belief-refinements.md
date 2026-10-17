# CWAH runtime and TRACE belief refinements

Date: 2026-10-17
Branch: `CWAH-TF`
Parent: `07b62c3`
Related commit: the commit introducing this record (`git log -1 --format=%h -- commit-records/2026-10-17-cwah-runtime-belief-refinements.md`).

## Changes

- Allow room revisits; count room entries and prioritize less-visited search
  candidates when no stronger location clue exists.
- Preserve legally observed movable objects independently of current goals;
  dynamically filter task candidates and invalidate stale locations.
- Save an atomic trajectory checkpoint after each fully recorded environment
  step, including communication and TRACE state. Handle SIGINT/SIGTERM/SIGHUP
  by finishing the current step and saving interrupted results.
- Merge repeated commitments by Bob/subtask/action, preserve the earliest
  acceptance and confirmation evidence, and settle each commitment once.
  Close pending commitments as unknown when Alice completes their task.
- Reject concrete commitments whose message predates task creation, or whose
  task creation time cannot be verified from public metadata.
- Assess Preference on every third shared Active/Reactive updater call; retain
  intermediate evidence. Accept concise descriptions without model-supplied
  evidence IDs, allow null descriptions to retain and rescore prior beliefs,
  and retain unknown uncertainty when no description exists.
- Structure the updater prompt, retain current Peer/Reply prompt edits, and
  organize portable profile configuration templates.
- Route OpenAI-compatible backends to Chat Completions independently of model
  names; pass model identifiers such as gpt-5 or provider aliases unchanged.

## Validation

- CWAH Conda environment: 61 unit tests passed.
- Tests cover interruption/checkpoint persistence, room revisits and task
  switches, commitment deduplication and temporal binding, Preference cadence
  and scoring, and arbitrary model-name Chat routing.
- Temporary real API prompt replays reproduced invalid action outputs and
  default/unknown Preference responses. These temporary payloads/results and
  API credentials are not included in Git.
- No new Unity episode was launched solely for this commit.

## Known limitations

- Invalid action parsing still includes fuzzy/random fallback and can replace
  the model's output with another action or empty communication.
- A valid all-default updater response can still reset an established
  Preference uncertainty to 1.0; its model/server origin is unresolved.
- Temporal commitment validation does not prove semantic responsibility or
  destination matching in the source message.
- GPT-5 routing was tested with mocked API responses; gateway-specific request
  parameter compatibility has not been verified with a real request.
- Graceful stop waits for the current step/API call. SIGKILL cannot finalize
  results; only the latest completed checkpoint remains available.
