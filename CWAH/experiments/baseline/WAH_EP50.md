# WAH Episode 50 with CollabBench teammate profiles

This is a WAH-data transfer baseline, **not** a strict CollabBench Task 26
reproduction. Use WAH episode index/task ID 50 (`prepare_food`, environment 6)
and source only the simulated teammate's profile text from CollabBench
`26_prepare_food` at the selected cluster. Never relabel the task data.

The optional environment variables `CWAH_PROFILE_EXPECTED_TASK=50_prepare_food`
and `CWAH_PROFILE_SOURCE_TASK=26_prepare_food` change only the profile lookup
for agent ID 2. The code fails if the actual task or profile does not match.
Without these variables, the original profile lookup is unchanged.

Place each local run in
`CWAH/experiments/baseline/prepare_food_profile<cluster>_<YYYYMMDD-HHMMSS>/`.
Keep `config.json` free of credentials and record the dataset SHA-256, actual
episode index/ID, source profile key, both effective model IDs, code revision,
and runtime settings. These folders are ignored because raw logs can contain
prompts, trajectories, and model output.
