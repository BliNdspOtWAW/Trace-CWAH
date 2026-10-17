# Continuous C-WAH local configurations

Copy the supplied `.json.example` files to names ending in `.json` in this
directory. Local JSON files are ignored by Git. Both examples use WAH Episode
44 and preserve its three initial `prepare_food` predicates. Each completed
subtask is replaced by randomly sampling a concrete movable object and one of
the configured legal destinations from the current scene state. Invalid or
conflicting samples are retried; a temporary generation failure is retried on
the next environment step. The Unity scene and all agent, dialogue, and TRACE
state remain live until the step budget.

Validate and run only this pair with:

```bash
python CWAH/experiment_runner/run_many.py \
  --config-dir CWAH/experiment_runner/configs/continuous \
  --check

python CWAH/experiment_runner/run_many.py \
  --config-dir CWAH/experiment_runner/configs/continuous \
  --max-parallel 2
```
