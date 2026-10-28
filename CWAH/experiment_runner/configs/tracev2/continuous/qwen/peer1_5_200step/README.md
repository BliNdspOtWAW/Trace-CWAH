# Qwen profiles 1–5, 200-step batch

This folder contains ten episode-44 experiment configs and a twelve-job batch:
one baseline and one TRACE run for each profile, plus TRACE without the
counterfactual probe for profiles 4 and 5. All jobs use the same continuous
subtask seed (`1`), temperature (`0`), model (`qwen3.8-27b`), episode and
200-step limit. The extra ablation jobs reuse the matching TRACE config;
`batch.json.example` supplies `no_counterfactual: true` per job. Do not run
those ablations with `run_many.py`, which does not apply that per-job switch.

The registry lists 13 existing Qwen services. It prefers PPU 12, 14, 15 and
6, then uses the other shared cards. Each experiment stays on one replica;
the scheduler spreads successive jobs across replicas. The batch allows three
concurrent Unity experiments because the local machine has 15 GiB total
memory. This limit does not mean only three remote services are usable.
`--check` verifies tunnels and model IDs. Raise local concurrency only after
measuring memory use and failed runs. Other users' work on shared cards must
not be stopped.

From Windows PowerShell, keep this SSH tunnel open:

```powershell
$ports = @(18044,18046,18047,18038,18032,18033,18034,18035,18036,18037,18041,18042,18043)
$forwards = foreach ($port in $ports) { '-L'; "127.0.0.1:${port}:127.0.0.1:${port}" }
ssh -o ExitOnForwardFailure=yes -N @forwards PPU-ali
```

From a WSL shell at the TRACE repository root:

```bash
conda activate CWAH
source CWAH/experiment_runner/env.sh qwen
batch_dir=CWAH/experiment_runner/configs/tracev2/continuous/qwen/peer1_5_200step
python CWAH/experiment_runner/batch_scheduler/run.py --batch "$batch_dir/batch.json.example" --replicas "$batch_dir/replicas.json.example" --check
python -u CWAH/experiment_runner/batch_scheduler/run.py --batch "$batch_dir/batch.json.example" --replicas "$batch_dir/replicas.json.example" | tee CWAH/experiments/tracev2/qwen_peer1_5_200step_v1.log
```

After an interrupted scheduler, use the same command with `--resume` once
the previous worker processes are no longer running. The scheduler refuses
to duplicate a live worker. Queue state and attempt logs are saved under
`CWAH/experiments/tracev2/batches/qwen_peer1_5_200step_v1/`.

Experiment results follow the standard tree:

```text
CWAH/experiments/tracev2/qwen/
  qwen3.8-27b_peer4_Unreliable-Committer_200step/
    baseline/qwen3.8-27b_peer4_Unreliable-Committer_200step_traceoff_<timestamp>/
    trace/qwen3.8-27b_peer4_Unreliable-Committer_200step_traceon_<timestamp>/
    trace/no_counterfactual_qwen3.8-27b_peer4_Unreliable-Committer_200step_traceon_<timestamp>/
```

The same layout applies to profile 5. The `config.json` inside each result
records whether `counterfactual_probe_disabled` was active. Setting
`max_experiments` above one for a replica is supported by the scheduler,
but it limits experiment processes, not HTTP requests; measure service queue
time and error rate before changing it. Agent decisions within an episode
remain sequential.

The counterfactual fragment is conditional on Alice's intent entropy being
greater than 0.85. After the run, inspect the TRACE trajectories for
`counterfactual_probe_prompt_injected`; a TRACE/no-counterfactual pair that
never crossed the threshold does not test the prompt's behavioral effect.
