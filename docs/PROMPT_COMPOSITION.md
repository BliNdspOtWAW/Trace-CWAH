# Prompt composition

Alice Active uses `LLM/prompt_Alice.txt`, now the Alice v2 body, or an explicitly
configured template. The previous Alice template is preserved as
`LLM/archived_prompts/prompt_Alice_pre_v2.txt`. Do not send these files directly
without rendering.
Bob Active and the generic Peer template use the same mode-specific output
contracts while retaining Bob's personality and ordinary communication format.

`LLM/prompt_composer.py` selects TRACE mode when Active planning receives a
nonempty belief context; the strict parser uses the same condition. TRACE-on
Active communication uses `delegate` or the broader `probe` type. Bob keeps its
existing role template and baseline protocol.

- Baseline: common body + `LLM/prompts/baseline/active_output.txt`.
- TRACE: common body + `LLM/prompts/trace/planner_context.txt` and structured
  belief values + `trace/communication.txt` decision guidance +
  `trace/active_output.txt` protocol requirements.
- When Alice's TRACE joint Intent entropy is strictly above 0.85, Active also
  inserts `trace/counterfactual_probe.txt` between the belief guidance and action
  selection. This asks whether an answer would change the next action and whether
  that change is worth a probe; high entropy alone does not force a probe. The
  decision record includes the entropy, threshold, and injection flag. Bob and
  TRACE-off Active never receive this fragment.
- Reactive: existing role-specific reply template and output format; TRACE adds belief
  values and `trace/reply_context.txt`, which contains the Reactive explanation.
  Baseline omits the optional TRACE section. Bob's reply template likewise omits
  the section in normal runs, while accepting explicit teammate context when supplied.
  Local Qwen and API models now select the same Alice/Bob Reply templates.
- Updater: `trace/updater.txt`.

Custom legacy Active templates remain supported with the former final TRACE override.
The revised built-in template has only the selected mode's output format. Complete
rendered prompts remain in trajectory call records; experiment config records hashes
of the composer and all prompt fragments. Restart experiments after editing templates.

Old Agent, Human, and generic Reply templates are in `LLM/archived_prompts/`.
The Peer template remains active as the CLI fallback.

For a single-file review of the assembled default flows, see
`docs/CWAH_PROMPT_FLOW_REVIEW.txt`. Regenerate it after editing any prompt with
`python CWAH/experiment_runner/render_prompt_flow.py` from the TRACE root.

## Runtime sequence

At an Active decision, TRACE-on Alice updates the belief from the legal event
interval, renders the structured values, and composes the Alice body with the
TRACE belief, communication guidance, and typed output contract. TRACE-off Alice
and Bob use the baseline output contract. The parser checks the selected action
and task ID; TRACE-on messages also require `delegate` or `probe` metadata.

An Active message is delivered as natural-language prose. The recipient then
uses the role-specific Reactive Reply template; TRACE-on Alice first updates
her belief again at this decision boundary. A Reply emits `intention`,
`new_subtask_id`, and `reply`, and can change a future macro objective without
cancelling the physical action already selected for this environment step.
The updater has its own JSON-only prompt and does not use the planner's output
contract. Bob's normal prompts receive no TRACE belief.

Alice's general Action Selection section explains the one-step cost of a
message; TRACE communication guidance then explains when a probe is useful.
TRACE-on probes additionally emit a local `probe_target` category. The category,
question, and linked reply are recorded for analysis, but the target label does
not update a belief by itself. Probe replies may inform Intent inference;
Capability and Reliability Beta counts still depend on public outcomes and
accepted-task settlement. There is no separate structured information-gain or
action-flip metric yet; the counterfactual comparison is prompt guidance.
