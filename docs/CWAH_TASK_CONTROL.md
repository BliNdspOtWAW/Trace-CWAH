# Continuous C-WAH public tasks and execution control (protocol v2)

The Continuous Manager owns a single `subtasks` table, retaining active and
completed records. Public records include IDs/classes, relation, lifecycle
timestamps and `completed_by`. Completion requires the post-action graph
relation; the matching submitted placement action identifies Alice/Bob, or
`unknown` if attribution is ambiguous. Internal `source_target_id` is excluded
from public snapshots because it could disclose hidden item locations.

Agents receive detached step-start snapshots. `active_goals` and
`completed_goals` are computed views, not separately mutable task state.
Agent-side persistent `satisfied` and `unsatisfied` fields are removed. Existing
environment/evaluator predicate checks remain in place; they are not agent
beliefs. Static-goal entry points have a legacy adapter and are not the new
Continuous task authority.

Private execution control contains:

```json
{"current_subtask_id": "S4", "current_skill": "[gograb] <apple> (375)", "skill_subtask_id": "S3"}
```

The macro target has switched to S4, but the current skill still serves S3.
Reactive never cancels that skill or changes this step's already selected
physical action. Skill completion/invalidation clears only the skill and its
origin; an active macro target survives for the next planning call.
While the S3 skill is still running, the control snapshot can show S4/S3 but
the agent normally continues that skill without another Active Planner call.
When the skill ends, `skill_subtask_id` becomes null; the next Active Planner
receives S4 as `current_subtask_id` with no running skill. The same transition
applies to Bob, including a running S2 `goput` followed by an accepted S4 request.

Active emits a closed `<subtask_id>` field along with the original cot/message/
action fields. The runtime validates that the ID is active and that grab/put
object and destination agree with it. Null is allowed for task-independent
exploration or communication. Invalid proposals are retried with the existing
bounded planning loop, not accepted as private task truth.
Custom old prompts without public-task/control placeholders and the new output
field are rejected during runner preflight instead of failing after model calls.

Reactive emits `<intention>`, `<new_subtask_id>` and `<reply>`. Null means retain
the macro target; an active ID proposes a subsequent target. Invalid IDs are
logged and the reply is not delivered as an accepted agreement. All backends
use the role-specific Reply prompts with the same three tagged fields.

One environment step is ordered as:

1. Capture each agent's partial observation and public task snapshot at t.
2. Finish selecting both agents' actions (no message delivery between them).
3. Deliver this step's active messages; invoke Reactive on the saved O(t).
4. Execute the frozen physical/message action dictionary.
5. Settle/replenish tasks at t+1 and record public completion evidence.
6. Consume the new snapshot/observation on the following decision step.

Reply changes dialogue and macro focus, not the already selected action.
Its context labels that action as selected but not executed. TRACE keeps its
existing Active/Reactive triggers, schema and rule calculations; own-action
evidence explicitly marks selection rather than completed execution.

Trajectory records include per-step control before/after Reactive and public
task snapshots. Result configuration records `decision_protocol_version: 2`
and timing semantics. The five-file result layout is retained; the compact
summary does not duplicate the full task table. Older trajectories are not
migrated. Tests use scripted generations, not evidence of real model quality.
