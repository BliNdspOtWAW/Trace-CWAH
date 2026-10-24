# TRACE v2: decision-coupled joint Intent and Beta statistics

Runtime branch: `CWAH-TF-v2`. No hidden Bob plan, action or profile enters Alice's
online TRACE inputs. Bob and TRACE-off agents retain their old message contract.

## Decision boundaries

An Active planning invocation or message-triggered Reactive planning invocation
is one decision. Both update TRACE immediately before the planner. Environment
steps that continue an existing skill collect legal observations but do not call
the Intent updater. Active retries are separate invocations and get separate
decision IDs. Same-environment-step Active/Reactive decisions are distinguishable.

The previous decision posterior is saved verbatim. Its support-aligned version
is supplied as the new prior; retired task mass is removed and new cells receive
their uniform share, with retained cells preserving relative mass. This support
adaptation is a heuristic, not a learned state transition. New evidence is the
ledger interval since the previous decision, not since the previous environment
step. Invalid generations retain the aligned prior and log the failure.

## Intent

The LLM emits `task_probs` and `skill_given_task`. The runtime validates exact
public candidate IDs and finite nonnegative positive-total weights, normalizes,
and forms `P(task,skill)=P(task)*P(skill|task)`. The task is Bob's current macro
objective; skill is the currently executing skill, which may still serve an
earlier task after a negotiated switch. They are estimates, not private truth.

Skills: goexplore, gocheck, gograb, goput, send_message, idle. Public active task
IDs are dynamic. Reserved `none` supports only goexplore/gocheck/send_message/idle.
No Bob-private feasibility mask is used. Initialization is uniform over legal
joint cells. Zero-probability cells remain in the entropy support denominator.
Normalized entropy is `-sum(p log p)/log(K)`; one-cell support has entropy zero.
Task entropy and conditional skill entropy are retained for diagnostics.

## Capability and Reliability

Both use Beta(1+S, 1+F), posterior mean and variance, with normalized uncertainty
`12*variance`. Initial mean is 0.5 and normalized variance is 1. Counts exclude
pending/unverifiable records. Low variance does not imply a high success rate.

Capability is global and task-level. Visible progress starts a record; public Bob
completion is success. All other cases, including failed primitive actions or
terminal-failure hints, are not task failures in this version: F_C remains zero.
Alice completing Bob's in-progress task is abandoned/unknown. This deliberately
optimistic capability estimate is not a measure of all attempted-action success.
Seeing Bob reappear holding an object is not treated as a fresh observed grab.

Reliability counts only Alice's typed delegate requests explicitly accepted by
Bob in a linked reply. LLM classifies acceptance, never success probabilities.
Acceptance must concern the entire bound subtask; unsolicited promises or partial
grab/search agreement do not establish a full-task delegation. Repeated requests
for the same concrete task do not inflate the responsibility count. Public Bob
completion fulfills it; Alice completion violates it. Unknown actor closes it as
unverifiable. No deadline exists: unfinished tasks at the episode budget remain
pending, not failures. First observed progress and completion timestamps are
recorded; they are not Bob's hidden execution start time.

Preference remains unknown and is excluded from policy and uncertainty. The
uncertainty value is the vector `[intent, capability, reliability]`; no averaged
overall score is computed. The generic DimensionBelief scalar uncertainty for
this vector uses its maximum only as a conservative serialization envelope.

## Communication

Only TRACE-on Alice's Active output additionally requires
`<communication_type>delegate|probe_intent</communication_type>` when choosing
send_message. delegate binds an active subtask_id; probe_intent requires null.
Missing/invalid metadata enters the existing bounded Active retry path. Delivery
keeps the ordinary natural-language payload, capped at 500 characters.

Local request metadata is attached to Alice's delivered ledger event. Bob's
ordinary immediate reply is linked locally; Bob needs no new output fields.
delegate creates a reliability record only after explicit acceptance is parsed.
probe_intent asks about intent only, records request/reply references under Intent,
and never creates commitments. Replies remain unverified claims. Reactive replies
cannot initiate a second typed active request. No communication cancels a running
skill or modifies the step's frozen physical action. Bob's existing Reactive may
still change its future macro target after either message type.

## Offline analysis status

The previous likelihood CLI and plotting implementation have been removed.
The independent replacement is documented in [Offline Intent agreement](OFFLINE_INTENT_AGREEMENT.md).
Entry: CWAH/experiment_runner/compare_intent_posteriors.py (prepare / infer / report).
Online Intent updates and legal evidence archives are unchanged. Both offline
branches retrospectively infer the same starting-decision Intent, allowing
within-interval state switches. A uses prior times relative compatibility; B
directly generates a posterior. Full-support base-2 JSD measures mechanism
agreement, not accuracy. Neither branch uses the next online posterior as its
answer.

## Validation

Scripted unit/integration tests cover normalization/entropy, candidate changes,
generation failures, multi-environment-step and same-step decisions, Beta updates,
delegation settlement, probe isolation, private-input exclusion, real Active
parsing, Arena delivery and skill preservation. No real-model quality or Unity
performance improvement is claimed by these tests.


## Confirmed refinements — 2026-10-25

Capability counts Bob's unique public task completions only; no other outcome
increments its failure count. Reliability's accepted-task rule is unchanged.

Checkpoint and episode exports now save `last_decision_belief` with its decision
ID/step, `final_rule_statistics`, and `unconsumed_intent_events`. Rules are computed
on detached copies using the complete legal ledger; export makes no model call,
creates no decision, advances no Intent cursor and changes no online state.
Repeated exports are idempotent. `final_belief` remains a compatibility alias of
`last_decision_belief`, explicitly marked by `final_belief_semantics`; it does not
claim to be an updated terminal Intent or Information. Final statistics contain
their evidence coverage. Unclassified last-minute delegate replies are explicitly
listed; rule-only export cannot invent LLM acceptance classifications.

Empty, missing or unmatched actions no longer randomly select an action. A parsed
no-action returns `None` to the environment, sends no message, consumes one normal
environment step, and clears no active macro target. No permanent wait skill is
installed; the agent can plan again on its next step. This does not pause Bob or
the environment. Explicit malformed task-bearing proposals retain bounded retries.

Intent-unconsumed terminal events remain archived separately and are not
relabeled as a new decision. Legal partial observations and decision/event
cursors are retained for future offline analysis.


## Peer observation versus memory boundary ? 2026-10-25

DecisionInformation now carries two detached structures:
- peer_observation (O_t_peer): observer/peer IDs, observation step, visibility and
  directly observed state. Invisible means state=null; no view yet means
  visible=null. State contains room, observed holding relations and position
  only when present in that partial graph.
- peer_memory (M_t_peer): last_observed_state and last_observed_step, retained
  only from prior direct observations. Messages never overwrite these fields.

PeerStateTracker refreshes observation at environment observation boundaries;
Intent still updates only before Active/Reactive decisions. Disappearance clears
current state but preserves timestamped history. Reappearance replaces memory
using the fresh view. Episode reset clears the tracker. Snapshots detach inputs
and outputs. Older observations cannot overwrite a newer current view.

Active progress no longer derives visibility from legacy opponent_last_room or
cached holdings. Active and Reactive both render the same labeled O/M structures;
their serialized shared_information includes the fields for TRACE and auditing.
TRACE observation-event summaries use peer node presence even when no room or
holding edge is available. Historical events are explicitly step-local facts,
not proof of current visibility. Baseline agents receive the same corrected
information boundary; no private teammate plan/state is added.

Replaying the original 50-step run's 19 decision boundaries identified five old
false-present statements at steps 21,22,23,24,25. With the new builder, those views
are invisible with no current peer state. At steps 24/25 the last direct sighting
is correctly retained as step 18 in the kitchen. This is a deterministic replay
of saved legal observations, not a fresh model-quality or performance claim.
Old experiment artifacts are retained unchanged.


## Active protocol versus message payload ? 2026-10-25

TRACE-on Active outputs now use a dedicated parser. cot/message/action envelopes
are isolated before top-level subtask_id and communication_type are read.
Controls in message/cot cannot override the top-level task/action/type.
Duplicate, malformed or misplaced controls are rejected through the existing
bounded Active retry loop. In particular, nested protocol tags in message are
not silently stripped or forwarded, because conflicting task IDs would make
repair ambiguous. Validated prose alone is wrapped into the send_message action;
ordinary object labels such as <apple> (375) remain intact.

The planner context includes a complete sibling-field example. Empty actions
still execute nothing. Bob/TRACE-off parsing keeps its previous contract.
Original raw generations remain in audit logs; this does not mean protocol tags
are delivered to peers. Old trajectories are unchanged.

Deterministic replay rejected all five metadata-leaking Active messages from the
two completed 50-step experiments. Regression tests cover nested/duplicate fields,
control tags in cot, malformed tags, option selection, no-ops and real Arena
delivery of prose with metadata retained only in Alice's local request record.


## Local Qwen validation limitation

The local Qwen3-8B TRACE-on 100-step validation was interrupted after 41 steps.
At the diagnostic checkpoint, 39 of 40 completed Intent updates retained their
prior; 39 updater calls reached the 1536-token output limit. Typical updater
latency was about 57 seconds. Active communication formatting was fixed, but
Updater schema compliance/output truncation remains unresolved. Do not treat
that run as a valid TRACE performance benchmark. Its temporary artifacts were
removed after recording this limitation; the regression suite remains.
