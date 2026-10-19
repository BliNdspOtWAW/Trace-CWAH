"""Translate evidence-backed TRACE state into planner context, not commands."""

import json


BELIEF_DIMENSIONS = (
    "Belief dimensions:\n\n"
    "- **Intent:** Bob's likely current high-level action or objective.\n"
    "- **Capability:** Bob's observed ability to execute tasks successfully.\n"
    "- **Preference:** Bob's inferred recurring behavioral tendencies in communication, "
    "coordination, delegation, initiative, and responsiveness.\n"
    "- **Reliability:** Bob's explicit task commitments and their verified fulfillment.\n"
    "- **Uncertainty:** The uncertainty of each estimate "
    "(0 = well supported, 1 = unknown or unsupported).\n\n"
    "Treat uncertain estimates cautiously. Distinguish inferred intentions, "
    "explicit commitments, and verified outcomes.\n\n"
)


def _recent_bob_evidence(ledger, limit):
    return [event.to_dict() for event in ledger.events
            if event.kind in ('peer_task_progress', 'peer_task_outcome')
            or (event.kind == 'message' and event.payload.get('sender') == ledger.peer_id)
            or (event.kind == 'task_completion'
                and event.payload.get('completed_by') == 'Bob')][-limit:]


def render_policy_context(state, ledger):
    dims = state.dimensions
    if dims["information"].status == "unknown":
        return "TRACE belief has no evidence yet. Do not assume anything about Bob."
    # Information is already rendered through the planner's normal Goal/Progress/
    # History fields. Inject only the inferred teammate belief here.
    recent = _recent_bob_evidence(ledger, 8)
    values = {}
    for name in ("intent", "capability", "preference", "reliability"):
        item = dims[name]
        values[name] = {
            "status": item.status,
            "value": item.value,
            "uncertainty": item.uncertainty,
            "evidence_ids": [e.reference for e in item.evidence[-4:]],
        }
    values["uncertainty"] = {
        "status": dims["uncertainty"].status,
        "value": dims["uncertainty"].value,
    }
    return (
        "TRACE teammate belief about Bob, derived from Alice's available information:\n\n"
        "Use this belief to inform coordination decisions when relevant. Adapt your actions, "
        "communication, delegation, or verification strategy based on the TRACE belief "
        "when appropriate.\n\n"
        + BELIEF_DIMENSIONS
        + "Recent attributable Bob evidence:\n" + json.dumps(recent, ensure_ascii=False) + "\n\n"
        + "Teammate belief:\n" + json.dumps(values, ensure_ascii=False)
    )


def render_reply_context(state, ledger):
    """Render teammate belief for message-triggered deliberation."""
    dims = state.dimensions
    if dims["information"].status == "unknown":
        return "TRACE belief has no evidence yet. Do not assume anything about Bob."
    values = {}
    for name in ("intent", "capability", "preference", "reliability"):
        item = dims[name]
        values[name] = {
            "status": item.status,
            "value": item.value,
            "uncertainty": item.uncertainty,
            "evidence_ids": [e.reference for e in item.evidence[-4:]],
        }
    uncertainty = dims["uncertainty"]
    values["uncertainty"] = {
        "status": uncertainty.status,
        "value": uncertainty.value,
    }
    recent = _recent_bob_evidence(ledger, 4)
    return (
        "TRACE teammate belief about Bob, derived from Alice's available information:\n\n"
        "Use this belief to inform your response and coordination decisions when relevant. "
        "Consider whether to maintain or revise your intended actions and coordination "
        "strategy based on the TRACE belief and the newly received message.\n\n"
        + BELIEF_DIMENSIONS
        + "Recent attributable Bob evidence:\n" + json.dumps(recent, ensure_ascii=False) + "\n\n"
        + "Teammate belief:\n" + json.dumps(values, ensure_ascii=False)
    )
