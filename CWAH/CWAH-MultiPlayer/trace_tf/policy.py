"""Translate evidence-backed TRACE state into planner context, not commands."""

import json


def render_policy_context(state, ledger):
    dims = state.dimensions
    if dims["information"].status == "unknown":
        return "TRACE belief has no evidence yet. Do not assume anything about Bob."
    # The planner already receives the full dialogue history separately. Keep the
    # observation ledger complete for audit, but inject only attributable cues.
    recent = [event.to_dict() for event in ledger.events
              if event.kind in ("message", "peer_task_outcome")][-8:]
    values = {name: {"status": item.status, "value": item.value,
                     "uncertainty": item.uncertainty,
                     "evidence_ids": [e.reference for e in item.evidence]}
              for name, item in dims.items()}
    values["information"]["value"] = {
        "event_count": len(ledger.events),
        "latest_event_id": ledger.events[-1].event_id if ledger.events else None,
        "full_history": "TRACE evidence ledger",
    }
    values["information"]["evidence_ids"] = []
    return (
        "TRACE teammate belief about Bob:\n"
        "Use the following evidence and beliefs to support coordination, "
        "task allocation, communication, and action planning when relevant.\n"
        "Information: Alice's observation, action, and communication history.\n"
        "Intent: Bob's likely current high-level action and communicated commitments.\n"
        "Capability: Bob's observed task-execution ability.\n"
        "Preference: Bob's inferred behavioral and collaboration tendencies.\n"
        "Reliability: Bob's coordination quality and fulfillment of commitments.\n"
        "Uncertainty: Uncertainty in each inference (0 = certain, 1 = unknown).\n"
        "Ground decisions in available evidence, account for uncertainty, "
        "and adapt coordination to Bob's observed behavior.\n"
        "Relevant evidence: " + json.dumps(recent, ensure_ascii=False) + "\n"
        "Teammate belief: " + json.dumps(values, ensure_ascii=False)
    )


def render_reply_context(state, ledger):
    """Render a compact teammate belief for message-triggered deliberation."""
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
    recent = [event.to_dict() for event in ledger.events
              if event.kind in ("message", "peer_task_outcome")][-4:]
    return (
        "Use this evidence-backed belief about Bob when it is relevant to the reply. "
        "Treat uncertain inferences cautiously and do not present them as verified facts.\n"
        "Recent attributable evidence: " + json.dumps(recent, ensure_ascii=False) + "\n"
        "Teammate belief: " + json.dumps(values, ensure_ascii=False)
    )
