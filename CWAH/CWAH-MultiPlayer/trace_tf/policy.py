"""Translate evidence-backed TRACE state into planner context, not commands."""

import json


def render_policy_context(state, ledger):
    dims = state.dimensions
    if dims["information"].status == "unknown":
        return "TRACE belief has no evidence yet. Do not assume anything about Bob."
    # Information is already rendered through the planner's normal Goal/Progress/
    # History fields. Inject only the inferred teammate belief here.
    recent = [event.to_dict() for event in ledger.events
              if event.kind in ("message", "peer_task_outcome")][-8:]
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
        "TRACE teammate belief about Bob, derived from the Information above:\n"
        "Use it for coordination, communication, and action planning when relevant.\n"
        "Intent: Bob's likely current high-level action.\n"
        "Capability: Bob's observed task-execution ability.\n"
        "Preference: Bob's inferred behavioral and collaboration tendencies.\n"
        "Reliability: Bob's explicit commitments and their verified fulfillment.\n"
        "Uncertainty: Uncertainty in each inference (0 = certain, 1 = unknown).\n"
        "Treat uncertain estimates cautiously and distinguish intent from commitment.\n"
        "Recent attributable Bob evidence: " + json.dumps(recent, ensure_ascii=False) + "\n"
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
        "Use this belief, derived from the same Information shown above, when relevant. "
        "Treat uncertain inferences cautiously and do not present them as verified facts.\n"
        "Recent attributable evidence: " + json.dumps(recent, ensure_ascii=False) + "\n"
        "Teammate belief: " + json.dumps(values, ensure_ascii=False)
    )
