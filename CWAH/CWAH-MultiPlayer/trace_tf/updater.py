"""One-call hybrid updater: legal history, LLM interpretation, rule settlement."""

import json
import re
import time
from pathlib import Path

from .events import EvidenceLedger
from .model import BeliefState, Evidence, EvidenceSource


PROMPT = Path(__file__).with_name("prompt_Updater.txt").read_text(encoding="utf-8")
SCORED = ("intent", "capability", "preference", "reliability")
ACTIONS = {"goexplore", "gocheck", "gograb", "goput", "send_message"}


def _unit(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        return 1.0
    return float(value)


class HybridUpdater:
    def __init__(self, planner, teammate_id=2):
        self.planner = planner
        self.ledger = EvidenceLedger(target_id=1, peer_id=teammate_id)
        self.state = BeliefState(teammate_id)
        self.cursor = 0
        self.update_log = []
        self.last_context = ""

    def _evidence(self, events):
        evidence = []
        for event in events:
            if event.kind == "message":
                source = (EvidenceSource.TEAMMATE_MESSAGE if event.payload["sender"] == 2
                          else EvidenceSource.OWN_MESSAGE)
            elif event.kind == "own_action":
                source = EvidenceSource.OWN_ACTION
            elif event.kind in ("peer_task_progress", "peer_task_outcome"):
                source = EvidenceSource.VISIBLE_PEER_OUTCOME
            elif event.kind == "task_completion" and event.payload.get("completed_by") == "Alice":
                source = EvidenceSource.OWN_ACTION_OUTCOME
            elif event.kind == "task_completion":
                source = EvidenceSource.VISIBLE_OBSERVATION
            else:
                source = EvidenceSource.VISIBLE_OBSERVATION
            evidence.append(Evidence(source, event.step, event.kind, event.event_id))
        return evidence

    def _apply(self, dimension, value, uncertainty, events):
        existing = {e.reference for e in self.state.dimensions[dimension].evidence}
        new = [e for e in self._evidence(events) if e.reference not in existing]
        if new:
            self.state = self.state.with_evidence(dimension, value, 1 - uncertainty,
                                                  new, uncertainty=uncertainty)

    def _prompt_events(self, events):
        result = []
        for event in events:
            payload = event.payload
            if event.kind == "observation":
                nodes = {node["id"]: node for node in payload["nodes"]}
                facts = []
                for edge in payload["edges"]:
                    if edge["from_id"] != self.ledger.peer_id:
                        continue
                    target = nodes.get(edge["to_id"], {})
                    relation = edge["relation_type"]
                    if relation in ("HOLDS_RH", "HOLDS_LH"):
                        facts.append("Bob visibly holds <{}> ({})".format(
                            target.get("class_name", "object"), edge["to_id"]
                        ))
                    elif relation == "INSIDE" and target.get("category") == "Rooms":
                        facts.append("Bob is visibly in the {}".format(target.get("class_name")))
                payload = {"semantic_peer_facts": facts or ["Bob is not currently visible"]}
            result.append({"event_id": event.event_id, "step": event.step,
                           "kind": event.kind, "payload": payload})
        return result

    def _compact_prior(self):
        result = {}
        for name, item in self.state.dimensions.items():
            if name in ("information", "uncertainty"):
                continue
            result[name] = {
                "status": item.status,
                "value": item.value,
                "uncertainty": item.uncertainty,
                "recent_evidence_ids": [e.reference for e in item.evidence[-8:]],
            }
        return result

    def _infer(self, information):
        # The ledger remains complete for audit. Inference receives the exact legal
        # Information snapshot shared with the planner plus compact Bob evidence.
        observations = [e for e in self.ledger.events if e.kind == "observation"]
        selected = [e for e in self.ledger.events
                    if e.kind in ("message", "peer_task_progress", "peer_task_outcome",
                                  "task_completion")][-20:]
        if observations:
            selected.append(observations[-1])
        selected.sort(key=lambda e: int(e.event_id[1:]))
        prompt = PROMPT.replace("$INFORMATION", json.dumps(information, ensure_ascii=False))
        prompt = prompt.replace("$BELIEF", json.dumps(self._compact_prior(), ensure_ascii=False))
        prompt = prompt.replace("$EVENTS", json.dumps(self._prompt_events(selected), ensure_ascii=False))
        sampling = self.planner.sampling_params
        params = dict(sampling) if isinstance(sampling, dict) else vars(sampling).copy()
        params["n"] = 1
        params["temperature"] = 0
        limit_key = "max_tokens" if self.planner.chat else "max_new_tokens"
        params[limit_key] = min(params.get(limit_key, 768), 768)
        request = [{"role": "user", "content": prompt}] if self.planner.chat else prompt
        started = time.monotonic()
        outputs, legacy_cost = self.planner.generator(request, params)
        latency = time.monotonic() - started
        raw = outputs[0] if outputs else ""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.I | re.S)
        data = json.loads(match.group(1) if match else raw.strip())
        if not isinstance(data, dict):
            raise ValueError("updater output must be an object")
        return data, {"model": self.planner.lm_id, "usage": self.planner.last_usage_metadata,
                      "legacy_cost_estimate": legacy_cost, "latency_seconds": latency,
                      "prompt": prompt, "raw_output": raw}

    def _valid_events(self, ids):
        by_id = self.ledger.by_id()
        if not isinstance(ids, list) or not all(isinstance(i, str) and i in by_id for i in ids):
            return []
        return [by_id[i] for i in dict.fromkeys(ids)]

    def _peer_evidence(self, events):
        """Alice's own request alone cannot establish Bob's intent/preference."""
        result = []
        for event in events:
            if event.kind == "message" and event.payload["sender"] == 2:
                result.append(event)
            elif event.kind in ("peer_task_progress", "peer_task_outcome"):
                result.append(event)
            elif event.kind == "task_completion" and event.payload.get("completed_by") == "Bob":
                result.append(event)
            elif event.kind == "observation" and any(
                node["id"] == 2 for node in event.payload["nodes"]
            ):
                result.append(event)
        return result

    def _commitments_from_reliability(self):
        reliability = self.state.dimensions["reliability"]
        if reliability.status != "supported" or not isinstance(reliability.value, dict):
            return {}
        return {
            item["commitment_id"]: json.loads(json.dumps(item))
            for item in reliability.value.get("commitment_events", [])
            if isinstance(item, dict) and item.get("commitment_id")
        }

    def _accept_commitments(self, proposals, commitments):
        if not isinstance(proposals, list):
            return
        by_id = self.ledger.by_id()
        for item in proposals:
            if not isinstance(item, dict):
                continue
            key = item.get("acceptance_event_id")
            accepted = by_id.get(key)
            action = item.get("action")
            obj = item.get("object")
            subtask_id = item.get("subtask_id")
            if (not accepted or accepted.kind != "message" or accepted.payload["sender"] != 2
                    or action not in ("grab", "place", "search")
                    or (obj is not None and not isinstance(obj, str))
                    or (action in ("grab", "place")
                        and subtask_id not in self.ledger.known_subtask_ids())
                    or (action == "search" and subtask_id is not None
                        and subtask_id not in self.ledger.known_subtask_ids())
                    or not self._valid_events(item.get("evidence_ids", []))):
                continue
            commitment_id = "{}:{}:{}".format(key, subtask_id or "none", action)
            proposed = item.get("proposed_status")
            if proposed not in ("pending", "fulfilled", "violated", "unverifiable"):
                proposed = "unverifiable"
            if commitment_id not in commitments:
                commitments[commitment_id] = {
                    "commitment_id": commitment_id, "acceptance_event_id": key,
                    "subtask_id": subtask_id, "action": action, "object": obj,
                    "status": "pending",
                    "evidence_ids": [key], "llm_proposed_status": proposed,
                    "llm_evidence_ids": list(dict.fromkeys(item["evidence_ids"])),
                }
            else:
                commitments[commitment_id]["llm_proposed_status"] = proposed
                commitments[commitment_id]["llm_evidence_ids"] = list(dict.fromkeys(item["evidence_ids"]))

    def _settle_commitments(self, commitments):
        for commitment in commitments.values():
            if commitment["status"] != "pending":
                continue
            accepted_step = self.ledger.by_id()[commitment["acceptance_event_id"]].step
            for event in self.ledger.events:
                outcome = event.payload
                if event.step < accepted_step:
                    continue
                matched = False
                if event.kind == "peer_task_progress":
                    task = outcome.get("task") or {}
                    matched = (
                        commitment["action"] == "grab"
                        and task.get("subtask_id") == commitment.get("subtask_id")
                    )
                elif event.kind == "peer_task_outcome":
                    task = outcome.get("task") or {}
                    matched = (
                        outcome.get("verified")
                        and outcome.get("action") == commitment["action"]
                        and task.get("subtask_id") == commitment.get("subtask_id")
                    )
                elif event.kind == "task_completion" and outcome.get("completed_by") == "Bob":
                    task = outcome.get("task") or {}
                    matched = (
                        commitment["action"] == "place"
                        and task.get("subtask_id") == commitment.get("subtask_id")
                    )
                if not matched:
                    continue
                if event.kind in ("peer_task_progress", "task_completion"):
                    commitment["status"] = "fulfilled"
                elif outcome.get("success") is True:
                    commitment["status"] = "fulfilled"
                elif outcome.get("success") is False and outcome.get("terminal") is True:
                    commitment["status"] = "violated"
                if commitment["status"] != "pending":
                    commitment["evidence_ids"].append(event.event_id)
                    break

    def _capability_executions(self):
        """Build task-level Bob execution records from legal progress/completion evidence."""
        executions = {}
        for event in self.ledger.events:
            payload = event.payload
            task = payload.get("task") or {}
            key = task.get("subtask_id")
            if not key:
                continue
            if event.kind == "peer_task_progress":
                item = executions.setdefault(key, {
                    "subtask_id": key, "task": task, "actor": "Bob",
                    "status": "in_progress", "outcome": "unknown",
                    "evidence_ids": [],
                })
                item["evidence_ids"].append(event.event_id)
                item["observed_actions"] = list(dict.fromkeys(
                    item.get("observed_actions", []) + [payload.get("observed_action")]
                ))
            elif event.kind == "task_completion":
                completed_by = payload.get("completed_by", "unknown")
                item = executions.get(key)
                if completed_by == "Bob":
                    item = item or {
                        "subtask_id": key, "task": task, "actor": "Bob",
                        "evidence_ids": [], "observed_actions": [],
                    }
                    executions[key] = item
                    item.update({"status": "completed", "outcome": "success",
                                 "completed_by": "Bob"})
                    item["evidence_ids"].append(event.event_id)
                elif item is not None and item["status"] == "in_progress":
                    item.update({"status": "abandoned", "outcome": "unknown",
                                 "completed_by": completed_by})
                    item["evidence_ids"].append(event.event_id)
        return executions

    def update(self, information=None):
        new = self.ledger.since(self.cursor)
        if not new:
            return self.state, None
        before = self.state.to_dict()
        self.cursor = len(self.ledger.events)
        if information is None:
            information = {"event_ids": [e.event_id for e in self.ledger.events],
                           "event_count": len(self.ledger.events)}
        else:
            information = json.loads(json.dumps(information))
            information["evidence_event_count"] = len(self.ledger.events)
        self._apply("information", information, 0.0, new)

        call = None
        inference = {}
        try:
            inference, call = self._infer(information)
        except Exception as exc:
            call = {"error": "{}: {}".format(type(exc).__name__, exc),
                    "model": self.planner.lm_id, "usage": self.planner.last_usage_metadata}

        commitments = self._commitments_from_reliability()
        self._accept_commitments(inference.get("commitments", []), commitments)
        self._settle_commitments(commitments)
        proposals = inference.get("uncertainty", {})
        if not isinstance(proposals, dict):
            proposals = {}
        uncertainties = {}

        for dimension in ("intent", "preference"):
            entry = inference.get(dimension, {})
            entry = entry if isinstance(entry, dict) else {}
            events = self._peer_evidence(self._valid_events(entry.get("evidence_ids", [])))
            value = entry.get("action") if dimension == "intent" else entry.get("description")
            if dimension == "intent" and value not in ACTIONS:
                value = None
            if dimension == "preference" and (not isinstance(value, str) or len(events) < 2):
                value = None
            if value is not None and events:
                floor = 1.0 / (len(events) + 1)
                u = max(_unit(proposals.get(dimension)), floor)
                detail = {"action": value, "target": entry.get("target")} if dimension == "intent" else {"description": value}
                self._apply(dimension, detail, u, events)
                uncertainties[dimension] = u
            else:
                if dimension == "intent" and self.state.dimensions[dimension].status == "supported":
                    prior_refs = {e.reference for e in self.state.dimensions[dimension].evidence}
                    withdrawal = [e for e in self._evidence(new) if e.reference not in prior_refs]
                    if withdrawal:
                        self.state = self.state.with_unknown("intent", withdrawal)
                prior = self.state.dimensions[dimension]
                uncertainties[dimension] = 1.0 if prior.status == "unknown" or prior.uncertainty is None else prior.uncertainty

        executions = self._capability_executions()
        successes = sum(item.get("outcome") == "success" for item in executions.values())
        failures = sum(item.get("outcome") == "failure" for item in executions.values())
        count = successes + failures
        capability_events = [self.ledger.by_id()[eid] for item in executions.values()
                             for eid in item.get("evidence_ids", [])
                             if eid in self.ledger.by_id()]
        if capability_events:
            # Capability is objectively counted, so evidence volume—not an
            # unconstrained LLM score—calibrates its uncertainty.
            u = 1.0 / (count + 1) if count else 1.0
            self._apply("capability", {"successes": successes, "failures": failures,
                                       "success_rate": successes / count if count else None,
                                       "task_execution_events": list(executions.values())},
                        u, capability_events)
            uncertainties["capability"] = u
        else:
            uncertainties["capability"] = 1.0

        fulfilled = sum(c["status"] == "fulfilled" for c in commitments.values())
        violated = sum(c["status"] == "violated" for c in commitments.values())
        resolved = fulfilled + violated
        u = 1.0 / (resolved + 1) if resolved else 1.0
        reliability_events = [self.ledger.by_id()[eid] for commitment in commitments.values()
                              for eid in commitment["evidence_ids"]
                              if eid in self.ledger.by_id()]
        if reliability_events:
            self._apply("reliability", {
                "fulfillments": fulfilled, "violations": violated,
                "fulfillment_rate": fulfilled / resolved if resolved else None,
                "commitment_events": list(commitments.values()),
            }, u, reliability_events)
        uncertainties["reliability"] = u

        overall = sum(uncertainties[name] for name in SCORED) / 4
        self._apply("uncertainty", {"per_dimension": uncertainties, "overall": overall},
                    overall, new)
        record = {"new_event_ids": [e.event_id for e in new], "belief_before": before,
                  "belief_after": self.state.to_dict(), "llm_call": call}
        self.update_log.append(record)
        return self.state, record
