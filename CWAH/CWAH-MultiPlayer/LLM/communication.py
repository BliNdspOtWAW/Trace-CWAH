"""Message-triggered reactive planning without directly changing environment actions."""

import re
import time
import json
import copy
from pathlib import Path
from trace_tf.information import DecisionInformation
from .telemetry import measured_generate


class ReplyFormatError(ValueError):
    """A failed parse with diagnostics, never an alternative executable reply."""

    def __init__(self, call):
        super().__init__("reactive planner must output non-empty <intention> and <reply>")
        self.call = call


def visible_object_summary(observation, limit=40):
    """Summarize only the recipient's partial observation, never the full scene."""
    nodes = observation.get("nodes", [])
    names = [f"<{node['class_name']}> ({node['id']})" for node in nodes
             if node.get("category") != "Rooms" and node.get("class_name") != "character"]
    return ", ".join(names[:limit]) + (" ..." if len(names) > limit else "") or "none visible"


def observed_self_state(observation, agent_id):
    """Read only the recipient's own room and held objects from a fresh partial view."""
    if agent_id is None or "nodes" not in observation or "edges" not in observation:
        return "Unknown", "Unknown"
    nodes = {node["id"]: node for node in observation["nodes"]}
    room = "Unknown"
    held = []
    for edge in observation["edges"]:
        if edge.get("from_id") != agent_id:
            continue
        target_id = edge.get("to_id")
        target = nodes.get(target_id, {})
        relation = edge.get("relation_type")
        if relation == "INSIDE" and target.get("category") == "Rooms":
            room = target["class_name"]
        elif relation in ("HOLDS_RH", "HOLDS_LH"):
            held.append(f"<{target.get('class_name', 'object')}> ({target_id})")
    return room, ", ".join(held) if held else "None"


class CommunicationModule:
    def __init__(self, planner):
        self.planner = planner
        role = 'Alice' if getattr(planner, 'agent_name', 'Alice') == 'Alice' else 'Bob'
        self.prompt_path = Path(getattr(
            planner, 'reply_prompt_template_path',
            Path(__file__).with_name(f'prompt_Reply_{role}.txt'),
        ))
        self.prompt_template = self.prompt_path.read_text(encoding='utf-8')

    def build_decision_information(
            self, incoming_message, dialogue_history, current_skill, observation,
            agent_id=None, progress="Unknown", action_history=(), step=0,
            source_event_ids=(), public_tasks=None, agent_control=None):
        observed_room, holding = observed_self_state(
            observation, agent_id if agent_id is not None else getattr(self.planner, "agent_id", None)
        )
        current_progress = (
            f"Current room: {observed_room}\n"
            f"Currently held objects: {holding}\n"
            f"Visible relevant objects: {visible_object_summary(observation)}\n"
            f"{progress or 'Unknown'}"
        )
        return DecisionInformation(
            planner_type="reactive",
            goal=self.planner.goal_desc or "Unknown",
            personality=self.planner.personality or "None",
            current_progress=current_progress,
            action_history=tuple(action_history),
            dialogue_history=tuple(dialogue_history),
            available_actions=(),
            step=int(step),
            source_event_ids=tuple(source_event_ids),
            current_skill=current_skill or "None",
            incoming_message=incoming_message,
            public_tasks=copy.deepcopy(public_tasks or {}),
            agent_control=copy.deepcopy(agent_control or {}),
        )

    def reply(self, incoming_message, dialogue_history, current_skill, current_room, observation,
              agent_id=None, progress="Unknown", action_history=(), belief_context="",
              decision_information=None):
        prompt = self.prompt_template
        information = decision_information or self.build_decision_information(
            incoming_message, dialogue_history, current_skill, observation,
            agent_id=agent_id, progress=progress, action_history=action_history,
        )
        values = {
            "$AGENT_NAME": self.planner.agent_name,
            "$OPPO_NAME": self.planner.oppo_name,
            "$ASSIGNED_PERSONALITY": information.personality,
            "$GOAL": information.goal,
            "$CURRENT_SKILL": information.current_skill,
            "$PROGRESS": information.current_progress,
            "$ACTION_HISTORY": ", ".join(information.action_history) or "None",
            "$DIALOGUE_HISTORY": "\n".join(information.dialogue_history) or "None",
            "$TRACE_BELIEF": belief_context or "TRACE is disabled. No teammate belief is available.",
            "$INCOMING_MESSAGE": incoming_message,
            "$PUBLIC_TASKS": json.dumps(information.public_tasks, ensure_ascii=False),
            "$AGENT_CONTROL": json.dumps(information.agent_control, ensure_ascii=False),
        }
        for placeholder, value in values.items():
            prompt = prompt.replace(placeholder, str(value))
        request = [{"role": "user", "content": prompt}] if self.planner.chat else prompt
        sampling = self.planner.sampling_params
        params = dict(sampling) if isinstance(sampling, dict) else vars(sampling).copy()
        params["n" if self.planner.chat else "num_return_sequences"] = 1
        if self.planner.chat:
            params["max_tokens"] = min(params.get("max_tokens", 256), 256)
        else:
            params["max_new_tokens"] = min(params.get("max_new_tokens", 256), 256)
        started = time.monotonic()
        outputs, legacy_cost = measured_generate(
            self.planner, request, params, 'reactive', information,
            sections={
                'goal': values['$GOAL'], 'progress': values['$PROGRESS'],
                'current_skill': values['$CURRENT_SKILL'],
                'action_history': values['$ACTION_HISTORY'],
                'dialogue_history': values['$DIALOGUE_HISTORY'],
                'trace_belief': values['$TRACE_BELIEF'], 'incoming_message': incoming_message,
            })
        elapsed = time.monotonic() - started
        raw = outputs[0] if outputs else ""
        intention_match = re.search(r"<intention>\s*(.*?)\s*</intention>", raw, re.I | re.S)
        reply_match = re.search(r"<reply>\s*(.*?)\s*</reply>", raw, re.I | re.S)
        intention = intention_match.group(1).strip()[:500] if intention_match else ""
        reply = reply_match.group(1).strip()[:500] if reply_match else ""
        task_match = re.search(r'<new_subtask_id>\s*(.*?)\s*</new_subtask_id>', raw, re.I | re.S)
        task_id = task_match.group(1).strip() if task_match else None
        call = {
            "prompt_file": str(self.prompt_path),
            "model": self.planner.lm_id,
            "prompt": prompt,
            "raw_output": raw,
            "intention": intention,
            "reply": reply,
            "new_subtask_id": None if task_id in (None, '', 'null', 'None') else task_id,
            "usage": self.planner.last_usage_metadata,
            "legacy_cost_estimate": legacy_cost,
            "latency_seconds": elapsed,
            "shared_information": information.to_dict(),
            "metrics_call_id": getattr(self.planner, 'last_call_metrics', {}).get('call_id'),
            "parse_status": "ok" if intention and reply else "error",
            "parse_error": None if intention and reply else "missing_or_empty_intention_or_reply",
        }
        if not intention or not reply or (information.public_tasks and (task_match is None or not task_id)):
            call['parse_status'] = 'error'
            call['parse_error'] = 'missing_or_empty_required_reactive_fields'
            raise ReplyFormatError(call)
        return reply, call
