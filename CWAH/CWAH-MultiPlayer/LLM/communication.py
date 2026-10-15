"""One-shot, action-free replies to active peer messages."""

import re
import time
from pathlib import Path


REPLY_TEMPLATE = (Path(__file__).with_name("prompt_Reply.txt")).read_text(encoding="utf-8")


def visible_object_summary(observation, limit=40):
    """Summarize only the recipient's partial observation, never the full scene."""
    nodes = observation.get("nodes", [])
    names = [f"<{node['class_name']}> ({node['id']})" for node in nodes
             if node.get("category") != "Rooms" and node.get("class_name") != "character"]
    return ", ".join(names[:limit]) + (" ..." if len(names) > limit else "") or "none visible"


class CommunicationModule:
    def __init__(self, planner):
        self.planner = planner

    def reply(self, incoming_message, dialogue_history, current_skill, current_room, observation):
        prompt = REPLY_TEMPLATE
        values = {
            "$AGENT_NAME": self.planner.agent_name,
            "$OPPO_NAME": self.planner.oppo_name,
            "$ASSIGNED_PERSONALITY": self.planner.personality or "None",
            "$GOAL": self.planner.goal_desc or "Unknown",
            "$CURRENT_SKILL": current_skill or "None",
            "$CURRENT_ROOM": current_room or "Unknown",
            "$VISIBLE_OBJECTS": visible_object_summary(observation),
            "$DIALOGUE_HISTORY": "\n".join(dialogue_history) or "None",
            "$INCOMING_MESSAGE": incoming_message,
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
        outputs, legacy_cost = self.planner.generator(request, params)
        elapsed = time.monotonic() - started
        raw = outputs[0] if outputs else ""
        match = re.search(r"<reply>\s*(.*?)\s*</reply>", raw, re.I | re.S)
        reply = match.group(1).strip()[:500] if match else None
        return reply, {
            "model": self.planner.lm_id,
            "prompt": prompt,
            "raw_output": raw,
            "usage": self.planner.last_usage_metadata,
            "legacy_cost_estimate": legacy_cost,
            "latency_seconds": elapsed,
        }
