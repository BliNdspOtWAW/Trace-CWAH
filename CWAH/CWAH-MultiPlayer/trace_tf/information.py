"""One legal decision context shared by TRACE and the planners."""

import copy
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionInformation:
    planner_type: str
    goal: str
    personality: str
    current_progress: str
    action_history: tuple = field(default_factory=tuple)
    dialogue_history: tuple = field(default_factory=tuple)
    available_actions: tuple = field(default_factory=tuple)
    available_actions_with_distance: tuple = field(default_factory=tuple)
    step: int = 0
    source_event_ids: tuple = field(default_factory=tuple)
    current_skill: str = None
    incoming_message: str = None
    public_tasks: dict = field(default_factory=dict)
    agent_control: dict = field(default_factory=dict)

    def to_dict(self):
        return copy.deepcopy({
            "planner_type": self.planner_type,
            "goal": self.goal,
            "personality": self.personality,
            "current_progress": self.current_progress,
            "action_history": list(self.action_history),
            "dialogue_history": list(self.dialogue_history),
            "available_actions": list(self.available_actions),
            "step": self.step,
            "source_event_ids": list(self.source_event_ids),
            "current_skill": self.current_skill,
            "incoming_message": self.incoming_message,
            "public_tasks": self.public_tasks,
            "agent_control": self.agent_control,
        })
