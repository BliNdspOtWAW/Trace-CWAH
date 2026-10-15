"""Action-free replies and complete peer dialogue, without Unity or API calls."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from LLM.communication import CommunicationModule
from LLM.LLM import LLM
from algos.arena_mp2 import ArenaMP


class FakeAgent:
    def __init__(self, name, response):
        self.name = name
        self.response = response
        self.history = []
        self.plan = "[gograb] <apple> (12)"
        self.reply_calls = 0

    def receive_dialogue_event(self, event):
        self.history.append(event.copy())

    def reply_to_message(self, message, observation):
        self.reply_calls += 1
        return self.response, {"usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}


class PeerCommunicationTests(unittest.TestCase):
    def test_active_message_gets_one_free_reply_without_changing_skill_or_step(self):
        arena = ArenaMP.__new__(ArenaMP)
        arena.env = SimpleNamespace(steps=8, get_observations=lambda: {0: {}, 1: {}})
        alice = FakeAgent("Alice", None)
        bob = FakeAgent("Bob", "I will check the kitchen.")
        arena.agents = [alice, bob]

        events = arena.process_messages(["I found an apple.", None])

        self.assertEqual([e["kind"] for e in events], ["active", "reply_call", "reply"])
        self.assertEqual(arena.env.steps, 8)
        self.assertEqual(bob.reply_calls, 1)
        self.assertEqual(alice.reply_calls, 0)
        self.assertEqual(alice.plan, "[gograb] <apple> (12)")
        self.assertEqual(bob.plan, "[gograb] <apple> (12)")
        self.assertEqual(len(alice.history), 2)
        self.assertEqual(len(bob.history), 2)

    def test_simultaneous_active_messages_get_one_reply_each(self):
        arena = ArenaMP.__new__(ArenaMP)
        arena.env = SimpleNamespace(steps=3, get_observations=lambda: {0: {}, 1: {}})
        arena.agents = [FakeAgent("Alice", "yes"), FakeAgent("Bob", "no")]
        events = arena.process_messages(["Question A", "Question B"])
        self.assertEqual(sum(e["kind"] == "reply" for e in events), 2)
        self.assertEqual([agent.reply_calls for agent in arena.agents], [1, 1])
        self.assertEqual([len(agent.history) for agent in arena.agents], [4, 4])

    def test_arena_step_delivers_reply_after_environment_step(self):
        arena = ArenaMP.__new__(ArenaMP)
        env = SimpleNamespace(steps=0)
        env.get_observations = Mock(return_value={0: {}, 1: {}})
        env.get_action_space = Mock(return_value={})

        def environment_step(actions):
            env.steps += 1
            return None, 0, False, {}, ["Found apple", None]

        env.step = Mock(side_effect=environment_step)
        arena.env = env
        arena.agents = [FakeAgent("Alice", None), FakeAgent("Bob", "Good find")]
        arena.get_actions = Mock(return_value=({0: "[send_message] <Found apple>", 1: "[walktowards] <kitchen> (3)"}, {0: {}, 1: {}}))
        arena.cnt_nouse_subgoal = 0
        arena.cnt_duplicate_subgoal = 0

        arena.step()

        self.assertEqual(env.step.call_count, 1)
        self.assertEqual(env.steps, 1)
        self.assertEqual([e["kind"] for e in arena.last_communication_events], ["active", "reply_call", "reply"])

    def test_reply_prompt_sees_complete_dialogue_and_only_partial_observation(self):
        planner = SimpleNamespace(
            agent_name="Bob", oppo_name="Alice", personality="organized",
            goal_desc="place apple", lm_id="test-model", chat=True,
            sampling_params={"max_tokens": 1024, "temperature": 0, "n": 1},
            last_usage_metadata={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            generator=Mock(return_value=(["<reply>Understood.</reply>"], 0)),
        )
        history = [f"Alice: message {i}" for i in range(5)]
        reply, call = CommunicationModule(planner).reply(
            "message 4", history, "[gocheck] <cabinet> (2)", "kitchen",
            {"nodes": [{"class_name": "apple", "id": 12, "category": "Objects"}]},
        )
        self.assertEqual(reply, "Understood.")
        self.assertIn("message 0", call["prompt"])
        self.assertIn("message 4", call["prompt"])
        self.assertIn("<apple> (12)", call["prompt"])
        self.assertEqual(planner.generator.call_args.args[1]["max_tokens"], 256)

    def test_both_agents_share_peer_prompt(self):
        args = SimpleNamespace(debug=False, act=False, big_5=None, api_base=None, api_key=None)
        alice = LLM("debug", "test", "unused", True, False, args, 1)
        bob = LLM("debug", "test", "unused", True, False, args, 2)
        self.assertIn("equal authority", alice.prompt_template)
        self.assertIn("equal authority", bob.prompt_template)
        self.assertNotIn("$AGENT_NAME", alice.prompt_template)
        self.assertNotIn("$AGENT_NAME", bob.prompt_template)

    def test_planning_prompt_contains_oldest_and_newest_messages(self):
        planner = LLM.__new__(LLM)
        planner.prompt_template = "Dialogue:\n$DIALOGUE_HISTORY\nGoal: $GOAL\n$PROGRESS\n$STEP\n$ACTION_HISTORY\n$ASSIGNED_PERSONALITY\n$ACTIONS_REFINE"
        planner.goal_desc = "place apple"
        planner.oppo_name = "Bob"
        planner.agent_name = "Alice"
        planner.personality = ""
        planner.chat = True
        planner.sampling_params = {"n": 1}
        planner.total_cost = 0
        planner.last_usage_metadata = None
        planner.debug = False
        planner.get_available_plans = Mock(return_value=("", "A. [send_message]", [], ["[send_message]"], 1))
        planner.generator = Mock(return_value=(["<action>A. [send_message]</action>"], 0))
        planner.parse_answer = Mock(return_value=("[send_message] <hello>", "", "hello"))
        planner.goal_location_with_r = "kitchen"
        planner.progress2text = Mock(return_value="none")
        history = [f"Alice: message {i}" for i in range(5)]
        planner.run(None, {}, [], [], {}, {}, None, [], history, [], None)
        prompt = planner.generator.call_args.args[0][0]["content"]
        self.assertIn("message 0", prompt)
        self.assertIn("message 4", prompt)


if __name__ == "__main__":
    unittest.main()
