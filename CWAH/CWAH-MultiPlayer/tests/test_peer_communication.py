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
        order = []
        alice = FakeAgent("Alice", None)
        original_receive = alice.receive_dialogue_event
        alice.observe_trace_outcome = lambda observation, step: order.append("post_step_observation")
        alice.receive_dialogue_event = lambda event: (order.append("message"), original_receive(event))[-1]
        arena.agents = [alice, FakeAgent("Bob", "Good find")]
        arena.get_actions = Mock(return_value=({0: "[send_message] <Found apple>", 1: "[walktowards] <kitchen> (3)"}, {0: {}, 1: {}}))
        arena.cnt_nouse_subgoal = 0
        arena.cnt_duplicate_subgoal = 0

        arena.step()

        self.assertEqual(env.step.call_count, 1)
        self.assertEqual(env.steps, 1)
        self.assertEqual(order[0], "post_step_observation")
        self.assertEqual([e["kind"] for e in arena.last_communication_events], ["active", "reply_call", "reply"])

    def test_reply_prompt_sees_complete_dialogue_and_only_partial_observation(self):
        planner = SimpleNamespace(
            agent_name="Bob", oppo_name="Alice", personality="organized",
            goal_desc="place apple", lm_id="test-model", chat=True,
            sampling_params={"max_tokens": 1024, "temperature": 0, "n": 1},
            last_usage_metadata={"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            generator=Mock(return_value=(["<intention>Continue checking the cabinet.</intention>"
                                          "<reply>Understood.</reply>"], 0)),
        )
        history = [f"Alice: message {i}" for i in range(5)]
        reply, call = CommunicationModule(planner).reply(
            "message 4", history, "[gocheck] <cabinet> (2)", "kitchen",
            {"nodes": [{"class_name": "apple", "id": 12, "category": "Objects"}]},
            progress="One object remains", action_history=["[goexplore] <kitchen> (11)"],
            belief_context="Bob likely intends to search the kitchen.",
        )
        self.assertEqual(reply, "Understood.")
        self.assertEqual(call["intention"], "Continue checking the cabinet.")
        self.assertIn("message 0", call["prompt"])
        self.assertIn("message 4", call["prompt"])
        self.assertIn("<apple> (12)", call["prompt"])
        self.assertIn("One object remains", call["prompt"])
        self.assertIn("[goexplore] <kitchen> (11)", call["prompt"])
        self.assertIn("Bob likely intends to search the kitchen.", call["prompt"])
        self.assertEqual(planner.generator.call_args.args[1]["max_tokens"], 256)

    def test_reply_uses_fresh_own_holding_not_peer_or_stale_room(self):
        planner = SimpleNamespace(
            agent_name="Alice", oppo_name="Bob", personality="", goal_desc="place apple",
            lm_id="test-model", chat=True, sampling_params={"max_tokens": 256, "n": 1},
            last_usage_metadata=None,
            generator=Mock(return_value=(["<intention>Place the apple.</intention>"
                                          "<reply>I have it and will place it.</reply>"], 0)),
        )
        observation = {
            "nodes": [
                {"id": 1, "class_name": "character", "category": "Characters"},
                {"id": 2, "class_name": "character", "category": "Characters"},
                {"id": 10, "class_name": "livingroom", "category": "Rooms"},
                {"id": 375, "class_name": "apple", "category": "Objects"},
                {"id": 372, "class_name": "pudding", "category": "Objects"},
            ],
            "edges": [
                {"from_id": 1, "to_id": 10, "relation_type": "INSIDE"},
                {"from_id": 1, "to_id": 375, "relation_type": "HOLDS_RH"},
                {"from_id": 2, "to_id": 372, "relation_type": "HOLDS_LH"},
            ],
            "hidden_profile": "PRIVATE_PROFILE_NEVER_ENTER_PROMPT",
        }
        _, call = CommunicationModule(planner).reply(
            "Please place the apple.", [], "[goput] <kitchentable> (72)",
            "kitchen", observation, agent_id=1,
        )
        self.assertIn("Current room: livingroom", call["prompt"])
        self.assertIn("Currently held objects: <apple> (375)", call["prompt"])
        self.assertNotIn("Currently held objects: <pudding> (372)", call["prompt"])
        self.assertIn("[goput] <kitchentable> (72)", call["prompt"])
        self.assertNotIn("PRIVATE_PROFILE_NEVER_ENTER_PROMPT", call["prompt"])

    def test_reactive_planner_cannot_choose_no_reply(self):
        planner = SimpleNamespace(
            agent_name="Alice", oppo_name="Bob", personality="", goal_desc="place apple",
            lm_id="test-model", chat=True, sampling_params={"max_tokens": 256, "n": 1},
            last_usage_metadata=None, generator=Mock(return_value=(["<no_reply/>"], 0)),
        )
        with self.assertRaisesRegex(ValueError, "non-empty <intention> and <reply>"):
            CommunicationModule(planner).reply(
                "Any update?", [], None, None, {"nodes": [], "edges": []}, agent_id=1,
            )

    def test_role_prompts_and_reply_prompts_are_selected(self):
        args = SimpleNamespace(debug=False, act=False, big_5=None, api_base=None, api_key=None)
        alice = LLM("debug", "test", "unused", True, False, args, 1)
        bob = LLM("debug", "test", "unused", True, False, args, 2)
        self.assertIn("equal authority", alice.prompt_template)
        self.assertIn("equal authority", bob.prompt_template)
        self.assertIn("reasonable consistency", alice.prompt_template)
        self.assertNotIn("$AGENT_NAME", alice.prompt_template)
        self.assertNotIn("$AGENT_NAME", bob.prompt_template)
        self.assertTrue(alice.prompt_template_path.endswith('prompt_Alice.txt'))
        self.assertTrue(bob.prompt_template_path.endswith('prompt_Bob.txt'))
        self.assertNotIn('$ASSIGNED_PERSONALITY', alice.prompt_template)
        self.assertIn('$ASSIGNED_PERSONALITY', bob.prompt_template)
        alice_reply = CommunicationModule(alice)
        bob_reply = CommunicationModule(bob)
        self.assertEqual(alice_reply.prompt_path.name, 'prompt_Reply_Alice.txt')
        self.assertEqual(bob_reply.prompt_path.name, 'prompt_Reply_Bob.txt')
        self.assertNotIn('$ASSIGNED_PERSONALITY', alice_reply.prompt_template)
        self.assertIn('$ASSIGNED_PERSONALITY', bob_reply.prompt_template)

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
