"""Config and integration checks that never start Unity or call a model."""

import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiment_runner"))
import run_experiment  # noqa: E402
import run_many  # noqa: E402
from run_experiment import validate  # noqa: E402

from LLM.LLM import LLM  # noqa: E402
from agents.LLM_agent import LLM_agent  # noqa: E402
from envs.unity_environment import UnityEnvironment  # noqa: E402
from trace_tf.events import EvidenceLedger  # noqa: E402
from trace_tf.model import BeliefState  # noqa: E402


class ExperimentRunnerTests(unittest.TestCase):
    def test_exact_object_id_is_not_corrupted_by_target_decoration(self):
        env = UnityEnvironment.__new__(UnityEnvironment)
        env.id2node = {72: {"id": 72, "class_name": "kitchentable"}}
        env.goal_class = {}

        goal = env.get_goal({"on_372_72": 1}, "LLM")

        self.assertEqual(goal, {
            "on_372_<kitchentable> (72)": [1, True, 2]
        })

    def test_wah_episode_and_profile_are_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            dataset = folder / "wah.pik"
            with dataset.open("wb") as stream:
                pickle.dump([{"task_id": 44, "task_name": "prepare_food", "env_id": 3,
                              "init_graph": {"nodes": [], "edges": []}, "task_goal": {}}], stream)
            executable = folder / "unity"
            executable.write_text("simulator placeholder", encoding="utf-8")
            executable.chmod(0o755)
            profile = folder / "peer.txt"
            profile.write_text("Bob acts independently.\n", encoding="utf-8")
            config = {"dataset": {"path": str(dataset), "episode_index": 0},
                      "executable_file": str(executable),
                      "bob_profile": {"id": "independent", "text_file": str(profile)},
                      "trace_mode": "on", "alice_model": "alice", "bob_model": "bob",
                      "temperature": 0, "max_tokens": 256, "max_episode_length": 250,
                      "num_runs": 1}
            selected = validate(config)
            self.assertEqual(selected["episode"]["task_id"], 44)
            self.assertEqual(selected["profile_path"], profile)
            self.assertEqual(selected["port_request"], "auto")
            self.assertEqual(selected["prompt_files"]["alice"].name, "prompt_Peer.txt")
            self.assertEqual(selected["prompt_files"]["bob"].name, "prompt_Peer.txt")
            with patch.dict(os.environ, {"CWAH_PROFILE_SOURCE_TASK": "26_prepare_food",
                                      "CWAH_PROFILE_EXPECTED_TASK": "50_prepare_food"}):
                args = SimpleNamespace(debug=False, act=False, big_5=None,
                                       teammate_profile_file=str(profile), api_base=None, api_key=None)
                bob = LLM("debug", "test-model", "unused", True, False, args, 2)
                alice = LLM("debug", "test-model", "unused", True, False, args, 1)
                for agent in (bob, alice):
                    with patch.object(agent, "goal2description", return_value=("goal", "table")):
                        agent.reset({"nodes": []}, [], {}, "<table> (1)", {}, "44_prepare_food")
                self.assertEqual(bob.personality, "Bob acts independently.")
                self.assertEqual(alice.personality, "")

    def test_role_specific_original_human_prompt(self):
        peer = run_experiment.GAME_ROOT / "LLM" / "prompt_Peer.txt"
        human = run_experiment.GAME_ROOT / "LLM" / "prompt_Human.txt"
        args = SimpleNamespace(debug=False, act=False, big_5=None, api_base=None, api_key=None,
                               alice_prompt_file=str(peer), bob_prompt_file=str(human))
        alice = LLM("debug", "test", "unused", True, False, args, 1)
        bob = LLM("debug", "test", "unused", True, False, args, 2)
        self.assertEqual(alice.prompt_template_path, str(peer))
        self.assertEqual(bob.prompt_template_path, str(human))
        self.assertIn("equal authority", alice.prompt_template)
        self.assertIn("You must send a message", bob.prompt_template)
        self.assertNotIn("$AGENT_NAME", bob.prompt_template)

    def test_official_configs_use_unified_peer_prompt(self):
        config_dir = Path(__file__).resolve().parents[2] / "experiment_runner" / "configs"
        configs = [json.loads(path.read_text()) for path in sorted(config_dir.glob("*.json"))]
        self.assertTrue(configs)
        for config in configs:
            selected = validate(config)
            self.assertEqual(selected["prompt_files"]["alice"].name, "prompt_Peer.txt")
            self.assertEqual(selected["prompt_files"]["bob"].name, "prompt_Peer.txt")

    def test_trace_switch_controls_updater(self):
        room = {"id": 11, "class_name": "kitchen", "category": "Rooms"}
        table = {"id": 12, "class_name": "table", "category": "Objects"}
        obs = {"nodes": [room, table], "edges": [{"from_id": 1, "relation_type": "INSIDE", "to_id": 11}]}
        for mode in ("off", "on"):
            with self.subTest(mode=mode):
                args = SimpleNamespace(debug=False, act=False, big_5=None, teammate_profile_file=None,
                                       trace_mode=mode, obs_type="partial", source="debug", lm_id="test-model",
                                       prompt_template_path="unused", communication=True, cot=False,
                                       api_base=None, api_key=None)
                agent = LLM_agent(1, 0, args)
                with patch.object(agent.LLM, "reset"):
                    agent.reset(obs, [], [], ["kitchen"], [room],
                                {"on_apple_<table> (12)": [1]}, "44_prepare_food")
                self.assertEqual(agent.trace is not None, mode == "on")

    def test_trace_updates_before_reactive_planner(self):
        order = []
        state = BeliefState(2)
        ledger = EvidenceLedger()
        trace = SimpleNamespace(
            state=state,
            ledger=ledger,
            cursor=0,
            update=Mock(side_effect=lambda *_args, **_kwargs: (order.append("trace") or (state, {
                "new_event_ids": ["e00000"]
            }))),
        )
        communication = SimpleNamespace(
            build_decision_information=Mock(return_value=SimpleNamespace(
                to_dict=lambda: {"planner_type": "reactive", "goal": "place apple"}
            )),
            reply=Mock(side_effect=lambda *args, **kwargs: (
                order.append("reactive") or ("I will continue.", {
                    "intention": "Continue the current task."
                })
            ))
        )
        agent = LLM_agent.__new__(LLM_agent)
        agent.current_room = {"class_name": "kitchen"}
        agent.trace = trace
        agent.satisfied = []
        agent.unsatisfied = {"on_apple_table": 1}
        agent.last_action = "[walktowards] <apple> (12)"
        agent.plan = "[gograb] <apple> (12)"
        agent.action_history = [agent.plan]
        agent.dialogue_history = ["Bob: I found the apple."]
        agent.communication_module = communication
        agent.agent_id = 1
        agent.steps = 3

        reply, call = agent.reply_to_message(
            "I found the apple.", {"nodes": [], "edges": []}
        )

        self.assertEqual(order, ["trace", "reactive"])
        self.assertEqual(reply, "I will continue.")
        self.assertTrue(call["trace_enabled"])
        self.assertTrue(call["trace_updated"])
        self.assertEqual(call["trace_new_event_ids"], ["e00000"])
        self.assertEqual(agent.plan, "[gograb] <apple> (12)")

    def test_continuous_goal_refresh_does_not_reset_plan_history_or_trace(self):
        trace = object()
        agent = LLM_agent.__new__(LLM_agent)
        agent.goal_location = "<kitchentable> (72)"
        agent.goal_location_id = 72
        agent.id_inside_room = {72: "kitchen"}
        agent.rooms_name = ["kitchen", "livingroom"]
        agent.goal_objects_name = ["apple", "kitchentable"]
        agent.unsatisfied = {"on_apple_<kitchentable> (72)": 1}
        agent.plan = "[gograb] <apple> (10)"
        agent.action_history = [agent.plan]
        agent.dialogue_history = ["Bob: I found the apple."]
        agent.trace = trace
        agent.LLM = SimpleNamespace(refresh_goal=Mock())

        agent.update_task_goal({
            "on_pudding_<kitchentable> (72)": [1, True, 2],
            "on_juice_<kitchentable> (72)": [1, True, 2],
        })

        self.assertEqual(agent.plan, "[gograb] <apple> (10)")
        self.assertEqual(agent.action_history, [agent.plan])
        self.assertEqual(agent.dialogue_history, ["Bob: I found the apple."])
        self.assertIs(agent.trace, trace)
        self.assertIn("pudding", agent.goal_objects_name)
        agent.LLM.refresh_goal.assert_called_once()

    def test_completed_subtask_is_retained_separately_from_active_tasks(self):
        agent = LLM_agent.__new__(LLM_agent)
        agent.completed_goals = []
        agent.trace = None
        completed = {
            "subtask_id": "s1", "object_id": 10, "object_class": "apple",
            "relation": "on", "target_id": 72, "target_class": "kitchentable",
            "status": "completed", "completed_step": 4,
        }

        agent.record_task_completions([completed], {"s1": "Bob"}, step=4)
        agent.record_task_completions([completed], {"s1": "Bob"}, step=4)

        self.assertEqual(len(agent.completed_goals), 1)
        self.assertEqual(agent.completed_goals[0]["completed_by"], "Bob")

    def test_planner_marks_completed_subtasks_as_history_not_active_goal(self):
        planner = LLM.__new__(LLM)
        planner.id2node = {
            10: {"id": 10, "class_name": "apple"},
            72: {"id": 72, "class_name": "kitchentable"},
            103: {"id": 103, "class_name": "fridge"},
        }
        completed = [{
            "subtask_id": "s1", "object_id": 10, "object_class": "apple",
            "relation": "on", "target_id": 72, "target_class": "kitchentable",
            "completed_by": "Bob",
        }]

        planner.refresh_goal(
            "<fridge> (103)", {"inside_10_103": 1},
            [{"relation": "inside", "object_id": 10, "object_class": "apple",
              "target_id": 103, "target_class": "fridge"}], completed,
        )

        self.assertIn("put <apple> (10) into <fridge> (103)", planner.goal_desc)
        self.assertIn("[s1]", planner.goal_desc)
        self.assertIn("already satisfied; do not repeat", planner.goal_desc)
        self.assertIn("<apple> (10) on <kitchentable> (72) by Bob", planner.goal_desc)

    def test_completion_attribution_uses_only_alice_action_and_public_completion(self):
        from algos.arena_mp2 import ArenaMP

        task = {
            "subtask_id": "s1", "object_id": 10, "relation": "on",
            "target_id": 72,
        }
        alice = ArenaMP._attribute_completed_subtasks(
            [task], {0: "[putback] <apple> (10) <kitchentable> (72)",
                     1: "[walktowards] <bedroom> (210)"},
        )
        bob = ArenaMP._attribute_completed_subtasks(
            [task], {0: "[walktowards] <bedroom> (210)",
                     1: "PRIVATE_BOB_ACTION_IS_NOT_NEEDED"},
        )

        self.assertEqual(alice["s1"], "Alice")
        self.assertEqual(bob["s1"], "Bob")

    def test_continuous_exact_inside_goal_produces_putin(self):
        agent = LLM_agent.__new__(LLM_agent)
        agent.rooms_name = ["kitchen", "livingroom"]
        agent.roomname2id = {"kitchen": 11, "livingroom": 267}
        agent.current_room = {"id": 11, "class_name": "kitchen"}
        agent.grabbed_objects = [10]
        agent.id2node = {
            10: {"id": 10, "class_name": "apple", "properties": ["GRABBABLE"], "states": []},
            103: {"id": 103, "class_name": "fridge", "properties": ["CONTAINERS"], "states": ["OPEN"]},
        }
        agent.id_inside_room = {103: "kitchen"}
        agent.reachable_objects = ["<fridge> (103)"]
        agent.active_goals = [{
            "predicate": "inside_10_103", "relation": "inside",
            "object_token": "10", "target_id": 103, "count": 1,
            "object_class": "apple", "target_class": "fridge",
        }]
        agent.plan = "[goput] <apple> (10) <fridge> (103)"

        self.assertEqual(agent._put_options(), [agent.plan])
        self.assertEqual(
            agent.goput(), "[putin] <apple> (10) <fridge> (103)"
        )

    def test_continuous_goal_refresh_prunes_obsolete_object_candidates(self):
        agent = LLM_agent.__new__(LLM_agent)
        agent.goal_location = "<kitchentable> (72)"
        agent.goal_location_id = 72
        agent.id_inside_room = {72: "kitchen", 368: "kitchen", 369: "bedroom"}
        agent.rooms_name = ["kitchen", "bedroom"]
        agent.goal_objects_name = ["poundcake", "cupcake", "kitchentable"]
        agent.id2node = {
            72: {"id": 72, "class_name": "kitchentable"},
            368: {"id": 368, "class_name": "cupcake"},
            369: {"id": 369, "class_name": "poundcake"},
        }
        agent.ungrabbed_objects = {
            "kitchen": [agent.id2node[368], agent.id2node[369]],
            "bedroom": [agent.id2node[369]],
        }
        agent.plan = None
        agent.action_history = []
        agent.dialogue_history = []
        agent.trace = None
        agent.LLM = SimpleNamespace(refresh_goal=Mock())

        agent.update_task_goal(
            {"on_368_<kitchentable> (72)": [1, True, 2]},
            [{"relation": "on", "object_id": 368, "object_class": "cupcake",
              "target_id": 72, "target_class": "kitchentable"}],
        )

        self.assertEqual([item["id"] for item in agent.ungrabbed_objects["kitchen"]], [368])
        self.assertEqual(agent.ungrabbed_objects["bedroom"], [])
        self.assertEqual(agent.goal_objects_name, ["cupcake", "kitchentable"])

    def test_invalid_grab_forgets_stale_locations_before_replanning(self):
        agent = LLM_agent.__new__(LLM_agent)
        stale = {"id": 369, "class_name": "poundcake"}
        agent.plan = "[gograb] <poundcake> (369)"
        agent.debug = False
        agent.grabbed_objects = []
        agent.opponent_grabbed_objects = []
        agent.current_room = {"class_name": "bedroom"}
        agent.id_inside_room = {369: "bedroom"}
        agent.id2node = {369: stale}
        agent.ungrabbed_objects = {"kitchen": [stale], "bedroom": []}
        agent.reachable_objects = []

        self.assertIsNone(agent.gograb())
        self.assertIsNone(agent.plan)
        self.assertEqual(agent.ungrabbed_objects, {"kitchen": [], "bedroom": []})

    def test_port_lock_blocks_another_local_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(run_experiment, "LOCK_ROOT", Path(temporary)), \
                    patch.object(run_experiment, "AUTO_PORTS", range(6390, 6392)), \
                    patch.object(run_experiment.socket, "socket"):
                with run_experiment.allocated_port("auto") as first:
                    with run_experiment.allocated_port("auto") as second:
                        self.assertEqual((first, second), (6390, 6391))
                    with self.assertRaisesRegex(ValueError, "reserved"):
                        with run_experiment.allocated_port(6390):
                            pass

    def test_batch_preflight_rejects_duplicate_ports(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "first.json").write_text('{"port": 6390}', encoding="utf-8")
            (folder / "second.json").write_text('{"port": 6390}', encoding="utf-8")
            (folder / "third.json.example").write_text('{"port": 6390}', encoding="utf-8")
            selected = {"episode": {"task_id": 44, "task_name": "prepare_food"},
                        "episode_index": 44, "profile_id": "peer17", "trace_mode": "off"}
            with patch.object(run_many, "validate", side_effect=lambda config: {
                    **selected, "port_request": config.get("port", "auto")}):
                with self.assertRaisesRegex(ValueError, "Duplicate port"):
                    run_many.preflight(folder)
                (folder / "second.json").write_text('{"port": 6391}', encoding="utf-8")
                self.assertEqual(run_many.main(["--config-dir", str(folder), "--check"]), 0)
                (folder / "first.json").write_text('{"port": "auto"}', encoding="utf-8")
                (folder / "second.json").write_text('{"port": "auto"}', encoding="utf-8")
                self.assertEqual(len(run_many.preflight(folder)), 2)

    def test_batch_runner_streams_to_terminal_without_batch_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            configs = folder / "configs"
            configs.mkdir()
            (configs / "one.json").write_text("{}", encoding="utf-8")
            selected = {
                "episode": {"task_id": 44, "task_name": "prepare_food"},
                "episode_index": 44, "profile_id": "peer17",
                "trace_mode": "off", "port_request": "auto",
            }
            process = SimpleNamespace(pid=1234, poll=Mock(return_value=0))
            with patch.object(run_many, "RUNNER_DIR", folder), \
                    patch.object(run_many, "validate", return_value=selected), \
                    patch.object(run_many.subprocess, "Popen", return_value=process) as popen:
                self.assertEqual(run_many.main([
                    "--config-dir", str(configs), "--max-parallel", "1"
                ]), 0)
            self.assertFalse((folder / "batch_logs").exists())
            self.assertNotIn("stdout", popen.call_args.kwargs)
            self.assertNotIn("stderr", popen.call_args.kwargs)

if __name__ == "__main__":
    unittest.main()
