"""Public task truth, macro transitions and pre-execution communication barriers."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agents.LLM_agent import LLM_agent
from algos.arena_mp2 import ArenaMP
from envs.continuous_subtasks import ContinuousSubtaskManager
from LLM.communication import CommunicationModule, ReplyFormatError
from LLM.LLM import LLM
from trace_tf.information import DecisionInformation


def tasks():
    return {
        'S3': dict(subtask_id='S3', object_id=375, object_class='apple', target_id=72,
                   target_class='kitchentable', relation='on', status='active', created_step=0,
                   completed_step=None, completed_by=None),
        'S4': dict(subtask_id='S4', object_id=370, object_class='pudding', target_id=72,
                   target_class='kitchentable', relation='on', status='active', created_step=0,
                   completed_step=None, completed_by=None),
    }


def agent():
    a = LLM_agent.__new__(LLM_agent)
    a.rooms_name = ['kitchen', 'bedroom']
    a.id_inside_room = {}
    a.id2node = {}
    a.trace = None
    a.LLM = SimpleNamespace(refresh_goal=Mock())
    a.set_public_tasks(tasks())
    return a


class TaskControlTests(unittest.TestCase):
    def test_real_agents_two_steps_reactive_switch_then_new_skill(self):
        """Exercise real preprocessing, prompt render, parsing and execution."""
        def node(i, name, category, properties=()):
            return {'id': i, 'class_name': name, 'category': category,
                    'properties': list(properties), 'states': []}
        graph = {'nodes': [node(1, 'character', 'Characters'), node(2, 'character', 'Characters'),
                          node(11, 'kitchen', 'Rooms'), node(72, 'kitchentable', 'Furniture', ['SURFACES']),
                          node(375, 'apple', 'Objects', ['GRABBABLE']),
                          node(370, 'pudding', 'Objects', ['GRABBABLE'])],
                 'edges': [{'from_id': i, 'to_id': 11, 'relation_type': 'INSIDE'}
                           for i in (1, 2, 72, 375, 370)] +
                          [{'from_id': i, 'to_id': j, 'relation_type': 'CLOSE'}
                           for i in (1, 2) for j in (72, 375, 370)]}
        manager = ContinuousSubtaskManager({'on_apple_72': 1, 'on_pudding_72': 1}, active_count=2,
            targets=[{'relation': 'on', 'object_id': 72, 'object_class': 'kitchentable', 'capacity': 6}])
        manager.initialize(graph)
        ids = {t['object_class']: t['subtask_id'] for t in manager.active_subtasks()}
        goal = {'on_375_<kitchentable> (72)': [1], 'on_370_<kitchentable> (72)': [1]}
        args = SimpleNamespace(debug=False, act=False, big_5=None, teammate_profile_file=None,
            trace_mode='off', obs_type='partial', source='debug', lm_id='test',
            prompt_template_path='unused', communication=True, cot=False, api_base=None, api_key=None)
        agents = [LLM_agent(i, i - 1, args) for i in (1, 2)]
        observation = lambda: dict(copy.deepcopy(graph), location=None, messages=[None, None])
        for a in agents:
            a.reset(observation(), [], ['apple', 'pudding'], ['kitchen'],
                    [graph['nodes'][2]], goal, 'test')
            a.set_public_tasks(manager.public_snapshot())
            a.LLM.chat = True
            a.LLM.sampling_params = {'max_tokens': 256, 'temperature': 0, 'n': 1}
        def active(task, skill, message=''):
            return ['<cot>test</cot><subtask_id>{}</subtask_id><message>{}</message><action>{}</action>'
                    .format(task, message, skill)], 0
        agents[0].LLM.generator = Mock(side_effect=[
            active(ids['apple'], '[gograb] <apple> (375)'),
            (['<intention>Switch after finishing.</intention><new_subtask_id>{}</new_subtask_id>'
              '<reply>I will handle the pudding after this skill.</reply>'.format(ids['pudding'])], 0),
            active(ids['pudding'], '[gograb] <pudding> (370)')])
        agents[1].LLM.generator = Mock(side_effect=[
            active(ids['apple'], '[send_message]', 'Please handle the pudding.'),
            active(ids['apple'], '[send_message]', '')])
        arena = ArenaMP.__new__(ArenaMP)
        arena.agents = agents
        arena.run_predefined_actions = False
        arena.cnt_nouse_subgoal = arena.cnt_duplicate_subgoal = 0
        env = SimpleNamespace(steps=0, continuous_subtasks_enabled=True, continuous_subtasks=manager,
            task_goal={0: goal, 1: goal}, agent_goals=['full', 'full'],
            get_goal=lambda *args: goal, get_action_space=lambda: {},
            get_observations=lambda: {0: observation(), 1: observation()})
        def execute(actions):
            if env.steps == 0:
                self.assertEqual(agents[0].control_context()['current_subtask_id'], ids['pudding'])
                self.assertEqual(agents[0].control_context()['skill_subtask_id'], ids['apple'])
            for index, action in actions.items():
                if action and action.startswith('[grab]'):
                    object_id = 375 if '(375)' in action else 370
                    graph['edges'].append({'from_id': index + 1, 'to_id': object_id, 'relation_type': 'HOLDS_RH'})
            env.steps += 1
            event = manager.advance(graph, env.steps, actions)
            return None, 0, False, {'continuous_subtasks': event}, [None, None]
        env.step = execute
        arena.env = env
        _, first, _ = arena.step()
        _, second, _ = arena.step()
        self.assertEqual(first[0], '[grab] <apple> (375)')
        self.assertEqual(second[0], '[grab] <pudding> (370)')
        self.assertEqual(agents[0].control_context()['skill_subtask_id'], ids['pudding'])
        self.assertEqual(manager.completed_count, 0)

    def test_reactive_switch_keeps_old_skill_and_its_origin(self):
        a = agent()
        a.accept_active_choice('S3', '[gograb] <apple> (375)')
        self.assertTrue(a.apply_reactive_target('S4'))
        self.assertEqual(a.control_context(), {
            'current_subtask_id': 'S4', 'current_skill': '[gograb] <apple> (375)',
            'skill_subtask_id': 'S3'})
        a.plan = None
        self.assertEqual(a.control_context()['current_subtask_id'], 'S4')
        self.assertIsNone(a.control_context()['skill_subtask_id'])
        a.accept_active_choice('S4', '[gograb] <pudding> (370)')
        self.assertEqual(a.control_context()['skill_subtask_id'], 'S4')

    def test_bob_reply_switches_macro_without_changing_running_s2_put(self):
        bob = agent()
        bob.agent_id = 2
        bob.set_public_tasks({
            'S2': dict(tasks()['S3'], subtask_id='S2'),
            'S4': tasks()['S4'],
        })
        running = '[goput] <apple> (375) <kitchentable> (72)'
        bob.accept_active_choice('S2', running)
        bob.steps = 1
        bob.current_room = {'class_name': 'kitchen'}
        bob.last_action = '[putback] <apple> (375) <kitchentable> (72)'
        bob.action_history, bob.dialogue_history = [], []
        bob._observe_peer = Mock()
        bob._peer_context = Mock(return_value={})
        bob.LLM = SimpleNamespace(
            agent_name='Bob', oppo_name='Alice', personality='helpful',
            goal_desc='place apple and pudding', lm_id='test', chat=True,
            sampling_params={'max_tokens': 256, 'n': 1},
            last_usage_metadata=None, generator=Mock(return_value=([
                '<intention>Finish S2, then work on S4.</intention>'
                '<new_subtask_id>S4</new_subtask_id>'
                '<reply>I accept S4 after finishing my current placement.</reply>'
            ], 0)),
        )
        bob.communication_module = CommunicationModule(bob.LLM)

        reply, call = bob.reply_to_message(
            'Please complete S4.', {'nodes': [], 'edges': []})
        self.assertIn('accept S4', reply)
        self.assertTrue(call['macro_changed'])
        self.assertEqual(bob.control_context(), {
            'current_subtask_id': 'S4', 'current_skill': running,
            'skill_subtask_id': 'S2'})
        self.assertEqual(bob.last_action,
                         '[putback] <apple> (375) <kitchentable> (72)')
        bob.plan = None  # The S2 skill finishes before another Active decision.
        self.assertEqual(bob.control_context(), {
            'current_subtask_id': 'S4', 'current_skill': None,
            'skill_subtask_id': None})

    def test_invalid_or_completed_target_changes_nothing(self):
        a = agent()
        a.accept_active_choice('S3', '[gograb] <apple> (375)')
        before = a.control_context()
        with self.assertRaises(ValueError):
            a.apply_reactive_target('S999')
        self.assertEqual(a.control_context(), before)
        a.public_tasks['S4']['status'] = 'completed'
        with self.assertRaises(ValueError):
            a.apply_reactive_target('S4')
        self.assertFalse(a.apply_reactive_target(None))
        self.assertEqual(a.control_context(), before)

    def test_public_snapshot_is_detached_and_views_are_derived(self):
        a = agent()
        snapshot = tasks()
        a.set_public_tasks(snapshot)
        snapshot['S3']['status'] = 'completed'
        self.assertEqual(len(a.active_goals), 2)
        a.set_public_tasks(snapshot)
        self.assertEqual(len(a.active_goals), 1)
        self.assertEqual(a.completed_goals[0]['subtask_id'], 'S3')
        self.assertNotIn('active_goals', a.__dict__)
        self.assertNotIn('completed_goals', a.__dict__)
        self.assertNotIn('satisfied', a.__dict__)
        self.assertNotIn('unsatisfied', a.__dict__)

    def test_active_skill_must_match_object_and_destination(self):
        a = agent()
        for task, skill in [('S3', '[gograb] <pudding> (370)'),
                            ('S4', '[goput] <pudding> (370) <desk> (213)'),
                            (None, '[gograb] <apple> (375)')]:
            with self.assertRaises(ValueError):
                a.accept_active_choice(task, skill)
        a.accept_active_choice('S3', '[goput] <apple> (375) <kitchentable> (72)')

    def test_completed_macro_is_retired_at_next_skill_boundary(self):
        a = agent()
        a.accept_active_choice('S3', '[gograb] <apple> (375)')
        snapshot = tasks()
        snapshot['S3'].update(status='completed', completed_step=3, completed_by='Bob')
        a.set_public_tasks(snapshot)
        self.assertIsNotNone(a.plan)
        a.plan = None
        a.set_public_tasks(snapshot)
        self.assertIsNone(a.control_context()['current_subtask_id'])

    def test_manager_owns_attribution_and_history_survives_later_move(self):
        graph = {'nodes': [
            {'id': 375, 'class_name': 'apple', 'properties': ['GRABBABLE'], 'states': []},
            {'id': 72, 'class_name': 'kitchentable', 'properties': ['SURFACES'], 'states': []},
            {'id': 213, 'class_name': 'desk', 'properties': ['SURFACES'], 'states': []}], 'edges': []}
        manager = ContinuousSubtaskManager({'on_apple_72': 1}, targets=[
            {'relation': 'on', 'object_id': 72, 'object_class': 'kitchentable', 'capacity': 4},
            {'relation': 'on', 'object_id': 213, 'object_class': 'desk', 'capacity': 4}])
        manager.initialize(graph)
        graph['edges'] = [{'from_id': 375, 'to_id': 72, 'relation_type': 'ON'}]
        event = manager.advance(graph, 1, {1: '[putback] <apple> (375) <kitchentable> (72)'})
        self.assertEqual(event['completed'][0]['completed_by'], 'Bob')
        original = manager.public_snapshot()['S1']
        self.assertNotIn('source_target_id', original)
        graph['edges'] = [{'from_id': 375, 'to_id': 213, 'relation_type': 'ON'}]
        manager.advance(graph, 2, {})
        self.assertEqual(manager.public_snapshot()['S1'], original)
        detached = manager.public_snapshot()
        detached['S1']['completed_by'] = 'Alice'
        self.assertEqual(manager.public_snapshot()['S1']['completed_by'], 'Bob')

    def test_unknown_attribution_is_not_guessed_as_bob(self):
        record = tasks()['S3']
        self.assertEqual(ContinuousSubtaskManager._completion_actor(record, {}), 'unknown')

    def test_manager_ambiguous_placement_is_unknown(self):
        record = tasks()['S3']
        action = '[putback] <apple> (375) <kitchentable> (72)'
        self.assertEqual(ContinuousSubtaskManager._completion_actor(record, {0: action, 1: action}), 'unknown')

    def test_completed_public_tasks_do_not_disappear_from_snapshot(self):
        a = agent()
        snapshot = tasks()
        snapshot['S3'].update(status='completed', completed_step=8, completed_by='Bob')
        a.set_public_tasks(snapshot)
        a.record_task_completions([snapshot['S3']], step=8)
        a.record_task_completions([snapshot['S3']], step=8)
        self.assertEqual(a.public_tasks['S3'], snapshot['S3'])
        self.assertEqual(len(a.completed_goals), 1)
        self.assertEqual(len(a._reported_completions), 1)

    def test_both_active_decisions_precede_message_delivery(self):
        order = []
        arena = ArenaMP.__new__(ArenaMP)
        arena.run_predefined_actions = False
        arena.task_goal = None
        def fake(index):
            def choose(obs, goal):
                self.assertEqual(order, ['active0'] if index == 1 else [])
                order.append('active' + str(index))
                return '[send_message] <hello>', {}
            return SimpleNamespace(agent_type='LLM', update_task_goal=Mock(), get_action=choose,
                receive_dialogue_event=lambda event: order.append('message'),
                reply_to_message=lambda *args: ('okay', {}))
        arena.agents = [fake(0), fake(1)]
        env = SimpleNamespace(steps=0, continuous_subtasks_enabled=False,
            task_goal={0: {}, 1: {}}, agent_goals=['full', 'full'],
            get_goal=lambda *args: {}, get_action_space=lambda: {},
            get_observations=lambda: {0: {}, 1: {}})
        def execute(actions):
            order.append('execute')
            env.steps += 1
            return None, 0, False, {}, ['hello', 'hello']
        env.step = execute
        arena.env = env
        arena.step()
        self.assertEqual(order[:2], ['active0', 'active1'])
        self.assertEqual(order[-1], 'execute')

    def test_reactive_reads_ot_before_frozen_physical_action(self):
        a = agent()
        a.accept_active_choice('S3', '[gograb] <apple> (375)')
        a.receive_dialogue_event = Mock()
        b = SimpleNamespace(receive_dialogue_event=Mock())
        observation = {0: {'stamp': 'O_t'}, 1: {'stamp': 'B_t'}}
        order = []
        def reply(message, obs):
            order.append('reactive')
            self.assertEqual(obs, {'stamp': 'O_t'})
            a.apply_reactive_target('S4')
            return 'I will handle S4 after my current skill.', {}
        a.reply_to_message = reply
        arena = ArenaMP.__new__(ArenaMP)
        arena.agents = [a, b]
        env = SimpleNamespace(steps=7, get_observations=Mock(return_value=observation),
                              get_action_space=Mock(return_value={}))
        chosen = {0: '[walktowards] <apple> (375)', 1: '[send_message] <Please handle S4>'}
        arena.get_actions = Mock(return_value=(chosen, {0: {}, 1: {}}))
        def execute(actions):
            order.append('physical')
            self.assertEqual(actions, chosen)
            self.assertEqual(a.control_context()['current_subtask_id'], 'S4')
            self.assertEqual(a.plan, '[gograb] <apple> (375)')
            env.steps += 1
            return None, 0, False, {}, [None, 'Please handle S4']
        env.step = execute
        arena.env = env
        _, actual, info = arena.step()
        self.assertEqual(order, ['reactive', 'physical'])
        self.assertEqual(env.get_observations.call_count, 1)
        self.assertEqual(actual, chosen)
        self.assertEqual(info[0]['agent_control_after_reactive']['skill_subtask_id'], 'S3')

    def test_reactive_requires_new_target_field_when_public_tasks_exist(self):
        planner = SimpleNamespace(agent_name='Alice', oppo_name='Bob', personality='',
            goal_desc='test', lm_id='test', chat=True, sampling_params={'max_tokens': 256},
            last_usage_metadata=None, generator=Mock(return_value=([
                '<intention>Continue.</intention><reply>Okay.</reply>'], 0)))
        info = DecisionInformation('reactive', 'test', '', '', public_tasks=tasks())
        with self.assertRaises(ReplyFormatError):
            CommunicationModule(planner).reply('test', [], None, None, {}, decision_information=info)
        planner.generator.return_value = ([
            '<intention>Switch later.</intention><new_subtask_id>S4</new_subtask_id><reply>Okay.</reply>'], 0)
        reply, call = CommunicationModule(planner).reply('test', [], None, None, {}, decision_information=info)
        self.assertEqual(call['new_subtask_id'], 'S4')


if __name__ == '__main__':
    unittest.main()
