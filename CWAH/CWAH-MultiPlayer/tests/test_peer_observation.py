"""Current partial views cannot be synthesized from historical peer memory."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agents.LLM_agent import LLM_agent
from LLM.LLM import LLM
from LLM.communication import CommunicationModule
from trace_tf.peer_observation import PeerStateTracker, observe_peer
from trace_tf.model import BeliefState
from trace_tf.events import EvidenceLedger


def graph(visible=True, holds=False, room='kitchen'):
    nodes = [{'id': 1, 'class_name': 'character', 'category': 'Characters'},
             {'id': 11, 'class_name': room, 'category': 'Rooms'},
             {'id': 12, 'class_name': 'apple', 'category': 'Objects', 'properties': ['GRABBABLE']}]
    edges = [{'from_id': 1, 'to_id': 11, 'relation_type': 'INSIDE'}]
    if visible:
        nodes.append({'id': 2, 'class_name': 'character', 'category': 'Characters',
                      'obj_transform': {'position': [1, 0, 1]}})
        edges.append({'from_id': 2, 'to_id': 11, 'relation_type': 'INSIDE'})
        if holds:
            edges.append({'from_id': 2, 'to_id': 12, 'relation_type': 'HOLDS_RH'})
    return {'nodes': nodes, 'edges': edges, 'hidden_profile': 'SECRET', 'peer_private_plan': 'SECRET_PLAN'}


def active_information(snapshot):
    planner = LLM.__new__(LLM)
    planner.single = False
    planner.oppo_name, planner.oppo_pronoun = 'Bob', 'he'
    planner.rooms = ['kitchen']
    planner.room_visit_counts = {}
    planner.goal_desc, planner.personality = 'put apple on table', ''
    planner.get_available_plans = Mock(return_value=('', '', [], [], 0))
    # Deliberately pass stale legacy parameters; they must not establish visibility.
    return planner.build_decision_information(None, {'id': 11, 'class_name': 'kitchen'}, [], [],
        {'kitchen': []}, {'kitchen': []}, None, [], [], [{'id': 12, 'class_name': 'apple'}],
        'kitchen', steps=snapshot['peer_observation']['step'], **snapshot)


class PeerObservationTests(unittest.TestCase):
    def test_disappearance_preserves_timestamped_memory_without_current_state(self):
        tracker = PeerStateTracker(1, 2)
        tracker.observe(graph(holds=True), 11)
        before = tracker.snapshot()['peer_memory']
        snapshot = tracker.observe(graph(visible=False), 24)
        self.assertFalse(snapshot['peer_observation']['visible'])
        self.assertIsNone(snapshot['peer_observation']['state'])
        self.assertEqual(snapshot['peer_memory'], before)
        self.assertEqual(snapshot['peer_memory']['last_observed_step'], 11)
        self.assertEqual(snapshot['peer_memory']['last_observed_state']['observed_held_objects'][0]['id'], 12)

    def test_reappearance_replaces_old_holdings_and_timestamp(self):
        tracker = PeerStateTracker(1, 2)
        tracker.observe(graph(holds=True), 0)
        tracker.observe(graph(visible=False), 1)
        snapshot = tracker.observe(graph(holds=False, room='bedroom'), 2)
        self.assertTrue(snapshot['peer_observation']['visible'])
        self.assertEqual(snapshot['peer_observation']['state']['observed_held_objects'], [])
        self.assertEqual(snapshot['peer_memory']['last_observed_step'], 2)
        self.assertEqual(snapshot['peer_memory']['last_observed_state']['room']['class_name'], 'bedroom')

    def test_visible_node_with_unknown_room_is_not_invisible(self):
        view = observe_peer({'nodes': [{'id': 2}], 'edges': []}, 1, 2, 3)
        self.assertTrue(view['visible'])
        self.assertIsNone(view['state']['room'])
        self.assertIsNone(view['state']['position'])

    def test_unseen_peer_edges_cannot_leak_current_state(self):
        partial = graph(visible=False)
        partial['edges'].append({'from_id': 2, 'to_id': 12, 'relation_type': 'HOLDS_RH'})
        self.assertIsNone(observe_peer(partial, 1, 2, 1)['state'])

    def test_snapshot_detaches_inputs_and_outputs_and_rejects_old_time(self):
        tracker = PeerStateTracker(1, 2)
        view = graph(holds=True)
        tracker.observe(view, 4)
        view['nodes'][-1]['obj_transform']['position'][0] = 999
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot['peer_observation']['state']['position'], [1, 0, 1])
        snapshot['peer_memory']['last_observed_state']['room']['class_name'] = 'WRONG'
        self.assertEqual(tracker.snapshot()['peer_memory']['last_observed_state']['room']['class_name'], 'kitchen')
        with self.assertRaises(ValueError):
            tracker.observe(graph(), 3)

    def test_active_progress_cannot_promote_stale_legacy_room_or_holdings(self):
        tracker = PeerStateTracker(1, 2)
        tracker.observe(graph(holds=True), 11)
        snapshot = tracker.observe(graph(visible=False), 24)
        info = active_information(snapshot)
        self.assertEqual(info.peer_observation, snapshot['peer_observation'])
        self.assertFalse(info.peer_observation['visible'])
        self.assertNotIn('I also see Bob', info.current_progress)
        self.assertIn('O_t_peer', info.current_progress)
        self.assertIn('M_t_peer', info.current_progress)
        self.assertEqual(info.to_dict()['peer_memory']['last_observed_step'], 11)
        snapshot['peer_memory']['last_observed_step'] = 999
        self.assertEqual(info.peer_memory['last_observed_step'], 11)

    def test_reactive_and_active_share_current_and_memory_boundary(self):
        tracker = PeerStateTracker(1, 2)
        tracker.observe(graph(holds=True), 11)
        partial = graph(visible=False)
        snapshot = tracker.observe(partial, 24)
        planner = SimpleNamespace(agent_name='Alice', oppo_name='Bob', goal_desc='task', personality='')
        reactive = CommunicationModule(planner).build_decision_information(
            'Bob says he is in kitchen', [], None, partial, agent_id=1, step=24, **snapshot)
        active = active_information(snapshot)
        self.assertEqual(reactive.peer_observation, active.peer_observation)
        self.assertEqual(reactive.peer_memory, active.peer_memory)
        self.assertNotIn('I also see Bob', reactive.current_progress)

    def test_message_cannot_change_observation_or_memory(self):
        agent = LLM_agent.__new__(LLM_agent)
        agent.agent_id = 1
        agent.peer_tracker = PeerStateTracker(1, 2)
        agent.peer_tracker.observe(graph(visible=False), 24)
        agent.trace = None
        agent.dialogue_events, agent.dialogue_history = [], []
        agent.agent_names = ['Zero', 'Alice', 'Bob']
        before = agent._peer_context()
        agent.receive_dialogue_event({'step': 24, 'sender': 1, 'receiver': 0, 'kind': 'active',
                                      'text': 'I am in the bedroom holding the apple.'})
        self.assertEqual(before, agent._peer_context())

    def test_identity_is_observer_relative_not_always_bob(self):
        tracker = PeerStateTracker(2, 1)
        snapshot = tracker.observe(graph(), 1)
        self.assertEqual(snapshot['peer_observation']['peer_id'], 1)
        self.assertEqual(snapshot['peer_observation']['observer_id'], 2)

    def test_agent_forwards_identical_information_to_trace_and_active_planner(self):
        agent = LLM_agent.__new__(LLM_agent)
        agent.agent_id, agent.opponent_agent_id, agent.steps = 1, 2, 24
        agent.peer_tracker = PeerStateTracker(1, 2)
        agent.peer_tracker.observe(graph(holds=True), 18)
        snapshot = agent.peer_tracker.observe(graph(visible=False), 24)
        agent.room_visit_counts = {'kitchen': 1}
        agent.current_room = {'id': 11, 'class_name': 'kitchen'}
        agent.grabbed_objects, agent.opponent_grabbed_objects = [], []
        agent.id2node = {}
        agent.id_inside_room = {2: 'kitchen'}
        agent.unchecked_containers = {'kitchen': []}
        agent.ungrabbed_objects = {'kitchen': []}
        agent.action_history, agent.dialogue_history, agent.public_tasks = [], [], {}
        agent.trace = SimpleNamespace(update=Mock(), state=BeliefState(2), ledger=EvidenceLedger())
        decision = active_information(snapshot)
        agent.LLM = SimpleNamespace(build_decision_information=Mock(return_value=decision),
                                    run=Mock(return_value=(None, {})))
        agent.LLM_plan(None)
        builder_input = agent.LLM.build_decision_information.call_args.kwargs
        self.assertEqual(builder_input['peer_observation'], snapshot['peer_observation'])
        self.assertEqual(builder_input['peer_memory'], snapshot['peer_memory'])
        planner_info = agent.LLM.run.call_args.kwargs['decision_information']
        self.assertEqual(agent.trace.update.call_args.args[0], planner_info.to_dict())


if __name__ == '__main__':
    unittest.main()
