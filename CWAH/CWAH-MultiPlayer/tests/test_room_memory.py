import unittest

from agents.LLM_agent import LLM_agent
from LLM.LLM import LLM


class RoomMemoryTests(unittest.TestCase):
    def test_inactive_object_remains_available_after_task_switch(self):
        agent = LLM_agent.__new__(LLM_agent)
        agent.agent_id = 1
        agent.rooms_name = ['kitchen', 'bedroom']
        agent.steps = 0
        agent.id_inside_room = {}
        agent.ungrabbed_objects = {'kitchen': [], 'bedroom': None}
        agent.unsatisfied = {'on_12_72': 1}
        observation = {
            'nodes': [
                {'id': 11, 'category': 'Rooms', 'class_name': 'kitchen'},
                {'id': 10, 'class_name': 'apple', 'properties': ['GRABBABLE']},
            ],
            'edges': [{'from_id': 1, 'to_id': 11, 'relation_type': 'INSIDE'}],
        }
        agent._remember_observation(observation)
        agent._prune_ungrabbed_memory()
        self.assertIn(10, agent.object_memory)
        self.assertEqual(agent.ungrabbed_objects['kitchen'], [])
        agent.unsatisfied = {'inside_10_103': 1}
        agent._prune_ungrabbed_memory()
        self.assertEqual(agent.ungrabbed_objects['kitchen'][0]['id'], 10)
        agent._remember_observation(observation)
        self.assertEqual(agent.room_visit_counts['kitchen'], 1)
        agent._last_observed_room = 'bedroom'
        agent._remember_observation(observation)
        self.assertEqual(agent.room_visit_counts['kitchen'], 2)

    def test_previously_visited_rooms_remain_search_candidates(self):
        planner = LLM.__new__(LLM)
        planner.rooms = ['kitchen', 'bedroom', 'livingroom']
        planner.roomname2id = {'kitchen': 11, 'bedroom': 210, 'livingroom': 267}
        planner.room_visit_counts = {'kitchen': 4, 'bedroom': 2, 'livingroom': 1}
        planner.current_room_name = 'bedroom'
        planner.communication = True
        _, _, _, actions, _ = planner.get_available_plans(
            None, [], {room: [] for room in planner.rooms},
            {room: [] for room in planner.rooms}, None, None,
        )
        self.assertEqual(actions[:2], [
            '[goexplore] <livingroom> (267)', '[goexplore] <kitchen> (11)',
        ])
