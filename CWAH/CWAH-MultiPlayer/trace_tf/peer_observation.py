"""Direct peer observation and timestamped memory; never infer from messages."""
import copy
import json


def observe_peer(observation, observer_id, peer_id, step):
    if type(step) is not int or step < 0:
        raise ValueError('observation step must be a nonnegative integer')
    nodes = {n['id']: n for n in observation.get('nodes', [])}
    visible = peer_id in nodes
    state = None
    if visible:
        room, held = None, []
        for edge in observation.get('edges', []):
            if edge.get('from_id') != peer_id:
                continue
            target = nodes.get(edge.get('to_id'))
            if target is None:
                continue
            relation = edge.get('relation_type')
            if relation == 'INSIDE' and target.get('category') == 'Rooms':
                room = {'id': target['id'], 'class_name': target.get('class_name')}
            elif relation in ('HOLDS_RH', 'HOLDS_LH'):
                held.append({'id': target['id'], 'class_name': target.get('class_name'),
                             'hand_relation': relation})
        state = {'room': room, 'observed_held_objects': sorted(held, key=lambda x: (x['id'], x['hand_relation'])),
                 'position': copy.deepcopy((nodes[peer_id].get('obj_transform') or {}).get('position'))}
    return {'observer_id': observer_id, 'peer_id': peer_id, 'step': step,
            'source': 'partial_observation', 'visible': visible, 'state': state}


class PeerStateTracker:
    def __init__(self, observer_id, peer_id):
        self.observer_id, self.peer_id = observer_id, peer_id
        self.current = {'observer_id': observer_id, 'peer_id': peer_id, 'step': None,
                        'source': 'no_observation_yet', 'visible': None, 'state': None}
        self.memory = {'observer_id': observer_id, 'peer_id': peer_id,
                       'last_observed_step': None, 'last_observed_state': None,
                       'source': 'partial_observation_memory'}

    def observe(self, observation, step):
        if self.current['step'] is not None and step < self.current['step']:
            raise ValueError('peer tracker cannot replace current view with an older observation')
        self.current = observe_peer(observation, self.observer_id, self.peer_id, step)
        if self.current['visible']:
            self.memory['last_observed_step'] = step
            self.memory['last_observed_state'] = copy.deepcopy(self.current['state'])
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy({'peer_observation': self.current, 'peer_memory': self.memory})


def render_peer_context(observation, memory, peer_name='teammate'):
    return (
        '\nStructured peer information about ' + peer_name + ':\n'
        'O_t_peer is ONLY the current direct partial observation. visible=false '
        'means no current observed peer state; null fields are unknown. '
        'observed_held_objects lists visible relations only.\n'
        'M_t_peer is timestamped last-observed memory, NOT current visibility, '
        'location or holdings. Do not promote historical state to present fact. '
        'Messages/self-reports update dialogue or beliefs, never these observation fields.\n'
        + json.dumps({'O_t_peer': observation or {}, 'M_t_peer': memory or {}}, ensure_ascii=False)
    )
