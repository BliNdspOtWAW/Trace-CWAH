"""Public task snapshots and private execution control; no world-state inference."""
import copy
import re


class TaskControl:
    def _control(self):
        if not hasattr(self, 'agent_control'):
            self.agent_control = {'current_subtask_id': None, 'current_skill': None,
                                  'skill_subtask_id': None}
        return self.agent_control

    @property
    def plan(self):
        return self._control()['current_skill']

    @plan.setter
    def plan(self, value):
        control = self._control()
        control['current_skill'] = value
        if value is None:
            control['skill_subtask_id'] = None
            if not self._is_active_task(control['current_subtask_id']):
                control['current_subtask_id'] = None

    @property
    def active_goals(self):
        """Derived execution view, never a second task authority."""
        return [dict(predicate='{}_{}_{}'.format(t['relation'], t.get('object_id', t.get('object_token')), t['target_id']),
                     relation=t['relation'], object_token=str(t.get('object_id', t.get('object_token'))),
                     target_id=t['target_id'], count=t.get('count', 1),
                     object_class=t.get('object_class'), target_class=t.get('target_class'))
                for t in getattr(self, 'public_tasks', {}).values() if t['status'] == 'active']

    @property
    def completed_goals(self):
        return [copy.deepcopy(t) for t in getattr(self, 'public_tasks', {}).values()
                if t['status'] == 'completed']

    def _active_requirements(self):
        return {t['predicate']: t['count'] for t in self.active_goals}

    def _matches_active_goal(self, node):
        return any(self._node_matches_goal_object(node, t['object_token']) for t in self.active_goals)

    def set_public_tasks(self, tasks):
        self.public_tasks = copy.deepcopy(tasks)
        self._uses_public_board = True
        self._refresh_task_snapshot()

    def update_task_goal(self, goal, subtask_records=None):
        """Legacy static-goal adapter. Continuous uses set_public_tasks exclusively."""
        if getattr(self, '_uses_public_board', False):
            return
        records = list(subtask_records or [])
        if not records:
            for predicate, value in goal.items():
                count = value[0] if isinstance(value, (list, tuple)) else value
                if count <= 0:
                    continue
                relation, token, target = self._parse_goal(predicate)
                records.append({'subtask_id': 'legacy:' + predicate, 'relation': relation,
                                'object_token': token, 'target_id': target, 'count': count})
        self.public_tasks = {}
        for i, record in enumerate(records):
            item = copy.deepcopy(record)
            item.setdefault('subtask_id', 'legacy:' + str(i))
            item.setdefault('status', 'active')
            self.public_tasks[item['subtask_id']] = item
        self._refresh_task_snapshot()

    def _refresh_task_snapshot(self):
        active = [t for t in self.public_tasks.values() if t['status'] == 'active']
        trace = getattr(self, 'trace', None)
        if trace is not None and hasattr(trace, 'ledger'):
            trace.ledger.set_active_tasks(active)
        if hasattr(self, 'ungrabbed_objects'):
            self._prune_ungrabbed_memory()
        nodes = getattr(self, 'id2node', {})
        self.goal_objects_name = []
        for t in self.active_goals:
            self.id_inside_room.setdefault(t['target_id'], self.rooms_name[:])
            object_name = t.get('object_class')
            if not object_name:
                object_name = (nodes.get(int(t['object_token']), {}).get('class_name')
                               if t['object_token'].isdigit() else t['object_token'])
            for name in (object_name, t.get('target_class') or nodes.get(t['target_id'], {}).get('class_name')):
                if name and name not in self.goal_objects_name:
                    self.goal_objects_name.append(name)
        if active:
            t = active[0]
            target_class = t.get('target_class') or nodes.get(t['target_id'], {}).get('class_name') or 'target'
            self.goal_location = '<{}> ({})'.format(target_class, t['target_id'])
            self.goal_location_id = t['target_id']
        if hasattr(self, 'LLM'):
            self.LLM.id2node = copy.deepcopy(nodes)
            self.LLM.refresh_goal(getattr(self, 'goal_location', None), self._active_requirements(),
                                  active, self.completed_goals)
        control = self._control()
        if control['current_skill'] is None and not self._is_active_task(control['current_subtask_id']):
            control['current_subtask_id'] = None

    def _is_active_task(self, task_id):
        return getattr(self, 'public_tasks', {}).get(task_id, {}).get('status') == 'active'

    def control_context(self):
        return copy.deepcopy(self._control())

    def accept_active_choice(self, task_id, skill):
        if task_id is not None and not self._is_active_task(task_id):
            raise ValueError('Active selected a non-active subtask: {}'.format(task_id))
        if skill.startswith(('[gograb]', '[goput]')):
            task = self.public_tasks.get(task_id)
            if not task:
                raise ValueError('Grab/place skill requires an active subtask_id')
            ids = [int(x) for x in re.findall(r'\((\d+)\)', skill)]
            token = str(task.get('object_id', task.get('object_token')))
            if not ids or (token.isdigit() and ids[0] != int(token)):
                raise ValueError('Skill object does not match selected subtask')
            if skill.startswith('[goput]') and (len(ids) != 2 or ids[1] != task['target_id']):
                raise ValueError('Skill destination does not match selected subtask')
        control = self._control()
        control['current_subtask_id'] = task_id
        control['current_skill'] = skill
        control['skill_subtask_id'] = task_id

    def apply_reactive_target(self, task_id):
        """A null proposal retains focus; switching never interrupts the skill."""
        if task_id is None:
            return False
        if not self._is_active_task(task_id):
            raise ValueError('Reactive selected a non-active subtask: {}'.format(task_id))
        control = self._control()
        changed = control['current_subtask_id'] != task_id
        control['current_subtask_id'] = task_id
        return changed
