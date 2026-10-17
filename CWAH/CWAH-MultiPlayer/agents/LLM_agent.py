from LLM import *
from LLM.communication import CommunicationModule
from trace_tf.updater import HybridUpdater
from trace_tf.policy import render_policy_context, render_reply_context


class LLM_agent:
	"""
	LLM agent class
	"""
	def __init__(self, agent_id, char_index, args):
		self.debug = args.debug
		self.agent_type = 'LLM'
		self.agent_names = ["Zero", "Alice", "Bob"]
		self.agent_id = agent_id
		if agent_id == 1 and getattr(args, 'trace_mode', 'off') == 'on' and getattr(args, 'obs_type', 'partial') != 'partial':
			raise ValueError('TRACE target requires partial observations; full/vision observations are not audited')
		self.opponent_agent_id = 3 - agent_id
		self.source = args.source
		self.lm_id = args.lm_id
		self.prompt_template_path = args.prompt_template_path
		self.communication = args.communication
		self.cot = args.cot
		self.args = args
		self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path, self.communication, self.cot, self.args, self.agent_id)
		self.communication_module = CommunicationModule(self.LLM)
		self.action_history = []
		self.dialogue_history = []
		self.dialogue_events = []
		self.trace = None
		self.containers_name = []
		self.goal_objects_name = []
		self.rooms_name = []
		self.roomname2id = {}
		self.unsatisfied = {}
		self.active_goals = []
		self.completed_goals = []
		self.steps = 0
		# self.location = None
		# self.last_location = None
		self.plan = None
		self.stuck = 0
		self.current_room = None
		self.last_room = None
		self.grabbed_objects = None
		self.opponent_grabbed_objects = []
		self.goal_location = None
		self.goal_location_id = None
		self.last_action = None
		self.id2node = {}
		self.id_inside_room = {}
		self.satisfied = []
		self.reachable_objects = []
		self.unchecked_containers = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.ungrabbed_objects = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}


	@property
	def all_relative_name(self) -> list:
		return self.containers_name + self.goal_objects_name + self.rooms_name + ['character']

	@staticmethod
	def _parse_goal(predicate):
		relation, object_token, target_token = predicate.split('_', 2)
		target_id = int(target_token.split('(')[-1].split(')')[0]) if '(' in target_token else int(target_token)
		return relation.lower(), object_token, target_id

	def _node_matches_goal_object(self, node, object_token):
		return node['id'] == int(object_token) if object_token.isdigit() else node['class_name'] == object_token

	def _matches_unsatisfied_goal(self, node):
		return any(
			count > 0 and self._node_matches_goal_object(node, self._parse_goal(predicate)[1])
			for predicate, count in getattr(self, 'unsatisfied', {}).items()
		)

	def _remove_cached_object(self, object_id):
		"""Invalidate every remembered room location for one observed/stale object."""
		for room, objects in getattr(self, 'ungrabbed_objects', {}).items():
			if objects is not None:
				self.ungrabbed_objects[room] = [item for item in objects if item['id'] != object_id]

	def _prune_ungrabbed_memory(self):
		"""Keep remote memory, but only expose objects relevant to active unfinished goals."""
		for room, objects in getattr(self, 'ungrabbed_objects', {}).items():
			if objects is not None:
				self.ungrabbed_objects[room] = [
					item for item in objects if self._matches_unsatisfied_goal(item)
				]

	def _refresh_active_goals(self, goal, subtask_records=None):
		records = {
			f"{item['relation']}_{item['object_id']}_{item['target_id']}": item
			for item in (subtask_records or [])
		}
		self.active_goals = []
		for predicate, value in goal.items():
			count = value[0] if isinstance(value, (list, tuple)) else value
			if count <= 0:
				continue
			relation, object_token, target_id = self._parse_goal(predicate)
			record = records.get(
				f"{relation}_{object_token}_{target_id}", {}
			)
			self.active_goals.append({
				'predicate': predicate,
				'relation': relation,
				'object_token': object_token,
				'target_id': target_id,
				'count': count,
				'object_class': record.get('object_class'),
				'target_class': record.get('target_class'),
			})

	def _put_options(self):
		options = []
		for object_id in self.grabbed_objects:
			obj = self.id2node.get(object_id)
			if obj is None:
				continue
			for goal in self.active_goals:
				if not self._node_matches_goal_object(obj, goal['object_token']):
					continue
				target = self.id2node.get(goal['target_id'])
				target_class = goal.get('target_class') or (target or {}).get('class_name')
				if target_class is None:
					continue
				options.append(
					f"[goput] <{obj['class_name']}> ({object_id}) "
					f"<{target_class}> ({goal['target_id']})"
				)
		return list(dict.fromkeys(options))
	
	def goexplore(self):
		target_room_id = int(self.plan.split(' ')[-1][1:-1])
		if self.current_room['id'] == target_room_id:
			self.plan = None
			return None
		return self.plan.replace('[goexplore]', '[walktowards]')
	
	
	def gocheck(self):
		assert len(self.grabbed_objects) < 2 # must have at least one free hands
		target_container_id = int(self.plan.split(' ')[-1][1:-1])
		target_container_name = self.plan.split(' ')[1]
		target_container_room = self.id_inside_room[target_container_id]
		if self.current_room['class_name'] != target_container_room:
			return f"[walktowards] <{target_container_room}> ({self.roomname2id[target_container_room]})"

		target_container = self.id2node[target_container_id]
		if 'OPEN' in target_container['states']:
			self.plan = None
			return None
		if f"{target_container_name} ({target_container_id})" in self.reachable_objects:
			return self.plan.replace('[gocheck]', '[open]') # conflict will work right?
		else:
			return self.plan.replace('[gocheck]', '[walktowards]')


	def gograb(self):
		target_object_id = int(self.plan.split(' ')[-1][1:-1])
		target_object_name = self.plan.split(' ')[1]
		if target_object_id in self.grabbed_objects:
			if self.debug:
				print(f"successful grabbed!")
			self.plan = None
			return None
		assert len(self.grabbed_objects) < 2 # must have at least one free hands

		target_object_room = self.id_inside_room[target_object_id]
		if self.current_room['class_name'] != target_object_room:
			return f"[walktowards] <{target_object_room}> ({self.roomname2id[target_object_room]})"

		if target_object_id not in self.id2node or target_object_id not in [w['id'] for w in self.ungrabbed_objects[target_object_room]] or target_object_id in [x['id'] for x in self.opponent_grabbed_objects]:
			if self.debug:
				print(f"not here any more!")
			self._remove_cached_object(target_object_id)
			self.plan = None
			return None
		if f"{target_object_name} ({target_object_id})" in self.reachable_objects:
			return self.plan.replace('[gograb]', '[grab]')
		else:
			return self.plan.replace('[gograb]', '[walktowards]')
	
	def goput(self):
		ids = [int(value) for value in __import__('re').findall(r'\((\d+)\)', self.plan)]
		if len(ids) < 2:
			raise ValueError(f"multi-target goput requires object and target ids: {self.plan}")
		object_id, target_id = ids[-2], ids[-1]
		if object_id not in self.grabbed_objects:
			self.plan = None
			return None
		goal = next((goal for goal in self.active_goals
			if goal['target_id'] == target_id
			and self._node_matches_goal_object(self.id2node[object_id], goal['object_token'])), None)
		if goal is None:
			self.plan = None
			return None
		target = self.id2node.get(target_id)
		target_class = goal.get('target_class') or (target or {}).get('class_name') or 'target'
		target_location = f"<{target_class}> ({target_id})"
		if type(self.id_inside_room[target_id]) is list:
			if len(self.id_inside_room[target_id]) == 0:
				print(f"never find the goal location {target_location}")
				self.id_inside_room[target_id] = self.rooms_name[:]
			target_room_name = self.id_inside_room[target_id][0]
		else:
			target_room_name = self.id_inside_room[target_id]

		if self.current_room['class_name'] != target_room_name:
			return f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
		if target_location not in self.reachable_objects:
			return f"[walktowards] {target_location}"
		y = self.id2node[target_id]
		if goal['relation'] == 'inside':
			if 'CLOSED' in y['states']:
				return f"[open] <{y['class_name']}> ({y['id']})"
			action = '[putin]'
		else:
			action = '[putback]'
		x = self.id2node[object_id]
		return f"{action} <{x['class_name']}> ({x['id']}) <{y['class_name']}> ({y['id']})"


	def LLM_plan(self, location):
		put_options = self._put_options()
		source_event_ids = self._trace_source_event_ids()
		information = self.LLM.build_decision_information(
			location, self.current_room, [self.id2node[x] for x in self.grabbed_objects],
			self.satisfied, self.unchecked_containers, self.ungrabbed_objects, None,
			self.action_history, self.dialogue_history, self.opponent_grabbed_objects,
			self.id_inside_room[self.opponent_agent_id], None, self.steps,
			put_options=put_options, source_event_ids=source_event_ids,
		)
		context = ""
		if self.trace is not None:
			self.trace.update(information.to_dict())
			context = render_policy_context(self.trace.state, self.trace.ledger)
		return self.LLM.run(
			location, self.current_room, [self.id2node[x] for x in self.grabbed_objects],
			self.satisfied, self.unchecked_containers, self.ungrabbed_objects, None,
			self.action_history, self.dialogue_history, self.opponent_grabbed_objects,
			self.id_inside_room[self.opponent_agent_id], None, self.steps,
			belief_context=context, put_options=put_options,
			decision_information=information,
		)

	def _trace_source_event_ids(self):
		if self.trace is None:
			return ()
		new = self.trace.ledger.since(getattr(self.trace, 'cursor', 0))
		ids = [event.event_id for event in new]
		observations = [event for event in self.trace.ledger.events if event.kind == 'observation']
		if observations and observations[-1].event_id not in ids:
			ids.append(observations[-1].event_id)
		return tuple(ids)


	def check_progress(self, state, goal_spec):
		unsatisfied = {}
		satisfied = []
		id2node = {node['id']: node for node in state['nodes']}

		for key, value in goal_spec.items():
			relation, object_token, target_id = self._parse_goal(key)
			cnt = value[0]
			for edge in state['edges']:
				if cnt == 0:
					break
				if (edge['relation_type'].lower() == relation
						and edge['to_id'] == target_id
						and self._node_matches_goal_object(id2node[edge['from_id']], object_token)):
					satisfied.append(id2node[edge['from_id']])
					cnt -= 1
					# if self.debug:
					# 	print(satisfied)
			if cnt > 0:
				unsatisfied[key] = cnt
		return satisfied, unsatisfied


	def filter_graph(self, obs):
		relative_id = [node['id'] for node in obs['nodes'] if node['class_name'] in self.all_relative_name]
		relative_id = [x for x in relative_id if all([x != y['id'] for y in self.satisfied])]
		new_graph = {
			"edges": [edge for edge in obs['edges'] if
					  edge['from_id'] in relative_id and edge['to_id'] in relative_id],
			"nodes": [node for node in obs['nodes'] if node['id'] in relative_id]
		}
	
		return new_graph
	
	def get_action(self, observation, goal):
		"""
		:param observation: {"edges":[{'from_id', 'to_id', 'relation_type'}],
		"nodes":[{'id', 'category', 'class_name', 'prefab_name', 'obj_transform':{'position', 'rotation', 'scale'}, 'bounding_box':{'center','size'}, 'properties', 'states'}],
		"messages": [None, None]
		}
		:param goal:{predicate:[count, True, 2]}
		:return:
		"""
		# Arena delivers every active and immediate-reply event exactly once.
		# Reading observation['messages'] here would duplicate active messages.
		if self.trace is not None:
			self.trace.ledger.observe(self.steps, observation)

		satisfied, unsatisfied = self.check_progress(observation, goal)
		self.unsatisfied = unsatisfied
		self.satisfied = satisfied
		self._prune_ungrabbed_memory()
		obs = self.filter_graph(observation)
		self.grabbed_objects = []
		opponent_grabbed_objects = []
		self.reachable_objects = []
		self.id2node = {x['id']: x for x in obs['nodes']}
		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			if x == self.agent_id:
				if r == 'INSIDE':
					self.current_room = self.id2node[y]
				elif r in ['HOLDS_RH', 'HOLDS_LH']:
					self.grabbed_objects.append(y)
				elif r == 'CLOSE':
					y = self.id2node[y]
					self.reachable_objects.append(f"<{y['class_name']}> ({y['id']})")
			elif x == self.opponent_agent_id and r in ['HOLDS_RH', 'HOLDS_LH']:
				opponent_grabbed_objects.append(self.id2node[y])

		unchecked_containers = []
		ungrabbed_objects = []
		for x in obs['nodes']:
			# A direct observation supersedes every remembered room location for
			# this object. The current-room cache is rebuilt below when relevant.
			self._remove_cached_object(x['id'])
			if x['id'] in self.grabbed_objects or x['id'] in [w['id'] for w in opponent_grabbed_objects]:
				continue
			self.id_inside_room[x['id']] = self.current_room['class_name']
			target_ids = {goal['target_id'] for goal in self.active_goals}
			if x['class_name'] in self.containers_name and 'CLOSED' in x['states'] and x['id'] not in target_ids:
				unchecked_containers.append(x)
			if (self._matches_unsatisfied_goal(x)
					and all([x['id'] != y['id'] for y in self.satisfied])
					and 'GRABBABLE' in x['properties']
					and x['id'] not in self.grabbed_objects
					and x['id'] not in [w['id'] for w in opponent_grabbed_objects]):
				ungrabbed_objects.append(x)

		for target_id in target_ids:
			self.id_inside_room.setdefault(target_id, self.rooms_name[:])
			if type(self.id_inside_room[target_id]) is list and self.current_room['class_name'] in self.id_inside_room[target_id]:
				self.id_inside_room[target_id].remove(self.current_room['class_name'])
				if len(self.id_inside_room[target_id]) == 1:
					self.id_inside_room[target_id] = self.id_inside_room[target_id][0]
		self.unchecked_containers[self.current_room['class_name']] = unchecked_containers[:]
		self.ungrabbed_objects[self.current_room['class_name']] = ungrabbed_objects[:]

		info = {'graph': obs,
				"obs": {
						 "grabbed_objects": self.grabbed_objects,
						 "opponent_grabbed_objects": opponent_grabbed_objects,
						 "reachable_objects": self.reachable_objects,
						 "progress": {
								"unchecked_containers": self.unchecked_containers,
								"ungrabbed_objects": self.ungrabbed_objects,
									  },
						"satisfied": self.satisfied,
						"current_room": self.current_room['class_name'],
						},
				}
		if self.id_inside_room[self.opponent_agent_id] == self.current_room['class_name']:
			self.opponent_grabbed_objects = opponent_grabbed_objects
		action = None
		LM_times = 0
		while action is None:
			if self.plan is None:
				if LM_times > 0:
					print(info)
				if LM_times > 3:
					raise Exception(f"retrying LM_plan too many times")
				plan, a_info = self.LLM_plan(observation['location'])
				if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
					print("No more things to do!")
					plan = f"[wait]"
				self.plan = plan
				self.action_history.append('[send_message]' if plan.startswith('[send_message]') else plan)
				if "plan" not in a_info.keys():
					a_info.update({"plan": self.plan})
				a_info.update({"steps": self.steps})
				info.update({"LLM": a_info})
				LM_times += 1
			if self.plan.startswith('[goexplore]'):
				action = self.goexplore()
			elif self.plan.startswith('[gocheck]'):
				action = self.gocheck()
			elif self.plan.startswith('[gograb]'):
				action = self.gograb()
			elif self.plan.startswith('[goput]'):
				action = self.goput()
			elif self.plan.startswith('[send_message]'):
				action = self.plan[:]
				self.plan = None
			elif self.plan.startswith('[wait]'):
				action = None
				break
			else:
				raise ValueError(f"unavailable plan {self.plan}")

		self.steps += 1
		info.update({"plan": self.plan,
					 })
		if action == self.last_action and self.current_room['class_name'] == self.last_room:
			self.stuck += 1
		else:
			self.stuck = 0
		self.last_action = action
		# self.last_location = self.location
		self.last_room = self.current_room
		if self.stuck > 20:
			print("Warning! stuck!")
			self.action_history[-1] += ' but unfinished'
			stuck_plan = self.plan
			self.plan = None
			if stuck_plan and stuck_plan.startswith('[goput]'):
				target_id = [int(value) for value in __import__('re').findall(r'\((\d+)\)', stuck_plan)][-1]
				target = self.id2node[target_id]
				target_location = f"<{target['class_name']}> ({target_id})"
				target_room = self.id_inside_room[target_id]
				target_room_name = target_room[0] if type(target_room) is list else target_room
				action = f"[walktowards] {target_location}"
				if self.current_room['class_name'] != target_room_name:
					action = f"[walktowards] <{target_room_name}> ({self.roomname2id[target_room_name]})"
			self.stuck = 0
	
		if self.trace is not None:
			self.trace.ledger.own_action(self.steps - 1, action)
		return action, info

	def receive_dialogue_event(self, event):
		self.dialogue_events.append(event.copy())
		self.dialogue_history.append(f"{self.agent_names[event['sender'] + 1]}: {event['text']}")
		if self.trace is not None:
			self.trace.ledger.message(event['step'], event['sender'], event['receiver'], event['text'], event['kind'])

	def observe_trace_outcome(self, observation, step):
		if self.trace is not None:
			self.trace.ledger.observe(step, observation)

	def update_task_goal(self, goal, subtask_records=None):
		"""Refresh a dynamic public goal without touching plans, history, or TRACE."""
		active = {key: value for key, value in goal.items() if value[0] > 0}
		if not active:
			self.active_goals = []
			self.unsatisfied = {}
			self.goal_objects_name = []
			self._prune_ungrabbed_memory()
			if self.trace is not None and hasattr(self.trace, 'ledger'):
				self.trace.ledger.set_active_tasks([])
			if self.goal_location is not None:
				self.LLM.refresh_goal(
					self.goal_location, {}, [], getattr(self, 'completed_goals', [])
				)
			return
		self._refresh_active_goals(active, subtask_records)
		if self.trace is not None and hasattr(self.trace, 'ledger'):
			self.trace.ledger.set_active_tasks(subtask_records)
		self.unsatisfied = {key: value[0] for key, value in active.items()}
		self._prune_ungrabbed_memory()
		first = self.active_goals[0]
		known_nodes = getattr(self, 'id2node', {})
		first_target = known_nodes.get(first['target_id'])
		first_target_class = first.get('target_class') or (first_target or {}).get('class_name') or 'target'
		self.goal_location = f"<{first_target_class}> ({first['target_id']})"
		self.goal_location_id = first['target_id']
		self.goal_objects_name = []
		for item in self.active_goals:
			self.id_inside_room.setdefault(item['target_id'], self.rooms_name[:])
			if item.get('object_class'):
				self.goal_objects_name.append(item['object_class'])
			elif item['object_token'].isdigit():
				node = known_nodes.get(int(item['object_token']))
				if node is not None:
					self.goal_objects_name.append(node['class_name'])
			else:
				self.goal_objects_name.append(item['object_token'])
			target = known_nodes.get(item['target_id'])
			target_class = item.get('target_class') or (target or {}).get('class_name')
			if target_class:
				self.goal_objects_name.append(target_class)
		self.goal_objects_name = list(dict.fromkeys(self.goal_objects_name))
		self.LLM.refresh_goal(
			self.goal_location, self.unsatisfied, subtask_records,
			getattr(self, 'completed_goals', [])
		)

	def record_task_completions(self, completed, completed_by=None, step=None):
		"""Retain public completion history; TRACE receives no hidden graph state."""
		for task in completed or []:
			record = dict(task)
			record['completed_by'] = (completed_by or {}).get(task.get('subtask_id'), 'unknown')
			if not any(item.get('subtask_id') == record.get('subtask_id')
					   for item in self.completed_goals):
				self.completed_goals.append(record)
			if self.trace is not None:
				self.trace.ledger.task_completion(
					step if step is not None else self.steps,
					task, record['completed_by'],
				)

	def reply_to_message(self, message, observation):
		room = self.current_room['class_name'] if self.current_room else None
		belief_context = "TRACE is disabled. No teammate belief is available."
		trace_record = None
		known_satisfied = [f"<{item['class_name']}> ({item['id']})" for item in self.satisfied]
		progress = (
			f"Last selected environment action: {self.last_action or 'None'}\n"
			f"Known satisfied goal objects: {', '.join(known_satisfied) or 'None'}\n"
			f"Known remaining goal requirements: {self.unsatisfied or 'Unknown'}"
		)
		information = self.communication_module.build_decision_information(
			message, self.dialogue_history, self.plan, observation,
			agent_id=self.agent_id, progress=progress,
			action_history=self.action_history[-10:], step=getattr(self, 'steps', 0),
			source_event_ids=self._trace_source_event_ids(),
		)
		if self.trace is not None:
			state, trace_record = self.trace.update(information.to_dict())
			belief_context = render_reply_context(state, self.trace.ledger)
		reply, call = self.communication_module.reply(
			message, self.dialogue_history, self.plan, room, observation,
			agent_id=self.agent_id, progress=progress,
			action_history=self.action_history[-10:], belief_context=belief_context,
			decision_information=information,
		)
		call.update({
			"planner_type": "reactive",
			"trace_enabled": self.trace is not None,
			"trace_revision": self.trace.state.revision if self.trace is not None else None,
			"trace_updated": trace_record is not None,
			"trace_new_event_ids": trace_record["new_event_ids"] if trace_record else [],
			"trace_context": belief_context,
			"current_skill": self.plan,
			"action_history": self.action_history[-10:],
		})
		return reply, call

	def reset(self, obs, containers_name, goal_objects_name, rooms_name, room_info, goal, task):
		# containers_name: ['bathroomcabinet', 'kitchencabinet', 'cabinet', 'fridge', 'stove', 'dishwasher', 'microwave']
        # goal_objects_name: ['apple', 'cupcake', 'juice', 'pudding', 'coffeetable']
        # rooms_name: ['livingroom', 'kitchen', 'bedroom', 'bathroom']
        # room_info: [{'id': 1, 'category': 'Characters', 'class_name': 'character', 'prefab_name': 'Female1', 'obj_transform': {'position': [6.15288, 1.25, -2.417234], 'rotation': [0.0, -0.8562313, 0.0, 0.516592741], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [6.19369459, 0.9142707, -2.387253], 'size': [0.806426048, 1.81487966, 0.94945]}, 'properties': [], 'states': []}, {'id': 2, 'category': 'Characters', 'class_name': 'character', 'prefab_name': 'Male1', 'obj_transform': {'position': [11.0559931, 1.25, -4.948724], 'rotation': [0.0, -0.6762676, 0.0, 0.73665607], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [11.0793667, 2.121052, -4.95072556], 'size': [0.628577232, 1.86971855, 1.38765383]}, 'properties': [], 'states': []}, {'id': 11, 'category': 'Rooms', 'class_name': 'kitchen', 'prefab_name': 'PRE_ROO_Kitchen_01', 'obj_transform': {'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0, 1.0], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [1.25, 1.25, 0.0], 'size': [8.0, 3.0, 8.0]}, 'properties': [], 'states': []}, {'id': 172, 'category': 'Rooms', 'class_name': 'bathroom', 'prefab_name': 'PRE_ROO_Bathroom_03', 'obj_transform': {'position': [1.228, 0.0, 6.25], 'rotation': [0.0, 0.7071068, 0.0, 0.7071067], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [2.478, 1.25, 6.25], 'size': [5.5, 3.0, 5.5]}, 'properties': [], 'states': []}, {'id': 210, 'category': 'Rooms', 'class_name': 'bedroom', 'prefab_name': 'PRE_ROO_Bedroom_03', 'obj_transform': {'position': [5.0, 0.0, -2.456], 'rotation': [0.0, 0.0, 0.0, 1.0], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [7.5, 1.25, -3.706], 'size': [5.5, 3.0, 5.5]}, 'properties': [], 'states': []}, {'id': 267, 'category': 'Rooms', 'class_name': 'livingroom', 'prefab_name': 'PRE_ROO_Livingroom_08', 'obj_transform': {'position': [11.25, 0.0, -6.1842], 'rotation': [0.0, -0.7071068, 0.0, 0.7071067], 'scale': [1.0, 1.0, 1.0]}, 'bounding_box': {'center': [13.75, 1.25, -4.9342], 'size': [8.0, 3.0, 8.0]}, 'properties': [], 'states': []}]
        # goal: {'on_pudding_<coffeetable> (268)': [1, True, 2], 'on_juice_<coffeetable> (268)': [1, True, 2], 'on_apple_<coffeetable> (268)': [1, True, 2], 'on_cupcake_<coffeetable> (268)': [2, True, 2]}
		self.steps = 0
		self.completed_goals = []
		self.containers_name = containers_name
		self.goal_objects_name = goal_objects_name
		self.rooms_name = rooms_name
		self.roomname2id = {x['class_name']: x['id'] for x in room_info}
		self.id2node = {x['id']: x for x in obs['nodes']}
		self.stuck = 0
		self.last_room = None
		self.unsatisfied = {k: v[0] for k, v in goal.items()}
		self._refresh_active_goals(goal)
		self.satisfied = []
		self.goal_location = list(goal.keys())[0].split('_')[-1]
		self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
		self.id_inside_room = {self.opponent_agent_id: None}
		for item in self.active_goals:
			self.id_inside_room[item['target_id']] = self.rooms_name[:]
		self.unchecked_containers = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.ungrabbed_objects = {
			"livingroom": None,
			"kitchen": None,
			"bedroom": None,
			"bathroom": None,
		}
		self.opponent_grabbed_objects = []
		for e in obs['edges']:
			x, r, y = e['from_id'], e['relation_type'], e['to_id']
			if x == self.agent_id and r == 'INSIDE':
				self.current_room = self.id2node[y]
		self.plan = None
		self.action_history = [f"[goexplore] <{self.current_room['class_name']}> ({self.current_room['id']})"]
		self.dialogue_history = []
		self.dialogue_events = []
		self.LLM.reset(obs,self.rooms_name, self.roomname2id, self.goal_location, self.unsatisfied, task)
		self.trace = HybridUpdater(self.LLM) if self.agent_id == 1 and getattr(self.args, 'trace_mode', 'off') == 'on' else None
