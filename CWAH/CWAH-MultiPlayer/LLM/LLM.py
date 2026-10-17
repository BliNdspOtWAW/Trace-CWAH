import random
import openai
import torch
import json
import os
import pandas as pd
from openai.error import OpenAIError
import backoff
from trace_tf.information import DecisionInformation

csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cwah.csv")
df = pd.read_csv(csv_path)

profile_dict = {}
examples_dict = {}
for _, row in df.iterrows():
    key = f"{row['Task']}_{row['Cluster']}"
    profile_dict[key] = row['Profile']
    examples_dict[key] = []


def resolve_profile_key(task):
    """Optionally reuse a profile on another task without changing task data."""
    source = os.getenv("CWAH_PROFILE_SOURCE_TASK")
    expected = os.getenv("CWAH_PROFILE_EXPECTED_TASK")
    if not source and not expected:
        return task
    if not source or not expected:
        raise ValueError("Both CWAH_PROFILE_SOURCE_TASK and CWAH_PROFILE_EXPECTED_TASK are required")
    actual, separator, cluster = task.rpartition("_")
    if not separator or actual != expected or not cluster:
        raise ValueError(f"Profile override expects {expected}_<cluster>, got {task}")
    key = f"{source}_{cluster}"
    if key not in profile_dict:
        raise ValueError(f"Missing profile for {key}")
    return key


class LLM:
    def __init__(self,
                 source,  # 'huggingface' or 'openai'
                 lm_id,
                 prompt_template_path,
                 communication,
                 cot,
                 sampling_parameters,
                 agent_id
                 ):
        self.goal_desc = None
        self.goal_location_with_r = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 1 else "Bob"
        self.oppo_name = "Alice" if agent_id == 2 else "Bob"
        self.oppo_pronoun = "she" if agent_id == 2 else "he"
        self.debug = sampling_parameters.debug
        self.goal_location = None
        self.active_subtasks = []
        self.goal_location_id = None
        self.roomname2id = {}
        self.rooms = []
        role_prompt = getattr(sampling_parameters,
                              'alice_prompt_file' if agent_id == 1 else 'bob_prompt_file', None)
        self.prompt_template_path = role_prompt or os.path.join(os.path.dirname(__file__), 'prompt_Peer.txt')
        self.single = False
        self.act = sampling_parameters.act
        self.big_5_act = sampling_parameters.big_5
        self.teammate_profile_file = getattr(sampling_parameters, 'teammate_profile_file', None)
        self.personality = None

        # Both players default to the peer policy; only Bob receives a profile.
        with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        self.prompt_template = self.prompt_template.replace("$AGENT_NAME", self.agent_name).replace("$OPPO_NAME", self.oppo_name)

        self.communication = communication
        self.cot = cot
        self.source = source
        # 支持按 agent/human 两套环境变量分别配置（脚本中导出的 CWAH_*）
        role_prefix = None
        if self.agent_id == 1:
            role_prefix = "CWAH_AGENT"
        elif self.agent_id == 2:
            role_prefix = "CWAH_HUMAN"

        def _get_env(suffix: str):
            if not role_prefix:
                return None
            value = os.getenv(f"{role_prefix}_{suffix}")
            return value if value not in (None, "") else None

        env_api_base = _get_env("API_BASE")
        env_lm_id = _get_env("LM_ID")
        env_api_key = _get_env("API_KEY")

        self.lm_id = env_lm_id or lm_id
        # 两个 agent 共用同一份 args；为了让 agent/human 能各用各的配置，这里让环境变量优先
        self.api_base = env_api_base or sampling_parameters.api_base
        self.api_key = env_api_key or getattr(sampling_parameters, "api_key", None)

        # The configured backend selects the protocol. Model names are opaque
        # API identifiers, including gateway aliases and future model names.
        self.chat = self.source == 'openai'
        self.OPENAI_KEY = None
        self.total_cost = 0
        self.last_usage_metadata = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.source == 'openai':
            if self.api_key is not None:
                openai.api_key = self.api_key
            if self.api_base is not None:
                openai.api_base = self.api_base
            if getattr(sampling_parameters, "debug", False):
                print(
                    f"[LLM] agent_id={self.agent_id} role_prefix={role_prefix} model={self.lm_id} "
                    f"api_base={self.api_base} api_key={'set' if self.api_key else 'unset'}"
                )
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        elif self.source == 'huggingface':
            self.sampling_params = {
                "max_new_tokens": sampling_parameters.max_tokens,
                "temperature": sampling_parameters.t,
                "top_p": sampling_parameters.top_p,
                "num_return_sequences": sampling_parameters.n,
                'use_cache': True,
                'return_dict_in_generate': True,
                'do_sample': True,
                'early_stopping': True,
            }
        elif self.source == "debug":
            self.sampling_params = sampling_parameters
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id, device):
            if source == 'huggingface':
                from transformers import AutoModelForCausalLM, AutoTokenizer, LLaMATokenizer, LLaMAForCausalLM
                print(f"loading huggingface model {lm_id}")
                if 'llama' in lm_id or 'alpaca' in lm_id:
                    tokenizer = LLaMATokenizer.from_pretrained(lm_id)
                    model = LLaMAForCausalLM.from_pretrained(lm_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, load_in_8bit=False).to(device)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(lm_id)
                    model = AutoModelForCausalLM.from_pretrained(lm_id, torch_dtype=torch.float16, pad_token_id=tokenizer.eos_token_id).to(device)
                print(f"loaded huggingface model {lm_id}")

            @backoff.on_exception(backoff.expo, OpenAIError)
            def _generate(prompt, sampling_params):
                usage = 0
                self.last_usage_metadata = None
                if source == 'openai':
                    try:
                        if self.chat:
                            # openai 的配置是全局的；这里每次请求前显式写回，确保 agent/human 各用各的配置
                            if self.api_key is not None:
                                openai.api_key = self.api_key
                            if self.api_base is not None:
                                openai.api_base = self.api_base
                            response = openai.ChatCompletion.create(
                                model=self.lm_id, messages=prompt, **sampling_params
                            )
                            generated_samples = [response['choices'][i]['message']['content'] for i in range(sampling_params['n'])]
                            response_usage = response.get('usage') or {}
                            self.last_usage_metadata = {
                                key: response_usage.get(key)
                                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                            }
                            if 'gpt-4' in self.lm_id:
                                usage = response['usage']['prompt_tokens'] * 0.03 / 1000 + response['usage']['completion_tokens'] * 0.06 / 1000
                            elif 'gpt-3.5' in self.lm_id:
                                usage = response['usage']['total_tokens'] * 0.002 / 1000
                        elif "text-" in lm_id:
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except OpenAIError as e:
                        print(e)
                        raise e
                elif source == 'huggingface':
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    prompt_len = input_ids.shape[-1]
                    output_dict = model.generate(input_ids, **sampling_params)
                    generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
                    for i, sample in enumerate(generated_samples):
                        stop_idx = sample.index('\n') if '\n' in sample else None
                        generated_samples[i] = sample[:stop_idx]
                elif source == "debug":
                    return ["navigation"], 0
                else:
                    raise ValueError("invalid source")
                return generated_samples, usage
            return _generate
        self.generator = lm_engine(self.source, self.lm_id, self.device)

    def reset(self, obs, rooms_name, roomname2id, goal_location, unsatisfied, task):
        self.rooms = rooms_name
        self.obs = obs
        self.roomname2id = roomname2id
        self.goal_location = goal_location
        self.active_subtasks = []
        self.task = task
        if self.teammate_profile_file is not None:
            if self.agent_id == 2:
                with open(self.teammate_profile_file, 'r', encoding='utf-8') as profile_file:
                    self.personality = profile_file.read().strip()
                if not self.personality:
                    raise ValueError(f'Empty teammate profile: {self.teammate_profile_file}')
            else:
                self.personality = ''
        elif self.big_5_act is not None:
          
            if self.agent_id == 1:
                self.personality = ""
            else:
                profile_key = resolve_profile_key(self.task)
                self.personality = profile_dict.get(profile_key, "")
                if profile_key != self.task:
                    print(f"[LLM] teammate profile: {self.task} -> {profile_key}")
                
        else:
            self.personality = ""
        self.id2node = {x['id']: x for x in obs['nodes']}
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)

    def refresh_goal(self, goal_location, unsatisfied, active_subtasks=None,
                     completed_subtasks=None):
        """Refresh public task text without resetting personality or histories."""
        self.goal_location = goal_location
        self.goal_location_id = int(self.goal_location.split(' ')[-1][1:-1])
        self.active_subtasks = list(active_subtasks or [])
        self.goal_desc, self.goal_location_with_r = self.goal2description(unsatisfied, None)
        completed = list(completed_subtasks or [])
        if completed:
            recent = []
            for item in completed[-20:]:
                recent.append(
                    "[{}] <{}> ({}) {} <{}> ({}) by {}".format(
                        item.get('subtask_id', 'unknown-subtask'),
                        item.get('object_class', 'object'), item.get('object_id'),
                        item.get('relation', 'at'), item.get('target_class', 'target'),
                        item.get('target_id'), item.get('completed_by', 'unknown'),
                    )
                )
            self.goal_desc += (
                " Completed subtask history (already satisfied; do not repeat unless "
                "listed again as active): " + "; ".join(recent) + "."
            )

    def goal2description(self, goals, goal_location_room):
        map_rel_to_pred = {'inside': 'into', 'on': 'onto'}
        records = {
            f"{item['relation']}_{item['object_id']}_{item['target_id']}": item
            for item in self.active_subtasks
        }
        descriptions = []
        for predicate, vl in goals.items():
            relation, obj1, obj2 = predicate.split('_', 2)
            count = vl
            if count == 0:
                continue
            if relation == 'holds':
                continue
            elif relation == 'sit':
                continue
            else:
                target_id = int(obj2.split('(')[-1].split(')')[0]) if '(' in obj2 else int(obj2)
                record = records.get(f"{relation}_{obj1}_{target_id}", {})
                target_class = record.get('target_class')
                if not target_class and '<' in obj2:
                    target_class = obj2.split('<', 1)[1].split('>', 1)[0]
                target = self.id2node.get(target_id, {"class_name": target_class or "target"})
                if obj1.isdigit():
                    object_id = int(obj1)
                    obj = self.id2node.get(object_id, {"class_name": record.get('object_class') or "object"})
                    object_text = f"<{obj['class_name']}> ({object_id})"
                else:
                    object_text = f"{count} {obj1}{'s' if count > 1 else ''}"
                subtask_prefix = (
                    f"[{record['subtask_id']}] " if record.get('subtask_id') else ""
                )
                descriptions.append(
                    f"{subtask_prefix}put {object_text} {map_rel_to_pred[relation]} "
                    f"<{target['class_name']}> ({target_id})"
                )
        if not descriptions:
            return "None.", "at their assigned destinations"
        return "Complete these shared subtasks: " + "; ".join(descriptions) + ".", "at their assigned destinations"

    def parse_answer(self, available_actions, text):
        import re
        import random
        think_content = None
        think_match = re.search(r'<cot[^>]*>\s*(.*?)\s*</cot\s*>', text, re.IGNORECASE | re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
        else:
            think_match = re.search(
                r'<cot[^>]*>\s*(.*?)(?=\n\s*(?:<action\b|[:<]?\s*message\b)|$)',
                text,
                re.IGNORECASE | re.DOTALL,
            )
            think_content = think_match.group(1).strip() if think_match else None

        # 提取 <message> 内容（兼容 ":message\"...\"" / "<message ...>..." / 缺失闭合）
        message_content = None
        message_match = re.search(r'<message[^>]*>\s*(.*?)\s*</message\s*>', text, re.IGNORECASE | re.DOTALL)
        if message_match:
            message_content = message_match.group(1).strip()
        else:
            # 优先截到 <action> 之前，避免把 action 内容吞进去
            action_pos = re.search(r'\n\s*<action\b', text, re.IGNORECASE)
            head = text[: action_pos.start()] if action_pos else text

            # 1) "<message ...> ... (无闭合)" 直到 <action> 或文本末尾
            message_match = re.search(r'<message[^>]*>\s*(.*?)\s*$', head, re.IGNORECASE | re.DOTALL)
            if message_match:
                message_content = message_match.group(1).strip()
            else:
                # 2) ":message\"...\"" 或 "message\"...\""（有时没有结尾引号）
                message_match = re.search(
                    r'(?:^|\n)\s*[:<]?\s*message\s*\"(.*?)(?=\"\s*(?:\n|$)|$)',
                    head,
                    re.IGNORECASE | re.DOTALL,
                )
                if message_match:
                    message_content = message_match.group(1).strip()
                else:
                    # 3) "message: ..." / "message= ..."
                    message_match = re.search(
                        r'(?:^|\n)\s*message\s*[:=]\s*(.*?)(?=\n|$)',
                        head,
                        re.IGNORECASE | re.DOTALL,
                    )
                    message_content = message_match.group(1).strip() if message_match else None
        if message_content is not None and message_content.strip() == "":
            message_content = None
        if think_content is not None and think_content.strip() == "":
            think_content = None

        # 提取 <action> 内容
        action_tag_match = re.search(r'<action>(.*?)</action>', text, re.IGNORECASE | re.DOTALL)
        if action_tag_match:
            action_content = action_tag_match.group(1).strip()
            # 兼容模型输出如 "A. [walktowards] ..."：先去掉前缀选项字母，避免误选 available_actions[0]
            action_content = re.sub(r'^\s*(?:\(?[A-Z]\)?)[\s\.\:\)]\s*', '', action_content).strip()

            # 精确匹配 available_actions
            for action in available_actions:
                if action in action_content or action_content in action:
                    chosen_action = action
                    break
            else:
                # 通过 A/B/C 选项匹配
                chosen_action = None
                action_lower = action_content.lower()
                if ("option " in action_lower) or re.fullmatch(r'[A-Z]', action_content.strip()) or re.fullmatch(r'\([A-Z]\)', action_content.strip()):
                    for i in range(len(available_actions)):
                        option = chr(ord('A') + i)
                        if f"option {option.lower()}" in action_lower or action_content.strip() in (option, f"({option})"):
                            chosen_action = available_actions[i]
                            break

            if chosen_action is None:
                # 尝试匹配动作类型或部分内容
                for action in available_actions:
                    parts = action.split(' ')
                    if len(parts) >= 2:
                        act_type = parts[0]
                        if act_type in action_content or any(part in action_content for part in parts[1:]):
                            chosen_action = action
                            break

            if chosen_action is None:
                # 模糊匹配
                for i in range(len(available_actions)):
                    action = available_actions[i]
                    if action in text:
                        chosen_action = action
                        break

            if chosen_action is None:
                # 最终警告匹配
                print("WARNING! Fuzzy match!")
                for i in range(len(available_actions)):
                    action = available_actions[i]
                    if self.communication and i == 0:
                        continue
                    parts = action.split(' ')
                    act = parts[0] if len(parts) > 0 else ""
                    name = parts[1] if len(parts) > 1 else ""
                    id = parts[2] if len(parts) > 2 else ""
                    option = chr(ord('A') + i)
                    if (
                        f"{option} " in text
                        or (act and act in text)
                        or (name and name in text)
                        or (id and id in text)
                    ):
                        chosen_action = action
                        break

            # 如果还是没有匹配，随机选择
            if chosen_action is None:
                print("WARNING! No available action parsed!!! Random choose one")
                chosen_action = random.choice(available_actions)

            # 如果动作是 send_message，则附加 message 内容
            if chosen_action.startswith('[send_message]'):
                chosen_action = f"{chosen_action} <{message_content or ''}>"

            return chosen_action, think_content, message_content

        # 如果没有 <action> 标签，则返回 None 或随机
        print("WARNING! No <action> tag found")
        return random.choice(available_actions), think_content, message_content

    def progress2text(self, current_room, grabbed_objects, unchecked_containers, ungrabbed_objects, goal_location_room, satisfied, opponent_grabbed_objects, opponent_last_room, room_explored):
        sss = {}
        for room, objs in ungrabbed_objects.items():
            cons = unchecked_containers[room]
            extra_obj = None
            if type(goal_location_room) is not list and goal_location_room == room:
                extra_obj = self.goal_location
            if objs is None and extra_obj is None and (room_explored is None or not room_explored[room]):
                sss[room] = f"The {room} is unexplored. "
                continue
            s = ""
            s_obj = ""
            s_con = ""
            if extra_obj is not None:
                s_obj = f"{extra_obj}, "
            if objs is not None and len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"<{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in objs])
                    s_obj += ss
            elif extra_obj is not None:
                s_obj = s_obj[:-2]
            if cons is not None and len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"an unchecked container <{x['class_name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['class_name']}> ({x['id']})" for x in cons])
                    s_con = f"unchecked containers " + ss
            if s_obj == "" and s_con == "":
                s += 'nothing'
                if room_explored is not None and not room_explored[room]:
                    s += ' yet'
            elif s_obj != "" and s_con != "":
                s += s_obj + ', and ' + s_con
            else:
                s += s_obj + s_con
            sss[room] = s
        if len(satisfied) == 0:
            s = ""
        else:
            s = f"{'I' if self.single else 'We'}'ve already found and put "
            s += ', '.join([f"<{x['class_name']}> ({x['id']})" for x in satisfied])
            s += ' ' + self.goal_location_with_r + '. '
        if len(grabbed_objects) == 0:
            s += "I'm holding nothing. "
        else:
            s += f"I'm holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
        s += f"I'm in the {current_room['class_name']}, where I found {sss[current_room['class_name']]}. "
        # opponent modeling
        if not self.single:
            ss = ""
            if len(opponent_grabbed_objects) == 0:
                ss += "nothing. "
            else:
                ss += f"<{opponent_grabbed_objects[0]['class_name']}> ({opponent_grabbed_objects[0]['id']}). "
                if len(opponent_grabbed_objects) == 2:
                    ss = ss[:-2] + f" and <{opponent_grabbed_objects[1]['class_name']}> ({opponent_grabbed_objects[1]['id']}). "
            if opponent_last_room is None:
                s += f"I don't know where {self.oppo_name} is. "
            elif opponent_last_room == current_room['class_name']:
                s += f"I also see {self.oppo_name} here in the {current_room['class_name']}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time I saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"
        for room in self.rooms:
            if room == current_room['class_name']:
                continue
            if 'unexplored' in sss[room]:
                s += sss[room]
            else:
                s += f"I found {sss[room]} in the {room}. "
        return s

    def get_available_plans(self, location, grabbed_objects, unchecked_containers,
                            ungrabbed_objects, message, room_explored, put_options=None):
        def get_distance_xy(pos1, pos2):
            import math
            dx = pos1[0] - pos2[0]
            dy = pos1[1] - pos2[1]
            dz = pos1[2] - pos2[2]
            return math.sqrt(dx * dx + dy * dy + dz * dz)
        available_plans = []
        available_plans_refine = []
        if self.communication and message is not None:
            available_plans.append(f"[send_message] <{message}>")
        visits = getattr(self, 'room_visit_counts', {})
        for room in sorted(self.rooms, key=lambda name: visits.get(name, 0)):
            if room == getattr(self, 'current_room_name', None):
                continue
            if location is not None:
                dist = get_distance_xy(location, self.id2node[self.roomname2id[room]]["obj_transform"]["position"])
                available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]}) - distance: {dist:.4f}")
            else:
                available_plans.append(f"[goexplore] <{room}> ({self.roomname2id[room]})")
        if len(grabbed_objects) < 2:
            for cl in unchecked_containers.values():
                if cl is None:
                    continue
                for container in cl:
                    if location is not None:
                        dist = get_distance_xy(location, container["obj_transform"]["position"])
                        available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']}) - distance: {dist:.4f}")
                    else:
                        available_plans.append(f"[gocheck] <{container['class_name']}> ({container['id']})")
            for ol in ungrabbed_objects.values():
                if ol is None:
                    continue
                for obj in ol:
                    if location is not None:
                        dist = get_distance_xy(location, obj["obj_transform"]["position"])
                        available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']}) - distance: {dist:.4f}")
                    else:
                        available_plans.append(f"[gograb] <{obj['class_name']}> ({obj['id']})")
        for option in put_options or []:
            available_plans.append(option)
        plans = ""
        plans_refine = ""
        available_plans.append(f"[send_message]")
        available_plans_refine = available_plans.copy()

        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"
        for i, plan in enumerate(available_plans):
            if "distance" in available_plans[i]:
                available_plans_refine[i] = plan.split(" - distance:")[0]
        for i, plan in enumerate(available_plans_refine):
            plans_refine += f"{chr(ord('A') + i)}. {plan}\n"
            
        print("**************plan*******************")
        print(plans)
        print("************plans_refine*************")
        print(plans_refine)
        print("*************************************")
        
        
        return plans, plans_refine, available_plans, available_plans_refine, len(available_plans)

    def run(self, location, current_room, grabbed_objects, satisfied, unchecked_containers,
            ungrabbed_objects, goal_location_room, action_history, dialogue_history,
            opponent_grabbed_objects, opponent_last_room, room_explored=None, steps=0,
            belief_context="", put_options=None, decision_information=None):
        info = {}
        information = decision_information or self.build_decision_information(
            location, current_room, grabbed_objects, satisfied, unchecked_containers,
            ungrabbed_objects, goal_location_room, action_history, dialogue_history,
            opponent_grabbed_objects, opponent_last_room, room_explored, steps,
            put_options=put_options,
        )
        progress_desc = information.current_progress
        action_history_desc = ", ".join(information.action_history) or 'None'
        dialogue_history_desc = '\n'.join(information.dialogue_history) or 'None'
        prompt = self.prompt_template.replace('$GOAL', information.goal)
        prompt = prompt.replace('$PROGRESS', progress_desc)
        prompt = prompt.replace('$STEP', str(information.step))
        prompt = prompt.replace('$ACTION_HISTORY', action_history_desc)
        prompt = prompt.replace('$ASSIGNED_PERSONALITY', information.personality)
        prompt = prompt.replace('$DIALOGUE_HISTORY', dialogue_history_desc)
        prompt = prompt.replace('$TRACE_BELIEF', belief_context)
        available_plans_list = list(information.available_actions_with_distance)
        available_plans_list_refine = list(information.available_actions)
        available_plans_refine = ''.join(
            f"{chr(ord('A') + i)}. {plan}\n"
            for i, plan in enumerate(available_plans_list_refine)
        )
        num = len(available_plans_list_refine)
        if num == 0:
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num, "plan": None})
            return plan, info
        # prompt = prompt.replace('$AVAILABLE_ACTIONS', available_plans)
        prompt = prompt.replace('$ACTIONS_REFINE', available_plans_refine)
        if self.debug:
            print(f"prompt:\n{prompt}")
        chat_prompt = [{"role": "user", "content": prompt}]
        info["raw_prompt"] = prompt
        info["trace_belief_context"] = belief_context
        info["raw_chat_prompt"] = chat_prompt
        outputs, usage = self.generator(chat_prompt if self.chat else prompt, self.sampling_params)
        output = outputs[0]
        self.total_cost += usage
        info["raw_output"] = output
        info["raw_outputs"] = outputs
        info["raw_usage"] = usage
        info["token_usage"] = self.last_usage_metadata
        if self.debug:
            print(f"output:\n{output}")
            print(f"total cost: {self.total_cost}")
        plan,think, message = self.parse_answer(available_plans_list_refine, output)
        if self.debug:
            print(f"plan: {plan}\n")
        # 追加所有prompt构造需要的主要变量到info，便于调试和后处理  $AVAILABLE_ACTIONS_REFINE
        info.update({
            "num_available_actions": num,
            # "prompts": prompt,
            "think": think,
            "message": message,
            "plan": plan,
            "total_cost": self.total_cost,
            "goal_desc": self.goal_desc,
            "progress_desc": progress_desc,
            "action_history_desc": action_history_desc,
            "dialogue_history_desc": dialogue_history_desc,
            "available_plans_list": available_plans_list_refine,
            "available_plans_list_distance": available_plans_list,
            "personality": self.personality,
            "step_now": steps,
            "oppo_name": self.oppo_name,
            "agent_name": self.agent_name
        })
        info["shared_information"] = information.to_dict()
        print("*************outputs******************")
        print(outputs)
        print("*************************************")
        return plan, info

    def build_decision_information(
            self, location, current_room, grabbed_objects, satisfied,
            unchecked_containers, ungrabbed_objects, goal_location_room,
            action_history, dialogue_history, opponent_grabbed_objects,
            opponent_last_room, room_explored=None, steps=0, put_options=None,
            source_event_ids=()):
        """Build once so TRACE and Active Planner see the same legal context."""
        progress = self.progress2text(
            current_room, grabbed_objects, unchecked_containers, ungrabbed_objects,
            goal_location_room, satisfied, opponent_grabbed_objects,
            opponent_last_room, room_explored,
        )
        visits = getattr(self, 'room_visit_counts', {})
        if visits:
            progress += (
                '\nRoom visits (counted on entry): ' + json.dumps(visits) +
                '. Revisit rooms when useful; prefer less-visited rooms when searching '
                'without a stronger location clue.'
            )
        _, _, with_distance, refined, _ = self.get_available_plans(
            location, grabbed_objects, unchecked_containers, ungrabbed_objects,
            None, room_explored, put_options=put_options,
        )
        return DecisionInformation(
            planner_type="active",
            goal=self.goal_desc or "Unknown",
            personality=self.personality or "None",
            current_progress=progress,
            action_history=tuple(action_history[-10:]),
            dialogue_history=tuple(dialogue_history),
            available_actions=tuple(refined),
            available_actions_with_distance=tuple(with_distance),
            step=int(steps),
            source_event_ids=tuple(source_event_ids),
        )
