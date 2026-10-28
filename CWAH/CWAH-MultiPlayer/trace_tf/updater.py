"""Decision-coupled TRACE v2: joint intent, Beta statistics, typed communication."""
import json
import re
from pathlib import Path
from .events import EvidenceLedger
from .model import BeliefState, Evidence, EvidenceSource
from .schema import V2_SCHEMA
from .peer_observation import observe_peer
from .probability import candidates, distribution, align_prior, beta_stats
from LLM.telemetry import measured_generate, record_decision

PROMPT = (Path(__file__).resolve().parent.parent / 'LLM' / 'prompts' / 'trace' / 'updater.txt').read_text(encoding='utf-8')


class HybridUpdater:
    def __init__(self, planner, teammate_id=2):
        self.planner = planner
        self.ledger = EvidenceLedger(target_id=1, peer_id=teammate_id)
        self.state = BeliefState(teammate_id, V2_SCHEMA)
        self.cursor = 0
        self.update_log = []
        self.call_count = 0
        self.intent = None
        self.pending_outgoing = None
        self.requests = {}
        self.commitments = {}

    def _evidence(self, events):
        evidence = []
        for event in events:
            if event.kind == "message":
                source = (EvidenceSource.TEAMMATE_MESSAGE if event.payload["sender"] == 2
                          else EvidenceSource.OWN_MESSAGE)
            elif event.kind == "own_action":
                source = EvidenceSource.OWN_ACTION
            elif event.kind in ("peer_task_progress", "peer_task_outcome"):
                source = EvidenceSource.VISIBLE_PEER_OUTCOME
            elif event.kind == "task_completion" and event.payload.get("completed_by") == "Alice":
                source = EvidenceSource.OWN_ACTION_OUTCOME
            elif event.kind == "task_completion":
                source = EvidenceSource.VISIBLE_OBSERVATION
            else:
                source = EvidenceSource.VISIBLE_OBSERVATION
            evidence.append(Evidence(source, event.step, event.kind, event.event_id))
        return evidence

    def _apply(self, dimension, value, uncertainty, events):
        existing = {e.reference for e in self.state.dimensions[dimension].evidence}
        new = [e for e in self._evidence(events) if e.reference not in existing]
        if new:
            self.state = self.state.with_evidence(dimension, value, 1 - uncertainty,
                                                  new, uncertainty=uncertainty)

    def _prompt_events(self, events):
        result = []
        for event in events:
            payload = event.payload
            if event.kind == "observation":
                peer = observe_peer(payload, self.ledger.target_id, self.ledger.peer_id, event.step)
                facts = []
                if peer['visible']:
                    facts.append('Bob is visible in this partial observation')
                    state = peer['state']
                    if state['room'] is not None:
                        facts.append('Bob is visibly in the ' + str(state['room']['class_name']))
                    facts.extend('Bob visibly holds <{}> ({})'.format(obj['class_name'], obj['id'])
                                 for obj in state['observed_held_objects'])
                else:
                    facts.append('Bob is not currently visible; current room and holdings are unknown')
                payload = {'peer_observation': peer, 'semantic_peer_facts': facts}
            result.append({"event_id": event.event_id, "step": event.step,
                           "kind": event.kind, "payload": payload})
        return result

    def _capability_executions(self):
        """Build task-level Bob execution records from legal progress/completion evidence."""
        executions = {}
        for event in self.ledger.events:
            payload = event.payload
            task = payload.get("task") or {}
            key = task.get("subtask_id")
            if not key:
                continue
            if event.kind == "peer_task_progress":
                item = executions.setdefault(key, {
                    "subtask_id": key, "task": task, "actor": "Bob",
                    "status": "in_progress", "outcome": "unknown",
                    "evidence_ids": [],
                })
                item["evidence_ids"].append(event.event_id)
                item["observed_actions"] = list(dict.fromkeys(
                    item.get("observed_actions", []) + [payload.get("observed_action")]
                ))
            elif event.kind == "task_completion":
                completed_by = payload.get("completed_by", "unknown")
                item = executions.get(key)
                if completed_by == "Bob":
                    item = item or {
                        "subtask_id": key, "task": task, "actor": "Bob",
                        "evidence_ids": [], "observed_actions": [],
                    }
                    executions[key] = item
                    item.update({"status": "completed", "outcome": "success",
                                 "completed_by": "Bob"})
                    item["evidence_ids"].append(event.event_id)
                elif item is not None and item["status"] == "in_progress":
                    item.update({"status": "abandoned", "outcome": "unknown",
                                 "completed_by": completed_by})
                    item["evidence_ids"].append(event.event_id)
        return executions

    def stage_communication(self, metadata):
        if metadata.get('type') == 'probe' and metadata.get('probe_target') not in (
                'intent', 'capability', 'reliability', 'information'):
            raise ValueError('probe requires a valid probe_target')
        if metadata.get('type') == 'delegate' and metadata.get('probe_target') is not None:
            raise ValueError('delegate must omit probe_target')
        self.pending_outgoing = dict(metadata)

    def receive_message(self, event):
        metadata = None
        if event['sender'] == 0 and event['kind'] == 'active':
            metadata, self.pending_outgoing = self.pending_outgoing, None
        reply_to = None
        if event['sender'] == 1 and event['kind'] == 'reply' and event.get('parent_sender') == 0:
            unanswered = [r for r in self.requests.values() if not r.get('reply_event_id')
                          and r['requested_step'] == event['step']]
            if unanswered:
                reply_to = unanswered[-1]['request_event_id']
        item = self.ledger.message(event['step'], event['sender'], event['receiver'],
                                   event['text'], event['kind'], metadata, reply_to)
        if metadata:
            self.requests[item.event_id] = dict(metadata, request_event_id=item.event_id,
                                               requested_step=item.step, status='requested')
        if reply_to:
            self.requests[reply_to]['reply_event_id'] = item.event_id
        return item

    def _accept_replies(self, proposals):
        if not isinstance(proposals, list):
            return
        by_id = self.ledger.by_id()
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            request = self.requests.get(proposal.get('request_event_id'))
            if not request or request['type'] != 'delegate' or request['status'] != 'requested':
                continue
            reply_id = proposal.get('reply_event_id')
            reply = by_id.get(reply_id)
            if (reply_id != request.get('reply_event_id') or not reply
                    or reply.payload.get('sender') != self.ledger.peer_id
                    or reply.payload.get('reply_to') != request['request_event_id']):
                continue
            response = proposal.get('response')
            if response not in ('accepted', 'rejected', 'unclear'):
                continue
            request['status'] = response
            request['classified_step'] = reply.step
            if response != 'accepted':
                continue
            task_id = request['subtask_id']
            created = self.ledger.subtask_created_step(task_id)
            if created is None or reply.step < created:
                request['status'] = 'unclear'
                continue
            # One responsibility per concrete task; repeat requests never inflate counts.
            self.commitments.setdefault(task_id, {
                'commitment_id': 'Bob:' + task_id, 'subtask_id': task_id,
                'request_event_id': request['request_event_id'],
                'acceptance_event_id': reply_id, 'accepted_step': reply.step,
                'started_step': None, 'completed_step': None,
                'status': 'pending', 'outcome': 'unknown',
                'evidence_ids': [request['request_event_id'], reply_id]})

    def _settle_commitments(self, commitments=None):
        for item in (self.commitments if commitments is None else commitments).values():
            if item['status'] != 'pending':
                continue
            accepted_index = int(item['acceptance_event_id'][1:])
            for event in self.ledger.events:
                if int(event.event_id[1:]) <= accepted_index:
                    continue
                if (event.payload.get('task') or {}).get('subtask_id') != item['subtask_id']:
                    continue
                if event.kind == 'peer_task_progress' and item['started_step'] is None:
                    item['started_step'] = event.step
                if event.kind != 'task_completion':
                    continue
                actor = event.payload.get('completed_by', 'unknown')
                item.update(completed_step=event.step, settlement_event_id=event.event_id,
                            status='fulfilled' if actor == 'Bob' else
                                   'violated' if actor == 'Alice' else 'unverifiable',
                            outcome='success' if actor == 'Bob' else
                                    'failure' if actor == 'Alice' else 'unknown')
                item['evidence_ids'].append(event.event_id)
                break

    def rule_statistics(self):
        """Settle a detached rule-only view, without a model call or decision."""
        executions = self._capability_executions()
        capability = beta_stats(sum(e['outcome'] == 'success' for e in executions.values()), 0)
        capability['task_execution_events'] = list(executions.values())
        commitments = json.loads(json.dumps(self.commitments))
        self._settle_commitments(commitments)
        reliability = beta_stats(sum(e['outcome'] == 'success' for e in commitments.values()),
                                 sum(e['outcome'] == 'failure' for e in commitments.values()))
        reliability['commitment_events'] = list(commitments.values())
        reliability['communication_requests'] = json.loads(json.dumps(list(self.requests.values())))
        return {'capability': capability, 'reliability': reliability,
                'through_event_count': len(self.ledger.events),
                'through_environment_step': max((e.step for e in self.ledger.events), default=None),
                'unclassified_delegate_reply_ids': [r['reply_event_id'] for r in self.requests.values()
                    if r['type'] == 'delegate' and r['status'] == 'requested' and r.get('reply_event_id')]}

    def archive_fields(self):
        """No side effects: final rules include the tail; Intent keeps its boundary."""
        last = self.state.to_dict()
        return {'last_decision_belief': last,
                'last_decision_id': self.call_count or None,
                'last_decision_environment_step': self.update_log[-1]['environment_step'] if self.update_log else None,
                'final_rule_statistics': self.rule_statistics(),
                'unconsumed_intent_events': [e.to_dict() for e in self.ledger.since(self.cursor)],
                'intent_event_cursor': self.cursor,
                'final_belief': json.loads(json.dumps(last)),
                'final_belief_semantics': 'legacy_alias_of_last_decision_belief'}

    def _infer(self, information, prior, support, new):
        # Retryable acceptance parsing: linked reply text remains available even
        # after an earlier generation/JSON failure consumed its decision interval.
        by_id = self.ledger.by_id()
        requests = []
        for request in self.requests.values():
            item = dict(request)
            for field in ('request_event_id', 'reply_event_id'):
                event = by_id.get(request.get(field))
                if event is not None:
                    item[field.replace('_id', '_text')] = event.payload['text']
            requests.append(item)
        values = {'$INFORMATION': information, '$PRIOR': prior, '$SUPPORT': support,
                  '$EVENTS': self._prompt_events(new), '$REQUESTS': requests}
        prompt = PROMPT
        for key, value in values.items():
            prompt = prompt.replace(key, json.dumps(value, ensure_ascii=False))
        sampling = self.planner.sampling_params
        params = dict(sampling) if isinstance(sampling, dict) else vars(sampling).copy()
        params['n' if self.planner.chat else 'num_return_sequences'] = 1
        params['temperature'] = 0
        thinking_enabled = getattr(self.planner, 'chat_thinking_updater',
                                   getattr(self.planner, 'chat_enable_thinking', False))
        updater_max_tokens = (getattr(self.planner, 'chat_updater_max_tokens', 2048)
                              if thinking_enabled else 1536)
        params['max_tokens' if self.planner.chat else 'max_new_tokens'] = max(
            updater_max_tokens, 100 * len(support))
        if self.planner.chat:
            params['_trace_call_type'] = 'updater'
        request = [{'role': 'user', 'content': prompt}] if self.planner.chat else prompt
        outputs, cost = measured_generate(self.planner, request, params, 'updater', information,
            sections={'information': values['$INFORMATION'], 'prior_belief': prior,
                      'events': values['$EVENTS'], 'communication_requests': values['$REQUESTS']},
            counts={'new_events': len(new), 'decision_id': self.call_count})
        raw = outputs[0] if outputs else ''
        call = {'model': self.planner.lm_id, 'prompt': prompt, 'raw_output': raw,
                'raw_chat_output': (getattr(self.planner, 'last_raw_chat_outputs', None) or [None])[0],
                'usage': self.planner.last_usage_metadata, 'legacy_cost_estimate': cost}
        try:
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError('updater output must be an object')
        except Exception as exc:
            call['error'] = str(exc)
            return {}, call
        return data, call

    def update(self, information=None):
        # Invoked exclusively at Active/Reactive planner boundaries, never by observe().
        new = self.ledger.since(self.cursor)
        if information is None and not new:
            return self.state, None
        before = self.state.to_dict()
        start_cursor = self.cursor
        information = json.loads(json.dumps(information or {
            'event_ids': [e.event_id for e in self.ledger.events]}))
        information['evidence_event_count'] = len(self.ledger.events)
        self.call_count += 1
        # A decision provenance record also allows same-environment-step decisions
        # with no new observation to update the immutable belief model.
        boundary = self.ledger.add(information.get('step', 0), 'decision', {
            'decision_id': self.call_count, 'planner_type': information.get('planner_type', 'unknown')})
        provenance = new + [boundary]
        tasks = information.get('public_tasks')
        if tasks is None:
            tasks = {t['subtask_id']: t for t in self.ledger._active_tasks}
        support = candidates(tasks)
        previous = self.intent
        prior = align_prior(previous, support)
        self._apply('information', information, 0.0, provenance)
        call = None
        inference = {}
        previous_call_id = (getattr(self.planner, 'last_call_metrics', None) or {}).get('call_id')
        try:
            inference, call = self._infer(information, prior, support, new)
        except Exception as exc:
            call = {'error': '{}: {}'.format(type(exc).__name__, exc), 'model': self.planner.lm_id}
        self._accept_replies(inference.get('delegation_responses', []))
        self._settle_commitments()
        try:
            entry = inference['intent']
            posterior = distribution(entry['task_probs'], entry['skill_given_task'], support)
            intent_status = 'updated'
        except (KeyError, TypeError, ValueError) as exc:
            posterior = prior
            intent_status = 'retained_prior'
            call['validation_error'] = str(exc)
        current_call_id = (getattr(self.planner, 'last_call_metrics', None) or {}).get('call_id')
        if current_call_id and current_call_id != previous_call_id:
            record_decision(self.planner, 'updater', intent_status == 'updated',
                            (call or {}).get('error') or (call or {}).get('validation_error'))
        self.intent = posterior
        probes = [dict(r) for r in self.requests.values() if r['type'] == 'probe']
        intent_value = dict(posterior, update_status=intent_status, probe_context=probes)
        ui = posterior['normalized_entropy']
        self._apply('intent', intent_value, ui, provenance)
        executions = self._capability_executions()
        c = beta_stats(sum(e['outcome'] == 'success' for e in executions.values()), 0)
        c['task_execution_events'] = list(executions.values())
        self._apply('capability', c, c['normalized_variance'], provenance)
        r = beta_stats(sum(e['outcome'] == 'success' for e in self.commitments.values()),
                       sum(e['outcome'] == 'failure' for e in self.commitments.values()))
        r['commitment_events'] = list(self.commitments.values())
        r['communication_requests'] = list(self.requests.values())
        self._apply('reliability', r, r['normalized_variance'], provenance)
        uncertainties = {'intent': ui, 'capability': c['normalized_variance'],
                         'reliability': r['normalized_variance']}
        self._apply('uncertainty', {'per_dimension': uncertainties,
                    'vector_order': ['intent', 'capability', 'reliability'],
                    'vector': [uncertainties[k] for k in ('intent', 'capability', 'reliability')]},
                    max(uncertainties.values()), provenance)
        self.cursor = len(self.ledger.events)
        record = {'decision_id': self.call_count, 'environment_step': information.get('step', 0),
                  'planner_type': information.get('planner_type', 'unknown'),
                  'interval_start_cursor': start_cursor, 'interval_end_cursor': self.cursor,
                  'new_event_ids': [e.event_id for e in new],
                  'legal_interval_events': [e.to_dict() for e in new],
                  'shared_information': information, 'previous_posterior': previous,
                  'aligned_prior': prior, 'posterior': posterior,
                  'intent_update_status': intent_status, 'support': support,
                  'belief_before': before, 'belief_after': self.state.to_dict(), 'llm_call': call}
        self.update_log.append(record)
        return self.state, record
