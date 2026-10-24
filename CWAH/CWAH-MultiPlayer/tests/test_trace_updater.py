"""TRACE-v2 protocol regression tests, no Unity, GPU or API calls."""
import copy
import json
import math
import unittest
import tempfile
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from trace_tf.updater import HybridUpdater
from trace_tf.probability import candidates, uniform, distribution, align_prior, beta_stats
from trace_tf.policy import render_policy_context, render_reply_context

TASK = {'subtask_id': 'S1', 'object_id': 12, 'created_step': 0, 'status': 'active',
        'object_class': 'apple', 'relation': 'on', 'target_id': 20, 'target_class': 'table'}

def obs(bob=True, holds=False):
    return {'nodes': [{'id': i, 'class_name': name, 'category': 'Objects'}
                      for i, name in [(1, 'character'), (12, 'apple')] + ([(2, 'character')] if bob else [])],
            'edges': [{'from_id': 2, 'to_id': 12, 'relation_type': 'HOLDS_RH'}] if holds else [],
            'hidden_profile': 'SECRET_PROFILE', 'peer_private_plan': 'SECRET_PLAN'}

def info(step=0, kind='active', tasks=None):
    return {'step': step, 'planner_type': kind, 'public_tasks': {'S1': TASK} if tasks is None else tasks}

def message(sender, text, kind='active', step=0, **extra):
    return dict(step=step, sender=sender, receiver=1-sender, text=text, kind=kind, **extra)

class TraceUpdaterTests(unittest.TestCase):
    def make(self):
        planner = SimpleNamespace(sampling_params={'n': 1, 'max_tokens': 1024}, chat=True,
            lm_id='test', last_usage_metadata={}, generator=Mock())
        updater = HybridUpdater(planner)
        updater.ledger.set_active_tasks([TASK])
        self.answer(updater, planner)
        return updater, planner

    def answer(self, updater, planner, responses=(), tasks=None):
        support = candidates({'S1': TASK} if tasks is None else tasks)
        dist = uniform(support)
        planner.generator.return_value = ([json.dumps({'intent': {
            'task_probs': dist['task_probs'], 'skill_given_task': dist['skill_given_task']},
            'delegation_responses': list(responses)})], 0)

    def delegate(self, updater, kind='delegate'):
        updater.stage_communication({'type': kind, 'subtask_id': 'S1' if kind == 'delegate' else None})
        request = updater.receive_message(message(0, 'Please complete S1' if kind == 'delegate' else 'What task?'))
        reply = updater.receive_message(message(1, 'I accept S1', 'reply', parent_sender=0))
        return {'request_event_id': request.event_id, 'reply_event_id': reply.event_id, 'response': 'accepted'}

    def test_beta_prior_success_failure_and_normalized_variance(self):
        self.assertEqual(beta_stats(0, 0)['mean'], 0.5)
        self.assertEqual(beta_stats(0, 0)['normalized_variance'], 1)
        self.assertAlmostEqual(beta_stats(1, 0)['mean'], 2/3)
        self.assertAlmostEqual(beta_stats(1, 0)['normalized_variance'], 2/3)
        self.assertAlmostEqual(beta_stats(0, 1)['mean'], 1/3)
        self.assertAlmostEqual(beta_stats(20, 20)['normalized_variance'], 3/43)

    def test_joint_entropy_and_exact_support_validation(self):
        support = candidates({'S1': TASK})
        value = uniform(support)
        self.assertAlmostEqual(sum(x['probability'] for x in value['joint']), 1)
        self.assertAlmostEqual(value['normalized_entropy'], 1)
        self.assertEqual(value['support_size'], 10)
        for invalid in (-1, True, float('nan'), float('inf')):
            bad = copy.deepcopy(value['skill_given_task'])
            bad['S1']['gograb'] = invalid
            with self.assertRaises(ValueError):
                distribution(value['task_probs'], bad, support)
        bad = dict(value['task_probs'], hidden_task=1)
        with self.assertRaises(ValueError):
            distribution(bad, value['skill_given_task'], support)

    def test_decision_boundaries_not_environment_observations(self):
        u, p = self.make()
        u.ledger.observe(0, obs())
        u.update(info())
        u.ledger.observe(1, obs())
        u.ledger.observe(2, obs(holds=True))
        self.assertEqual(p.generator.call_count, 1)
        _, record = u.update(info(2, 'reactive'))
        self.assertEqual(record['decision_id'], 2)
        self.assertEqual(record['planner_type'], 'reactive')
        self.assertEqual([e['step'] for e in record['legal_interval_events'] if e['kind'] == 'observation'], [1, 2])
        u.update(info(2, 'active'))
        self.assertEqual(p.generator.call_count, 3)
        self.assertEqual(u.update_log[-1]['new_event_ids'], [])
        u.update()
        self.assertEqual(p.generator.call_count, 3)

    def test_private_fields_never_in_prompt_and_preference_disabled(self):
        u, p = self.make()
        u.ledger.observe(0, obs())
        state, record = u.update(info())
        self.assertNotIn('SECRET', record['llm_call']['prompt'])
        self.assertEqual(state.dimensions['preference'].status, 'unknown')
        self.assertEqual(state.dimensions['uncertainty'].value['vector_order'], ['intent', 'capability', 'reliability'])
        self.assertNotIn('overall', state.dimensions['uncertainty'].value)
        self.assertIn('probe_intent', render_policy_context(state, u.ledger))
        self.assertNotIn('<communication_type>', render_reply_context(state, u.ledger))

    def test_invalid_llm_output_retains_aligned_prior(self):
        u, p = self.make()
        u.ledger.observe(0, obs())
        u.update(info())
        old = copy.deepcopy(u.intent)
        p.generator.return_value = (['not json'], 0)
        _, record = u.update(info(1))
        self.assertEqual(record['intent_update_status'], 'retained_prior')
        self.assertEqual(u.intent['joint'], old['joint'])
        self.assertEqual(record['llm_call']['raw_output'], 'not json')

    def test_task_retirement_and_new_task_have_valid_mass(self):
        old = uniform(candidates({'S1': TASK}))
        support = candidates({'S2': dict(TASK, subtask_id='S2')})
        aligned = align_prior(old, support)
        self.assertNotIn('S1', aligned['task_probs'])
        self.assertGreater(aligned['task_probs']['S2'], 0)
        self.assertAlmostEqual(sum(x['probability'] for x in aligned['joint']), 1)

    def test_delegation_settles_only_public_completion_and_counts_once(self):
        for actor, expected in [('Bob', 2/3), ('Alice', 1/3), ('unknown', 0.5)]:
            with self.subTest(actor=actor):
                u, p = self.make()
                accepted = self.delegate(u)
                self.answer(u, p, [accepted])
                u.update(info())
                self.assertEqual(u.commitments['S1']['status'], 'pending')
                u.ledger.add(1, 'peer_task_progress', {'task': TASK, 'observed_action': 'grab'})
                u.update(info(1))
                self.assertEqual(u.commitments['S1']['status'], 'pending')
                u.ledger.task_completion(2, TASK, actor)
                state, _ = u.update(info(2))
                self.assertAlmostEqual(state.dimensions['reliability'].value['mean'], expected)
                stats = state.dimensions['reliability'].value
                u.update(info(3))
                self.assertEqual(u.state.dimensions['reliability'].value, stats)

    def test_probe_unsolicited_claim_and_unlinked_reply_do_not_create_commitments(self):
        u, p = self.make()
        proposal = self.delegate(u, 'probe_intent')
        self.answer(u, p, [proposal])
        state, _ = u.update(info())
        self.assertEqual(u.commitments, {})
        self.assertEqual(len(state.dimensions['intent'].value['probe_context']), 1)
        u.receive_message(message(1, 'I will complete S1'))
        u.update(info(1))
        self.assertEqual(u.commitments, {})
        u2, p2 = self.make()
        proposal = self.delegate(u2)
        proposal['reply_event_id'] = proposal['request_event_id']
        self.answer(u2, p2, [proposal])
        u2.update(info())
        self.assertEqual(u2.commitments, {})

    def test_reject_and_unclear_do_not_count_as_failures(self):
        for response in ('rejected', 'unclear'):
            u, p = self.make()
            proposal = self.delegate(u)
            proposal['response'] = response
            self.answer(u, p, [proposal])
            u.update(info())
            u.ledger.task_completion(1, TASK, 'Alice')
            state, _ = u.update(info(1))
            self.assertEqual(state.dimensions['reliability'].value['failures'], 0)

    def test_capability_alice_completion_unknown_bob_completion_success(self):
        for actor, expected in [('Alice', 0.5), ('Bob', 2/3)]:
            u, p = self.make()
            u.ledger.observe(0, obs())
            u.ledger.observe(1, obs(holds=True))
            u.ledger.task_completion(2, TASK, actor)
            state, _ = u.update(info(2))
            self.assertAlmostEqual(state.dimensions['capability'].value['mean'], expected)
            self.assertEqual(state.dimensions['capability'].value['failures'], 0)

    def test_visibility_reentry_is_not_fresh_grab(self):
        u, _ = self.make()
        u.ledger.observe(0, obs(bob=False))
        u.ledger.observe(1, obs(holds=True))
        self.assertFalse(any(e.kind == 'peer_task_progress' for e in u.ledger.events))

    def test_explicit_terminal_failure_and_pending_episode_are_distinct(self):
        u, p = self.make()
        u.ledger.add(0, 'peer_task_outcome', {'task': TASK, 'verified': True,
                     'terminal': True, 'success': False})
        state, _ = u.update(info())
        self.assertEqual(state.dimensions['capability'].value['failures'], 0)
        self.assertAlmostEqual(state.dimensions['capability'].value['mean'], 0.5)
        u, p = self.make()
        proposal = self.delegate(u)
        self.answer(u, p, [proposal])
        u.update(info())
        state, _ = u.update(info(1000))
        self.assertEqual(u.commitments['S1']['status'], 'pending')
        self.assertEqual(state.dimensions['reliability'].value['failures'], 0)

    def test_repeated_accepted_delegation_does_not_duplicate_responsibility(self):
        u, p = self.make()
        proposal = self.delegate(u)
        self.answer(u, p, [proposal])
        u.update(info())
        proposal = self.delegate(u)
        self.answer(u, p, [proposal])
        u.update(info())
        self.assertEqual(len(u.commitments), 1)
        u.ledger.task_completion(1, TASK, 'Bob')
        state, _ = u.update(info(1))
        self.assertEqual(state.dimensions['reliability'].value['successes'], 1)

    def test_late_unrelated_reply_cannot_attach_to_old_failed_delivery(self):
        u, _ = self.make()
        u.stage_communication({'type': 'delegate', 'subtask_id': 'S1'})
        u.receive_message(message(0, 'Please complete S1', step=0))
        reply = u.receive_message(message(1, 'Yes', 'reply', step=5, parent_sender=0))
        self.assertIsNone(reply.payload['reply_to'])

    def test_joint_distribution_can_express_task_skill_dependency(self):
        support = candidates({'S1': TASK, 'S2': dict(TASK, subtask_id='S2')})
        rows = {t: dict.fromkeys(skills, 0) for t, skills in support.items()}
        rows['S1']['gograb'] = 1
        rows['S2']['goput'] = 1
        rows['none']['idle'] = 1
        dist = distribution({'S1': 0.6, 'S2': 0.4, 'none': 0}, rows, support)
        joint = {(x['subtask_id'], x['skill']): x['probability'] for x in dist['joint']}
        self.assertEqual(joint['S1', 'goput'], 0)
        self.assertEqual(joint['S1', 'gograb'], 0.6)
        self.assertAlmostEqual(dist['conditional_skill_entropy'], 0)



    def test_acceptance_can_recover_after_invalid_generation(self):
        u, p = self.make()
        proposal = self.delegate(u)
        p.generator.return_value = (['bad json'], 0)
        u.update(info())
        self.assertFalse(u.commitments)
        self.answer(u, p, [proposal])
        _, record = u.update(info(1))
        self.assertIn('I accept S1', record['llm_call']['prompt'])
        self.assertEqual(u.commitments['S1']['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
