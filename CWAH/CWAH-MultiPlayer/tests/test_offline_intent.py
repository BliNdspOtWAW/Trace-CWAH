"""Offline inference protocol: legal boundaries, full support, numerical edge cases."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from offline_intent.dataset import build_intervals, digest, prepare
from offline_intent.inference import evaluate, infer, prompt_for
from offline_intent.metrics import bayes, jsd, parse_scores
from offline_intent.report import report


def fixture():
    candidates = [{'subtask_id': 'S1', 'skill': 'gograb'}, {'subtask_id': 'none', 'skill': 'idle'}]
    events = [
        {'event_id': 'e0', 'step': 0, 'kind': 'observation', 'payload': {'nodes': [], 'edges': []}},
        {'event_id': 'e1', 'step': 0, 'kind': 'decision', 'payload': {'decision_id': 1}},
        {'event_id': 'e2', 'step': 0, 'kind': 'message', 'payload': {'text': 'I will do S1', 'sender': 2, 'receiver': 1}},
        {'event_id': 'e3', 'step': 0, 'kind': 'decision', 'payload': {'decision_id': 2}},
        {'event_id': 'e4', 'step': 1, 'kind': 'message', 'payload': {'text': 'FUTURE_DO_NOT_READ'}},
    ]
    def update(i, start, end):
        return {'decision_id': i, 'interval_start_cursor': start, 'interval_end_cursor': end,
                'new_event_ids': [e['event_id'] for e in events[start:end - 1]],
                'environment_step': 0, 'planner_type': 'active' if i == 1 else 'reactive',
                'support': {'S1': ['gograb'], 'none': ['idle']},
                'posterior': {'joint': [dict(c, probability=p) for c, p in zip(candidates, [0.75, 0.25])]},
                'shared_information': {'goal': 'move apple', 'belief_before': 'PRIOR_SECRET', 'full_graph': 'PRIVATE'},
                'llm_call': {'prompt': 'PRIVATE_PROMPT'}}
    return {'evidence_events': events, 'belief_updates': [update(1, 0, 2), update(2, 2, 4)]}


def response(values, field='likelihoods'):
    key = 'score'
    candidates = [{'subtask_id': 'S1', 'skill': 'gograb'}, {'subtask_id': 'none', 'skill': 'idle'}]
    return {'raw_output': json.dumps({field: [dict(c, **{key: v}) for c, v in zip(candidates, values)]}),
            'finish_reason': 'stop'}


class OfflineIntentTests(unittest.TestCase):
    def test_jsd_symmetry_zero_disjoint_and_zero_cells(self):
        self.assertEqual(jsd([1, 0], [1, 0]), 0)
        self.assertEqual(jsd([1, 0], [0, 1]), 1)
        self.assertAlmostEqual(jsd([0.7, 0.3], [0.2, 0.8]), jsd([0.2, 0.8], [0.7, 0.3]))
        self.assertEqual(bayes([0, 1], [5, 3]), [0, 1])
        self.assertAlmostEqual(bayes([0.75, 0.25], [1, 3])[0], 0.5)
        for a, b in [([1, 0], [0, 1]), ([0.5, 0.5], [0, 0])]:
            with self.assertRaises(ValueError):
                bayes(a, b)

    def test_identity_matching_and_invalid_scores(self):
        row = next(build_intervals(fixture(), 'episode'))
        raw = json.loads(response([1, 3])['raw_output'])
        raw['likelihoods'].reverse()
        values, _, _ = parse_scores(json.dumps(raw), row['common']['candidates'], 'likelihoods')
        self.assertEqual(values, [1, 3])
        for values in ([0, 0], [-1, 2], [True, 2], [float('nan'), 1]):
            with self.assertRaises(ValueError):
                parse_scores(response(values)['raw_output'], row['common']['candidates'], 'likelihoods')
        raw['likelihoods'][1] = raw['likelihoods'][0]
        with self.assertRaises(ValueError):
            parse_scores(json.dumps(raw), row['common']['candidates'], 'likelihoods')
        raw, normalized, total = parse_scores(response([3, 7], 'posterior_scores')['raw_output'],
                                              row['common']['candidates'], 'posterior_scores')
        self.assertEqual(raw, [3, 7])
        self.assertEqual(total, 10)
        self.assertEqual(normalized, [0.3, 0.7])

    def test_same_step_boundaries_and_future_and_prior_exclusion(self):
        row = next(build_intervals(fixture(), 'episode'))
        self.assertEqual([e['event_id'] for e in row['common']['new_evidence']], ['e2'])
        self.assertEqual(row['start_step'], row['end_step'])
        a, b = prompt_for(row, 'A'), prompt_for(row, 'B')
        for prompt in (a, b):
            self.assertNotIn('FUTURE_DO_NOT_READ', prompt)
            self.assertNotIn('PRIVATE', prompt)
            self.assertNotIn('PRIOR_SECRET', prompt)
            self.assertIn('may switch', prompt)
        payload_a = json.loads(a.split('\nINPUT:\n')[1])
        payload_b = json.loads(b.split('\nINPUT:\n')[1])
        self.assertNotIn('prior', payload_a)
        self.assertEqual(payload_b.pop('prior')[0]['probability'], 0.75)
        self.assertEqual(payload_a, payload_b)

    def test_B_normalizes_scores_without_multiplying_prior_again(self):
        row = next(build_intervals(fixture(), 'episode'))
        outputs = iter([response([1, 1]), response([1, 3], 'posterior_scores')])
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate(row, lambda _: next(outputs), Path(folder)/'calls.jsonl', retries=0)
        self.assertEqual(result['status'], 'ok')
        for actual, expected in zip(result['branches']['A']['posterior'], row['prior']):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result['branches']['B']['posterior'], [0.25, 0.75])
        self.assertEqual(result['branches']['B']['raw_sum'], 4)
        self.assertEqual(result['branches']['B']['normalization'], 'score_only')
        self.assertGreater(result['jsd_bits'], 0)

    def test_corrupt_cursor_rejected(self):
        trace = fixture()
        trace['belief_updates'][1]['interval_start_cursor'] = 1
        with self.assertRaises(ValueError):
            list(build_intervals(trace, 'episode'))

    def test_all_observation_frames_preserved_and_empty_intervals_allowed(self):
        trace = fixture()
        observation = {'event_id': 'extra', 'step': 0, 'kind': 'observation',
                       'payload': {'nodes': [{'id': 2, 'class_name': 'character',
                                  'obj_transform': {'position': [1, 0, 0]}}], 'edges': []}}
        trace['evidence_events'].insert(3, observation)
        trace['belief_updates'][1]['interval_end_cursor'] = 5
        trace['belief_updates'][1]['new_event_ids'].append('extra')
        row = next(build_intervals(trace, 'episode'))
        self.assertEqual([e['event_id'] for e in row['common']['new_evidence']], ['e2', 'extra'])
        trace = fixture()
        trace['evidence_events'].pop(2)
        trace['belief_updates'][1].update(interval_end_cursor=3, new_event_ids=[])
        row = next(build_intervals(trace, 'episode'))
        self.assertEqual(row['common']['new_evidence'], [])

    def test_report_pending_and_failed_never_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'trajectory.json', root/'analysis'
            source.write_text(json.dumps(fixture()))
            prepare(source, output)
            summary = report(output)
            self.assertEqual(summary['pending'], 1)
            self.assertIsNone(summary['mean_jsd_bits'])
            infer(output, 'test', retries=0, client=lambda _: response([0, 0]))
            summary = report(output)
            self.assertEqual(summary['failed'], 1)
            self.assertEqual(summary['failure_rate_among_attempted'], 1)
            self.assertIsNone(summary['mean_jsd_bits'])

    def test_full_22_support_never_topk(self):
        trace = fixture()
        update = trace['belief_updates'][0]
        update['support'] = {t: ['goexplore', 'gocheck', 'gograb', 'goput', 'send_message', 'idle'] for t in ('S1', 'S2', 'S3')}
        update['support']['none'] = ['goexplore', 'gocheck', 'send_message', 'idle']
        update['posterior']['joint'] = [dict(subtask_id=t, skill=s, probability=1/22)
                                        for t, skills in update['support'].items() for s in skills]
        row = next(build_intervals(trace, 'episode'))
        self.assertEqual(len(row['prior']), 22)
        self.assertEqual(len(json.loads(prompt_for(row, 'A').split('\nINPUT:\n')[1])['candidates']), 22)

    def test_failed_branch_never_becomes_zero_jsd(self):
        row = next(build_intervals(fixture(), 'episode'))
        outputs = iter([response([0, 0]), response([0.5, 0.5], 'posterior_scores')])
        with tempfile.TemporaryDirectory() as folder:
            result = evaluate(row, lambda _: next(outputs), Path(folder)/'calls.jsonl', retries=0)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['jsd_bits'])

    def test_prepare_infer_resume_and_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'trajectory.json', root/'analysis'
            source.write_text(json.dumps(fixture()))
            manifest = prepare(source, output)
            self.assertEqual(manifest['terminal_tails'][0]['unpaired_terminal_events'], 1)
            calls = []
            def client(prompt):
                calls.append(prompt)
                return response([1, 3]) if 'relative LIKELIHOOD' in prompt else response([0.5, 0.5], 'posterior_scores')
            self.assertEqual(infer(output, 'test', client=client), 1)
            self.assertEqual(infer(output, 'test', client=client), 0)
            self.assertEqual(len(calls), 2)
            summary = report(output)
            self.assertEqual(summary['valid'], 1)
            self.assertAlmostEqual(summary['mean_jsd_bits'], 0)
            self.assertTrue((output/'posterior_agreement.png').is_file())
            with self.assertRaises(ValueError):
                infer(output, 'changed-model', client=client)

    def test_compact_layout_and_legacy_migration_preserve_records(self):
        from offline_intent.storage import data_directory
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'trajectory.json', root/'analysis'
            source.write_text(json.dumps(fixture()))
            prepare(source, output)
            details = output/'details'
            for name in ('manifest.json', 'intervals.jsonl'):
                (details/name).rename(output/name)
            legacy_bytes = (output/'intervals.jsonl').read_bytes()
            for name in ('jsd_curve.png', 'jsd_curve.pdf', 'metrics.csv'):
                (output/name).write_text('old generated export')
            report(output)
            self.assertEqual((details/'intervals.jsonl').read_bytes(), legacy_bytes)
            self.assertEqual({p.name for p in output.iterdir()}, {'details', 'posterior_agreement.png'})
            (output/'manifest.json').write_text('conflict')
            with self.assertRaises(ValueError):
                data_directory(output)
            self.assertEqual((output/'manifest.json').read_text(), 'conflict')

    def test_resume_keeps_successful_branch_after_other_branch_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root/'trajectory.json', root/'analysis'
            source.write_text(json.dumps(fixture()))
            prepare(source, output)
            answers = iter([response([1, 3]), response([0, 0], 'posterior_scores')])
            infer(output, 'test', retries=0, client=lambda _: next(answers))
            calls = []
            def client(prompt):
                calls.append(prompt)
                return response([0.5, 0.5], 'posterior_scores')
            infer(output, 'test', retries=0, client=client)
            self.assertEqual(len(calls), 1)
            self.assertIn('FULL JOINT posterior', calls[0])


if __name__ == '__main__':
    unittest.main()
