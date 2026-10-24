"""Two independent calls with identical legal context and fixed start support."""
import datetime
import json
import math
import os
import time
import uuid
from pathlib import Path
from .dataset import VERSION, digest
from .metrics import bayes, entropy, jsd, parse_scores
from .storage import data_directory

RULES = '''You are performing OFFLINE retrospective inference about Bob.
The hidden state is (subtask, skill) at the START decision, never the end decision.
Bob may switch task or skill DURING the interval. Evaluate the complete subsequent
evidence allowing these switches; do not assume the start Intent persists throughout.
Use only the supplied Alice-legal history and ordered new evidence. Do not invent
hidden actions. Observation appearance/disappearance is visibility, not proof of
grab/place/movement. Messages are claims, not verified execution. Alice's selected
actions are not necessarily successful actions. Repeated views are not independent
new facts. Treat derived outcomes and their underlying observations as the same
evidence, not separate independent confirmations. Full candidate support is fixed
at START even if tasks complete or new tasks appear later. Return every supplied
candidate exactly once by subtask_id and skill. Do not include prose or evidence
fields in the JSON. These are inference distributions, not ground truth.
'''


def prompt_for(row, branch):
    if branch == 'A':
        instruction = '''For EACH starting candidate, assess how compatible the ENTIRE
subsequent evidence is with that starting state, conditional on the history.
Output relative LIKELIHOOD/compatibility scores, NOT posterior probabilities.
Do not estimate or multiply by a prior. Scores need not sum to one and are not
claimed to be calibrated likelihoods. Observationally indistinguishable candidates
should receive equal scores. With no informative new evidence, use equal scores.
Output exactly {"likelihoods":[{"subtask_id":"...","skill":"...","score":1.0},...]}.
Scores must be finite and nonnegative with positive total mass.
'''
        payload = row['common']
    elif branch == 'B':
        instruction = '''Given the original prior and the SAME history and subsequent
evidence, re-estimate Bob's Intent at the START decision. Do not predict his end
Intent. Use the supplied prior as the starting belief; do not count the starting
history a second time as independent new evidence. With no informative new
evidence, output scores proportional to the prior. Output the FULL JOINT posterior
SCORES incorporating the supplied prior and new evidence, NOT likelihood scores.
The program will NORMALIZE your scores directly; it will NOT multiply by prior
again. Do not normalize or perform probability-sum arithmetic yourself.
Output exactly {"posterior_scores":[{"subtask_id":"...","skill":"...","score":1.0},...]}.
Scores must be finite and nonnegative, with positive total mass. They need not
sum to one. Preserve relative posterior plausibility across ALL candidates.
'''
        payload = dict(row['common'], prior=[dict(c, probability=p)
                       for c, p in zip(row['common']['candidates'], row['prior'])])
    else:
        raise ValueError('branch must be A or B')
    return RULES + instruction + '\nINPUT:\n' + json.dumps(payload, ensure_ascii=False, allow_nan=False)


class ChatClient:
    """Independent stateless Chat endpoint; no online planner state or GPU model."""
    def __init__(self, model, temperature=0, max_tokens=4096, timeout=180):
        self.model, self.temperature, self.max_tokens, self.timeout = model, temperature, max_tokens, timeout
        self.base = os.environ.get('CWAH_OFFLINE_API_BASE') or os.environ.get('CWAH_AGENT_API_BASE')
        self.key = os.environ.get('CWAH_OFFLINE_API_KEY') or os.environ.get('CWAH_AGENT_API_KEY')
        if not self.base or not self.key:
            raise ValueError('load API environment or set CWAH_OFFLINE_API_BASE/API_KEY')
        wire = os.environ.get('CWAH_AGENT_WIRE_API', 'chat')
        if wire != 'chat' and not os.environ.get('CWAH_OFFLINE_API_BASE'):
            raise ValueError('this offline client supports Chat Completions; use a Chat endpoint')

    def __call__(self, prompt):
        import requests
        response = requests.post(self.base.rstrip('/') + '/chat/completions',
            headers={'Authorization': 'Bearer ' + self.key},
            json={'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                  'temperature': self.temperature, 'max_tokens': self.max_tokens},
            timeout=(15, self.timeout))
        if response.status_code >= 400:
            # Do not dump server response bodies, request headers or credentials.
            raise RuntimeError('offline API HTTP ' + str(response.status_code))
        data = response.json()
        choice = data['choices'][0]
        return {'raw_output': choice['message']['content'], 'usage': data.get('usage'),
                'finish_reason': choice.get('finish_reason'), 'response_model': data.get('model')}


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def evaluate(row, client, calls_path, retries=2, cached=None, on_success=None):
    result = {'interval_id': row['interval_id'], 'input_hash': row['input_hash'],
              'common_hash': row['common_hash'], 'prior': row['prior'],
              'candidates': row['common']['candidates'], 'new_event_count': row['new_event_count'],
              'start_step': row['start_step'], 'end_step': row['end_step'], 'branches': {}}
    for branch, field in (('A', 'likelihoods'), ('B', 'posterior_scores')):
        if cached and branch in cached:
            result['branches'][branch] = cached[branch]
            continue
        base_prompt = prompt_for(row, branch)
        error = None
        for attempt in range(retries + 1):
            prompt = base_prompt + ('\nYour previous output was invalid: ' + error + '\nCorrect the output schema.' if error else '')
            call = {'call_id': uuid.uuid4().hex, 'interval_id': row['interval_id'],
                    'input_hash': row['input_hash'], 'common_hash': row['common_hash'],
                    'branch': branch, 'attempt': attempt + 1, 'prompt': prompt,
                    'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            started = time.monotonic()
            try:
                response = client(prompt)
                call.update(response)
                if response.get('finish_reason') == 'length':
                    raise ValueError('model output truncated by token budget')
                raw, normalized, total = parse_scores(response['raw_output'], row['common']['candidates'], field)
                posterior = bayes(row['prior'], raw) if branch == 'A' else normalized
                saved = {'status': 'ok', 'call_id': call['call_id'], 'raw_values': raw,
                         'raw_sum': total, 'posterior': posterior,
                         'score_semantics': 'relative_likelihood' if branch == 'A' else 'relative_posterior',
                         'normalization': 'prior_times_score' if branch == 'A' else 'score_only'}
                result['branches'][branch] = saved
                call['status'] = 'ok'
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
                call.update(status='failed', error=error)
            call['elapsed_seconds'] = time.monotonic() - started
            append(calls_path, call)
            if call['status'] == 'ok':
                if on_success:
                    on_success(branch, saved)
                break
        else:
            result['branches'][branch] = {'status': 'failed', 'error': error}
    a, b = (result['branches'][key] for key in ('A', 'B'))
    if a['status'] == b['status'] == 'ok':
        result.update(status='ok', jsd_bits=jsd(a['posterior'], b['posterior']),
                      entropy_A_bits=entropy(a['posterior']), entropy_B_bits=entropy(b['posterior']),
                      jsd_A_prior=jsd(a['posterior'], row['prior']),
                      jsd_B_prior=jsd(b['posterior'], row['prior']),
                      B_mass_on_zero_prior=sum(q for p, q in zip(row['prior'], b['posterior']) if p == 0))
    else:
        result.update(status='failed', jsd_bits=None)
    return result


def infer(directory, model, temperature=0, max_tokens=4096, retries=2, limit=None, client=None):
    directory = data_directory(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['protocol'] != VERSION:
        raise ValueError('protocol version mismatch')
    settings = {'protocol': VERSION, 'model': model, 'temperature': temperature,
                'max_tokens': max_tokens, 'retries': retries, 'prompt_hash': digest(RULES + prompt_for(
                    {'common': {'candidates': []}, 'prior': []}, 'A') + prompt_for(
                    {'common': {'candidates': []}, 'prior': []}, 'B')),
                'endpoint': os.environ.get('CWAH_OFFLINE_API_BASE') or os.environ.get('CWAH_AGENT_API_BASE')}
    settings_path = directory / 'inference_config.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError('inference settings changed; prepare a new analysis directory')
    rows = read_rows(directory / 'intervals.jsonl')
    for row in rows:
        saved_hash = row['input_hash']
        if digest({k: v for k, v in row.items() if k != 'input_hash'}) != saved_hash or digest(row['common']) != row['common_hash']:
            raise ValueError('frozen interval hash mismatch')
    if not settings_path.exists():
        settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    results_path, cache_path = directory / 'results.jsonl', directory / 'branch_cache.jsonl'
    old = read_rows(results_path) if results_path.exists() else []
    latest = {r['interval_id']: r for r in old}
    cache = {}
    for item in read_rows(cache_path) if cache_path.exists() else []:
        cache[(item['input_hash'], item['branch'])] = item['result']
    if client is None:
        client = ChatClient(model, temperature, max_tokens)
    count = 0
    for row in rows:
        prior_result = latest.get(row['interval_id'])
        if prior_result and prior_result['input_hash'] != row['input_hash']:
            raise ValueError('result input hash mismatch')
        if prior_result and prior_result['status'] == 'ok':
            continue
        if limit is not None and count >= limit:
            break
        cached = {b: cache[(row['input_hash'], b)] for b in ('A', 'B') if (row['input_hash'], b) in cache}
        def save(branch, result):
            append(cache_path, {'input_hash': row['input_hash'], 'branch': branch, 'result': result})
        result = evaluate(row, client, directory / 'calls.jsonl', retries, cached, save)
        append(results_path, result)
        print(row['interval_id'], result['status'], result['jsd_bits'], flush=True)
        count += 1
    return count
