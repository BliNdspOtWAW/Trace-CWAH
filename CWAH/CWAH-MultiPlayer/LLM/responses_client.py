"""Small Responses transport for the existing OpenAI 0.x environment."""
import time
import requests


def generate_responses(base_url, api_key, model, messages, params, effort='high',
                       max_retries=20, output_budget=8192):
    if not api_key:
        raise ValueError('Responses API key is empty; fill PROXY_OPENAI_API_KEY first')
    outputs = []
    totals = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    totals.update(cached_input_tokens=None, reasoning_tokens=None, request_count=0,
                  retry_count=0, token_usage_available=True)
    for _ in range(params.get('n', 1)):
        body = {'model': model, 'input': messages, 'store': False,
                'max_output_tokens': max(output_budget, params.get('max_tokens', 1024)),
                'reasoning': {'effort': effort}}
        for attempt in range(max_retries + 1):
            totals['request_count'] += 1
            if attempt:
                totals['retry_count'] += 1
            try:
                response = requests.post(
                    base_url.rstrip('/') + '/responses', json=body,
                    headers={'Authorization': 'Bearer ' + api_key}, timeout=(15, 180),
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == max_retries:
                    raise RuntimeError('Responses connection/timeout retry limit reached') from None
            else:
                if response.status_code < 400:
                    break
                if response.status_code not in (408, 429) and response.status_code < 500:
                    raise RuntimeError('Responses API HTTP {}'.format(response.status_code))
                if attempt == max_retries:
                    raise RuntimeError('Responses API HTTP {}; retries exhausted'.format(response.status_code))
            time.sleep(min(2 ** attempt, 30))
        data = response.json()
        text = ''.join(
            part.get('text', '') for item in data.get('output', [])
            if item.get('type') == 'message' and item.get('role') == 'assistant'
            for part in item.get('content', []) if part.get('type') == 'output_text'
        )
        if data.get('error') or data.get('status') in ('failed', 'incomplete') or not text.strip():
            raise RuntimeError('Responses returned no complete assistant text (status={})'.format(data.get('status')))
        outputs.append(text)
        usage = data.get('usage') or {}
        if not all(isinstance(usage.get(key), int) for key in ('input_tokens', 'output_tokens', 'total_tokens')):
            totals['token_usage_available'] = False
        for target, detail in [('cached_input_tokens', 'input_tokens_details'),
                               ('reasoning_tokens', 'output_tokens_details')]:
            value = (usage.get(detail) or {}).get('cached_tokens' if target == 'cached_input_tokens' else 'reasoning_tokens')
            if isinstance(value, int):
                totals[target] = (totals[target] or 0) + value
        for target, source in [('prompt_tokens', 'input_tokens'),
                               ('completion_tokens', 'output_tokens'), ('total_tokens', 'total_tokens')]:
            totals[target] += usage.get(source) or 0
    return outputs, totals
