"""Full-support probabilities and base-2 Jensen-Shannon divergence."""
import math


def normalize(values):
    values = list(values)
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float))
                         or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('expected finite nonnegative numeric values')
    largest = max(values)
    if largest <= 0:
        raise ValueError('zero total mass')
    scaled = [x / largest for x in values]
    total = math.fsum(scaled)
    return [x / total for x in scaled]


def validate_prior(values):
    result = normalize(values)
    if not math.isclose(math.fsum(values), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError('saved prior is not normalized')
    return result


def bayes(prior, likelihood):
    if len(prior) != len(likelihood):
        raise ValueError('support length mismatch')
    p = normalize(prior)
    normalize(likelihood)  # validation; keep raw scores for stable log products
    weights = likelihood
    logs = [math.log(a) + math.log(b) if a and b else -math.inf
            for a, b in zip(p, weights)]
    peak = max(logs)
    if peak == -math.inf:
        raise ValueError('likelihood has no overlap with prior support')
    return normalize(math.exp(x - peak) for x in logs)


def jsd(a, b):
    if len(a) != len(b):
        raise ValueError('support length mismatch')
    a, b = normalize(a), normalize(b)
    # log2(p)-log2(m) avoids ratio overflow for subnormal probabilities.
    value = math.fsum(0.5 * p * (math.log2(p) - (math.log2(p + q) - 1))
                     for row, other in ((a, b), (b, a))
                     for p, q in zip(row, other) if p > 0)
    return min(1.0, max(0.0, value))


def entropy(values):
    return -math.fsum(p * math.log2(p) for p in normalize(values) if p)


def identity(candidate):
    return candidate['subtask_id'], candidate['skill']


def parse_scores(raw, candidates, field):
    """Identity matching, never positional or Top-k normalization."""
    import json
    text = raw.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {field} or not isinstance(data[field], list):
        raise ValueError('expected exactly one ' + field + ' list')
    rows = {}
    if field not in ('likelihoods', 'posterior_scores'):
        raise ValueError('unknown score field')
    value_key = 'score'
    for item in data[field]:
        if not isinstance(item, dict) or set(item) != {'subtask_id', 'skill', value_key}:
            raise ValueError('invalid score row fields')
        key = identity(item)
        if key in rows:
            raise ValueError('duplicate candidate')
        rows[key] = item[value_key]
    if set(rows) != {identity(c) for c in candidates}:
        raise ValueError('candidate identities must exactly match full start support')
    values = [rows[identity(c)] for c in candidates]
    normalized = normalize(values)
    total = math.fsum(values)
    return values, normalized, total
