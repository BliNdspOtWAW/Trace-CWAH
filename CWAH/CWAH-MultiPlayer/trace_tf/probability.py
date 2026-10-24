"""Validated finite intent distributions and Beta(1,1) statistics."""
import math
import copy

SKILLS = ('goexplore', 'gocheck', 'gograb', 'goput', 'send_message', 'idle')
NO_TASK = 'none'


def normalize(weights):
    values = list(weights)
    if not values or any(isinstance(x, bool) or not isinstance(x, (int, float))
                         or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('probability weights must be finite nonnegative numbers')
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError('probability weights need positive finite mass')
    return [x / total for x in values]


def candidates(tasks):
    """Public active tasks only. None cannot perform task-bound grab/put."""
    result = {key: list(SKILLS) for key, task in tasks.items()
              if task.get('status', 'active') == 'active'}
    if NO_TASK in result:
        raise ValueError('reserved task ID: none')
    result[NO_TASK] = ['goexplore', 'gocheck', 'send_message', 'idle']
    return result


def distribution(task_probs, skill_given_task, support):
    if not isinstance(task_probs, dict) or set(task_probs) != set(support):
        raise ValueError('task_probs must exactly match public candidates')
    if not isinstance(skill_given_task, dict) or set(skill_given_task) != set(support):
        raise ValueError('skill_given_task must exactly match public candidates')
    tasks = dict(zip(support, normalize(task_probs[t] for t in support)))
    conditional, joint = {}, []
    for task, skills in support.items():
        row = skill_given_task[task]
        if not isinstance(row, dict) or set(row) != set(skills):
            raise ValueError('conditional skill row must exactly match legal skills')
        conditional[task] = dict(zip(skills, normalize(row[s] for s in skills)))
        joint.extend({'subtask_id': task, 'skill': s,
                      'probability': tasks[task] * conditional[task][s]} for s in skills)
    entropy = -sum(x['probability'] * math.log(x['probability'])
                   for x in joint if x['probability'] > 0)
    task_entropy = -sum(p * math.log(p) for p in tasks.values() if p > 0)
    return {'task_probs': tasks, 'skill_given_task': conditional, 'joint': joint,
            'normalized_entropy': min(1.0, max(0.0, entropy / math.log(len(joint)))) if len(joint) > 1 else 0.0,
            'task_entropy': task_entropy, 'conditional_skill_entropy': max(0.0, entropy - task_entropy),
            'support_size': len(joint)}


def uniform(support):
    # Uniform joint, rather than uniform tasks with unequal row sizes.
    return distribution({t: len(ss) for t, ss in support.items()},
                        {t: dict.fromkeys(ss, 1) for t, ss in support.items()}, support)


def align_prior(prior, support):
    """Retain overlapping mass; reserve a uniform share for new cells."""
    if not prior:
        return uniform(support)
    old = {(x['subtask_id'], x['skill']): x['probability'] for x in prior['joint']}
    cells = [(t, s) for t, ss in support.items() for s in ss]
    if set(cells) == set(old):
        return copy.deepcopy(prior)
    new = [cell for cell in cells if cell not in old]
    retained = sum(old.get(cell, 0) for cell in cells)
    if retained <= 0:
        return uniform(support)
    new_mass = len(new) / len(cells)
    weights = {cell: (1 / len(cells) if cell in new else
                      old.get(cell, 0) / retained * (1 - new_mass)) for cell in cells}
    rows = {t: {s: weights[t, s] for s in ss} for t, ss in support.items()}
    totals = {t: sum(row.values()) for t, row in rows.items()}
    return distribution(totals, {t: row if totals[t] else dict.fromkeys(row, 1)
                                 for t, row in rows.items()}, support)


def beta_stats(successes, failures):
    alpha, beta = 1 + successes, 1 + failures
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return {'successes': successes, 'failures': failures, 'alpha': alpha, 'beta': beta,
            'mean': alpha / (alpha + beta), 'variance': variance,
            'normalized_variance': min(1.0, max(0.0, 12 * variance))}
