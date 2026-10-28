"""Stable, human-readable paths for CWAH experiment runs."""

import re
from pathlib import Path


def safe_name(value):
    name = str(value).replace('\\', '/').rsplit('/', 1)[-1]
    name = re.sub(r'[^A-Za-z0-9._-]+', '-', name).strip('._-')
    if not name or name in ('.', '..'):
        raise ValueError('model/profile name cannot form a result path')
    return name


def model_family(model):
    value = str(model).lower()
    if 'qwen' in value:
        return 'qwen'
    if 'gpt' in value:
        return 'gpt'
    if 'deepseek' in value:
        return 'deepseek'
    return 'other'


def run_parts(alice_model, bob_model, profile_id, max_steps, trace_mode, timestamp):
    alice = safe_name(alice_model)
    bob = safe_name(bob_model)
    profile = safe_name(profile_id)
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('max_steps must be a positive integer')
    if trace_mode not in ('on', 'off'):
        raise ValueError('trace_mode must be on or off')
    provider = model_family(alice_model)
    if model_family(bob_model) != provider:
        provider = 'mixed'
    model = alice if alice == bob else f'alice-{alice}_bob-{bob}'
    group = f'{model}_{profile}_{max_steps}step'
    mode_dir = 'trace' if trace_mode == 'on' else 'baseline'
    run_id = f'{group}_trace{trace_mode}_{timestamp}'
    return provider, group, mode_dir, run_id


def run_directory(root, alice_model, bob_model, profile_id, max_steps, trace_mode, timestamp):
    provider, group, mode, run_id = run_parts(
        alice_model, bob_model, profile_id, max_steps, trace_mode, timestamp)
    return Path(root) / provider / group / mode / run_id
