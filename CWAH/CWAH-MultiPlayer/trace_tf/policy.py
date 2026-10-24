"""TRACE-v2 planner context without private peer state."""
import json

COMMUNICATION_INSTRUCTIONS = '''
TRACE-v2 communication (only this TRACE-equipped agent uses this local protocol):
For [send_message], also output <communication_type>delegate</communication_type>
or <communication_type>probe_intent</communication_type> with closing tags.
delegate requests Bob undertake one COMPLETE active public subtask; select its ID
in <subtask_id>. A request is not acceptance. probe_intent asks only Bob's current
task/skill and creates no commitment; use null subtask_id. Consider recent probes
and communication cost; high entropy alone does not require repeated questions.
Keep messages within 500 characters. Neither kind cancels a running skill or
changes already selected physical actions. Physical actions have empty messages.
Protocol fields must be TOP-LEVEL siblings, exactly once. Never put
communication_type, subtask_id, action, or other protocol tags inside message or
cot. Message contains only prose addressed to Bob (ordinary object labels such
as <apple> (375) are allowed). Misplaced/duplicate fields are rejected.
Valid delegate example (use an actual active task ID):
<cot>Request a division of work.</cot>
<communication_type>delegate</communication_type>
<message>Please complete S4 by placing the pudding on the kitchen table.</message>
<action>[send_message]</action>
<subtask_id>S4</subtask_id>
For probe_intent use the same sibling layout, type probe_intent, subtask_id null,
and a plain-language question about current task/skill.
'''

def render_policy_context(state, ledger):
    values = {}
    for name in ('intent', 'capability', 'reliability', 'uncertainty'):
        item = state.dimensions[name]
        values[name] = {'status': item.status, 'value': item.value, 'uncertainty': item.uncertainty}
    return ('TRACE-v2 teammate belief about Bob. Intent is a joint macro-subtask/current-skill '
            'distribution, not observed truth. Capability/reliability mean are Beta means; '
            'normalized_variance=12*variance expresses parameter uncertainty, not failure '
            'probability. Reliability measures whether Bob ultimately completes an accepted '
            'task; Alice completing it counts as failure. Pending/unknown are excluded. '
            'Preference is disabled. Uncertainty vector is [intent entropy, capability '
            'variance, reliability variance], all in [0,1] with different meanings. '
            'Use beliefs to choose execution, delegation or an intent probe.\n'
            + json.dumps(values, ensure_ascii=False) + '\n' + COMMUNICATION_INSTRUCTIONS)

def render_reply_context(state, ledger):
    return render_policy_context(state, ledger).split(COMMUNICATION_INSTRUCTIONS)[0] + (
        '\nRespond to this message; replies do not initiate a new local delegate/probe. '
        'Retain current skill. Task switches guide later execution.')
