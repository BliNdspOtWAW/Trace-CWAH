# TRACE v2 configurations

Copy a .json.example to .json and select TRACE on/off independently. Results go
to CWAH/experiments/tracev2/trace or baseline. The supplied Peer23 templates use
the archived tracev1 profile explicitly, not one of the new tracev2 profiles.
New profile texts are in profiles/tracev2; select them explicitly in bob_profile.

Local Qwen TRACE-on still needs output-protocol validation: the interrupted
41-step check repeatedly exhausted its updater output budget and retained the
prior. These are reproducible configuration templates, not validated benchmark
settings. See docs/TRACE_V2.md and commit-records/trace-v2 for limitations.
