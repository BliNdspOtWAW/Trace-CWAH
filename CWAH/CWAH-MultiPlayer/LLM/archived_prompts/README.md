# Archived prompts

These templates are kept for reproducing older experiments. The current runner
uses `../prompt_Alice.txt`, `../prompt_Bob.txt`, and the role-specific Reply
templates. `../prompt_Peer.txt` remains the CLI fallback and is still active.

- `prompt_Alice_pre_v2.txt` is the Alice template from commit `1895dee`, before
  the Alice v2 body replaced `../prompt_Alice.txt`.
- `prompt_Agent.txt` and `prompt_Human.txt` are the old role templates. They do
  not implement the current task-control output contract.
- `prompt_Reply.txt` is the old generic Reply template. The current runtime
  selects `prompt_Reply_Alice.txt` or `prompt_Reply_Bob.txt` for every backend.

Use an archived template only with an explicit experiment configuration and
record its path. The current runner may reject archived Active templates that
do not satisfy the task-control protocol.
