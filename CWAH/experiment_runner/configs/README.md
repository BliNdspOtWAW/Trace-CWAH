Place any number of local `*.json` experiment configurations here. They are ignored by Git. Copy either `.json.example` file or `../config.example.json`, and edit its Episode, Bob profile, models, and TRACE mode independently. `"port": "auto"` (or omitting `port`) safely allocates a distinct port for each concurrent experiment. The batch launcher only selects files ending exactly in `.json`; documentation and `.json.example` files are not run. See [the runner guide](../README.md).

The supplied Episode 44 examples cover Peer17 (planning/coordination) and
Peer24 (independent/low-communication) with TRACE off and on. This gives a
2-profile × 2-mode comparison while keeping all other settings fixed.
