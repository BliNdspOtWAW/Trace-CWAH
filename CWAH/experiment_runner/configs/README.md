Place any number of local `*.json` experiment configurations here. They are ignored by Git. Copy either `.json.example` file or `../config.example.json`, and edit its Episode, Bob profile, models, and TRACE mode independently. `"port": "auto"` (or omitting `port`) safely allocates a distinct port for each concurrent experiment. The batch launcher only selects files ending exactly in `.json`; documentation and `.json.example` files are not run. See [the runner guide](../README.md).

`config_lib/` stores Episode 44 templates for Peer8, Peer17, Peer23 and Peer24,
with TRACE off/on. `continuous/` stores the selected continuous batch templates.
Copy templates to local `.json` files, and check their model/provider and step
budget before running. Scanning is non-recursive; select the intended directory
explicitly with `--config-dir`.
