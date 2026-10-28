# TRACE repository instructions

TRACE is a monorepo for exploratory research. `EntCollabBench/` and
`CWAH/` are independent benchmark implementations; do not move or rename
them without an explicit request.

- Keep API keys, `.env` files, tokens, Docker secrets, and machine-local
  paths out of Git. Update an `.env.example` when a variable is needed.
- Treat `CWAH/virtualhome/` and `CWAH/executable/` as external local
  dependencies. Record their source and revision in documentation, but do
  not commit their Git history or large binaries.
- Before changing or removing a nested `.git`, record its remote, HEAD,
  branch, status, and local modifications, then make a recoverable backup.
- Do not use `git reset --hard` or overwrite local experiment changes.
- Keep generated benchmark runtime data and results ignored unless a small,
  intentionally curated artifact is explicitly requested.
- Before declaring a baseline stable, run the relevant smoke test and the
  benchmark pipeline, and document any known blocker.
