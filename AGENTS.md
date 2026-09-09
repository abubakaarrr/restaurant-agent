# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Canonical restaurant facts live in `db/fixtures/harbor_and_hearth.v1.json`;
  `app/restaurant_knowledge.py` validates/query-normalizes them and `db/seed.py`
  projects them into PostgreSQL without external calls by default.
- Use `requirements-dev.txt` and the commands in `README.md` for local tests.
  Because `/tests/` is ignored for new files, explicitly force-add any new
  permanent test module after verifying it is task-owned.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
