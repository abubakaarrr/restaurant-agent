# Restaurant Agent

## Architecture and safety

- Retell managed Conversation Flow is the primary voice path; see `config/retell-agent.pilot.json` and `app/prompts/retell/`. The LangGraph custom-LLM WebSocket is rollback-only. Deployment examples disable it, although the bare fallback in `app/config.py` is currently enabled, so configure `ENABLE_LEGACY_RETELL_CUSTOM_LLM=false` explicitly.
- Keep `VOICE_LIVE_WRITES_ENABLED=false` unless a separately approved release task has cleared the documented gates.
- Booking and order-table mutation routes require idempotency keys. Booking creation, booking updates/cancellation, and final order confirmation use a server-owned pending-readback and later-affirmation gate. Confirmed-order item changes only check a caller-confirmed flag; guest-note writes and pending-order fulfillment changes lack a confirmation/readback gate. Treat stronger enforcement for those gaps as future work; see `app/services/restaurant.py`.
- Lifecycle webhooks require timestamped HMAC verification and replay deduplication; see `app/security.py` and `app/call_analytics.py`.
- Do not use live restaurant/customer data, place calls, provision providers, or deploy during local development. `scripts/provision_retell.py --apply` and `scripts/reset_demo.py` mutate external or database state and are not verification commands.
- Values copied from `.env.example` for `VOICE_TOOL_SECRET`, `RETELL_WS_TOKEN`, `SESSION_SECRET`, and `DASHBOARD_API_KEY` are placeholders, not usable secrets. Before a separately approved local-environment bootstrap, clear or replace all four, then run `scripts/bootstrap_local_secrets.py --apply`; keep generated values untracked, never print them, and never commit them. Do not run it or create secrets during governance or orientation tasks.

## Development

- Use Python 3.11+ and install the tracked runtime dependencies with `python -m pip install -r requirements.txt`. Docker uses PostgreSQL 16 with pgvector; follow `README.md` and `DEPLOY.md`, and use `python scripts/migrate.py --initialize-schema` only for an empty database.
- The current baseline does not track `requirements-dev.txt` or `tests/`; both paths are ignored in `.gitignore`. Do not report the README's pytest commands as passing until the test assets and development dependencies are restored. Database integration tests require a pre-created, disposable, isolated `TEST_DATABASE_URL` and may truncate application tables there.
- Make changes on a feature branch and deliver them through review. Never edit an applied migration; add a new file under `db/migrations/`.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
