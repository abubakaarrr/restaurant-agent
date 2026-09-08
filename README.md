# Restaurant AI Receptionist

Production-pilot backend for a restaurant phone and website receptionist.

The primary route is **Retell managed Conversation Flow** for telephony,
transcription, turn-taking, voice, transfer, and call operations. This FastAPI
service owns trusted restaurant data and exposes authenticated, idempotent
booking/order/menu tools. The older LangGraph custom-LLM WebSocket remains
feature-flagged for one rollback release.

## Architecture

```mermaid
flowchart LR
    Phone[Phone caller] --> Retell[Retell managed voice]
    Web[Website widget] --> Retell
    Retell --> Flow[Conversation Flow]
    Flow --> API[Authenticated voice tool API]
    API --> Service[Restaurant service]
    Service --> DB[(PostgreSQL)]
    Retell --> Staff[Warm staff transfer]
    Retell --> Webhook[Signed lifecycle webhook]
    Webhook --> DB
```

Production call audio does not pass through this server. Retell owns the audio
path; the backend returns structured business results over HTTPS.

Phase 1 does not provide deterministic cancellation reversal for the managed
Conversation Flow. That stateful boundary is supported only by the local text
and self-hosted streaming transports.

Phase 1 restaurant facts come from the versioned synthetic fixture
`db/fixtures/harbor_and_hearth.v1.json`. `db/seed.py` deterministically projects
that source into normalized menu rows and versioned restaurant-knowledge
records; it does not call an embedding or delivery provider unless the separate
legacy `--with-embeddings` option is explicitly used.

## Safety properties

- Write tools default off with `VOICE_LIVE_WRITES_ENABLED=false`.
- Every booking, cancellation, order mutation, and order confirmation requires
  an idempotency key.
- Orders remain drafts until the caller approves a complete itemized readback.
- Order-level instructions and allergy notes are part of confirmation integrity
  and persist with item options, removals, substitutions, fulfillment, and fees.
- Booking creation locks the selected table and prevents a double booking.
- Fuzzy menu matches return candidates without mutating an order.
- Human transfer uses a server-configured E.164 number, never caller/model text.
- Retell lifecycle webhooks use timestamped HMAC verification and replay
  rejection.
- Call telemetry excludes phone numbers and transcripts by default and follows
  a configured retention period.
- The deterministic behavior reducer makes no age, accent, disability, or
  emotion classifications.

## Local setup

Requirements:

- Python 3.11+
- PostgreSQL 16 with pgvector, or Docker
- OpenAI key only if testing the legacy LangGraph rollback adapter
- Retell account for managed voice deployment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
python scripts/bootstrap_local_secrets.py --apply
```

For Docker, set both `POSTGRES_PASSWORD` and an encoded `DATABASE_URL` in
`.env`, then:

```powershell
docker compose up -d db
python scripts/migrate.py --initialize-schema
python db/seed.py
```

For an existing database, never rerun the base schema:

```powershell
python scripts/migrate.py
```

Start the API:

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

The health endpoint returns 503 until the pilot migration is installed.

## Managed voice tool API

All routes are under `/api/voice-tools` and require
`X-Voice-Tool-Secret`. Write routes also require `Idempotency-Key`.

Capabilities include:

- versioned menu, ingredients, allergens, modifiers, and conservative item matching;
- grounded restaurant identity, hours, policies, seating, and amenities;
- availability, booking creation, verified lookup, and cancellation;
- dine-in, pickup, and synthetic local-delivery draft flows with no live courier integration;
- order draft add/update/remove, order-level notes, and complete canonical summaries;
- explicit versioned order confirmation;
- authenticated tool health with write-flag status.

Retell desired state and endpoint mappings live in
`config/retell-agent.pilot.json`. Managed-flow node prompts live under
`app/prompts/retell/`.

## Rollback adapter

The legacy Retell custom-LLM route is
`wss://HOST/retell-ws/CALL_ID?token=RETELL_WS_TOKEN` and is disabled with
`ENABLE_LEGACY_RETELL_CUSTOM_LLM=false`.

It now includes:

- cancellable generation and guaranteed response closure;
- deterministic silence handling that never replays the previous action;
- behavior adaptation from explicit requests and timing proxies;
- staff transfer and end-call controls;
- first-response and cancellation timing events;
- bounded tool loops and provider-neutral restaurant services.

Vapi and self-hosted Chatterbox are disabled in the live route by default.
Chatterbox remains available only through Docker's `offline-tts` profile.

## Tests

Unit and protocol suite:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q -p pytest_asyncio.plugin
```

Database race/idempotency suite:

```powershell
$env:RUN_DB_INTEGRATION = "1"
$env:TEST_DATABASE_URL = "postgresql://postgres:password@localhost:5432/restaurant_agent"
python -m pytest -q -p pytest_asyncio.plugin tests/test_database_integration.py tests/test_app_security_integration.py
```

The suite covers behavior transitions, API authentication, CSRF, webhook
signatures, replay protection, reminders, handoff, booking concurrency, order
corrections, idempotency, migration/provisioning validation, and bakeoff gates.

## Pilot operations

- [Retell setup, number provisioning, transfer, and rollback](docs/RETELL_SETUP.md)
- [Client website widget](docs/CLIENT_WEBSITE.md)
- [Voice bakeoff](docs/VOICE_BAKEOFF.md)
- [Direct ElevenAgents challenger](docs/ELEVENAGENTS_CHALLENGER.md)
- [Supervised staff/customer pilot](docs/PILOT_RUNBOOK.md)
- [Server deployment](DEPLOY.md)

Existing Retell baseline metrics are collected without PII:

```powershell
python scripts/collect_retell_baseline.py --output artifacts/retell-baseline.json
```

Do not enable live writes or expand unattended hours until the checked-in
release gates pass on real carrier calls.
