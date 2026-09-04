# Production deployment

This deploys the restaurant backend on one Linux host with Docker Compose and
Apache. Retell hosts telephony/audio/turn-taking; this server handles secure
restaurant tools, the operator dashboard, lifecycle webhooks, and PostgreSQL.

## 1. Prerequisites

- Linux host with Docker Engine and Compose
- DNS name pointing to the host
- Apache with `proxy`, `proxy_http`, `proxy_wstunnel`, `ssl`, `headers`, and
  `rewrite`
- TLS certificate for the API hostname
- Retell managed Conversation Flow and a webhook-enabled API key
- Staff transfer number in E.164

Chatterbox is not part of production. Its container is behind the optional
`offline-tts` profile and requires an NVIDIA host.

## 2. Secrets and configuration

```bash
cp .env.production.example .env
chmod 600 .env
```

Replace every placeholder. Generate independent random values for
`VOICE_TOOL_SECRET`, `RETELL_WS_TOKEN`, `SESSION_SECRET`, and
`DASHBOARD_API_KEY`. Do not reuse the dashboard key as a provider secret.

For the first deploy keep:

```env
VOICE_LIVE_WRITES_ENABLED=false
ENABLE_LEGACY_VAPI=false
ENABLE_LEGACY_RETELL_CUSTOM_LLM=false
STORE_CALL_TRANSCRIPTS=false
CALL_RECORDING_ENABLED=false
```

`DATABASE_URL` must use host `db` in Compose and URL-encode special password
characters. `ALLOWED_ORIGINS` must contain exact HTTPS operator-dashboard
origins, never `*`.

Production startup rejects weak/default credentials, missing Retell/tool
secrets, non-HTTPS CORS origins, invalid transfer numbers, and incomplete widget
configuration.

## 3. Database and application

```bash
docker compose build web
docker compose up -d db
docker compose run --rm web python scripts/migrate.py
docker compose run --rm web python db/seed.py
docker compose up -d web
```

The Compose `db` service loads `db/schema.sql` when it creates a new database
volume. Use `--initialize-schema` only for an empty external database that was
not initialized by Compose. On upgrades:

```bash
docker compose build web
docker compose run --rm web python scripts/migrate.py
docker compose up -d web
```

The default seed is idempotent and populates only live tables/menu data. Legacy
pgvector chunks are outside the critical call path; rebuild them only when
needed with `python db/seed.py --with-embeddings`.

Migrations are checksum-tracked and advisory-locked. Never edit an applied SQL
migration; add a new migration.

Check:

```bash
docker compose ps
curl -fsS http://127.0.0.1:8000/health
```

`/health` returns 503 when the pilot schema is missing.

## 4. Apache and TLS

Update the hostname and certificate paths in
`deploy/apache-restaurant-agent.conf`, then:

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel ssl headers rewrite
sudo cp deploy/apache-restaurant-agent.conf \
  /etc/apache2/sites-available/restaurant-agent.conf
sudo a2ensite restaurant-agent
sudo apache2ctl configtest
sudo systemctl reload apache2
```

The vhost:

- redirects HTTP to HTTPS;
- proxies only to the app bound on `127.0.0.1:8000`;
- sets a 15-minute proxy timeout for the rollback WebSocket;
- adds HSTS and baseline browser headers;
- excludes secret-bearing rollback WebSocket URLs from access logs.

If the rollback adapter is enabled, configure Retell's custom-LLM URL as:

```text
wss://API_HOST/retell-ws/{call_id}?token=RETELL_WS_TOKEN
```

Managed Conversation Flow does not use this WebSocket.

## 5. Retell managed flow

Configure each custom function from `config/retell-agent.pilot.json`:

- base path: `https://API_HOST/api/voice-tools`;
- header: `X-Voice-Tool-Secret`;
- unique `Idempotency-Key` on each write;
- lifecycle webhook:
  `https://API_HOST/api/retell/webhook`.

Retell webhook requests must reach the application with the original raw body
and `X-Retell-Signature` unchanged.

Follow `docs/RETELL_SETUP.md` to publish/pin the flow, provision a number, set
transfer/fallback behavior, run carrier tests, and rehearse rollback.

## 6. Website widget

The client website uses Retell's hosted widget directly; it never receives this
server's Retell API key. Configure a domain-restricted public key and reCAPTCHA
v3 according to `docs/CLIENT_WEBSITE.md`.

For the local preview route, set:

```env
WIDGET_ENABLED=true
WIDGET_MODE=hybrid
RETELL_PUBLIC_KEY=public_key_only
RETELL_AGENT_ID=voice_agent_id
RETELL_CHAT_AGENT_ID=
WIDGET_ALLOWED_DOMAINS=https://www.restaurant.example
RECAPTCHA_SITE_KEY=site_key_only
```

Open `/widget-demo`. Callback mode additionally requires the owned Retell phone
number and an approved HTTPS terms/privacy URL.

## 7. Release sequence

1. Deploy with live writes off.
2. Verify tool reads, webhook signatures, call telemetry, dashboard auth, and
   backups.
3. Run all automated tests against staging PostgreSQL.
4. Run the voice bakeoff and save only real provider call IDs/measurements.
5. Place at least 20 real carrier transfer calls covering answer, no-answer,
   busy, voicemail, and after-hours.
6. Run 100+ staff/sandbox calls.
7. Enable `VOICE_LIVE_WRITES_ENABLED=true` only when idempotency and full
   confirmation gates pass.
8. Run the first 100 customer calls during staffed hours with immediate
   transfer and rollback available.

Any duplicate/unconfirmed write, privacy disclosure, invented fact, unsafe
allergy response, or broken transfer disables live writes immediately.

## 8. Backups and retention

```bash
docker compose exec db pg_dump -U postgres -Fc restaurant_agent \
  > restaurant_agent_$(date +%F).dump
```

Test restores regularly. Keep encrypted backups outside the host and apply the
approved retention schedule.

Call-event cleanup runs at application startup using
`CALL_DATA_RETENTION_DAYS`. Transcripts are not persisted unless
`STORE_CALL_TRANSCRIPTS=true`; enable that only after privacy approval.

## 9. Operations and rollback

```bash
docker compose logs -f web
docker compose restart web
```

For upgrades, follow the database and application sequence in section 3 so the
current `web` image is built before migrations run.

Application rollback:

1. Set `VOICE_LIVE_WRITES_ENABLED=false`.
2. Rebind the Retell number to the previous pinned managed-flow version or the
   published staff-transfer-only version.
3. Place a real inbound verification call.
4. Roll back the web container only if database migrations are backward
   compatible; never delete the database volume.

Do not use `docker compose down -v` in production. Releasing a Retell number is
also destructive and stops ownership/charges; the provisioning script
intentionally cannot release numbers.
