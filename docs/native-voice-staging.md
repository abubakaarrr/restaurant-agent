# QA browser voice on the existing staging service

This deployment reuses the existing PostgreSQL container, seeded database,
restaurant runtime volume, web service on 127.0.0.1:8000, and Apache HTTPS domain.
QA calls write bookings/orders into that database. Use fictional caller data.
Do not initialize the schema, reseed, or remove Docker volumes.

## 1. Save the current release before switching from main

Run in Bash on the staging server:

```bash
cd /var/www/dev/restaurant-agent/restaurant-agent
git status --short
# Stop here if tracked files have local modifications.
umask 077
backup_dir="$HOME/restaurant-agent-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$backup_dir"
cp .env "$backup_dir/env"
cp docker-compose.yml "$backup_dir/docker-compose.yml"
git rev-parse HEAD > "$backup_dir/commit"
old_image=$(docker inspect --format '{{.Image}}' "$(docker compose ps -q web)")
docker image tag "$old_image" restaurant-agent-web:before-browser-qa
printf '%s\n' "$backup_dir" > "$HOME/.restaurant-agent-last-backup"
git fetch origin
git switch development
git pull --ff-only origin development
```

If no local development branch exists, git switch development tracks
origin/development automatically. Do not use git reset --hard on local changes.

## 2. Update .env

Keep the existing DATABASE_URL, POSTGRES_PASSWORD, GEMINI_API_KEY, OPENAI_API_KEY,
LOGIN_USERNAME, and strong LOGIN_PASSWORD. Set LOGIN_PASSWORD to at least 12
characters if it is weaker. Replace conflicting existing entries for these values:

```dotenv
APP_ENV=staging
WEB_APP_MODULE=app.native_voice.staging
ALLOWED_ORIGINS=https://agent.servicesground.com
NATIVE_VOICE_STAGING_ENABLED=true
NATIVE_VOICE_REALTIME_ENABLED=true
NATIVE_VOICE_ALLOW_SHARED_DATABASE=true
NATIVE_VOICE_DATABASE_WRITE_ENABLED=true
VOICE_LIVE_WRITES_ENABLED=true
ENABLE_LEGACY_RETELL_CUSTOM_LLM=false
ENABLE_LEGACY_VAPI=false
ENABLE_PUBLIC_WEB_CALLS=false
WIDGET_ENABLED=false
LANGCHAIN_TRACING_V2=false
```

Use nano .env. Do not set a different NATIVE_VOICE_DATABASE_URL: the explicitly
enabled staging mode intentionally uses the existing DATABASE_URL for both
dashboard and voice. The database account must be able to run existing migrations
and ALTER DATABASE for the authorization marker (the existing Compose postgres
account can).

Generate new session/API secrets and the QA marker once. Remove any old definitions
of these three variables from .env first. These commands write values to the private
file without displaying them; changing SESSION_SECRET signs out old sessions:

```bash
printf '\nSESSION_SECRET=%s\n' "$(openssl rand -hex 32)" >> .env
printf 'DASHBOARD_API_KEY=%s\n' "$(openssl rand -hex 32)" >> .env
printf 'NATIVE_VOICE_DATABASE_MARKER=%s\n' "$(openssl rand -hex 32)" >> .env
chmod 600 .env
```

## 3. Build, validate configuration, back up data, migrate, start

Run one command at a time; do not continue after a failure.
Building leaves the old web container running.

```bash
docker compose build web
docker compose run --rm --no-deps web python -c 'from app.config import settings; settings.validate_native_voice_staging(); print("QA configuration OK")'
docker compose stop web
backup_dir=$(cat "$HOME/.restaurant-agent-last-backup")
docker compose exec -T db pg_dumpall -U postgres > "$backup_dir/database.sql"
test -s "$backup_dir/database.sql"
docker compose run --rm --no-deps web python scripts/migrate.py
docker compose run --rm --no-deps web python scripts/mark_native_voice_staging.py
docker compose up -d --no-deps --force-recreate web
docker compose ps
docker compose logs --tail=80 web
curl --fail --silent --show-error http://127.0.0.1:8000/health
curl --fail --silent --show-error https://agent.servicesground.com/info
```

The marker script alters only a database-level QA authorization setting, not
restaurant rows. Startup checks the configured DB address/name/port and marker
on an actual connection. Native production mode remains prohibited.

The new code introduces no SQL schema migrations. The existing migration runner
applies any older migrations the server has not yet run; never use
--initialize-schema on this seeded database. Migration errors must be resolved,
not ignored.

## 4. Apache and browser verification

Keep the existing TLS certificate and HTTP proxy to port 8000. Apache must also
forward WebSocket Upgrade requests for /voice-gemini. If the existing HTTPS virtual
host does not already handle upgrades, enable proxy_wstunnel and add these rules
INSIDE the agent.servicesground.com HTTPS virtual host, BEFORE a general ProxyPass /:

```apache
ProxyPass        /voice-gemini ws://127.0.0.1:8000/voice-gemini
ProxyPassReverse /voice-gemini ws://127.0.0.1:8000/voice-gemini
```

```bash
sudo a2enmod proxy proxy_http proxy_wstunnel
sudo apache2ctl configtest
# Only after Syntax OK:
sudo systemctl reload apache2
```

Log in at https://agent.servicesground.com and hard-refresh. Start a call and
allow the microphone; the browser must connect to wss://agent.servicesground.com/voice-gemini.
Test a reservation, a correction after confirmation, menu questions, order
confirmation, interruption, and end/new call. Confirm actual records on the dashboard.
There is a two-concurrent-call limit per process; run one web worker.
The /info HTTP check alone does not prove WebSocket/provider/audio success.

The server's Apache configuration and headset behavior cannot be verified from
the local development environment. Do not call the handover verified until those
checks pass. The local Docker engine was unavailable; build is verified on the server.

## Roll back the application

This preserves the DB and any QA records; it does not automatically undo migrations.

```bash
cd /var/www/dev/restaurant-agent/restaurant-agent
backup_dir=$(cat "$HOME/.restaurant-agent-last-backup")
cp "$backup_dir/env" .env
chmod 600 .env
git switch --detach "$(cat "$backup_dir/commit")"
cat > /tmp/restaurant-agent-rollback.yml <<'YAML'
services:
  web:
    image: restaurant-agent-web:before-browser-qa
YAML
docker compose -f docker-compose.yml -f /tmp/restaurant-agent-rollback.yml up -d --no-deps --no-build --force-recreate web
curl --fail --silent --show-error http://127.0.0.1:8000/health
```

Keep the private backup until QA handover succeeds. Restoring the SQL dump is a
separate, destructive operation and is not part of routine application rollback.
