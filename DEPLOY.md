# Deploying the Restaurant AI Agent to a Contabo Linux Server (Docker)

This guide deploys two containers with Docker Compose:

- **db** – PostgreSQL 16 with the pgvector extension (auto-creates tables on first boot)
- **web** – the FastAPI + LangGraph agent (bound to `127.0.0.1:8000`)

Your existing **Apache** server reverse-proxies your domain (with the Certbot SSL you
already have) to the app. HTTPS is required for Vapi voice + browser microphone access.

---

## 1. Install Docker on the server

SSH into your Contabo server, then:

```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
docker --version
docker compose version
```

---

## 2. Copy the project to the server

From your local machine (PowerShell), upload the project (replace IP/user):

```bash
scp -r "e:\restrurent agent\restaurant-agent\*" root@YOUR_SERVER_IP:/var/www/dev/restaurant-agent/
```

Or clone it from git if you have pushed it to a repository.

Then on the server:

```bash
cd /var/www/dev/restaurant-agent
```

---

## 3. Create the `.env` file on the server

```bash
nano .env
```

Paste (fill in your real keys):

```env
# Database password used by BOTH the db container and the app
POSTGRES_PASSWORD=change_this_to_a_strong_password

# DATABASE_URL is overridden automatically by docker-compose to point at the
# internal db container, so you do NOT need to set it here.

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

RESTAURANT_NAME=La Casa Restaurant
RESTAURANT_TIMEZONE=UTC

VAPI_PUBLIC_KEY=your_vapi_public_key
VAPI_ASSISTANT_ID=your_vapi_assistant_id

APP_ENV=production
```

Save with `Ctrl+O`, `Enter`, then `Ctrl+X`.

---

## 4. Build and start

```bash
docker compose up -d --build
```

Check status (wait until db is healthy):

```bash
docker compose ps
docker compose logs -f web
```

The tables are created automatically on the first DB boot from `db/schema.sql`.

---

## 5. Seed the database (run ONCE)

This inserts tables, menu items, and embeds the knowledge files (needs the OpenAI key):

```bash
docker compose exec web python db/seed.py
```

You should see counts of tables, menu items, and embedded chunks.

---

## 6. Test it locally on the server

```bash
curl http://127.0.0.1:8000/health
```

(The app is bound to localhost only — it's reached publicly through Apache, next step.)

---

## 7. Point your existing Apache at the app

The app listens on `127.0.0.1:8000`. Apache handles the public domain + your
Certbot SSL and forwards requests to it.

1. Enable the required Apache modules (once):
   ```bash
   sudo a2enmod proxy proxy_http ssl headers rewrite
   sudo systemctl restart apache2
   ```

2. Make sure `agent.servicesground.com` has a TLS certificate. If it isn't covered yet:
   ```bash
   sudo certbot --apache -d agent.servicesground.com
   ```

3. Copy the provided vhost (already filled in with your domain):
   ```bash
   sudo cp deploy/apache-restaurant-agent.conf /etc/apache2/sites-available/restaurant-agent.conf
   ```

4. Enable the site and reload:
   ```bash
   sudo a2ensite restaurant-agent
   sudo apache2ctl configtest
   sudo systemctl reload apache2
   ```

Your agent is now live at `https://agent.servicesground.com`.

Update Vapi's Custom LLM URL to:
`https://agent.servicesground.com/vapi/llm`

> Note: Apache must NOT buffer the `/vapi/llm/chat/completions` streaming response.
> The provided vhost sets `proxy-sendchunked` to keep the Server-Sent Events flowing.
> If you ever see Vapi "LLM failed" errors, confirm `mod_deflate` is not compressing
> that endpoint.

---

## Everyday commands

```bash
docker compose logs -f web          # tail app logs
docker compose restart web          # restart app after a code change
docker compose up -d --build web    # rebuild app after code changes
docker compose down                 # stop everything (keeps the data volume)
docker compose down -v              # stop AND delete the database volume (wipes data)
```

### Re-seeding after a wipe
If you ever run `docker compose down -v`, the DB is empty again. After bringing it
back up, run the seed step (#5) once more.

### Backups
```bash
docker compose exec db pg_dump -U postgres restaurant_agent > backup_$(date +%F).sql
```
