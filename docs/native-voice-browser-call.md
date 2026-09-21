# Local native voice call

Open **http://localhost:8765**, click **Start call**, and grant microphone access.
Speak naturally; a 1.4-second pause sends your turn. The agent speaks back and
keeps the same session until you end the call. Speaking during its reply stops
playback and requests cancellation. Use headphones to reduce speaker echo.

The page shows caller/assistant transcripts, response time from the end of your
speech to the start of playback, call duration, and application order state with
tool outcomes. The pause can be adjusted from 0.9 to 3 seconds. Turns are limited
to 60 seconds; this is a browser energy-based detector, not production turn-taking.

The API key stays on the server. Audio goes to OpenAI; only the separately marked
test database is used for orders/bookings. Use made-up contact details. Transcript
text is displayed in the current browser page; existing canonical business-state
persistence is unchanged. End call stops microphone tracks and closes the provider
connection. It does not delete completed test actions. A new call gets a new session.

## Start in the existing WSL setup

```bash
cd /home/abubakar/.treehouse/restaurant-agent-88b0e0/4/restaurant-agent
/home/abubakar/.cache/restaurant-native-voice-testenv/bin/python scripts/native_voice_ui.py \
  --key-env /var/www/codex-workspace/tools/firstmate/projects/restaurant-agent/.env \
  --database-env /home/abubakar/.cache/restaurant-native-voice-01M2Y02/acceptance.env \
  --test-clock 2026-09-23T18:00:00
```

It binds only to 127.0.0.1:8765, independently of the existing app/Retell routes.
The optional fixed restaurant clock is clearly shown in Call settings. This
example uses Wednesday at 6 p.m. so the synthetic restaurant is open. Omit
`--test-clock` to use real restaurant time, including normal closure rules.
Do not start another copy if the page already works. Close the foreground process
with Ctrl+C when running manually. The operator-started background instance logs
only safe error categories to ~/.cache/restaurant-native-voice-ui.log.

On a fresh checkout, use Python 3.11/3.12, install requirements-dev.txt, and supply
the two environment files above with your own paths. The key file needs
OPENAI_API_KEY. The database file needs NATIVE_VOICE_DATABASE_URL and
NATIVE_VOICE_DATABASE_MARKER matching PostgreSQL's
app.native_voice_disposable_marker database setting. Initialize/migrate/seed only
an explicitly separate disposable database before starting. This launcher does
not perform migrations or copy a production environment.

## What to test yourself

- Speak for 20–30 seconds with several details; check the transcript and readback.
- Change quantities, remove ingredients, add notes, then ask for a summary.
- Interrupt mid-reply and change your request.
- Ask about an invented dish or unavailable date. Check that it does not invent facts.
- Complete an order or booking only after hearing the full readback and agreeing.
- Compare response times for greetings, menu questions, full readbacks and confirmation.

The browser adapter defers confirmation eligibility until matching playback
completion is acknowledged after the audio duration has elapsed. Interrupted or
disconnected playback does not authorize confirmation. Suppressed assistant claims
are not shown as accepted replies. Cross-origin sockets and untrusted hosts are
rejected; this is not an internet-facing authenticated service.

## Known limits

This is a complete local browser call path, not a production deployment. Speech
is collected per turn, and generated audio is buffered for existing claim checks.
That adds both the configured silence timeout and the known readback delay.
Browser echo cancellation and the simple energy detector need human testing with
the actual microphone. A quiet voice/noisy room can cause missed or split turns.
Do not describe synthetic transport checks as a human microphone acceptance test.
The UI has no production phone transport, new-session booking identity provisioning,
or production concurrency guarantee. Keep the documented Phase 1 reliability and
latency limits in view when deciding whether to deliver to customers.

Protocol reference: https://developers.openai.com/api/docs/guides/realtime-conversations
