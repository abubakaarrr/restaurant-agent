# Browser voice demo

The original authenticated restaurant dashboard and Gemini voice engine run in
one process at http://localhost:8766. No separate 8765 or 8788 service is needed.

## Start

Install requirements-dev.txt in a virtual environment. Prepare three private,
untracked configuration files:

- key.env: GEMINI_API_KEY and OPENAI_API_KEY
- database.env: NATIVE_VOICE_DATABASE_URL and NATIVE_VOICE_DATABASE_MARKER
- auth.env: LOGIN_USERNAME, LOGIN_PASSWORD, and optionally SESSION_SECRET

The database must be the explicitly marked disposable PostgreSQL database
required by app/native_voice/database_guard.py. Do not use a production database.
The ordinary DATABASE_URL is disabled. The authenticated dashboard reads the
same verified pool as the voice session; dashboard HTTP writes are disabled.

Run from the repository root:

    python scripts/voice_dashboard.py --key-env /private/key.env --database-env /private/database.env --auth-env /private/auth.env

Sign in with the configured credentials. Start a call to create a fresh session.
An explicit goodbye stops microphone capture, finishes the goodbye audio, then
ends the call. Refresh the page after updating the build. Loopback binding,
origin checks, authentication and the two-session limit remain enforced.

## Design

Gemini 3.8 Live handles conversation. The server owns session identity, captured
reservation fields, quantity reconciliation, transaction authorization and
backend state. OpenAI gpt-4o-mini-tts (Marin) speaks controlled readbacks and
transaction results. Canonical facts come from the restaurant fixture and service.

Reservation and food confirmations are separate. Supported pre-order items can
be retained before table creation, then prepared as a dine-in draft. The food is
not confirmed until its own current readback has played and a later approval
arrives. Ambiguous or conditional approvals cannot authorize writes.

The shared streaming module supplies the domain/TTS primitives used by Gemini;
it is not a separately launched service.

## Verify

    python -m pytest -q tests/test_native_voice_gemini.py tests/test_native_voice_reliability.py tests/test_native_voice_dashboard.py tests/test_native_voice_phase1.py tests/test_pending_confirmation.py tests/test_native_voice_streaming.py
    node tests/test_native_voice_farewell.mjs

Database/provider integration tests require the explicit disposable environment.
Unit tests do not establish acoustic reliability. Native generated conversation
can still drift in wording or language; no zero-hallucination or production
readiness claim is made. Complex food modifications require structured tool
interpretation. No telephone integration is included.
