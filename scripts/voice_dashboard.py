"""Run the original dashboard and native voice engine together on loopback port 8766."""
import argparse
from contextlib import asynccontextmanager
import os
from pathlib import Path
import secrets
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-env", type=Path, required=True)
    parser.add_argument("--database-env", type=Path, required=True)
    parser.add_argument("--auth-env", type=Path, required=True,
                        help="Private file containing LOGIN_USERNAME and LOGIN_PASSWORD")
    args = parser.parse_args()
    from dotenv import dotenv_values
    keys = dotenv_values(args.key_env)
    database = dotenv_values(args.database_env)
    auth = dotenv_values(args.auth_env)
    gemini = keys.get("GEMINI_API_KEY") or keys.get("GOOGLE_API_KEY") or keys.get("GOOGLE_GENAI_API_KEY")
    if not gemini or not keys.get("OPENAI_API_KEY"):
        parser.error("Gemini and OpenAI credentials are required")
    if not all(database.get(k) for k in ("NATIVE_VOICE_DATABASE_URL", "NATIVE_VOICE_DATABASE_MARKER")):
        parser.error("A marked disposable database is required")
    if not all(auth.get(k) for k in ("LOGIN_USERNAME", "LOGIN_PASSWORD")):
        parser.error("Dashboard login credentials are required")
    os.environ.update(
        APP_ENV="development", DATABASE_URL="postgresql://disabled:disabled@127.0.0.1:1/disabled",
        NATIVE_VOICE_DATABASE_URL=database["NATIVE_VOICE_DATABASE_URL"],
        NATIVE_VOICE_DATABASE_MARKER=database["NATIVE_VOICE_DATABASE_MARKER"],
        NATIVE_VOICE_DATABASE_WRITE_ENABLED="true", NATIVE_VOICE_REALTIME_ENABLED="true",
        VOICE_LIVE_WRITES_ENABLED="true", GEMINI_API_KEY=gemini, OPENAI_API_KEY=keys["OPENAI_API_KEY"],
        ENABLE_LEGACY_VAPI="false", ENABLE_LEGACY_RETELL_CUSTOM_LLM="false",
        ENABLE_PUBLIC_WEB_CALLS="false", LANGCHAIN_TRACING_V2="false",
        LOGIN_USERNAME=auth["LOGIN_USERNAME"], LOGIN_PASSWORD=auth["LOGIN_PASSWORD"],
        SESSION_SECRET=auth.get("SESSION_SECRET") or secrets.token_urlsafe(48),
        DASHBOARD_API_KEY=secrets.token_urlsafe(48),
        ALLOWED_ORIGINS="http://localhost:8766,http://127.0.0.1:8766",
    )
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import app.config as config
    config.Settings.model_config["env_file"] = None
    config.get_settings.cache_clear()
    config.settings = config.get_settings()
    from app.main import app
    import app.db_pool as dashboard_pool
    from app.native_voice.database_guard import get_native_voice_pool, close_native_voice_pool
    from app.native_voice.dashboard_voice import attach_local_voice
    from fastapi.responses import JSONResponse
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    attach_local_voice(app)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @asynccontextmanager
    async def lifespan(application):
        # Both surfaces use the already verified disposable pool. The ordinary
        # database configuration remains disabled, preserving the native guard.
        pool = await get_native_voice_pool()
        dashboard_pool._pool = pool
        try:
            yield
        finally:
            dashboard_pool._pool = None
            await close_native_voice_pool()

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def read_only_dashboard(request, call_next):
        if request.method not in {"GET", "HEAD"} and request.url.path not in {"/login", "/logout"}:
            return JSONResponse({"detail": "Use the voice call to change demo reservations and orders."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8766, proxy_headers=False, access_log=False)


if __name__ == "__main__":
    main()
