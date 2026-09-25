"""Explicit QA entry point: uvicorn app.native_voice.staging:app."""
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from app.config import settings

settings.validate_native_voice_staging()

from app.main import app
import app.db_pool as dashboard_pool
from app.native_voice.database_guard import get_native_voice_pool, close_native_voice_pool
from app.native_voice.dashboard_voice import attach_local_voice
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

origin = settings.cors_origins[0]
attach_local_voice(app, allowed_origin=origin)
# Apache may preserve the public Host or forward its loopback upstream Host.
# The WebSocket Origin is checked independently against the exact public origin.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=[
    urlsplit(origin).hostname, "localhost", "127.0.0.1",
])


@asynccontextmanager
async def lifespan(application):
    try:
        pool = await get_native_voice_pool()
        if not await pool.fetchval(
            "SELECT to_regclass('public.voice_action_idempotency') IS NOT NULL"
        ):
            raise RuntimeError("Apply database migrations before starting QA voice")
        dashboard_pool._pool = pool
        yield
    finally:
        dashboard_pool._pool = None
        await close_native_voice_pool()


app.router.lifespan_context = lifespan


@app.middleware("http")
async def qa_dashboard(request, call_next):
    if request.method not in {"GET", "HEAD"} and request.url.path not in {"/login", "/logout"}:
        return JSONResponse({"detail": "Use the voice call to change QA reservations and orders."}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response
