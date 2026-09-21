"""Loopback-only browser harness; deliberately separate from app.main."""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, replace
import json
import logging
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.native_voice.adapter import NativeVoiceAdapter, RealtimeConfig, connect_development_adapter
from app.native_voice.database_guard import close_native_voice_pool

LOG = logging.getLogger(__name__)
ASSETS = Path(__file__).with_name("ui")
MAX_AUDIO_BYTES = 24_000 * 2 * 60


class BrowserVoiceAdapter(NativeVoiceAdapter):
    """Defer confirmation eligibility until the browser finishes playback."""

    async def _release_pending_readbacks(self, text: str, state_version: int) -> None:
        self.browser_readback = (text, state_version)

    def discard_readback(self) -> None:
        self.browser_readback = None

    async def complete_playback(self) -> None:
        pending = getattr(self, "browser_readback", None)
        self.discard_readback()
        if pending is not None:
            await super()._release_pending_readbacks(*pending)


def browser_realtime_config(language: str = "en") -> RealtimeConfig:
    if language != "en":
        raise ValueError("unsupported_call_language")
    from app.config import get_settings
    settings = get_settings()
    base = RealtimeConfig(
        model=settings.native_voice_realtime_model,
        voice=settings.native_voice_realtime_voice,
    )
    return replace(
        base,
        language=language,
        instructions=base.instructions + (
            "\n\nCALL LANGUAGE: ENGLISH ONLY. "
            "Speak only English throughout this call, including greetings, clarifications, "
            "tool-related replies and final answers. Do not switch languages based on "
            "accent, names, short foreign words, background audio or uncertain speech. "
            "If the caller is unclear, ask them to repeat in English rather than guessing "
            "another language. This call is configured for English; do not translate "
            "verified restaurant readbacks."
        ),
    )


async def connect_browser_adapter(session_id: str, *, language: str = "en") -> BrowserVoiceAdapter:
    return await connect_development_adapter(
        session_id=session_id,
        config=browser_realtime_config(language),
        adapter_class=BrowserVoiceAdapter,
    )


def same_local_origin(socket: WebSocket) -> bool:
    host = socket.headers.get("host", "")
    origin = socket.headers.get("origin", "")
    parsed = urlsplit(origin)
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.netloc == host
        and not parsed.path and not parsed.query and not parsed.fragment
    )


def create_app(*, connector=connect_browser_adapter, turn_timeout: float = 120) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        yield
        await close_native_voice_pool()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.state.active_sessions = 0

    @app.middleware("http")
    async def local_headers(request, call_next):
        response = await call_next(request)
        response.headers.update({
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; worker-src 'self'; frame-ancestors 'none'",
            "Permissions-Policy": "microphone=(self), camera=()",
        })
        return response

    @app.get("/")
    async def index():
        return FileResponse(ASSETS / "index.html")

    @app.get("/assets/{name}")
    async def asset(name: str):
        if name not in {"app.js", "style.css", "capture.js", "audio.mjs"}:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return FileResponse(ASSETS / name)

    @app.get("/info")
    async def info():
        from app.restaurant_knowledge import get_restaurant_knowledge
        facts = get_restaurant_knowledge()
        return {
            "restaurant": facts.identity.get("name", "Restaurant"),
            "clock": os.getenv("NATIVE_VOICE_UI_TEST_CLOCK", "Real restaurant time"),
            "model": "gpt-realtime",
            "database": "Separate test database",
        }

    @app.websocket("/voice")
    async def voice(socket: WebSocket):
        language = socket.query_params.get("language", "en")
        if language != "en" or not same_local_origin(socket) or app.state.active_sessions >= 2:
            await socket.close(code=1008)
            return
        await socket.accept()
        app.state.active_sessions += 1
        adapter = None
        task = None
        epoch = 0
        awaiting = None
        send_lock = asyncio.Lock()

        async def send(payload):
            async with send_lock:
                await socket.send_json(payload)

        async def process(audio, generation):
            nonlocal awaiting
            started = time.monotonic()
            try:
                adapter.discard_readback()
                result = await asyncio.wait_for(adapter.submit_audio(audio), turn_timeout)
                if generation != epoch:
                    return
                allowed = result.speech is not None and result.speech.allowed
                audio = result.audio if allowed else b""
                token = uuid.uuid4().hex
                awaiting = (token, time.monotonic() + len(audio) / 48_000) if audio else None
                # Never show rejected assistant claims as an accepted response.
                await send({
                    "type": "result", "token": token,
                    "caller": result.turn.transcript if result.turn else "",
                    "assistant": result.transcript if allowed else "",
                    "audio": base64.b64encode(audio).decode("ascii"),
                    "allowed": allowed,
                    "reasons": list(result.speech.reasons) if result.speech else ["response_incomplete"],
                    "elapsed_ms": round((time.monotonic() - started) * 1000),
                    "state": asdict(adapter.state),
                    "tools": [{"name": t.name, "success": t.success, "verified": t.readback_verified, "error": t.error} for t in result.tool_outcomes],
                })
                if not audio:
                    adapter.discard_readback()
                    await send({"type": "ready"})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Do not expose provider exceptions, secrets or caller data.
                LOG.warning("Native browser turn failed: %s", type(exc).__name__)
                await send({"type": "error", "message": "The voice turn failed. End this session and start a new one. Any completed test actions remain saved."})
                await socket.close(code=1011)

        try:
            adapter = await asyncio.wait_for(connector("browser-" + uuid.uuid4().hex, language=language), 20)
            await asyncio.wait_for(adapter.start(), 20)
            await send({"type": "ready", "language": language})
            while True:
                message = await asyncio.wait_for(socket.receive(), timeout=600)
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    audio = message["bytes"]
                    if (task is not None and not task.done()) or awaiting is not None:
                        await send({"type": "notice", "message": "Wait for the reply to finish, or use Stop response."})
                        continue
                    if len(audio) < 12_000 or len(audio) > MAX_AUDIO_BYTES or len(audio) % 2:
                        await send({"type": "notice", "message": "Record between 0.25 and 60 seconds of speech."})
                        await send({"type": "ready"})
                        continue
                    await send({"type": "processing"})
                    task = asyncio.create_task(process(audio, epoch))
                    continue
                text = message.get("text") or ""
                if len(text) > 1024:
                    await socket.close(code=1009)
                    break
                try:
                    control = json.loads(text)
                    if not isinstance(control, dict):
                        raise ValueError()
                except (ValueError, TypeError):
                    await socket.close(code=1008)
                    break
                if control.get("type") == "played":
                    if awaiting is not None and control.get("token") == awaiting[0]:
                        if time.monotonic() + 0.1 < awaiting[1]:
                            await send({"type": "notice", "message": "Playback has not finished."})
                            continue
                        adapter.mark_audio_played(MAX_AUDIO_BYTES)
                        await adapter.complete_playback()
                        awaiting = None
                        await send({"type": "ready"})
                elif control.get("type") == "interrupt":
                    epoch += 1
                    awaiting = None
                    adapter.discard_readback()
                    adapter.mark_audio_played(0)
                    await adapter.interrupt()
                    if task is not None and not task.done():
                        await asyncio.wait_for(asyncio.shield(task), 8)
                    adapter.discard_readback()
                    await send({"type": "ready"})
                elif control.get("type") == "end":
                    break
                else:
                    await socket.close(code=1008)
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            LOG.warning("Native browser session stopped: %s", type(exc).__name__)
            with suppress(Exception):
                await send({"type": "error", "message": "Cannot continue this test session. Check the local server and test database, then reconnect."})
        finally:
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            if adapter is not None:
                adapter.discard_readback()
                with suppress(Exception):
                    await adapter.close()
            app.state.active_sessions -= 1
            with suppress(Exception):
                await socket.close()

    return app
