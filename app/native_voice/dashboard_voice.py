"""Authenticated, loopback-only voice routes hosted by the restaurant dashboard."""
import asyncio
from contextlib import suppress
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

LOG = logging.getLogger(__name__)


def same_local_origin(socket):
    origin = urlsplit(socket.headers.get("origin", ""))
    return (origin.scheme == "http" and origin.hostname in {"localhost", "127.0.0.1"}
            and origin.netloc == socket.headers.get("host")
            and not origin.path and not origin.query and not origin.fragment)


def attach_local_voice(app, *, connector=None):
    from app.config import settings
    if settings.is_production:
        raise RuntimeError("local_voice_dashboard_requires_development")
    if connector is None:
        from app.native_voice.gemini_live import connect_gemini_session
        connector = connect_gemini_session
    app.state.native_voice_dashboard = True
    app.state.dashboard_voice_sessions = 0
    assets = Path(__file__).with_name("ui")

    @app.get("/assets/{name}")
    async def voice_asset(name: str):
        if name not in {"gemini.js", "capture.js"}:
            return JSONResponse({"error": "not_found"}, status_code=404)
        return FileResponse(assets / name, headers={"Cache-Control": "no-store"})

    @app.get("/info")
    async def voice_info():
        from app.restaurant_knowledge import get_restaurant_knowledge
        from app.native_voice.gemini_live import MODEL
        return {"restaurant": get_restaurant_knowledge().identity["name"],
                "clock": "Real restaurant time", "model": MODEL,
                "database": "Separate test database"}

    @app.websocket("/voice-gemini")
    async def voice(socket: WebSocket):
        pace = socket.query_params.get("pace", "natural")
        if (not socket.scope.get("session", {}).get("authenticated")
                or not same_local_origin(socket) or pace not in {"natural", "patient"}
                or app.state.dashboard_voice_sessions >= 2):
            await socket.close(code=1008)
            return
        await socket.accept()
        app.state.dashboard_voice_sessions += 1
        session = None
        try:
            session = await connector(send=socket.send_json, pace=pace)
            await session.start()
            while True:
                event = await asyncio.wait_for(socket.receive(), 600)
                if event["type"] == "websocket.disconnect":
                    break
                if event.get("bytes") is not None:
                    await session.append_audio(event["bytes"])
                    continue
                raw = event.get("text") or ""
                if len(raw) > 1024:
                    break
                control = json.loads(raw)
                if not isinstance(control, dict):
                    break
                kind = control.get("type")
                if kind == "interrupt":
                    await session.interrupt()
                elif kind == "played":
                    await session.played(control.get("token"))
                elif kind == "end":
                    break
                else:
                    break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            LOG.warning("Local voice session stopped: %s", type(exc).__name__)
            with suppress(Exception):
                await socket.send_json({"type": "error", "message": "The voice connection stopped. Start a new call to reconnect."})
        finally:
            if session is not None:
                with suppress(Exception):
                    await session.close()
            app.state.dashboard_voice_sessions -= 1
            with suppress(Exception):
                await socket.close()
