"""Regression coverage for the single-process authenticated dashboard voice routes."""
from unittest.mock import AsyncMock
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.sessions import SessionMiddleware
from starlette.websockets import WebSocketDisconnect
from app.native_voice.dashboard_voice import attach_local_voice

def fixture():
    app = FastAPI()
    app.add_middleware(SessionMiddleware, secret_key="test-only")
    sessions = []
    async def connector(*, send, pace):
        session = AsyncMock()
        async def start(): await send({"type":"ready"})
        session.start.side_effect = start
        async def played(token): await send({"type":"played_ack", "token":token})
        session.played.side_effect = played
        sessions.append(session)
        return session
    attach_local_voice(app, connector=connector)
    @app.get("/test-login")
    async def login(request: Request):
        request.session["authenticated"] = True
        return {}
    client = TestClient(app, base_url="http://localhost:8766")
    return app, client, sessions

def test_dashboard_hosts_engine_without_proxy():
    app, client, sessions = fixture()
    client.get("/test-login")
    with client.websocket_connect("ws://localhost:8766/voice-gemini", headers={"origin":"http://localhost:8766"}) as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(b"\x00\x00")
        ws.send_json({"type":"played","token":"readback"})
        assert ws.receive_json() == {"type":"played_ack","token":"readback"}
        ws.send_json({"type":"end"})
    sessions[0].append_audio.assert_awaited_once_with(b"\x00\x00")
    sessions[0].close.assert_awaited_once()
    assert app.state.dashboard_voice_sessions == 0

@pytest.mark.parametrize("authenticated,origin,query",[
    (False,"http://localhost:8766",""), (True,"http://evil.example",""),
    (True,"http://localhost:8765",""), (True,"http://localhost:8766","?pace=invalid")])
def test_rejects_unauthenticated_or_cross_origin_connections(authenticated,origin,query):
    _,client,sessions=fixture()
    if authenticated: client.get("/test-login")
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("ws://localhost:8766/voice-gemini"+query,headers={"origin":origin}):
            pass
    assert exc.value.code == 1008
    assert not sessions

def test_only_current_assets_are_served():
    _,client,_=fixture()
    assert client.get("/assets/gemini.js").status_code==200
    assert client.get("/assets/capture.js").status_code==200
    assert client.get("/assets/streaming.js").status_code==404
    assert client.get("/info").json()["model"]=="gemini-3.8-live"

def test_session_limit_does_not_open_another_provider():
    app,client,sessions=fixture()
    client.get("/test-login")
    app.state.dashboard_voice_sessions=2
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("ws://localhost:8766/voice-gemini",headers={"origin":"http://localhost:8766"}): pass
    assert not sessions
