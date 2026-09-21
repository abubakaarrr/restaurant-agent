"""Browser transport boundaries; provider calls remain in separate acceptance."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.native_voice.adapter import NativeVoiceAdapter, VoiceTurnResult
from app.native_voice.contracts import OrderState
from app.native_voice.protocol import MemoryRealtimeTransport
from app.native_voice.speech import SpeechDecision
from app.native_voice.state_store import InMemoryOrderStateStore
from app.native_voice.ui_server import BrowserVoiceAdapter, create_app


ORIGIN = {"origin": "http://localhost"}


class FakeAdapter:
    def __init__(self, *, block=False, allowed=True):
        self.state = OrderState()
        self.submissions = 0
        self.completed = 0
        self.closed = False
        self.interrupted = False
        self.block = block
        self.allowed = allowed
        self.event = asyncio.Event()

    async def start(self):
        pass

    async def close(self):
        self.closed = True

    def discard_readback(self):
        pass

    def mark_audio_played(self, value):
        pass

    async def complete_playback(self):
        self.completed += 1

    async def interrupt(self):
        self.interrupted = True
        self.event.set()

    async def submit_audio(self, audio):
        self.submissions += 1
        if self.block:
            await self.event.wait()
        return VoiceTurnResult(
            turn=SimpleNamespace(transcript="hello"),
            audio=b"\0\0" * 2400 if self.allowed else b"",
            transcript="Hello" if self.allowed else "An unsupported claim",
            speech=SpeechDecision(allowed=self.allowed, text="Hello", audio=b"", reasons=() if self.allowed else ("unverified",)),
        )


def client_for(fake):
    async def connector(session_id, *, language="en"):
        assert session_id.startswith("browser-")
        assert language == "en"
        return fake
    return TestClient(create_app(connector=connector), base_url="http://localhost")


def test_page_and_assets_have_local_security_headers():
    with client_for(FakeAdapter()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Start call" in page.text
        assert page.headers["cache-control"] == "no-store"
        assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
        assert client.get("/assets/audio.mjs").status_code == 200
        assert client.get("/assets/secret.env").status_code == 404
        assert client.get("/", headers={"host": "evil.example"}).status_code == 400


@pytest.mark.parametrize("origin", [None, "https://evil.example", "http://localhost.evil", "http://localhost:9999", "null"])
def test_cross_origin_websocket_is_rejected_before_connecting(origin):
    fake = FakeAdapter()
    with client_for(fake) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("ws://localhost/voice", headers={} if origin is None else {"origin": origin}):
                pass
        assert fake.submissions == 0


def test_playback_must_finish_before_a_second_turn():
    fake = FakeAdapter()
    with client_for(fake) as client:
        with client.websocket_connect("ws://localhost/voice", headers=ORIGIN) as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(b"\0\0" * 12000)
            assert ws.receive_json()["type"] == "processing"
            result = ws.receive_json()
            assert result["type"] == "result" and result["audio"]
            assert fake.completed == 0
            ws.send_bytes(b"\0\0" * 12000)
            assert ws.receive_json()["type"] == "notice"
            assert fake.submissions == 1
            ws.send_json({"type": "played", "token": result["token"]})
            assert ws.receive_json()["type"] == "ready"
            assert fake.completed == 1
            ws.send_json({"type": "end"})
    assert fake.closed


def test_interruption_cannot_release_pending_readback():
    fake = FakeAdapter()
    with client_for(fake) as client:
        with client.websocket_connect("ws://localhost/voice", headers=ORIGIN) as ws:
            ws.receive_json()
            ws.send_bytes(b"\0\0" * 12000)
            ws.receive_json()
            result = ws.receive_json()
            ws.send_json({"type": "interrupt"})
            assert ws.receive_json()["type"] == "ready"
            assert fake.interrupted and fake.completed == 0
            ws.send_json({"type": "played", "token": result["token"]})
            ws.send_bytes(b"\0\0" * 12000)
            assert ws.receive_json()["type"] == "processing"
            ws.receive_json()
            assert fake.completed == 0


def test_inflight_interruption_suppresses_stale_result():
    fake = FakeAdapter(block=True)
    with client_for(fake) as client:
        with client.websocket_connect("ws://localhost/voice", headers=ORIGIN) as ws:
            ws.receive_json()
            ws.send_bytes(b"\0\0" * 12000)
            assert ws.receive_json()["type"] == "processing"
            ws.send_json({"type": "interrupt"})
            assert ws.receive_json()["type"] == "ready"
            assert fake.interrupted and fake.completed == 0


def test_rejected_claim_is_not_exposed_as_assistant_reply():
    fake = FakeAdapter(allowed=False)
    with client_for(fake) as client:
        with client.websocket_connect("ws://localhost/voice", headers=ORIGIN) as ws:
            ws.receive_json()
            ws.send_bytes(b"\0\0" * 12000)
            ws.receive_json()
            result = ws.receive_json()
            assert result["assistant"] == result["audio"] == ""
            assert result["reasons"] == ["unverified"]
            assert ws.receive_json()["type"] == "ready"


@pytest.mark.asyncio
async def test_browser_adapter_defers_base_confirmation_release(monkeypatch):
    calls = []
    async def released(self, text, version):
        calls.append((text, version))
    monkeypatch.setattr(NativeVoiceAdapter, "_release_pending_readbacks", released)
    adapter = BrowserVoiceAdapter(session_id="test", transport=MemoryRealtimeTransport(), state_store=InMemoryOrderStateStore())
    await adapter._release_pending_readbacks("Confirm this order?", 4)
    assert not calls
    adapter.discard_readback()
    await adapter.complete_playback()
    assert not calls
    await adapter._release_pending_readbacks("Updated order?", 5)
    await adapter.complete_playback()
    await adapter.complete_playback()
    assert calls == [("Updated order?", 5)]

def test_early_playback_acknowledgement_cannot_authorize_confirmation():
    from dataclasses import replace
    fake = FakeAdapter()
    original_submit = fake.submit_audio
    async def longer_response(audio):
        return replace(await original_submit(audio), audio=b"\0\0" * 48000)
    fake.submit_audio = longer_response
    with client_for(fake) as client:
        with client.websocket_connect("ws://localhost/voice", headers=ORIGIN) as ws:
            ws.receive_json()
            ws.send_bytes(b"\0\0" * 12000)
            ws.receive_json()
            result = ws.receive_json()
            ws.send_json({"type": "played", "token": result["token"]})
            assert ws.receive_json()["type"] == "notice"
            assert fake.completed == 0
            ws.send_json({"type": "interrupt"})
            assert ws.receive_json()["type"] == "ready"
            assert fake.completed == 0


@pytest.mark.parametrize("audio", [b"", b"x" * 12001, b"x" * (2880000 + 2)])
def test_invalid_audio_cannot_reach_provider(audio):
    fake = FakeAdapter()
    with client_for(fake) as client:
        with client.websocket_connect("ws://localhost/voice", headers=ORIGIN) as ws:
            ws.receive_json()
            ws.send_bytes(audio)
            assert ws.receive_json()["type"] == "notice"
            assert ws.receive_json()["type"] == "ready"
            assert fake.submissions == 0


def test_english_setting_controls_provider_input_and_reply_policy():
    from app.native_voice.ui_server import browser_realtime_config
    config = browser_realtime_config("en")
    session = config.session_update()["session"]
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert "Speak only English throughout this call" in session["instructions"]
    assert "accent, names" in session["instructions"]
    assert "exact_speech_required=true" in session["instructions"]


def test_language_selector_and_explicit_english_session():
    with client_for(FakeAdapter()) as client:
        page = client.get("/")
        assert 'id="language"' in page.text and 'value="en" selected>English' in page.text
        with client.websocket_connect("ws://localhost/voice?language=en", headers=ORIGIN) as ws:
            assert ws.receive_json()["language"] == "en"


@pytest.mark.parametrize("language", ["fr", "auto", "ignore-instructions"])
def test_unapproved_language_is_rejected_before_provider_connection(language):
    called = []
    async def connector(session_id, **kwargs):
        called.append(session_id)
        return FakeAdapter()
    with TestClient(create_app(connector=connector), base_url="http://localhost") as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("ws://localhost/voice?language=" + language, headers=ORIGIN):
                pass
    assert not called
