from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest
from fastapi import WebSocketDisconnect

import app.retell_handler as handler
from app.behavior import BehaviorState
from app.call_flags import CallControl
from app.config import settings


class FakeWebSocket:
    def __init__(self, messages: list[dict], *, disconnect_delay: float = 0.0):
        self.messages = [json.dumps(message) for message in messages]
        self.sent: list[str] = []
        self.disconnect_delay = disconnect_delay

    async def receive_text(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        if self.disconnect_delay:
            await asyncio.sleep(self.disconnect_delay)
        raise WebSocketDisconnect()

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


@pytest.fixture(autouse=True)
def isolated_behavior_store(monkeypatch: pytest.MonkeyPatch):
    async def fake_load(_call_id: str):
        return BehaviorState()

    async def fake_save(_call_id: str, _state: BehaviorState):
        return None

    monkeypatch.setattr(handler, "load_behavior_state", fake_load)
    monkeypatch.setattr(handler, "save_behavior_state", fake_save)
    monkeypatch.setattr(
        "app.transfer_availability._now",
        lambda timezone_info: datetime(2026, 9, 8, 12, tzinfo=timezone_info),
    )


@pytest.mark.asyncio
async def test_reminder_never_replays_previous_user_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def forbidden_stream(*args, **kwargs):
        nonlocal called
        called = True
        if False:
            yield ""

    monkeypatch.setattr(handler, "stream_agent_tokens", forbidden_stream)
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    websocket = FakeWebSocket(
        [
            {
                "interaction_type": "reminder_required",
                "response_id": 7,
                "transcript": [
                    {"role": "user", "content": "Book the table now"}
                ],
            }
        ]
    )
    await handler.handle_retell_connection(websocket, "call-reminder")
    payloads = [json.loads(item) for item in websocket.sent]
    responses = [item for item in payloads if item.get("response_type") == "response"]
    assert called is False
    assert responses[-1]["response_id"] == 7
    assert "Take your time" in responses[-1]["content"]
    assert responses[-1]["content_complete"] is True


@pytest.mark.asyncio
async def test_transfer_uses_only_server_configured_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream(*args, **kwargs):
        yield "I'll connect you with the restaurant team now."

    monkeypatch.setattr(handler, "stream_agent_tokens", fake_stream)
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        handler,
        "consume_call_control",
        lambda call_id: CallControl(
            "transfer", "human_requested", "+14155550123"
        ),
    )
    monkeypatch.setattr(handler, "current_staff_transfer_number", lambda: "")
    monkeypatch.setattr(settings, "staff_transfer_number", "+14155550123")
    websocket = FakeWebSocket(
        [
            {
                "interaction_type": "response_required",
                "response_id": 8,
                "transcript": [
                    {"role": "user", "content": "Please get a manager"}
                ],
            }
        ],
        disconnect_delay=0.05,
    )
    await handler.handle_retell_connection(websocket, "call-transfer")
    payloads = [json.loads(item) for item in websocket.sent]
    completion = [
        item
        for item in payloads
        if item.get("response_type") == "response"
        and item.get("content_complete")
    ][-1]
    assert completion["transfer_number"] == "+14155550123"
    assert completion["transfer_caller_id"] is True
    assert completion["no_interruption_allowed"] is True


@pytest.mark.asyncio
async def test_behavior_transfer_carries_the_resolved_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "app.behavior.resolve_handoff_destination",
        lambda reason: {
            "owner": "staff",
            "channel": "voice_transfer",
            "transfer_number": "+15035550149",
            "can_transfer": True,
        },
    )
    monkeypatch.setattr(handler, "current_staff_transfer_number", lambda: "")
    websocket = FakeWebSocket(
        [
            {
                "interaction_type": "response_required",
                "response_id": 81,
                "transcript": [
                    {"role": "user", "content": "Connect me to a person"}
                ],
            }
        ]
    )
    await handler.handle_retell_connection(websocket, "call-resolved-transfer")
    completion = [
        json.loads(item)
        for item in websocket.sent
        if json.loads(item).get("content_complete")
    ][-1]
    assert completion["transfer_number"] == "+15035550149"
    assert "connect you" in completion["content"].casefold()


@pytest.mark.asyncio
async def test_unintelligible_audio_is_repaired_without_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def forbidden_stream(*args, **kwargs):
        nonlocal called
        called = True
        if False:
            yield ""

    monkeypatch.setattr(handler, "stream_agent_tokens", forbidden_stream)
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    websocket = FakeWebSocket(
        [
            {
                "interaction_type": "response_required",
                "response_id": 9,
                "transcript": [
                    {"role": "user", "content": "(unintelligible audio)"}
                ],
            }
        ]
    )
    await handler.handle_retell_connection(websocket, "call-repair")
    payloads = [json.loads(item) for item in websocket.sent]
    assert called is False
    assert any("say it one more time" in item.get("content", "") for item in payloads)


@pytest.mark.asyncio
async def test_generation_failure_closes_response_and_uses_static_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_stream(*args, **kwargs):
        raise RuntimeError("provider unavailable")
        if False:
            yield ""

    monkeypatch.setattr(handler, "stream_agent_tokens", failing_stream)
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    monkeypatch.setattr(settings, "staff_transfer_number", "+14155550123")
    websocket = FakeWebSocket(
        [
            {
                "interaction_type": "response_required",
                "response_id": 10,
                "transcript": [
                    {"role": "user", "content": "Can you check a table?"}
                ],
            }
        ],
        disconnect_delay=0.05,
    )
    await handler.handle_retell_connection(websocket, "call-failure")
    payloads = [json.loads(item) for item in websocket.sent]
    completion = [
        item
        for item in payloads
        if item.get("response_type") == "response"
        and item.get("content_complete")
    ][-1]
    assert completion["response_id"] == 10
    assert completion["transfer_number"] == "+14155550123"
    assert "technical issue" in completion["content"]
