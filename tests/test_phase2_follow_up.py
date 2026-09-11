from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocketDisconnect

import app.caller_turn as caller_turn_module
import app.retell_handler as handler
import app.tools.db as db_tools
from app.behavior import BehaviorState
from app.call_memory import clear_call_memory, get_reservation_draft
from app.spoken_delivery import SpokenTextBuffer, sanitize_spoken_text


class SyntheticWebSocket:
    def __init__(self, messages: list[dict]) -> None:
        self.messages = [json.dumps(message) for message in messages]
        self.sent: list[str] = []

    async def receive_text(self) -> str:
        if self.messages:
            return self.messages.pop(0)
        raise WebSocketDisconnect()

    async def send_text(self, payload: str) -> None:
        self.sent.append(payload)


def _response_messages(websocket: SyntheticWebSocket) -> list[dict]:
    return [
        message
        for payload in websocket.sent
        if (message := json.loads(payload)).get("response_type") == "response"
    ]


def _patch_local_call_state(
    monkeypatch: pytest.MonkeyPatch,
    state_store: dict[str, BehaviorState] | None = None,
) -> dict[str, BehaviorState]:
    states = state_store if state_store is not None else {}

    async def load(call_id: str) -> BehaviorState:
        return states.get(call_id, BehaviorState())

    async def save(call_id: str, state: BehaviorState) -> None:
        states[call_id] = state

    monkeypatch.setattr(handler, "load_behavior_state", load)
    monkeypatch.setattr(handler, "save_behavior_state", save)
    monkeypatch.setattr(caller_turn_module, "hydrate_call_memory", AsyncMock())
    monkeypatch.setattr(handler, "_record_background", lambda *args, **kwargs: None)
    return states


@pytest.mark.parametrize(
    ("utterance", "party_size", "expected_route"),
    [
        ("I need a reservation for 11 people.", 11, "private-dining"),
        ("Can you book a table for 12?", 12, "private-dining"),
        ("We have 24 guests; can you reserve a table?", 24, "private-dining"),
        ("We have 25 guests.", 25, "can't reserve"),
    ],
)
@pytest.mark.asyncio
async def test_large_party_voice_path_routes_without_inventory_or_booking(
    monkeypatch: pytest.MonkeyPatch,
    utterance: str,
    party_size: int,
    expected_route: str,
) -> None:
    call_id = f"large-party-{party_size}"
    clear_call_memory(call_id)
    _patch_local_call_state(monkeypatch)

    async def forbidden_stream(*args, **kwargs):
        raise AssertionError("large-party request reached the standard agent/tool path")
        if False:
            yield ""

    availability = AsyncMock(side_effect=AssertionError("availability was checked"))
    create_booking = AsyncMock(side_effect=AssertionError("booking was attempted"))
    monkeypatch.setattr(handler, "stream_agent_tokens", forbidden_stream)
    monkeypatch.setattr(
        db_tools.restaurant_service,
        "check_availability",
        availability,
    )
    monkeypatch.setattr(db_tools.restaurant_service, "create_booking", create_booking)
    websocket = SyntheticWebSocket(
        [
            {
                "interaction_type": "response_required",
                "response_id": party_size,
                "transcript": [{"role": "user", "content": utterance}],
            }
        ]
    )

    await handler.handle_retell_connection(websocket, call_id)

    responses = _response_messages(websocket)
    assert len(responses) == 1
    assert responses[0]["content_complete"] is True
    assert expected_route in responses[0]["content"]
    assert "Booking confirmed" not in responses[0]["content"]
    assert availability.await_count == 0
    assert create_booking.await_count == 0
    assert get_reservation_draft(call_id)["party_size"] == 0
    clear_call_memory(call_id)


@pytest.mark.asyncio
async def test_ten_guest_voice_path_and_post_large_party_retry_remain_normal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = "large-party-retry"
    clear_call_memory(call_id)
    _patch_local_call_state(monkeypatch)
    streamed: list[str] = []

    async def normal_stream(_call_id: str, user_text: str, *args, **kwargs):
        streamed.append(user_text)
        yield "I can help with that standard reservation."

    monkeypatch.setattr(handler, "stream_agent_tokens", normal_stream)
    websocket = SyntheticWebSocket(
        [
            {
                "interaction_type": "response_required",
                "response_id": 10,
                "transcript": [
                    {"role": "user", "content": "I need a reservation for 10 people."}
                ],
            },
            {
                "interaction_type": "response_required",
                "response_id": 11,
                "transcript": [
                    {"role": "user", "content": "I need a reservation for 11 people."}
                ],
            },
            {
                "interaction_type": "response_required",
                "response_id": 12,
                "transcript": [
                    {"role": "user", "content": "What time do you close?"}
                ],
            },
        ]
    )

    await handler.handle_retell_connection(websocket, call_id)

    assert streamed == [
        "I need a reservation for 10 people.",
        "What time do you close?",
    ]
    assert get_reservation_draft(call_id)["party_size"] == 0
    responses = _response_messages(websocket)
    assert any("private-dining" in row["content"] for row in responses)
    assert responses[-1]["content_complete"] is True
    clear_call_memory(call_id)


@pytest.mark.asyncio
async def test_compound_large_party_request_is_routed_before_standard_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(caller_turn_module, "hydrate_call_memory", AsyncMock())
    result = await caller_turn_module.process_caller_turn(
        "large-party-compound",
        "Book a table for 12 on Friday at seven under Maya.",
    )
    assert result["handled"] is True
    assert result["kind"] == "large_party_route_required"
    assert result["party_size"] == 12


@pytest.mark.parametrize("party_size", [11, 12, 24, 25])
@pytest.mark.asyncio
async def test_large_party_draft_tool_rejects_before_state_mutation(
    monkeypatch: pytest.MonkeyPatch,
    party_size: int,
) -> None:
    call_id = f"large-draft-{party_size}"
    clear_call_memory(call_id)
    persist = AsyncMock(side_effect=AssertionError("large-party draft was persisted"))
    availability = AsyncMock(side_effect=AssertionError("availability was checked"))
    monkeypatch.setattr(db_tools, "hydrate_call_memory", AsyncMock())
    monkeypatch.setattr(db_tools.restaurant_service, "persist_call_state", persist)
    monkeypatch.setattr(db_tools.restaurant_service, "check_availability", availability)

    result = await db_tools.update_reservation_draft.ainvoke(
        {
            "session_id": call_id,
            "name": "Maya",
            "date": "2026-09-18",
            "time": "19:00",
            "party_size": party_size,
        }
    )

    expected_code = (
        "large_party_route_required"
        if party_size <= 24
        else "large_party_capacity_exceeded"
    )
    assert result.startswith(expected_code)
    assert persist.await_count == 0
    assert availability.await_count == 0
    assert get_reservation_draft(call_id)["party_size"] == 0
    clear_call_memory(call_id)


def test_r66_split_sequential_marker_retains_context_until_sanitized() -> None:
    complete = sanitize_spoken_text("You can choose 1. burger. 2. salad.")
    assert complete == "You can choose 1, burger. 2, salad."

    stream = SpokenTextBuffer()
    assert stream.feed("You can choose 1. burger. ") == ()
    assert stream.feed("2") == ()
    delivered = stream.feed(". salad.")
    assert "".join((*delivered, *stream.flush())) == complete


def test_r67_currency_ranges_are_not_classified_as_bullets() -> None:
    text = "Lunch is $12 - $16, and dinner is $20 - $24. Specials are $12–$16."
    assert sanitize_spoken_text(text) == text

    stream = SpokenTextBuffer()
    delivered = [
        part
        for chunk in ("Lunch is $12 - $16, ", "and dinner is $20 - $24.")
        for part in stream.feed(chunk)
    ]
    delivered.extend(stream.flush())
    assert "".join(delivered) == "Lunch is $12 - $16, and dinner is $20 - $24."


@pytest.mark.asyncio
async def test_r68_reconnect_does_not_replay_opening_greeting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = "reconnect-greeting"
    states = _patch_local_call_state(monkeypatch)
    first = SyntheticWebSocket(
        [{"interaction_type": "response_required", "response_id": 1, "transcript": []}]
    )
    await handler.handle_retell_connection(first, call_id)
    assert states[call_id].opening_greeting_sent is True

    second = SyntheticWebSocket(
        [{"interaction_type": "response_required", "response_id": 2, "transcript": []}]
    )
    await handler.handle_retell_connection(second, call_id)

    responses = _response_messages(first) + _response_messages(second)
    assert sum("you've reached" in row["content"] for row in responses) == 1
    assert "didn't catch that" in responses[-1]["content"]


@pytest.mark.asyncio
async def test_r69_completed_interaction_exception_emits_fallback_and_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_call_state(monkeypatch)
    events: list[tuple[str, dict]] = []

    async def fail_interaction(*args, **kwargs):
        raise RuntimeError("synthetic interaction failure")

    monkeypatch.setattr(handler, "process_caller_turn", fail_interaction)
    monkeypatch.setattr(handler, "current_staff_transfer_number", lambda: "")
    monkeypatch.setattr(
        handler,
        "_record_background",
        lambda _call_id, event_type, **kwargs: events.append((event_type, kwargs)),
    )
    websocket = SyntheticWebSocket(
        [
            {
                "interaction_type": "response_required",
                "response_id": 69,
                "transcript": [{"role": "user", "content": "Can you help?"}],
            }
        ]
    )

    await handler.handle_retell_connection(websocket, "interaction-failure")

    completions = [
        row for row in _response_messages(websocket) if row["content_complete"]
    ]
    assert len(completions) == 1
    assert "technical issue" in completions[0]["content"]
    assert any(
        event_type == "generation_error"
        and event.get("payload") == {"stage": "interaction_task"}
        for event_type, event in events
    )


def test_r70_single_line_item_is_sanitized_without_changing_numeric_speech() -> None:
    assert sanitize_spoken_text("1. Hearth Burger.") == "1, Hearth Burger."
    unchanged = (
        "7. That's your pickup time.",
        "Confirmation 123. Your table is ready.",
        "We open at 3. Dinner begins at 4.",
        "The range is 12 - 16.",
    )
    assert tuple(sanitize_spoken_text(text) for text in unchanged) == unchanged

    stream = SpokenTextBuffer()
    assert stream.feed("1.") == ()
    assert stream.feed(" Hearth") == ()
    delivered = stream.feed(" Burger.")
    assert "".join((*delivered, *stream.flush())) == "1, Hearth Burger."
