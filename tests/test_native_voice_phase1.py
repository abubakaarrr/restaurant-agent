"""Offline Phase 1 safety and correctness gates for native voice."""

from __future__ import annotations

import base64
import sys

import pytest

from app.native_voice.adapter import InterruptionController, NativeVoiceAdapter, RealtimeConfig
from app.native_voice.contracts import (
    CorrectionRecord,
    OrderItemState,
    OrderPatch,
    OrderState,
    UnresolvedField,
)
from app.native_voice.protocol import EventRecorder, MemoryRealtimeTransport
from app.native_voice.speech import SpeechGate, ToolEvidence
from app.native_voice.state_store import InMemoryOrderStateStore, StateVersionConflict
from app.native_voice.tools import ToolBridge, ToolOutcome, _order_readback_hash, realtime_tool_definitions
from app.native_voice.turns import TurnAssembler


class FakeExecutor:
    def __init__(self, result=None, readback=None, error: Exception | None = None):
        self.result = result
        self.readback_result = readback
        self.error = error
        self.calls = []

    async def invoke(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        if self.error:
            raise self.error
        return self.result

    async def readback(self, name, arguments, result):
        return self.readback_result


def item(item_id: str, name: str, quantity: int, *, line_id: str = "", **kwargs) -> OrderItemState:
    return OrderItemState(
        canonical_item_id=item_id,
        item_name=name,
        quantity=quantity,
        line_id=line_id,
        **kwargs,
    )


def order_readback(**overrides):
    readback = {
        "order_id": 1,
        "call_id": "call-1",
        "booking_id": 0,
        "status": "pending",
        "draft_version": 1,
        "total": 0.0,
        "items": [],
        "proposed_items": [],
        "fulfillment": "pickup",
        "fulfillment_type": "pickup",
        "fulfillment_details": {},
        "order_notes": "",
        "allergy_notes": "",
        "unresolved_fields": [],
        "state_version": 1,
        "readback_committed": True,
    }
    readback.update(overrides)
    readback["readback_hash"] = _order_readback_hash(readback)
    return readback


def test_turn_deltas_are_provisional_until_one_versioned_completion():
    assembler = TurnAssembler()
    assembler.start("turn-1")
    assembler.add_delta("two ")
    assembler.add_delta("burgers")
    assert assembler.completed("turn-1") is None
    completed = assembler.finalize(turn_id="turn-1")
    assert completed.transcript == "two burgers"
    assert completed.version == 1
    assert assembler.finalize("ignored replacement", turn_id="turn-1") == completed


def test_compound_order_state_retains_every_field_and_correction_history():
    state = OrderState().apply(
        OrderPatch(
            source_turn_id="turn-1",
            items=(
                item(
                    "menu.main.hearth-burger",
                    "Hearth Burger",
                    2,
                    modifiers=("modifier.side-fries",),
                    removals=("onion jam",),
                    substitutions=("modifier.gluten-aware-bun",),
                    source_turn_ids=("turn-1",),
                    line_id="burger-1",
                    notes="extra sauce",
                ),
                item("menu.dessert.apple-crisp", "Skillet Apple Crisp", 1, source_turn_ids=("turn-1",), line_id="dessert-1"),
            ),
            order_notes="birthday message",
            allergy_notes="peanut allergy; shared kitchen acknowledged",
            guest_notes="please bring water",
            fulfillment="delivery",
            fulfillment_details={"address": "10 Synthetic Street", "instructions": "side door"},
        )
    )
    assert state.version == 1
    assert state.items[0].canonical_item_id == "menu.main.hearth-burger"
    assert state.items[0].quantity == 2
    assert state.items[0].removals == ("onion jam",)
    assert state.fulfillment_details == {"address": "10 Synthetic Street", "instructions": "side door"}

    corrected = state.apply(
        OrderPatch(
            source_turn_id="turn-2",
            items=(item("menu.main.hearth-burger", "Hearth Burger", 1, line_id="burger-1", source_turn_ids=("turn-2",)),),
            corrections=(CorrectionRecord("burger-1.quantity", 2, 1, "turn-2"),),
        )
    )
    assert corrected.version == 2
    assert corrected.items[0].quantity == 1
    assert corrected.items[0].modifiers == ("modifier.side-fries",)
    assert corrected.items[0].removals == ("onion jam",)
    assert corrected.items[0].notes == "extra sauce"
    assert corrected.corrections[-1].previous_value == 2
    assert corrected.source_turn_ids == ("turn-1", "turn-2")

    cleared = corrected.apply(
        OrderPatch(
            source_turn_id="turn-3",
            items=(item("menu.main.hearth-burger", "Hearth Burger", 1, line_id="burger-1", modifiers=(), removals=(), substitutions=(), notes=""),),
        )
    )
    assert not cleared.items[0].modifiers
    assert not cleared.items[0].removals
    assert cleared.items[0].notes == ""


def test_ambiguous_fact_is_unresolved_until_explicitly_cleared():
    state = OrderState().apply(
        OrderPatch(
            source_turn_id="turn-1",
            unresolved_fields=(UnresolvedField("item", "ambiguous menu item", ("Hearth Burger", "Portobello"), "turn-1"),),
        )
    )
    assert state.status == "needs_clarification"
    assert state.finalized_turn_ids == ("turn-1",)
    resolved = state.apply(
        OrderPatch(
            source_turn_id="turn-2",
            items=(item("menu.sandwich.portobello", "Charred Portobello Sandwich", 1, source_turn_ids=("turn-2",)),),
            resolved_fields=("item",),
        )
    )
    assert not resolved.unresolved_fields
    assert resolved.items[0].canonical_item_id == "menu.sandwich.portobello"


def test_finalized_turn_cannot_be_applied_twice():
    state = OrderState().apply(
        OrderPatch(source_turn_id="turn-1", items=(item("menu.na.lemonade", "House Lemonade", 1),))
    )
    with pytest.raises(ValueError, match="already been applied"):
        state.apply(OrderPatch(source_turn_id="turn-1", items=(item("menu.na.lemonade", "House Lemonade", 2),)))


@pytest.mark.asyncio
async def test_structured_state_survives_adapter_restart_without_model_history():
    store = InMemoryOrderStateStore()
    current = await store.load("call-1")
    next_state = current.apply(OrderPatch(source_turn_id="turn-1", items=(item("menu.na.lemonade", "House Lemonade", 2),)))
    await store.save("call-1", next_state, expected_version=0)
    restarted = await store.load("call-1")
    assert restarted.version == 1
    assert restarted.items[0].canonical_item_id == "menu.na.lemonade"
    with pytest.raises(StateVersionConflict):
        await store.save("call-1", next_state, expected_version=0)


@pytest.mark.asyncio
async def test_tool_bridge_requires_readback_and_replays_idempotently():
    executor = FakeExecutor(
        result={"ok": True, "order_id": 7, "status": "pending", "draft_version": 2},
        readback=order_readback(
            order_id=7,
            draft_version=2,
            state_version=2,
            items=[
                {
                    "order_item_id": 1,
                    "item_id": "menu.na.lemonade",
                    "item_name": "House Lemonade",
                    "quantity": 2,
                    "modifiers": [],
                    "removals": [],
                    "substitutions": [],
                    "notes": "",
                }
            ],
        ),
    )
    bridge = ToolBridge(executor)
    args = {"session_id": "call-1", "item_name": "House Lemonade", "quantity": 2}
    first = await bridge.invoke(call_id="tool-1", name="add_order_item", arguments=args, turn_id="turn-1", state_version=1)
    replay = await bridge.invoke(call_id="tool-1", name="add_order_item", arguments=args, turn_id="turn-1", state_version=1)
    assert first.success and first.readback_verified
    assert replay.replayed and len(executor.calls) == 1
    assert not replay.as_evidence(turn_id="turn-1").speakable

    conflict = await bridge.invoke(call_id="tool-1", name="add_order_item", arguments={**args, "quantity": 3}, turn_id="turn-2", state_version=2)
    assert not conflict.success and conflict.error == "idempotency_conflict"

    second_call = await bridge.invoke(call_id="tool-2", name="add_order_item", arguments=args, turn_id="turn-1", state_version=1)
    assert not second_call.success and second_call.error == "replayed_finalized_turn"


@pytest.mark.asyncio
async def test_tool_bridge_rejects_mutation_before_turn_finalization_and_incomplete_readback():
    executor = FakeExecutor(
        result={"ok": True, "order_id": 7, "status": "pending"},
        readback={"order_id": 7, "status": "pending"},
    )
    bridge = ToolBridge(executor)
    not_finalized = await bridge.invoke(
        call_id="tool-before-turn",
        name="set_order_notes",
        arguments={"session_id": "call-1", "order_notes": "birthday"},
        turn_id="",
        state_version=1,
    )
    assert not not_finalized.success and not executor.calls
    incomplete = await bridge.invoke(
        call_id="tool-incomplete-readback",
        name="confirm_order",
        arguments={"session_id": "call-1", "expected_draft_version": 2},
        turn_id="turn-1",
        state_version=1,
    )
    assert not incomplete.readback_verified


@pytest.mark.asyncio
async def test_tool_bridge_rejects_cross_session_and_error_readbacks():
    executor = FakeExecutor(result="saved", readback="order_not_found: no order")
    bridge = ToolBridge(executor)
    bridge.bind_session("call-a")
    cross_session = await bridge.invoke(
        call_id="tool-cross",
        name="set_order_notes",
        arguments={"session_id": "call-b", "order_notes": "hello"},
        turn_id="turn-1",
        state_version=1,
    )
    assert not cross_session.success and cross_session.error == "session_scope_mismatch"
    failed_readback = await bridge.invoke(
        call_id="tool-note",
        name="set_order_notes",
        arguments={"session_id": "call-a", "order_notes": "hello"},
        turn_id="turn-2",
        state_version=1,
    )
    assert not failed_readback.readback_verified


@pytest.mark.asyncio
async def test_availability_parser_preserves_negative_authoritative_result():
    outcome = await ToolBridge(
        FakeExecutor(result="Hearth Burger is not available at $21.")
    ).invoke(
        call_id="tool-availability",
        name="check_menu_item_availability",
        arguments={"item_name": "Hearth Burger"},
        turn_id="turn-1",
        state_version=1,
    )
    assert outcome.facts["availability"] == "unavailable"


@pytest.mark.asyncio
async def test_tool_failure_or_readback_mismatch_cannot_unlock_success_speech():
    failed = ToolBridge(FakeExecutor(result={"ok": True}, readback=None))
    outcome = await failed.invoke(call_id="tool-2", name="confirm_order", arguments={"session_id": "call-1"}, turn_id="turn-1", state_version=1)
    assert outcome.success and not outcome.readback_verified
    evidence = outcome.as_evidence(turn_id="turn-1")
    decision = SpeechGate().evaluate("Your order is placed.", b"audio", evidence=[evidence], current_state_version=1)
    assert not decision.allowed

    exception = ToolBridge(FakeExecutor(error=TimeoutError()))
    timed_out = await exception.invoke(call_id="tool-3", name="confirm_order", arguments={"session_id": "call-1"}, turn_id="turn-1", state_version=1)
    assert not timed_out.success and "tool_exception" in timed_out.error


def test_speech_gate_blocks_hallucinated_items_prices_availability_and_success():
    gate = SpeechGate()
    for text in (
        "Your booking is confirmed.",
        "The Dragon Burger costs $99.00.",
        "The 9 PM patio slot is available.",
    ):
        assert not gate.evaluate(text, b"audio", current_state_version=1).allowed

    evidence = ToolEvidence(
        action="create_booking",
        call_id="tool-4",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={"availability": "available", "prices": {"Hearth Burger": 21.0}, "items": ["Hearth Burger"]},
    )
    assert gate.evaluate("Your booking is confirmed.", b"audio", evidence=[evidence], current_state_version=1).allowed
    assert not gate.evaluate("The 9 PM patio slot is available.", b"audio", evidence=[evidence], current_state_version=1).allowed
    assert not gate.evaluate("Hearth Burger is available.", b"audio", evidence=[evidence], current_state_version=2).allowed
    unavailable = ToolEvidence(
        action="check_menu_item_availability",
        call_id="tool-5",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={"availability": "unavailable", "items": ["Hearth Burger"]},
    )
    assert not gate.evaluate("Hearth Burger is available.", b"audio", evidence=[unavailable], current_state_version=1).allowed
    assert gate.evaluate("Hearth Burger is unavailable.", b"audio", evidence=[unavailable], current_state_version=1).allowed
    assert not gate.evaluate("Hearth Burger is not currently available.", b"audio", evidence=[evidence], current_state_version=1).allowed
    alias = ToolEvidence(
        action="check_menu_item_availability",
        call_id="tool-6",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={"availability": "available", "items": ["burger"], "prices": {"burger": 21.0}},
    )
    assert not gate.evaluate("Dragon Burger costs $21.", b"audio", evidence=[alias], current_state_version=1).allowed
    assert gate.evaluate("The Dragon Burger costs $99.00.", b"audio", evidence=[evidence], current_state_version=1).allowed is False


def test_structured_menu_facts_remain_item_bound():
    outcome = ToolBridge(
        FakeExecutor(
            result={
                "status": "current",
                "items": [
                    {"item_id": "menu.hearth", "name": "Hearth Burger", "price": 21.0, "available": True},
                    {"item_id": "menu.dragon", "name": "Dragon Burger", "price": 24.0, "available": False},
                ],
            }
        )
    )

    async def run():
        return await outcome.invoke(
            call_id="menu-1",
            name="get_full_menu",
            arguments={},
            turn_id="turn-1",
            state_version=1,
        )

    import asyncio

    evidence = asyncio.run(run()).as_evidence(turn_id="turn-1")
    gate = SpeechGate()
    assert gate.evaluate("Hearth Burger costs $21.", b"audio", evidence=[evidence], current_state_version=1).allowed
    assert not gate.evaluate("Dragon Burger is available.", b"audio", evidence=[evidence], current_state_version=1).allowed


@pytest.mark.asyncio
async def test_booking_tools_fail_closed_without_verified_server_scope():
    executor = FakeExecutor(result="Booking #7")
    bridge = ToolBridge(executor)
    bridge.bind_session("call-1")
    outcome = await bridge.invoke(
        call_id="booking-1",
        name="lookup_booking",
        arguments={"booking_id": 7},
        turn_id="turn-1",
        state_version=1,
    )
    assert not outcome.success and outcome.error == "booking_scope_unverified"
    assert not executor.calls


def test_realtime_config_is_native_audio_and_strictly_development_scoped():
    session = RealtimeConfig().session_update()["session"]
    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == "marin"
    assert all(tool["parameters"]["additionalProperties"] is False for tool in realtime_tool_definitions())
    names = {tool["name"] for tool in realtime_tool_definitions()}
    assert {"set_order_notes", "update_order_item", "remove_order_item", "update_confirmed_booking"} <= names


@pytest.mark.asyncio
async def test_adapter_emits_native_audio_after_final_turn_and_records_protocol_events():
    output = b"native-pcm-audio"
    transport = MemoryRealtimeTransport(
        [
            {"type": "session.updated"},
            {"type": "response.created", "response_id": "response-1"},
            {"type": "conversation.item.input_audio_transcription.delta", "delta": "hello"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"},
            {"type": "response.output_audio.delta", "response_id": "response-1", "delta": base64.b64encode(output).decode()},
            {"type": "response.output_audio_transcript.delta", "response_id": "response-1", "delta": "How can I help?"},
            {"type": "response.done", "response_id": "response-1"},
        ]
    )
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    result = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-1")
    assert result.audio == output
    assert result.turn and result.turn.transcript == "hello"
    assert any(event.event_type == "turn_finalized" for event in adapter.recorder.events)
    assert any(event.event_type == "success_speakable" for event in adapter.recorder.events)
    assert transport.sent[0]["type"] == "session.update"
    assert transport.sent[-1]["type"] == "response.create"


@pytest.mark.asyncio
async def test_adapter_suppresses_audio_without_assistant_transcript():
    output = b"unsupported-audio"
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response_id": "response-1"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"},
            {"type": "response.output_audio.delta", "response_id": "response-1", "delta": base64.b64encode(output).decode()},
            {"type": "response.done", "response_id": "response-1"},
        ]
    )
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    result = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-1", transcript="caller text")
    assert result.audio == b""
    assert result.speech and "assistant_transcript_missing" in result.speech.reasons


@pytest.mark.asyncio
async def test_adapter_tool_result_readback_unlocks_only_matching_final_response():
    executor = FakeExecutor(
        result={"ok": True, "status": "confirmed", "order_id": 3, "draft_version": 2},
        readback=order_readback(order_id=3, status="confirmed", draft_version=2, state_version=2),
    )
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response_id": "response-1"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "yes"},
            {"type": "response.function_call_arguments.done", "response_id": "response-1", "call_id": "tool-1", "name": "confirm_order", "arguments": '{"session_id":"call-1","expected_draft_version":2,"caller_approved_full_readback":true}'},
            {"type": "response.done", "response_id": "response-1"},
            {"type": "response.created", "response_id": "response-2"},
            {"type": "response.output_audio.delta", "response_id": "response-2", "delta": base64.b64encode(b"confirmed-audio").decode()},
            {"type": "response.output_audio_transcript.delta", "response_id": "response-2", "delta": "Your order is placed."},
            {"type": "response.done", "response_id": "response-2"},
        ]
    )
    adapter = NativeVoiceAdapter(
        session_id="call-1",
        transport=transport,
        state_store=InMemoryOrderStateStore(),
        tool_bridge=ToolBridge(executor),
    )
    result = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-1")
    assert result.audio == b"confirmed-audio"
    assert result.tool_outcomes[0].readback_verified
    assert any(event.event_type == "database_readback_verified" for event in adapter.recorder.events)


@pytest.mark.asyncio
async def test_interruption_cancels_audio_and_rejects_stale_generation():
    transport = MemoryRealtimeTransport()
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    await adapter.start()
    generation = adapter.interruptions.generation
    await adapter.interrupt()
    assert adapter.interruptions.generation == generation + 1
    assert {event["type"] for event in transport.sent} >= {"response.cancel", "output_audio_buffer.clear"}
    assert not adapter.interruptions.accepts(generation=generation, response_id="old-response")
    assert any(event.event_type == "interruption" for event in adapter.recorder.events)


def test_interruption_rejects_cancelled_response_until_new_response_created():
    controller = InterruptionController()
    generation = controller.begin_response("response-1")
    next_generation = controller.interrupt("response-1")
    assert not controller.accepts(generation=next_generation, response_id="response-1")
    assert controller.accepts(
        generation=next_generation,
        response_id="response-2",
        allow_new_response=True,
    )


@pytest.mark.asyncio
async def test_interruption_truncates_buffered_assistant_item():
    output = b"\x00\x01" * 240
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response_id": "response-1"},
            {"type": "response.output_item.added", "response_id": "response-1", "item": {"id": "item-1", "type": "message", "role": "assistant"}},
            {"type": "response.output_audio.delta", "response_id": "response-1", "item_id": "item-1", "delta": base64.b64encode(output).decode()},
            {"type": "input_audio_buffer.speech_started"},
        ]
    )
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-1")
    truncate = next(event for event in transport.sent if event["type"] == "conversation.item.truncate")
    assert truncate == {
        "type": "conversation.item.truncate",
        "item_id": "item-1",
        "content_index": 0,
        "audio_end_ms": 0,
    }


@pytest.mark.asyncio
async def test_unresolved_state_blocks_mutating_tool_calls():
    store = InMemoryOrderStateStore()
    unresolved = OrderState().apply(
        OrderPatch(
            source_turn_id="turn-0",
            unresolved_fields=(UnresolvedField("item", "ambiguous", ("one", "two"), "turn-0"),),
        )
    )
    await store.save("call-1", unresolved, expected_version=0)
    executor = FakeExecutor(result={"ok": True})
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response_id": "response-1"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "yes"},
            {"type": "response.function_call_arguments.done", "response_id": "response-1", "call_id": "tool-1", "name": "add_order_item", "arguments": "{\"session_id\":\"call-1\",\"item_name\":\"Hearth Burger\"}"},
            {"type": "response.done", "response_id": "response-1"},
            {"type": "response.created", "response_id": "response-2"},
            {"type": "response.output_audio_transcript.delta", "response_id": "response-2", "delta": "Which item did you mean?"},
            {"type": "response.done", "response_id": "response-2"},
        ]
    )
    adapter = NativeVoiceAdapter(
        session_id="call-1",
        transport=transport,
        state_store=store,
        tool_bridge=ToolBridge(executor),
    )
    result = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-1")
    assert result.tool_outcomes[0].error == "clarification_required"
    assert not executor.calls


def test_production_entrypoint_does_not_import_native_voice():
    sys.modules.pop("app.native_voice", None)
    import app.main  # noqa: F401

    assert "app.native_voice" not in sys.modules
