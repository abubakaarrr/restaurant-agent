"""Offline Phase 1 safety and correctness gates for native voice."""

from __future__ import annotations

import base64
import asyncio
from datetime import datetime
import hashlib
import importlib
import json
import os
import sys
import uuid

import pytest

from app.native_voice.adapter import InterruptionController, NativeVoiceAdapter, RealtimeConfig
from app.native_voice.contracts import (
    CorrectionRecord,
    OrderItemState,
    OrderPatch,
    OrderState,
    UnresolvedField,
)
from app.native_voice.database_guard import (
    NativeVoiceDatabaseGuardError,
    close_native_voice_pool,
    get_native_voice_pool,
    validate_native_voice_database,
    verify_native_voice_database_connection,
)
from app.native_voice.protocol import EventRecorder, MemoryRealtimeTransport
from app.native_voice.speech import SpeechGate, ToolEvidence
from app.native_voice.state_store import CallSessionOrderStateStore, InMemoryOrderStateStore, StateVersionConflict
from app.native_voice.tools import OfflineToolExecutor, ToolBridge, ToolOutcome, _order_readback_hash, realtime_tool_definitions
from app.native_voice.tools import RestaurantToolExecutor
from app.call_memory import clear_call_memory
from app.services.restaurant import RestaurantService
from app.native_voice.turns import CompletedCallerTurn, TurnAssembler


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


class NativeServiceFake:
    def __init__(self):
        self.calls = []
        self.summary = order_readback(order_id=7, call_id="native-call")

    async def load_call_state(self, session_id):
        return {"state": {}}

    async def get_order_summary(self, *, call_id, arm_confirmation=True):
        return self.summary

    async def add_order_item(self, **kwargs):
        self.calls.append(kwargs)
        item_id = len(self.calls)
        self.summary = order_readback(
            order_id=7,
            call_id="native-call",
            draft_version=item_id,
            state_version=item_id,
            items=[
                {
                    "order_item_id": item_id,
                    "item_id": "menu.main.hearth-burger",
                    "item_name": "Hearth Burger",
                    "quantity": 1,
                    "modifiers": [],
                    "removals": [],
                    "substitutions": [],
                    "notes": "",
                }
            ],
        )
        return {"ok": True, "order_id": 7, "order_item_id": len(self.calls), "status": "pending", "draft_version": len(self.calls)}

    async def add_guest_note(self, **kwargs):
        self.summary["order_notes"] = kwargs["note"]
        self.summary["guest_notes"] = kwargs["note"]
        self.summary["readback_hash"] = _order_readback_hash(self.summary)
        return {"saved": True, "guest_notes": kwargs["note"], "note_owner": "order"}

    async def lookup_booking(self, **kwargs):
        return {
            "booking_id": 7,
            "customer_name": "Ada Lovelace",
            "customer_phone": "+14155550123",
            "status": "confirmed",
            "date": "2026-09-19",
            "time": "19:00",
            "party_size": 2,
        }

    async def persist_call_state(self, *args, **kwargs):
        return None


async def fake_order_scope(session_id):
    return {"order_id": 7, "booking_id": 0}


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


@pytest.mark.skipif(os.getenv("RUN_DB_INTEGRATION") != "1", reason="disposable native PostgreSQL required")
@pytest.mark.asyncio
async def test_native_postgresql_executor_and_state_store_are_actual_boundaries():
    session = f"native-test-{uuid.uuid4().hex}"
    pool = await get_native_voice_pool()
    store = CallSessionOrderStateStore()
    try:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM call_sessions WHERE session_id = $1", session)
        first = OrderState().apply(OrderPatch(source_turn_id="turn-1", items=(item("menu.na.lemonade", "House Lemonade", 1),)))
        await store.save(session, first, expected_version=0)
        second = first.apply(OrderPatch(source_turn_id="turn-2", items=(item("menu.main.hearth-burger", "Hearth Burger", 1),)))
        await store.save(session, second, expected_version=1)
        assert (await store.load(session)).version == 2
        service = RestaurantService(pool_provider=get_native_voice_pool)
        bridge = ToolBridge(RestaurantToolExecutor(service=service), session_id=session)
        outcome = await bridge.invoke(call_id="menu-1", name="get_full_menu", arguments={}, turn_id="turn-menu", state_version=0)
        assert outcome.success and outcome.facts.get("items")
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM call_sessions WHERE session_id = $1", session)
        await close_native_voice_pool()


@pytest.mark.skipif(os.getenv("RUN_DB_INTEGRATION") != "1", reason="disposable native PostgreSQL required")
@pytest.mark.asyncio
async def test_native_adapter_affirmation_advances_server_confirmation_turn():
    session = f"native-confirm-test-{uuid.uuid4().hex}"
    pool = await get_native_voice_pool()
    booking_id = None
    try:
        async with pool.acquire() as conn:
            table_id = await conn.fetchval("SELECT id FROM tables ORDER BY id LIMIT 1")
            booking_id = await conn.fetchval(
                """
                INSERT INTO bookings
                    (customer_name, customer_phone, table_id, booked_at, party_size, status)
                VALUES ($1, $2, $3, $4, $5, 'confirmed')
                RETURNING id
                """,
                "Ada Lovelace",
                "+14155550123",
                table_id,
                datetime(2026, 9, 28, 19, 0),
                2,
            )
            await conn.execute(
                "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
                session,
                json.dumps({
                    "booking_id": booking_id,
                    "customer_name": "Ada Lovelace",
                    "customer_phone": "+14155550123",
                }),
            )
        class CanonicalReadbackTransport(MemoryRealtimeTransport):
            async def receive(self):
                event = dict(await super().receive())
                if event.get("type") == "response.output_audio_transcript.done" and event.get("response_id") in (None, "r1b"):
                    for sent in reversed(self.sent):
                        item = sent.get("item") or {}
                        if item.get("type") != "function_call_output":
                            continue
                        output = json.loads(item.get("output") or "{}")
                        if output.get("speech"):
                            event["transcript"] = output["speech"]
                            break
                return event

        events = [
            {"type": "response.created", "response": {"id": "r1"}},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i1", "transcript": "Cancel my reservation"},
            {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "c1", "name": "cancel_booking", "arguments": json.dumps({"session_id": session, "booking_id": booking_id, "caller_confirmed": False})}},
            {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
            {"type": "response.created", "response": {"id": "r1b"}},
            {"type": "response.output_audio_transcript.done", "response_id": "r1b", "transcript": "ignored"},
            {"type": "response.done", "response": {"id": "r1b", "status": "completed"}},
            {"type": "response.created", "response": {"id": "r2"}},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "i2", "transcript": "Yes, cancel my reservation"},
            {"type": "response.output_item.done", "item": {"type": "function_call", "call_id": "c2", "name": "cancel_booking", "arguments": json.dumps({"session_id": session, "booking_id": booking_id, "caller_confirmed": True})}},
            {"type": "response.done", "response": {"id": "r2", "status": "completed"}},
            {"type": "response.created", "response": {"id": "r2b"}},
            {"type": "response.output_audio_transcript.done", "transcript": "Your reservation was cancelled."},
            {"type": "response.done", "response": {"id": "r2b", "status": "completed"}},
        ]
        adapter = NativeVoiceAdapter(
            session_id=session,
            transport=CanonicalReadbackTransport(events),
            state_store=CallSessionOrderStateStore(),
        )
        first = await adapter.submit_audio(b"synthetic", turn_id="turn-1")
        second = await adapter.submit_audio(b"synthetic", turn_id="turn-2")
        async with pool.acquire() as conn:
            status = await conn.fetchval("SELECT status FROM bookings WHERE id = $1", booking_id)
        assert status == "cancelled"
        assert first.turn and second.turn
    finally:
        async with pool.acquire() as conn:
            if booking_id:
                await conn.execute("DELETE FROM voice_action_idempotency WHERE call_id = $1", session)
                await conn.execute("DELETE FROM bookings WHERE id = $1", booking_id)
            await conn.execute("DELETE FROM call_sessions WHERE session_id = $1", session)
        await close_native_voice_pool()


@pytest.mark.skipif(os.getenv("RUN_DB_INTEGRATION") != "1", reason="disposable native PostgreSQL required")
@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_readback", [False, True])
async def test_native_order_confirmation_requires_delivered_readback(invalid_readback):
    session = f"native-order-confirm-test-{uuid.uuid4().hex}"
    pool = await get_native_voice_pool()
    order_id = None

    class OrderReadbackTransport(MemoryRealtimeTransport):
        def prepare(self, approved):
            suffix = "yes" if approved else "request"
            self.incoming = [
                {"type": "response.created", "response": {"id": f"order-{suffix}"}},
                {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Yes, place my order." if approved else "Read back my order."},
                {"type": "response.function_call_arguments.done", "response_id": f"order-{suffix}", "call_id": f"order-call-{suffix}", "name": "confirm_order" if approved else "get_order_summary", "arguments": json.dumps({"session_id": session, **({"expected_draft_version": self.draft_version, "caller_approved_full_readback": True} if approved else {})})},
                {"type": "response.done", "response": {"id": f"order-{suffix}", "status": "completed"}},
                {"type": "response.created", "response": {"id": f"order-final-{suffix}"}},
                {"type": "response.output_audio.delta", "response_id": f"order-final-{suffix}", "delta": base64.b64encode(b"synthetic-audio").decode()},
                {"type": "response.output_audio_transcript.done", "response_id": f"order-final-{suffix}", "canonical_speech": not approved},
                {"type": "response.done", "response": {"id": f"order-final-{suffix}", "status": "completed"}},
            ]

        async def receive(self):
            event = dict(await super().receive())
            if event.pop("canonical_speech", False):
                outputs = [
                    json.loads(sent["item"]["output"])
                    for sent in self.sent
                    if sent.get("type") == "conversation.item.create"
                    and sent.get("item", {}).get("type") == "function_call_output"
                ]
                speech = outputs[-1]["speech"]
                event["transcript"] = (
                    f"Would you like me to confirm order {self.order_id + 1}?"
                    if invalid_readback
                    else speech
                )
            return event

    try:
        service = RestaurantService(pool_provider=get_native_voice_pool)
        await service.add_order_item(
            call_id=session,
            idempotency_key=f"{session}-seed",
            item_name="Hearth Burger",
            quantity=1,
            modifier_ids=["modifier.side-fries"],
            customer_name="Synthetic Order Guest",
            customer_phone="+15035550109",
        )
        summary = await service.get_order_summary(call_id=session)
        order_id = int(summary["order_id"])
        transport = OrderReadbackTransport([])
        transport.order_id = order_id
        transport.draft_version = int(summary["draft_version"])
        adapter = NativeVoiceAdapter(session_id=session, transport=transport)
        try:
            transport.prepare(False)
            pending = await adapter.submit_audio(b"synthetic", turn_id="order-readback")
            assert bool(pending.speech and pending.speech.allowed) is not invalid_readback
            transport.prepare(True)
            approved = await adapter.submit_audio(b"synthetic", turn_id="order-approval")
            status = await pool.fetchval("SELECT status FROM orders WHERE id = $1", order_id)
            if invalid_readback:
                assert not approved.tool_outcomes[0].success
                assert status == "pending"
            else:
                assert approved.tool_outcomes[0].success
                assert status == "confirmed"
        finally:
            await adapter.close()
    finally:
        async with pool.acquire() as conn:
            if order_id:
                await conn.execute("DELETE FROM voice_action_idempotency WHERE call_id = $1", session)
                await conn.execute("DELETE FROM orders WHERE id = $1", order_id)
            await conn.execute("DELETE FROM call_sessions WHERE session_id = $1", session)
        await close_native_voice_pool()


@pytest.mark.skipif(os.getenv("RUN_DB_INTEGRATION") != "1", reason="disposable native PostgreSQL required")
@pytest.mark.asyncio
async def test_native_reservation_draft_readback_then_create_booking():
    session = f"native-create-test-{uuid.uuid4().hex}"
    pool = await get_native_voice_pool()
    booking_id = None

    class ReservationTransport(MemoryRealtimeTransport):
        def prepare(self, create):
            suffix = "create" if create else "draft"
            arguments = (
                {"session_id": session, "name": "Synthetic Reservation Guest", "phone": "+15035550110", "date": "2026-09-29", "time": "19:00", "party_size": 2, "caller_confirmed": True}
                if create
                else {"session_id": session}
            )
            self.incoming = [
                {"type": "response.created", "response": {"id": f"reservation-{suffix}"}},
                {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Yes, book it." if create else "Read back the reservation."},
                {"type": "response.function_call_arguments.done", "response_id": f"reservation-{suffix}", "call_id": f"reservation-call-{suffix}", "name": "create_booking" if create else "get_reservation_draft", "arguments": json.dumps(arguments)},
                {"type": "response.done", "response": {"id": f"reservation-{suffix}", "status": "completed"}},
                {"type": "response.created", "response": {"id": f"reservation-final-{suffix}"}},
                {"type": "response.output_audio_transcript.done", "response_id": f"reservation-final-{suffix}", "canonical_speech": not create},
                {"type": "response.done", "response": {"id": f"reservation-final-{suffix}", "status": "completed"}},
            ]

        async def receive(self):
            event = dict(await super().receive())
            if event.pop("canonical_speech", False):
                outputs = [
                    json.loads(sent["item"]["output"])
                    for sent in self.sent
                    if sent.get("type") == "conversation.item.create"
                    and sent.get("item", {}).get("type") == "function_call_output"
                ]
                event["transcript"] = outputs[-1]["speech"]
            return event

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
                session,
                json.dumps({
                    "reservation_draft": {
                        "customer_name": "Synthetic Reservation Guest",
                        "customer_phone": "+15035550110",
                        "date": "2026-09-29",
                        "time": "19:00",
                        "party_size": 2,
                        "notes": "",
                    }
                }),
            )
        transport = ReservationTransport([])
        adapter = NativeVoiceAdapter(session_id=session, transport=transport)
        try:
            transport.prepare(False)
            pending = await adapter.submit_audio(b"synthetic", turn_id="reservation-draft")
            assert pending.tool_outcomes[0].success
            assert pending.speech and pending.speech.allowed
            assert pending.speech.text == pending.tool_outcomes[0].confirmation_text, (
                pending.speech.text,
                pending.tool_outcomes[0].confirmation_text,
            )
            transport.prepare(True)
            booked = await adapter.submit_audio(b"synthetic", turn_id="reservation-create")
            outcome = booked.tool_outcomes[0]
            assert outcome.success, f"{outcome.error}: {outcome.result}"
            booking_id = int(outcome.result["booking_id"])
            assert outcome.success and outcome.readback_verified
            status = await pool.fetchval("SELECT status FROM bookings WHERE id = $1", booking_id)
            assert status == "confirmed"
        finally:
            await adapter.close()
    finally:
        async with pool.acquire() as conn:
            if booking_id:
                await conn.execute("DELETE FROM voice_action_idempotency WHERE call_id = $1", session)
                await conn.execute("DELETE FROM bookings WHERE id = $1", booking_id)
            await conn.execute("DELETE FROM call_sessions WHERE session_id = $1", session)
        await close_native_voice_pool()


@pytest.mark.asyncio
async def test_tool_bridge_requires_readback_and_replays_idempotently():
    executor = FakeExecutor(
        result={"ok": True, "order_id": 7, "order_item_id": 1, "status": "pending", "draft_version": 2},
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
    bridge = ToolBridge(executor, scope_resolver=fake_order_scope)
    bridge.bind_session("call-1")
    args = {"session_id": "call-1", "item_name": "House Lemonade", "quantity": 2}
    first = await bridge.invoke(call_id="tool-1", name="add_order_item", arguments=args, turn_id="turn-1", state_version=1)
    replay = await bridge.invoke(call_id="tool-1", name="add_order_item", arguments=args, turn_id="turn-1", state_version=1)
    assert first.success and first.readback_verified
    assert replay.replayed and len(executor.calls) == 1
    assert not replay.as_evidence(turn_id="turn-1").speakable

    conflict = await bridge.invoke(call_id="tool-1", name="add_order_item", arguments={**args, "quantity": 3}, turn_id="turn-2", state_version=2)
    assert not conflict.success and conflict.error == "idempotency_conflict"

    second_call = await bridge.invoke(call_id="tool-2", name="add_order_item", arguments=args, turn_id="turn-1", state_version=1)
    assert second_call.replayed and second_call.success and len(executor.calls) == 1
    state_shifted_replay = await bridge.invoke(call_id="tool-3", name="add_order_item", arguments=args, turn_id="turn-1", state_version=9)
    assert state_shifted_replay.replayed and state_shifted_replay.success and len(executor.calls) == 1


@pytest.mark.asyncio
async def test_offline_order_scope_never_contacts_restaurant_service():
    executor = OfflineToolExecutor(
        result={"ok": True, "order_id": 7, "order_item_id": 1, "status": "pending", "draft_version": 2},
        readback=order_readback(
            order_id=7,
            call_id="offline-call",
            draft_version=2,
            state_version=2,
            items=[
                {
                    "order_item_id": 1,
                    "item_id": "menu.na.burger",
                    "item_name": "Hearth Burger",
                    "quantity": 1,
                    "modifiers": [],
                    "removals": [],
                    "substitutions": [],
                    "notes": "",
                }
            ],
        ),
        order_scope={"order_id": 7, "booking_id": 0},
    )
    bridge = ToolBridge(executor, session_id="offline-call")
    outcome = await bridge.invoke(
        call_id="offline-order",
        name="add_order_item",
        arguments={"session_id": "offline-call", "item_name": "Hearth Burger", "quantity": 1},
        turn_id="turn-1",
        state_version=1,
    )
    assert outcome.success and outcome.readback_verified
    assert len(executor.calls) == 1


@pytest.mark.asyncio
async def test_offline_booking_scope_and_cancellation_retry_are_authorized_without_database():
    result = {
        "booking_id": 7,
        "customer_name": "Ada Lovelace",
        "customer_phone": "+14155550123",
        "status": "cancelled",
        "date": "2026-09-19",
        "time": "19:00",
        "party_size": 2,
    }
    executor = OfflineToolExecutor(result=result, readback={**result, "readback_committed": True}, booking_identity={**result, "status": "confirmed"})
    bridge = ToolBridge(executor, session_id="offline-call")
    arguments = {"session_id": "offline-call", "booking_id": 7, "caller_confirmed": True}
    first = await bridge.invoke(call_id="cancel-1", name="cancel_booking", arguments=arguments, turn_id="turn-cancel", state_version=1)
    retry = await bridge.invoke(call_id="cancel-2", name="cancel_booking", arguments=arguments, turn_id="turn-cancel", state_version=2)
    assert first.success and first.readback_verified
    assert retry.replayed and retry.success and len(executor.calls) == 1


@pytest.mark.asyncio
async def test_native_executor_scopes_idempotency_to_session_and_finalized_turn():
    from app.native_voice.tools import RestaurantToolExecutor

    service = NativeServiceFake()
    bridge = ToolBridge(RestaurantToolExecutor(service=service), session_id="native-call")
    arguments = {"session_id": "native-call", "item_name": "Hearth Burger", "quantity": 1}
    first = await bridge.invoke(
        call_id="native-1", name="add_order_item", arguments=arguments, turn_id="turn-1", state_version=1
    )
    same_turn = await bridge.invoke(
        call_id="native-2", name="add_order_item", arguments=arguments, turn_id="turn-1", state_version=2
    )
    next_turn = await bridge.invoke(
        call_id="native-3", name="add_order_item", arguments=arguments, turn_id="turn-2", state_version=3
    )
    assert first.success and first.readback_verified
    assert same_turn.replayed and same_turn.success
    assert next_turn.success and next_turn.readback_verified
    assert len(service.calls) == 2


@pytest.mark.asyncio
async def test_native_booking_lookup_requires_supplied_phone_match():
    from app.native_voice.tools import RestaurantToolExecutor

    service = NativeServiceFake()
    bridge = ToolBridge(RestaurantToolExecutor(service=service), session_id="native-call")
    outcome = await bridge.invoke(
        call_id="booking-1",
        name="lookup_booking",
        arguments={"booking_id": 7, "customer_phone": "+14155550999"},
        turn_id="turn-1",
        state_version=1,
    )
    assert not outcome.success
    assert outcome.error == "booking_scope_unverified"


@pytest.mark.asyncio
async def test_native_anonymous_order_guest_note_uses_authoritative_order_scope():
    from app.native_voice.tools import RestaurantToolExecutor

    service = NativeServiceFake()
    bridge = ToolBridge(RestaurantToolExecutor(service=service), session_id="native-call")
    outcome = await bridge.invoke(
        call_id="note-1",
        name="add_guest_note",
        arguments={"session_id": "native-call", "note": "Please include utensils"},
        turn_id="turn-note",
        state_version=1,
    )
    assert outcome.success and outcome.readback_verified
    assert service.summary["order_notes"] == "Please include utensils"


@pytest.mark.asyncio
async def test_tool_bridge_rejects_mutation_before_turn_finalization_and_incomplete_readback():
    executor = FakeExecutor(
        result={"ok": True, "order_id": 7, "status": "pending"},
        readback={"order_id": 7, "status": "pending"},
    )
    bridge = ToolBridge(executor, scope_resolver=fake_order_scope)
    bridge.bind_session("call-1")
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
    bridge = ToolBridge(executor, scope_resolver=fake_order_scope)
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
async def test_tool_bridge_rejects_proposed_order_items_as_uncommitted():
    proposed = {
        "order_id": 7,
        "call_id": "call-1",
        "booking_id": 0,
        "status": "pending",
        "draft_version": 2,
        "total": 21.0,
        "items": [],
        "proposed_items": [{
            "order_item_id": "proposal-1",
            "item_id": "menu.hearth",
            "item_name": "Hearth Burger",
            "quantity": 1,
            "modifiers": [],
            "removals": [],
            "substitutions": [],
            "notes": "",
        }],
        "fulfillment": "pickup",
        "fulfillment_type": "pickup",
        "fulfillment_details": {},
        "order_notes": "",
        "allergy_notes": "",
        "unresolved_fields": [],
        "state_version": 2,
        "readback_committed": True,
    }
    proposed["readback_hash"] = _order_readback_hash(proposed)
    bridge = ToolBridge(
        FakeExecutor(
            result={"ok": True, "status": "pending"},
            readback=proposed,
        ),
        scope_resolver=fake_order_scope,
    )
    bridge.bind_session("call-1")
    outcome = await bridge.invoke(
        call_id="proposal-1",
        name="add_order_item",
        arguments={"session_id": "call-1", "item_name": "Hearth Burger", "quantity": 1},
        turn_id="turn-proposal",
        state_version=1,
    )
    assert not outcome.readback_verified


@pytest.mark.asyncio
async def test_anonymous_pickup_scope_keeps_contact_data_server_bound(monkeypatch):
    async def current_order(session_id):
        assert session_id == "call-1"
        return {"order_id": 7, "booking_id": 0}

    executor = FakeExecutor(
        result={"ok": True, "order_id": 7, "order_item_id": 1, "status": "pending", "draft_version": 2},
        readback=order_readback(
            order_id=7,
            call_id="call-1",
            draft_version=2,
            state_version=2,
            items=[
                {
                    "order_item_id": 1,
                    "item_id": "menu.na.lemonade",
                    "item_name": "House Lemonade",
                    "quantity": 1,
                    "modifiers": [],
                    "removals": [],
                    "substitutions": [],
                    "notes": "",
                }
            ],
        ),
    )
    bridge = ToolBridge(executor, scope_resolver=current_order)
    bridge.bind_session("call-1")
    outcome = await bridge.invoke(
        call_id="tool-contact",
        name="add_order_item",
        arguments={
            "session_id": "call-1",
            "item_name": "House Lemonade",
            "customer_name": "  Ada   Lovelace ",
            "customer_phone": "415 555 0123",
        },
        turn_id="turn-contact",
        state_version=1,
    )
    assert outcome.success and outcome.readback_verified
    assert executor.calls[0][1]["customer_name"] == "Ada Lovelace"
    assert executor.calls[0][1]["customer_phone"] == "+14155550123"
    assert executor.calls[0][1]["session_id"] == "call-1"


@pytest.mark.asyncio
async def test_anonymous_guest_note_stays_bound_to_current_session():
    async def no_order_scope(session_id):
        return None

    bridge = ToolBridge(
        FakeExecutor(result={"saved": True}, readback=order_readback(guest_notes="extra napkins")),
        session_id="call-1",
        scope_resolver=no_order_scope,
    )

    async def no_verified_booking():
        return None, "booking_scope_unverified"

    bridge._verified_booking_identity = no_verified_booking
    outcome = await bridge.invoke(
        call_id="guest-note-1",
        name="add_guest_note",
        arguments={"session_id": "call-1", "note": "extra napkins"},
        turn_id="turn-note",
        state_version=1,
    )
    assert outcome.success and outcome.readback_verified
    assert bridge.executor.calls[0][1]["session_id"] == "call-1"
    assert bridge.executor.calls[0][1]["booking_id"] == 0
    assert "customer_name" not in bridge.executor.calls[0][1]


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
    failed = ToolBridge(FakeExecutor(result={"ok": True}, readback=None), session_id="call-1", scope_resolver=fake_order_scope)
    outcome = await failed.invoke(call_id="tool-2", name="confirm_order", arguments={"session_id": "call-1"}, turn_id="turn-1", state_version=1)
    assert not outcome.success and not outcome.readback_verified
    evidence = outcome.as_evidence(turn_id="turn-1")
    decision = SpeechGate().evaluate("Your order is placed.", b"audio", evidence=[evidence], current_state_version=1)
    assert not decision.allowed

    exception = ToolBridge(FakeExecutor(error=TimeoutError()), session_id="call-1", scope_resolver=fake_order_scope)
    timed_out = await exception.invoke(call_id="tool-3", name="confirm_order", arguments={"session_id": "call-1"}, turn_id="turn-1", state_version=1)
    assert not timed_out.success and "tool_exception" in timed_out.error


def test_speech_gate_blocks_hallucinated_items_prices_availability_and_success():
    gate = SpeechGate()
    for text in (
        "Your booking is confirmed.",
        "Your reservation is all set.",
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
        facts={"availability": "available", "prices": {"Hearth Burger": 21.0}, "items": ["Hearth Burger"], "status": "confirmed"},
    )
    assert gate.evaluate("Your booking is confirmed.", b"audio", evidence=[evidence], current_state_version=1).allowed
    menu_evidence = ToolEvidence(
        action="get_full_menu",
        call_id="menu-1",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={"items": ["Hearth Burger"]},
    )
    assert not gate.evaluate("Your reservation is all set.", b"audio", evidence=[menu_evidence], current_state_version=1).allowed
    assert not gate.evaluate("The 9 PM patio slot is available.", b"audio", evidence=[evidence], current_state_version=1).allowed
    assert not gate.evaluate("Hearth Burger is available.", b"audio", evidence=[evidence], current_state_version=2).allowed
    order_edit = ToolEvidence(
        action="set_order_notes",
        call_id="tool-notes",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={"items": ["Hearth Burger"]},
    )
    assert not gate.evaluate("Your order was submitted.", b"audio", evidence=[order_edit], current_state_version=1).allowed
    booking_details = ToolEvidence(
        action="create_booking",
        call_id="tool-booking-details",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={"status": "confirmed", "date": "2025-05-09", "time": "7:00 PM", "subject": {"booking_id": 7}},
    )
    assert not gate.evaluate(
        "Your booking is confirmed for 9 PM on Friday.",
        b"audio",
        evidence=[booking_details],
        current_state_version=1,
    ).allowed
    assert gate.evaluate(
        "Your booking is confirmed for 7 PM on Friday.",
        b"audio",
        evidence=[booking_details],
        current_state_version=1,
    ).allowed
    order_details = ToolEvidence(
        action="add_order_item",
        call_id="tool-order-details",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={
            "items": [{"item_name": "Hearth Burger", "quantity": 2, "modifiers": ["side fries"]}],
            "status": "pending",
        },
    )
    assert gate.evaluate("Two Hearth Burgers with side fries were added.", b"audio", evidence=[order_details], current_state_version=1).allowed
    assert not gate.evaluate("Three Hearth Burgers with bacon were added.", b"audio", evidence=[order_details], current_state_version=1).allowed
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
    assert not gate.evaluate("Hearth Burger was added to your order.", b"audio", current_state_version=1).allowed

    dietary = ToolEvidence(
        action="check_menu_item_availability",
        call_id="tool-dietary",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={
            "canonical_items": [{"name": "Hearth Burger", "allergens": ["dairy"]}],
            "items": ["Hearth Burger"],
        },
    )
    assert gate.evaluate("Hearth Burger contains dairy.", b"audio", evidence=[dietary], current_state_version=1).allowed
    assert not gate.evaluate("Hearth Burger contains peanuts.", b"audio", evidence=[dietary], current_state_version=1).allowed
    assert not gate.evaluate("Hearth Burger contains dairy and peanuts.", b"audio", evidence=[dietary], current_state_version=1).allowed
    assert not gate.evaluate("Hearth Burger has 900 calories.", b"audio", evidence=[dietary], current_state_version=1).allowed
    assert not gate.evaluate("Hearth Burger does not contain dairy.", b"audio", evidence=[dietary], current_state_version=1).allowed


@pytest.mark.asyncio
async def test_concurrent_audio_submissions_are_serialized():
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "first"},
            {"type": "response.output_audio_transcript.done", "response_id": "response-1", "transcript": "first reply"},
            {"type": "response.done", "response": {"id": "response-1", "status": "completed"}},
            {"type": "response.created", "response": {"id": "response-2"}},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "second"},
            {"type": "response.output_audio_transcript.done", "response_id": "response-2", "transcript": "second reply"},
            {"type": "response.done", "response": {"id": "response-2", "status": "completed"}},
        ]
    )
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    first, second = await asyncio.gather(
        adapter.submit_audio(b"first", turn_id="turn-1"),
        adapter.submit_audio(b"second", turn_id="turn-2"),
    )
    assert first.turn and first.turn.transcript == "first"
    assert second.turn and second.turn.transcript == "second"


def test_event_recorder_redacts_transcripts_arguments_and_personal_fields():
    recorder = EventRecorder()
    event = recorder.record(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "My name is Ada and my phone is 555-0100",
            "arguments": '{"customer_phone":"555-0100"}',
            "customer_phone": "555-0100",
            "item": {"content": [{"type": "input_text", "text": "My address is 123 Main Street"}]},
        }
    )
    payload = event.payload
    assert "transcript" not in payload
    assert "arguments" not in payload
    assert "555-0100" not in str(payload)
    assert "123 Main Street" not in str(payload)
    assert payload["item"]["content_parts"][0]["text_chars"] > 0
    assert payload["transcript_chars"] > 0
    assert payload["arguments_present"] is True


@pytest.mark.asyncio
async def test_interrupted_finalization_does_not_commit_turn():
    store = InMemoryOrderStateStore()
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response": {"id": "response-cancel"}},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "add a burger"},
            {"type": "response.done", "response": {"id": "response-cancel", "status": "completed"}},
        ]
    )
    adapter = None

    async def extractor(turn, state):
        await adapter.interrupt()
        return OrderPatch(source_turn_id=turn.turn_id, items=(item("menu.hearth", "Hearth Burger", 1),))

    adapter = NativeVoiceAdapter(
        session_id="call-1",
        transport=transport,
        state_store=store,
        facts_extractor=extractor,
    )
    await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-cancel")
    assert (await store.load("call-1")).finalized_turn_ids == ()


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
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "conversation.item.input_audio_transcription.delta", "delta": "hello"},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"},
            {"type": "response.output_audio.delta", "response_id": "response-1", "delta": base64.b64encode(output).decode()},
            {"type": "response.output_audio_transcript.delta", "response_id": "response-1", "delta": "How can I help?"},
            {"type": "response.output_audio_transcript.done", "response_id": "response-1", "transcript": "How can I help?"},
            {"type": "response.done", "response": {"id": "response-1", "status": "completed"}},
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
async def test_adapter_reconciles_input_transcript_after_response_done():
    output = b"late-transcript-audio"
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response": {"id": "response-late"}},
            {"type": "response.output_audio.delta", "response_id": "response-late", "delta": base64.b64encode(output).decode()},
            {"type": "response.output_audio_transcript.done", "response_id": "response-late", "transcript": "How can I help?"},
            {"type": "response.done", "response": {"id": "response-late", "status": "completed"}},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"},
        ]
    )
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    result = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-late")
    assert result.audio == output
    assert result.turn and result.turn.transcript == "hello"


@pytest.mark.asyncio
async def test_adapter_quarantines_transcript_after_non_completed_response():
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response": {"id": "response-cancelled"}},
            {"type": "input_audio_buffer.committed", "item_id": "item-old"},
            {"type": "response.done", "response": {"id": "response-cancelled", "status": "cancelled"}},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "item-old", "transcript": "old caller"},
            {"type": "response.created", "response": {"id": "response-current"}},
            {"type": "input_audio_buffer.committed", "item_id": "item-current"},
            {"type": "conversation.item.input_audio_transcription.completed", "item_id": "item-current", "transcript": "current caller"},
            {"type": "response.output_audio_transcript.done", "response_id": "response-current", "transcript": "How can I help?"},
            {"type": "response.done", "response": {"id": "response-current", "status": "completed"}},
        ]
    )
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    first = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-old")
    second = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-current")
    assert first.turn is None
    assert second.turn and second.turn.transcript == "current caller"


@pytest.mark.asyncio
async def test_adapter_suppresses_audio_without_assistant_transcript():
    output = b"unsupported-audio"
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello"},
            {"type": "response.output_audio.delta", "response_id": "response-1", "delta": base64.b64encode(output).decode()},
            {"type": "response.done", "response": {"id": "response-1", "status": "completed"}},
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
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "yes"},
            {"type": "response.function_call_arguments.done", "response_id": "response-1", "call_id": "tool-1", "name": "confirm_order", "arguments": '{"session_id":"call-1","expected_draft_version":2,"caller_approved_full_readback":true}'},
            {"type": "response.done", "response": {"id": "response-1", "status": "completed"}},
            {"type": "response.created", "response": {"id": "response-2"}},
            {"type": "response.output_audio.delta", "response_id": "response-2", "delta": base64.b64encode(b"confirmed-audio").decode()},
                {"type": "response.output_audio_transcript.delta", "response_id": "response-2", "delta": "Your order is confirmed."},
                {"type": "response.output_audio_transcript.done", "response_id": "response-2", "transcript": "Your order is confirmed."},
            {"type": "response.done", "response": {"id": "response-2", "status": "completed"}},
        ]
    )
    adapter = NativeVoiceAdapter(
        session_id="call-1",
        transport=transport,
        state_store=InMemoryOrderStateStore(),
        tool_bridge=ToolBridge(executor, scope_resolver=fake_order_scope),
    )
    result = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-1")
    assert result.audio == b"confirmed-audio"
    assert result.tool_outcomes[0].readback_verified
    assert any(event.event_type == "database_readback_verified" for event in adapter.recorder.events)
    tool_output = next(
        event["item"]["output"]
        for event in transport.sent
        if event.get("type") == "conversation.item.create"
    )
    payload = json.loads(tool_output)
    assert {"status", "result_id", "clarification_state", "speech", "facts"} <= set(payload)
    assert payload["facts"]["subject"]["order_id"] == 3
    assert "result" not in payload
    assert "customer" not in tool_output


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
            {"type": "response.created", "response": {"id": "response-1"}},
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
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "conversation.item.input_audio_transcription.completed", "transcript": "yes"},
            {"type": "response.function_call_arguments.done", "response_id": "response-1", "call_id": "tool-1", "name": "add_order_item", "arguments": "{\"session_id\":\"call-1\",\"item_name\":\"Hearth Burger\"}"},
            {"type": "response.done", "response": {"id": "response-1", "status": "completed"}},
            {"type": "response.created", "response": {"id": "response-2"}},
            {"type": "response.output_audio_transcript.delta", "response_id": "response-2", "delta": "Which item did you mean?"},
            {"type": "response.output_audio_transcript.done", "response_id": "response-2", "transcript": "Which item did you mean?"},
            {"type": "response.done", "response": {"id": "response-2", "status": "completed"}},
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
    previous_native = sys.modules.pop("app.native_voice", None)
    previous_main = sys.modules.pop("app.main", None)
    try:
        importlib.import_module("app.main")
        assert "app.native_voice" not in sys.modules
    finally:
        sys.modules.pop("app.main", None)
        if previous_main is not None:
            sys.modules["app.main"] = previous_main
        if previous_native is not None:
            sys.modules["app.native_voice"] = previous_native


def test_booking_claims_require_structured_date_time_and_reference_evidence():
    evidence = ToolEvidence(
        action="create_booking",
        call_id="booking-1",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={
            "status": "confirmed",
            "date": "2026-09-19",
            "time": "19:00",
            "timezone": "America/Los_Angeles",
            "reference": "7",
            "subject": {"booking_id": 7},
        },
    )
    gate = SpeechGate()
    assert gate.evaluate(
        "Your reservation is confirmed for 09/19/2026 at 19:00, booking reference 7.",
        b"audio",
        evidence=[evidence],
        current_state_version=1,
    ).allowed
    assert not gate.evaluate(
        "Your reservation is confirmed for 09/20/2026 at 20:00, booking reference 7.",
        b"audio",
        evidence=[evidence],
        current_state_version=1,
    ).allowed
    lookup = ToolEvidence(
        action="lookup_booking",
        call_id="booking-lookup",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts=evidence.facts,
    )
    assert not gate.evaluate(
        "Your reservation is on 2026-09-20 at 20:00.",
        b"audio",
        evidence=[lookup],
        current_state_version=1,
    ).allowed


def test_read_only_facts_do_not_require_mutation_confirmation_envelope():
    outcome = ToolOutcome(
        name="get_full_menu",
        call_id="menu-1",
        arguments={},
        result={"ok": True},
        success=True,
        readback_verified=True,
        facts={"prices": {"Hearth Burger": 21.0}, "canonical_items": [{"name": "Hearth Burger"}]},
        state_version=1,
    )
    evidence = outcome.as_evidence(turn_id="turn-1")
    assert not evidence.confirmation_text
    assert SpeechGate().evaluate(
        "Hearth Burger costs $21.00.", b"audio", evidence=[evidence], current_state_version=1
    ).allowed


@pytest.mark.asyncio
async def test_inmemory_adapter_uses_offline_executor(monkeypatch):
    monkeypatch.delenv("NATIVE_VOICE_DATABASE_URL", raising=False)
    adapter = NativeVoiceAdapter(
        session_id="offline-call",
        transport=MemoryRealtimeTransport(),
        state_store=InMemoryOrderStateStore(),
    )
    await adapter.start()


@pytest.mark.asyncio
async def test_empty_readback_notes_replace_application_memory():
    store = InMemoryOrderStateStore()
    await store.save("call-1", OrderState(version=1, order_notes="old note"), expected_version=0)
    adapter = NativeVoiceAdapter(
        session_id="call-1",
        transport=MemoryRealtimeTransport(),
        state_store=store,
        tool_bridge=ToolBridge(FakeExecutor()),
    )
    adapter._completed_turn = CompletedCallerTurn("turn-clear", 2, "clear the note", 0.0)
    adapter._outcomes = [
        ToolOutcome(
            name="set_order_notes",
            call_id="notes-1",
            arguments={"notes": ""},
            result={"ok": True},
            success=True,
            readback_verified=True,
            readback={"order_notes": "", "readback_committed": True},
        )
    ]
    state = await adapter._sync_order_memory()
    assert state is not None and state.order_notes == ""
    assert (await store.load("call-1")).order_notes == ""


@pytest.mark.asyncio
async def test_committed_mutation_replays_after_adapter_restart():
    store = InMemoryOrderStateStore()
    arguments = {
        "session_id": "call-1",
        "booking_id": 7,
        "customer_name": "Ada Lovelace",
        "customer_phone": "+14155550123",
        "caller_confirmed": True,
    }
    first = NativeVoiceAdapter(
        session_id="call-1",
        transport=MemoryRealtimeTransport(),
        state_store=store,
        tool_bridge=ToolBridge(OfflineToolExecutor()),
    )
    first._completed_turn = CompletedCallerTurn("turn-cancel", 1, "cancel it", 0.0)
    committed = ToolOutcome(
        name="cancel_booking",
        call_id="cancel-1",
        arguments=arguments,
        result={"booking_id": 7, "status": "cancelled"},
        success=True,
        readback_verified=True,
        readback={
            "booking_id": 7,
            "customer_name": "Ada Lovelace",
            "customer_phone": "+14155550123",
            "status": "cancelled",
            "date": "2026-09-19",
            "time": "19:00",
            "readback_committed": True,
        },
        facts={"status": "cancelled", "booking_id": 7, "reference": "7"},
    )
    committed = first._with_confirmation(committed)
    committed = await first._persist_committed_outcome(committed)
    assert committed.state_version == first.state.version

    restarted = NativeVoiceAdapter(
        session_id="call-1",
        transport=MemoryRealtimeTransport(),
        state_store=store,
        tool_bridge=ToolBridge(OfflineToolExecutor()),
    )
    restarted.state = await store.load("call-1")
    restarted._completed_turn = first._completed_turn
    replay = await restarted._durable_replay("cancel-2", "cancel_booking", arguments)
    assert replay is not None
    assert replay.replayed and replay.success and replay.readback_verified
    assert replay.result == committed.result
    restarted._replayed_finalized_turns.add("turn-cancel")
    same_operation = await restarted._run_tool("cancel-3", "cancel_booking", arguments, generation=0)
    different_operation = await restarted._run_tool(
        "cancel-4",
        "cancel_booking",
        {**arguments, "reason": "different"},
        generation=0,
    )
    assert same_operation.replayed and same_operation.success
    assert not different_operation.success and different_operation.error == "replayed_finalized_turn"


@pytest.mark.asyncio
async def test_malformed_readback_is_a_structured_failed_outcome():
    bridge = ToolBridge(FakeExecutor(result={"ok": True}, readback={"booking_id": "bad"}))
    outcome = await bridge.invoke(
        call_id="booking-1",
        name="create_booking",
        arguments={},
        turn_id="turn-1",
        state_version=1,
    )
    assert not outcome.success
    assert not outcome.readback_verified
    assert outcome.error == "database_readback_mismatch"


def test_native_voice_database_guard_requires_separate_approved_disposable_database(monkeypatch):
    monkeypatch.delenv("NATIVE_VOICE_DATABASE_URL", raising=False)
    with pytest.raises(NativeVoiceDatabaseGuardError):
        validate_native_voice_database()

    native_url = "postgresql://native:password@127.0.0.1:5432/native_voice"
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_URL", native_url)
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_WRITE_ENABLED", "true")
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_MARKER", "preprovisioned-disposable-marker")
    monkeypatch.setenv("DATABASE_URL", "postgresql://app:password@localhost:5432/native_voice")
    with pytest.raises(NativeVoiceDatabaseGuardError):
        validate_native_voice_database()


@pytest.mark.asyncio
async def test_native_voice_database_marker_is_verified_server_side(monkeypatch):
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_URL", "postgresql://native:password@localhost:5432/native_voice")
    monkeypatch.setenv("NATIVE_VOICE_DATABASE_MARKER", "preprovisioned-disposable-marker")

    class Pool:
        async def fetchrow(self, query):
            return {
                "server_host": "127.0.0.1",
                "server_port": 5432,
                "database_name": "native_voice",
                "marker": "preprovisioned-disposable-marker",
            }

    await verify_native_voice_database_connection(Pool())

    class WrongPool:
        async def fetchrow(self, query):
            return {
                "server_host": "127.0.0.1",
                "server_port": 5432,
                "database_name": "native_voice",
                "marker": "customer-database",
            }

    with pytest.raises(NativeVoiceDatabaseGuardError):
        await verify_native_voice_database_connection(WrongPool())


def test_item_confirmation_uses_the_committed_operation_item():
    outcome = ToolOutcome(
        name="add_order_item",
        call_id="item-2",
        arguments={"item_name": "House Lemonade"},
        result={"order_item_id": 2},
        success=True,
        readback_verified=True,
        readback={
            "items": [
                {"order_item_id": 1, "item_name": "Hearth Burger", "quantity": 1},
                {"order_item_id": 2, "item_name": "House Lemonade", "quantity": 2},
            ]
        },
        facts={"items": [{"item_name": "Hearth Burger"}, {"item_name": "House Lemonade"}]},
    )
    payload = NativeVoiceAdapter._model_tool_output(outcome)
    assert "House Lemonade" in payload["speech"]
    assert "Hearth Burger" not in payload["speech"]


@pytest.mark.asyncio
async def test_public_native_draft_update_maps_name_and_phone_aliases():
    class DraftService:
        def __init__(self):
            self.updates = None

        async def update_reservation_draft_native(self, session_id, updates):
            self.updates = (session_id, dict(updates))
            return {"customer_name": updates["customer_name"], "customer_phone": updates["customer_phone"]}

    service = DraftService()
    executor = RestaurantToolExecutor(service=service)
    result = await executor.invoke(
        "update_reservation_draft",
        {"session_id": "draft-alias", "name": "Ada Lovelace", "phone": "+14155550123"},
    )

    assert result == {"customer_name": "Ada Lovelace", "customer_phone": "+14155550123"}
    assert service.updates == (
        "draft-alias",
        {"customer_name": "Ada Lovelace", "customer_phone": "+14155550123"},
    )


@pytest.mark.asyncio
async def test_native_reservation_draft_readback_includes_complete_notes(monkeypatch):
    session_id = "draft-complete-readback"
    clear_call_memory(session_id)
    service = RestaurantService(pool_provider=lambda: None)

    async def load_call_state(call_id):
        return {
            "state": {
                "reservation_draft": {
                    "customer_name": "Ada Lovelace",
                    "customer_phone": "+14155550123",
                    "date": "2026-10-01",
                    "time": "19:00",
                    "party_size": 2,
                    "seating_preference": "patio",
                    "occasion": "birthday",
                    "dietary": "vegetarian",
                    "extra_notes": "window table",
                }
            }
        }

    monkeypatch.setattr(service, "load_call_state", load_call_state)
    draft = await service.get_reservation_draft(session_id, arm_confirmation=True)

    assert draft["proposed"]["notes"] == (
        "seating: patio; occasion: birthday; dietary: vegetarian; window table"
    )
    clear_call_memory(session_id)


@pytest.mark.asyncio
async def test_native_booking_update_readback_resolves_live_omitted_fields(monkeypatch):
    session_id = "booking-update-readback"
    clear_call_memory(session_id)
    service = RestaurantService(pool_provider=lambda: None)

    async def lookup_booking(**kwargs):
        return {
            "booking_id": 17,
            "customer_name": "Ada Lovelace",
            "customer_phone": "+14155550123",
            "date": "2026-10-01",
            "time": "19:00",
            "party_size": 4,
            "status": "confirmed",
            "location": "patio",
            "notes": "occasion: birthday",
        }

    async def persist_call_state(*args, **kwargs):
        return None

    monkeypatch.setattr(service, "lookup_booking", lookup_booking)
    monkeypatch.setattr(service, "persist_call_state", persist_call_state)
    pending = await service.update_confirmed_booking(
        call_id=session_id,
        idempotency_key="booking-update-readback",
        booking_id=17,
        confirmed=False,
        customer_name="Grace Hopper",
    )

    proposed = pending["proposed"]
    assert proposed["date"] == "2026-10-01"
    assert proposed["time"] == "19:00"
    assert proposed["party_size"] == 4
    assert proposed["customer_name"] == "Grace Hopper"
    assert proposed["customer_phone"] == "+14155550123"
    assert proposed["preferred_location"] == "patio"
    assert proposed["notes"] == "occasion: birthday"
    sentence = NativeVoiceAdapter._confirmation_sentence(
        ToolOutcome(
            name="update_confirmed_booking",
            call_id="update-readback",
            arguments={},
            result=pending,
            success=True,
            pending=True,
            facts={},
        )
    )
    assert "date 2026-10-01, time 19:00" in sentence
    assert "party size 4" in sentence
    assert "Grace Hopper" in sentence
    assert "occasion: birthday" in sentence
    clear_call_memory(session_id)


def test_native_order_confirmation_reads_back_all_item_effects():
    sentence = NativeVoiceAdapter._confirmation_sentence(
        ToolOutcome(
            name="get_order_summary",
            call_id="order-readback",
            arguments={},
            result={"pending": True, "pending_confirmation_hash": "hash", "order_id": 7},
            success=True,
            pending=True,
            facts={
                "order_id": 7,
                "canonical_items": [
                    {
                        "name": "Hearth Burger",
                        "quantity": 1,
                        "modifiers": ["fries"],
                        "removals": ["onion jam"],
                        "substitutions": ["gluten-free bun"],
                        "notes": "cut in half",
                    }
                ],
            },
        )
    )

    assert "modifiers ['fries']" in sentence
    assert "removals ['onion jam']" in sentence
    assert "substitutions ['gluten-free bun']" in sentence
    assert "note cut in half" in sentence


def test_native_model_facts_preserve_unresolved_candidates():
    output = NativeVoiceAdapter._model_tool_output(
        ToolOutcome(
            name="get_reservation_draft",
            call_id="unresolved-readback",
            arguments={},
            result={"pending": True, "pending_confirmation_hash": "hash"},
            success=True,
            pending=True,
            facts={
                "unresolved_fields": [
                    {
                        "field": "date",
                        "reason": "ambiguous",
                        "candidates": ["Friday", "Saturday"],
                        "source_turn_id": "turn-1",
                    }
                ]
            },
        )
    )

    assert output["facts"]["unresolved_fields"] == [
        {
            "field": "date",
            "reason": "ambiguous",
            "candidates": ["Friday", "Saturday"],
            "source_turn_id": "turn-1",
        }
    ]


def test_native_booking_update_speech_includes_every_effective_field():
    sentence = NativeVoiceAdapter._confirmation_sentence(
        ToolOutcome(
            name="update_confirmed_booking",
            call_id="complete-update-readback",
            arguments={},
            result={
                "pending": True,
                "proposed": {
                    "booking_id": 17,
                    "date": "2026-10-01",
                    "time": "19:00",
                    "party_size": 2,
                    "preferred_location": "",
                    "customer_name": "Ada Lovelace",
                    "customer_phone": "+14155550123",
                    "seating_preference": "",
                    "extra_notes": "",
                    "require_approval_for_paid_items": False,
                },
            },
            success=True,
            pending=True,
            facts={},
        )
    )

    assert "callback phone +14155550123" in sentence
    assert "preferred location cleared" in sentence
    assert "seating preference cleared" in sentence
    assert "extra notes cleared" in sentence
    assert "paid-item approval off" in sentence


def test_consequential_speech_requires_exact_application_confirmation():
    confirmation = "Your reservation is confirmed."
    evidence = ToolEvidence(
        action="create_booking",
        call_id="booking-1",
        turn_id="turn-1",
        state_version=1,
        success=True,
        readback_verified=True,
        facts={"status": "confirmed"},
        confirmation_text=confirmation,
        confirmation_hash=hashlib.sha256(confirmation.casefold().encode()).hexdigest(),
    )
    gate = SpeechGate()
    assert not gate.evaluate("Your reservation is all set.", b"audio", evidence=[evidence], current_state_version=1).allowed
    assert gate.evaluate(confirmation, b"audio", evidence=[evidence], current_state_version=1).allowed


@pytest.mark.asyncio
async def test_ga_function_call_done_dispatches_nested_item():
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response": {"id": "response-tool"}},
            {
                "type": "response.output_item.done",
                "response_id": "response-tool",
                "item": {
                    "id": "item-tool",
                    "type": "function_call",
                    "call_id": "call-tool",
                    "name": "get_full_menu",
                    "arguments": "{}",
                },
            },
            {"type": "response.done", "response": {"id": "response-tool", "status": "incomplete"}},
        ]
    )
    adapter = NativeVoiceAdapter(
        session_id="call-1",
        transport=transport,
        state_store=InMemoryOrderStateStore(),
    )
    await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-tool")
    assert any(
        event.event_type == "tool_requested"
        and event.payload.get("call_id") == "call-tool"
        and event.payload.get("name") == "get_full_menu"
        for event in adapter.recorder.events
    )


@pytest.mark.asyncio
async def test_realtime_failure_is_terminal_and_quarantines_input():
    transport = MemoryRealtimeTransport(
        [
            {"type": "response.created", "response": {"id": "response-failed"}},
            {"type": "input_audio_buffer.committed", "item_id": "item-failed"},
            {"type": "conversation.item.input_audio_transcription.failed", "item_id": "item-failed", "error": {"code": "transcription_failed"}},
        ]
    )
    adapter = NativeVoiceAdapter(session_id="call-1", transport=transport, state_store=InMemoryOrderStateStore())
    result = await adapter.submit_audio(b"synthetic-pcm", turn_id="turn-failed")
    assert result.speech and not result.speech.allowed
    assert "item-failed" in adapter._quarantined_input_item_ids
    assert adapter.interruptions.generation == 1


@pytest.mark.asyncio
async def test_unresolved_reservation_allows_only_scoped_correction(monkeypatch):
    store = InMemoryOrderStateStore()
    state = OrderState().apply(
        OrderPatch(
            source_turn_id="turn-0",
            unresolved_fields=(UnresolvedField("date", "ambiguous date", ("Friday", "Saturday"), "turn-0"),),
        )
    )
    await store.save("call-1", state, expected_version=0)
    executor = FakeExecutor(
        result={"ok": True},
        readback={
            "readback_committed": True,
            "customer_name": "Ada Lovelace",
            "customer_phone": "+14155550123",
            "date": "2026-09-19",
            "time": "19:00",
            "party_size": 2,
        },
    )
    adapter = NativeVoiceAdapter(
        session_id="call-1",
        transport=MemoryRealtimeTransport(),
        state_store=store,
        tool_bridge=ToolBridge(executor),
    )
    adapter.state = state
    adapter._completed_turn = CompletedCallerTurn("turn-1", 1, "correct the date", 0.0)
    outcome = await adapter._run_tool(
        "correction-1",
        "update_reservation_draft",
        {"session_id": "call-1", "date": "2026-09-19"},
        generation=0,
    )
    assert outcome.success and outcome.readback_verified
    assert not (await store.load("call-1")).unresolved_fields
    assert executor.calls
