"""Native voice public booking lifecycle regression."""

import base64
import json
import os
import uuid
from datetime import datetime, timedelta

import pytest

from app.native_voice.adapter import NativeVoiceAdapter
from app.native_voice.database_guard import close_native_voice_pool, get_native_voice_pool
from app.native_voice.protocol import MemoryRealtimeTransport
from app.services.restaurant import _restaurant_now
from app.restaurant_knowledge import get_restaurant_knowledge
from zoneinfo import ZoneInfo


@pytest.fixture(autouse=True)
def native_open_restaurant_clock(monkeypatch):
    if os.getenv("RUN_DB_INTEGRATION") == "1":
        timezone_info = ZoneInfo(get_restaurant_knowledge().identity["timezone"])
        monkeypatch.setattr(
            "app.services.restaurant._restaurant_now",
            lambda: datetime(2026, 9, 23, 18, 0, tzinfo=timezone_info),
        )


class ToolSpeechTransport(MemoryRealtimeTransport):
    def prepare(self, name, arguments, transcript, turn_id):
        self.incoming = [
            {"type": "response.created", "response": {"id": turn_id + "-tool"}},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": transcript,
            },
            {
                "type": "response.function_call_arguments.done",
                "response_id": turn_id + "-tool",
                "call_id": turn_id + "-call",
                "name": name,
                "arguments": json.dumps(arguments),
            },
            {"type": "response.done", "response": {"id": turn_id + "-tool", "status": "completed"}},
            {"type": "response.created", "response": {"id": turn_id + "-speech"}},
            {
                "type": "response.output_audio.delta",
                "response_id": turn_id + "-speech",
                "delta": base64.b64encode(b"\x00\x00" * 2400).decode(),
            },
            {
                "type": "response.output_audio_transcript.done",
                "response_id": turn_id + "-speech",
                "canonical_speech": True,
            },
            {"type": "response.done", "response": {"id": turn_id + "-speech", "status": "completed"}},
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
            assert outputs, "no tool output for spoken response"
            event["transcript"] = outputs[-1].get("speech", "")
        return event


@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="explicit disposable native PostgreSQL required",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("with_alias_whitespace", [False, True])
@pytest.mark.parametrize("note_resolution", ["normal", "clear", "replace"])
async def test_public_native_booking_lifecycle(with_alias_whitespace, note_resolution):
    session_id = "public-booking-flow-" + uuid.uuid4().hex
    moment = _restaurant_now() + timedelta(days=2)
    while moment.weekday() == 0:
        moment += timedelta(days=1)
    date = moment.date().isoformat()
    transport = ToolSpeechTransport()
    adapter = NativeVoiceAdapter(session_id=session_id, transport=transport)
    pool = await get_native_voice_pool()
    booking_id = 0

    async def call(name, arguments, transcript, turn_id):
        transport.prepare(name, arguments, transcript, turn_id)
        result = await adapter.submit_audio(b"\x00\x00" * 2400, turn_id=turn_id)
        assert result.tool_outcomes, "no tool outcome for " + name
        return result, result.tool_outcomes[0]

    draft_args = {
        "name": "Synthetic Booking Guest",
        "phone": "+15035550108",
        "date": date,
        "time": "19:00",
        "party_size": 2,
        "occasion": "birthday",
        "dietary": "vegan",
        "require_approval_for_paid_items": True,
    }
    if with_alias_whitespace:
        draft_args["name"] = "  Synthetic   Booking Guest "
        draft_args["phone"] = " +1 (503) 555-0108 "

    try:
        _, drafted = await call(
            "update_reservation_draft",
            draft_args,
            "A table for two at seven, under Synthetic Booking Guest.",
            "draft",
        )
        assert drafted.success, f"draft failed: {drafted.error}"
        readback, proposal = await call(
            "get_reservation_draft",
            {},
            "Please read back my reservation details.",
            "readback",
        )
        assert proposal.pending
        assert readback.audio and readback.speech and readback.speech.allowed
        _, created = await call(
            "create_booking",
            {
                **draft_args,
                "notes": "occasion: birthday; dietary: vegan",
                "caller_confirmed": True,
            },
            "Yes, those details are correct. Book it.",
            "create",
        )
        assert created.success and created.readback_verified, created.error
        booking_id = int(created.result["booking_id"])
        row = await pool.fetchrow(
            "SELECT status, party_size, booked_at, notes, require_approval_for_paid_items FROM bookings WHERE id = $1",
            booking_id,
        )
        assert row["status"] == "confirmed"
        assert row["party_size"] == 2
        assert row["booked_at"].date().isoformat() == date
        assert "occasion: birthday" in row["notes"]
        assert "dietary: vegan" in row["notes"]
        assert row["require_approval_for_paid_items"] is True

        # Knowing the exact booking ID, name and phone is not session ownership.
        from app.native_voice.tools import RestaurantToolExecutor, ToolBridge
        from app.services.restaurant import RestaurantService
        stranger = ToolBridge(
            RestaurantToolExecutor(service=RestaurantService(pool_provider=get_native_voice_pool)),
            session_id="unverified-" + uuid.uuid4().hex,
        )
        denied = await stranger.invoke(
            call_id="foreign-lookup", name="lookup_booking",
            arguments={"booking_id": booking_id, "customer_name": "Synthetic Booking Guest",
                       "customer_phone": "+15035550108"},
            turn_id="foreign", state_version=0,
        )
        assert not denied.success and denied.error == "booking_scope_unverified"
        assert denied.result is None

        _, availability = await call(
            "check_table_availability",
            {"date": date, "time": "19:00", "party_size": 3},
            "Can we change that reservation to three guests?",
            "availability",
        )
        assert availability.success, availability.error
        update_args = {"booking_id": booking_id, "party_size": 3, "caller_confirmed": False}
        update_readback, update_proposal = await call(
            "update_confirmed_booking",
            update_args,
            "Please propose the change to three guests.",
            "update-proposal",
        )
        assert update_proposal.pending
        assert update_readback.audio and update_readback.speech and update_readback.speech.allowed
        assert "occasion: birthday" in update_readback.transcript
        assert "dietary: vegan" in update_readback.transcript
        assert "paid-item approval on" in update_readback.transcript
        assert await pool.fetchval("SELECT party_size FROM bookings WHERE id = $1", booking_id) == 2
        _, updated = await call(
            "update_confirmed_booking",
            {**update_args, "caller_confirmed": True},
            "Yes, that is correct.",
            "update-approval",
        )
        assert updated.success and updated.readback_verified, updated.error
        after = await pool.fetchrow(
            "SELECT party_size, booked_at, notes, require_approval_for_paid_items FROM bookings WHERE id = $1",
            booking_id,
        )
        assert after["party_size"] == 3
        assert after["booked_at"] == row["booked_at"]
        assert "occasion: birthday" in after["notes"]
        assert "dietary: vegan" in after["notes"]
        assert after["require_approval_for_paid_items"] is True

        rename_args = {
            "booking_id": booking_id,
            "customer_name": "Synthetic Updated Guest",
            "caller_confirmed": False,
        }
        renamed_readback, rename_proposal = await call(
            "update_confirmed_booking",
            rename_args,
            "Please update the reservation name to Synthetic Updated Guest.",
            "rename-proposal",
        )
        assert rename_proposal.pending
        assert renamed_readback.audio and renamed_readback.speech and renamed_readback.speech.allowed
        assert "Synthetic Updated Guest".casefold() in renamed_readback.transcript.casefold()
        _, renamed = await call(
            "update_confirmed_booking",
            {**rename_args, "caller_confirmed": True},
            "Yes, that is correct.",
            "rename-approval",
        )
        assert renamed.success and renamed.readback_verified, renamed.error
        assert await pool.fetchval("SELECT customer_name FROM bookings WHERE id = $1", booking_id) == "Synthetic Updated Guest"

        if note_resolution != "normal":
            _, appended = await call(
                "add_guest_note",
                {"booking_id": booking_id, "note": "dietary: sesame allergy"},
                "Please add this guest note: dietary: sesame allergy.",
                "duplicate-dietary-note",
            )
            assert appended.success and appended.readback_verified, appended.error
            ambiguous_notes = await pool.fetchval(
                "SELECT notes FROM bookings WHERE id = $1", booking_id
            )
            assert "dietary: vegan" in ambiguous_notes
            assert "dietary: sesame allergy" in ambiguous_notes
            blocked_readback, blocked = await call(
                "update_confirmed_booking",
                {**rename_args, "customer_name": "Another Synthetic Name"},
                "Please change only the reservation name.",
                "ambiguous-name-change",
            )
            assert not blocked.success and not blocked.pending
            assert "booking_notes_ambiguous" in str(blocked.error)
            assert "cleared" not in blocked_readback.transcript.casefold()
            assert "multiple different dietary" in blocked_readback.transcript
            assert blocked_readback.audio and blocked_readback.speech.allowed, blocked_readback.speech
            assert await pool.fetchval(
                "SELECT notes FROM bookings WHERE id = $1", booking_id
            ) == ambiguous_notes
            assert await pool.fetchval(
                "SELECT customer_name FROM bookings WHERE id = $1", booking_id
            ) == "Synthetic Updated Guest"

        replacement_dietary = "sesame allergy" if note_resolution == "replace" else ""
        clear_args = {
            "booking_id": booking_id,
            "dietary": replacement_dietary,
            "caller_confirmed": False,
        }
        cleared_readback, cleared_proposal = await call(
            "update_confirmed_booking",
            clear_args,
            "Clear the dietary request from the reservation.",
            "dietary-clear-proposal",
        )
        assert cleared_proposal.pending
        assert (
            "dietary request sesame allergy" if replacement_dietary
            else "dietary request cleared"
        ) in cleared_readback.transcript
        _, cleared = await call(
            "update_confirmed_booking",
            {**clear_args, "caller_confirmed": True},
            "Yes, that is correct.",
            "dietary-clear-approval",
        )
        assert cleared.success and cleared.readback_verified, cleared.error
        cleared_row = await pool.fetchrow(
            "SELECT notes, require_approval_for_paid_items FROM bookings WHERE id = $1",
            booking_id,
        )
        assert "occasion: birthday" in cleared_row["notes"]
        assert "dietary: vegan" not in cleared_row["notes"]
        if replacement_dietary:
            assert cleared_row["notes"].count("dietary:") == 1
            assert "dietary: sesame allergy" in cleared_row["notes"]
        else:
            assert "dietary:" not in cleared_row["notes"]
        assert cleared_row["require_approval_for_paid_items"] is True
    finally:
        await adapter.close()
        async with pool.acquire() as conn:
            if booking_id:
                await conn.execute("DELETE FROM voice_action_idempotency WHERE call_id = $1", session_id)
                await conn.execute("DELETE FROM bookings WHERE id = $1", booking_id)
            await conn.execute("DELETE FROM call_sessions WHERE session_id = $1", session_id)
        await close_native_voice_pool()


def test_booking_notes_ambiguity_requires_explicit_resolution():
    from app.reservation_draft import AmbiguousBookingNotes, compose_notes, draft_from_booking

    booking = {"notes": "occasion: birthday; dietary: vegan; dietary: sesame allergy"}
    for omitted in ({}, {"dietary": None}, {"occasion": ""}):
        with pytest.raises(AmbiguousBookingNotes) as error:
            draft_from_booking(booking, note_updates=omitted)
        assert error.value.fields == ("dietary",)
    cleared = draft_from_booking(booking, note_updates={"dietary": ""})
    assert cleared["occasion"] == "birthday"
    assert cleared["dietary"] == ""
    assert compose_notes(cleared) == "occasion: birthday"
    replaced = draft_from_booking(booking, note_updates={"dietary": "sesame allergy"})
    assert compose_notes(replaced) == "occasion: birthday; dietary: sesame allergy"


def test_identical_booking_note_entries_do_not_create_ambiguity():
    from app.reservation_draft import compose_notes, draft_from_booking

    draft = draft_from_booking(
        {"notes": "occasion: birthday; dietary: vegan; dietary: vegan; bring a card"}
    )
    assert compose_notes(draft) == "occasion: birthday; dietary: vegan; bring a card"


def test_booking_clarification_cannot_authorize_other_speech():
    import hashlib
    from dataclasses import replace
    from app.native_voice.speech import SpeechGate, ToolEvidence

    text = "Please specify the complete dietary values to keep."
    evidence = ToolEvidence(
        action="update_confirmed_booking", call_id="clarify", turn_id="turn",
        state_version=3, success=False, readback_verified=False,
        facts={"safe_clarification": text}, confirmation_text=text,
        confirmation_hash=hashlib.sha256(text.casefold().encode()).hexdigest(),
    )
    gate = SpeechGate()
    assert gate.evaluate(text, b"audio", evidence=[evidence], current_state_version=3).allowed
    for candidate in (replace(evidence, facts={}), replace(evidence, state_version=2),
                      replace(evidence, replayed=True), replace(evidence, confirmation_hash="bad")):
        assert not gate.evaluate(text, b"audio", evidence=[candidate], current_state_version=3).allowed
    assert not gate.evaluate(
        text + " Your reservation is confirmed.", b"audio",
        evidence=[evidence], current_state_version=3
    ).allowed


def test_model_tool_schemas_do_not_request_server_session_identity():
    from app.native_voice.tools import realtime_tool_definitions

    for tool in realtime_tool_definitions():
        assert "session_id" not in tool["parameters"]["properties"]
        assert "session_id" not in tool["parameters"]["required"]
