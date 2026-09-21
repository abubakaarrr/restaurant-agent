"""Native voice public booking lifecycle regression."""

import base64
import json
import os
import uuid
from datetime import timedelta

import pytest

from app.native_voice.adapter import NativeVoiceAdapter
from app.native_voice.database_guard import close_native_voice_pool, get_native_voice_pool
from app.native_voice.protocol import MemoryRealtimeTransport
from app.services.restaurant import _restaurant_now


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
async def test_public_native_booking_lifecycle(with_alias_whitespace):
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
        transport.prepare(name, {"session_id": session_id, **arguments}, transcript, turn_id)
        result = await adapter.submit_audio(b"\x00\x00" * 2400, turn_id=turn_id)
        assert result.tool_outcomes, "no tool outcome for " + name
        return result, result.tool_outcomes[0]

    draft_args = {
        "name": "Synthetic Booking Guest",
        "phone": "+15035550108",
        "date": date,
        "time": "19:00",
        "party_size": 2,
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
            {**draft_args, "caller_confirmed": True},
            "Yes, those details are correct. Book it.",
            "create",
        )
        assert created.success and created.readback_verified, created.error
        booking_id = int(created.result["booking_id"])
        row = await pool.fetchrow(
            "SELECT status, party_size, booked_at FROM bookings WHERE id = $1",
            booking_id,
        )
        assert row["status"] == "confirmed"
        assert row["party_size"] == 2
        assert row["booked_at"].date().isoformat() == date

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
        assert await pool.fetchval("SELECT party_size FROM bookings WHERE id = $1", booking_id) == 2
        _, updated = await call(
            "update_confirmed_booking",
            {**update_args, "caller_confirmed": True},
            "Yes, that is correct.",
            "update-approval",
        )
        assert updated.success and updated.readback_verified, updated.error
        after = await pool.fetchrow(
            "SELECT party_size, booked_at FROM bookings WHERE id = $1",
            booking_id,
        )
        assert after["party_size"] == 3
        assert after["booked_at"] == row["booked_at"]

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
    finally:
        await adapter.close()
        async with pool.acquire() as conn:
            if booking_id:
                await conn.execute("DELETE FROM voice_action_idempotency WHERE call_id = $1", session_id)
                await conn.execute("DELETE FROM bookings WHERE id = $1", booking_id)
            await conn.execute("DELETE FROM call_sessions WHERE session_id = $1", session_id)
        await close_native_voice_pool()
