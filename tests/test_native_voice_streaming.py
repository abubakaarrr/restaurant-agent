"""Offline streaming controller race and playback-contract regressions."""
import asyncio
import base64
from contextlib import suppress
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.native_voice.adapter import InterruptionController, NativeVoiceAdapter
from app.native_voice.contracts import OrderState, OrderItemState
from app.native_voice.protocol import MemoryRealtimeTransport
from app.native_voice.state_store import InMemoryOrderStateStore
from app.native_voice.streaming import StreamingVoiceSession, streaming_config
from app.native_voice.tools import ToolOutcome


class QueueTransport:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.sent_event = asyncio.Event()
        self.closed = False

    async def receive(self):
        event = await self.incoming.get()
        self.incoming.task_done()
        return event

    async def send(self, event):
        self.sent.append(event)
        self.sent_event.set()

    async def close(self):
        self.closed = True

    async def feed(self, *events):
        for event in events:
            await self.incoming.put(event)
        await asyncio.wait_for(self.incoming.join(), 1)
        await asyncio.sleep(0)


class PlaybackDomain:
    def __init__(self):
        self.state = OrderState()
        self.interruptions = InterruptionController()
        self._outcomes = ["current"]
        self.releases = []

    async def _release_pending_readbacks(self, text, version):
        self.releases.append((text, version, tuple(self._outcomes)))


def session_for(domain=None):
    events = []

    async def send(event):
        events.append(event)

    session = StreamingVoiceSession(
        domain=domain or PlaybackDomain(),
        transport=QueueTransport(),
        speech_client=SimpleNamespace(),
        send=send,
        config={},
    )
    return session, events


async def cancel_task(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


def test_late_response_done_cannot_complete_current_turn():
    async def scenario():
        session, _ = session_for()
        reader = asyncio.create_task(session.read_events())
        response = asyncio.create_task(session.response(0))
        await session.transport.sent_event.wait()
        await session.transport.feed(
            {"type": "response.created", "response": {"id": "current"}},
            {"type": "response.done", "response": {"id": "old", "status": "completed", "output": [{"type": "function_call", "name": "confirm_order"}]}},
        )
        assert not response.done()
        assert session.active_response == "current"
        await session.transport.feed(
            {"type": "response.created", "response": {"id": "unexpected"}},
            {"type": "response.done", "response": {"id": "unexpected", "status": "completed"}},
        )
        assert not response.done()
        assert {"type": "response.cancel", "response_id": "unexpected"} in session.transport.sent
        await session.transport.feed({"type": "response.done", "response": {"id": "current", "status": "completed", "output": []}})
        assert (await response)["id"] == "current"
        await cancel_task(reader)
    asyncio.run(scenario())


def test_response_timeout_retires_connection_and_disallows_next_response(monkeypatch):
    original_wait_for = asyncio.wait_for

    async def timeout_model_response(awaitable, timeout):
        if timeout == 30:
            awaitable.cancel()
            raise TimeoutError
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", timeout_model_response)

    async def scenario():
        session, _ = session_for()
        with pytest.raises(TimeoutError):
            await session.response(0)
        assert session.failed and session.transport.closed
        with pytest.raises(RuntimeError, match="voice_session_failed"):
            await session.response(1)
        assert len(session.transport.sent) == 1
    asyncio.run(scenario())


def test_interrupted_transcription_does_not_delay_new_turn():
    async def scenario():
        session, _ = session_for()
        processed = asyncio.Event()
        seen = []

        async def run_turn(item_id, transcript, epoch, *, initial_response=None):
            if initial_response is not None:
                await initial_response
            seen.append((item_id, transcript, epoch))
            processed.set()

        session.run_turn = run_turn
        async def response(epoch):
            return {"status": "cancelled", "output": []}
        session.response = response
        worker = asyncio.create_task(session.work())
        await session.queue.put(("old", 0))
        # Wait until the old unresolved transcript is actively being awaited.
        for _ in range(20):
            if "old" in session.transcripts:
                break
            await asyncio.sleep(0)
        assert "old" in session.transcripts
        await session.interrupt()
        session.transcript_future("new").set_result("Actually, make that two.")
        await session.queue.put(("new", session.epoch))
        await asyncio.wait_for(processed.wait(), 1)
        assert seen == [("new", "Actually, make that two.", 1)]
        assert not session.transcript_future("old").cancelled()
        await cancel_task(worker)
    asyncio.run(scenario())


def test_old_queued_turn_is_discarded_without_waiting_for_transcription():
    async def scenario():
        session, _ = session_for()
        session.epoch = 1
        worker = asyncio.create_task(session.work())
        await session.queue.put(("obsolete", 0))
        await asyncio.wait_for(session.queue.join(), 1)
        assert "obsolete" not in session.transcripts
        await cancel_task(worker)
    asyncio.run(scenario())


def test_mutation_after_summary_never_speaks_stale_direct_readback():
    async def scenario():
        domain = NativeVoiceAdapter(
            session_id="stream-test-stale-summary",
            transport=MemoryRealtimeTransport(),
            state_store=InMemoryOrderStateStore(),
        )
        session, _ = session_for(domain)
        spoken = []
        responses = [
            {"status": "completed", "output": [
                {"type": "function_call", "name": "get_order_summary", "call_id": "summary", "arguments": "{}"},
                {"type": "function_call", "name": "add_order_item", "call_id": "add", "arguments": "{}"},
            ]},
            {"status": "completed", "output": [
                {"type": "function_call", "name": "get_order_summary", "call_id": "fresh-summary", "arguments": "{}"},
            ]},
        ]
        response_count = 0

        async def response(epoch):
            nonlocal response_count
            response_count += 1
            return responses.pop(0)

        async def run_tool(call_id, name, args, generation):
            if name == "get_order_summary":
                return ToolOutcome(name, call_id, args, {}, False,
                    pending=True, state_version=domain.state.version,
                    confirmation_text=("Fresh two-item readback. Shall I confirm?" if call_id == "fresh-summary" else "Old one-item readback. Shall I confirm?"))
            domain.state = replace(domain.state, version=domain.state.version + 1)
            return ToolOutcome(name, call_id, args, {}, True,
                readback_verified=True, state_version=domain.state.version)

        async def passthrough(outcome):
            return outcome

        async def sync_order_memory(generation=None):
            return None

        async def speak(text, epoch, **kwargs):
            spoken.append(text)

        domain._run_tool = run_tool
        domain._with_confirmation = lambda outcome: outcome
        domain._persist_committed_outcome = passthrough
        domain._sync_order_memory = sync_order_memory
        session.response = response
        session.speak = speak
        await session.run_turn("turn-a", "One burger and add a lemonade.", 0)
        assert response_count == 2
        assert spoken == ["Fresh two-item readback. Shall I confirm?"]
    asyncio.run(scenario())


def utterance(session, **overrides):
    value = {
        "token": "spoken", "epoch": session.epoch, "text": "Verified full readback",
        "version": session.domain.state.version, "outcomes": (ToolOutcome("snapshot", "snapshot", {}, None, False),),
        "bytes": 48000, "started": 90.0, "done": True, "played": False,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize("case", ["early", "unfinished", "wrong-token", "old-epoch", "interrupted"])
def test_invalid_playback_ack_cannot_release_confirmation(monkeypatch, case):
    monkeypatch.setattr("app.native_voice.streaming.time.monotonic", lambda: 100.0)

    async def scenario():
        session, _ = session_for()
        session.utterance = utterance(session)
        token = "spoken"
        if case == "early":
            session.utterance["started"] = 99.5
        elif case == "unfinished":
            session.utterance["done"] = False
        elif case == "wrong-token":
            token = "other-response"
        elif case == "old-epoch":
            session.epoch += 1
        elif case == "interrupted":
            await session.interrupt()
        assert await session.played(token) is False
        assert session.domain.releases == []
    asyncio.run(scenario())


def test_completed_playback_releases_snapshot_once(monkeypatch):
    monkeypatch.setattr("app.native_voice.streaming.time.monotonic", lambda: 100.0)

    async def scenario():
        session, _ = session_for()
        session.utterance = utterance(session)
        original = session.domain._outcomes
        assert await session.played("spoken") is True
        assert session.domain.releases == [("Verified full readback", 0, session.utterance["outcomes"])]
        assert session.domain._outcomes is original
        assert await session.played("spoken") is False
        assert len(session.domain.releases) == 1
    asyncio.run(scenario())


def test_playback_of_stale_state_never_releases_confirmation(monkeypatch):
    monkeypatch.setattr("app.native_voice.streaming.time.monotonic", lambda: 100.0)

    async def scenario():
        session, _ = session_for()
        session.utterance = utterance(session)
        session.domain.state = replace(session.domain.state, version=1)
        await session.played("spoken")
        assert session.domain.releases == []
    asyncio.run(scenario())


def test_audio_chunks_are_forwarded_without_local_commit():
    async def scenario():
        session, _ = session_for()
        audio = b"\x00\x00" * 1200
        await session.append_audio(audio)
        await session.append_audio(audio)
        assert [e["type"] for e in session.transport.sent] == ["input_audio_buffer.append"] * 2
        assert base64.b64decode(session.transport.sent[0]["audio"]) == audio
    asyncio.run(scenario())


@pytest.mark.parametrize(("pace", "microphone", "eagerness"), [
    ("natural", "near_field", None),
    ("patient", "far_field", "low"),
])
def test_provider_owns_continuous_turn_detection(pace, microphone, eagerness):
    session = streaming_config(pace=pace, microphone=microphone)["session"]
    audio = session["audio"]["input"]
    assert session["output_modalities"] == ["text"]
    expected = (
        {"type": "semantic_vad", "eagerness": "low",
         "create_response": False, "interrupt_response": False}
        if pace == "patient" else
        {"type": "server_vad", "threshold": .5, "prefix_padding_ms": 300,
         "silence_duration_ms": 500, "create_response": False, "interrupt_response": False}
    )
    assert audio["turn_detection"] == expected
    assert audio["noise_reduction"] == {"type": microphone}
    assert audio["transcription"]["language"] == "en"
    assert any(tool["name"] == "remember_request" for tool in session["tools"])

def test_interruption_during_tool_commit_preserves_verified_operation():
    async def scenario():
        domain = NativeVoiceAdapter(
            session_id="stream-test-interrupted-write",
            transport=MemoryRealtimeTransport(),
            state_store=InMemoryOrderStateStore(),
        )
        session, events = session_for(domain)
        tool_started, finish_tool = asyncio.Event(), asyncio.Event()
        spoken = []

        async def response(epoch):
            return {"status": "completed", "output": [
                {"type": "function_call", "name": "add_order_item", "call_id": "committing", "arguments": "{}"},
            ]}

        async def run_tool(call_id, name, args, generation):
            tool_started.set()
            await finish_tool.wait()
            return ToolOutcome(name, call_id, args, {"order_id": 7}, True,
                readback={"items": [], "order_id": 7, "status": "draft"},
                readback_verified=True, state_version=domain.state.version)

        async def speak(text, epoch, **kwargs):
            spoken.append(text)

        domain._run_tool = run_tool
        domain._with_confirmation = lambda outcome: outcome
        session.response = response
        session.speak = speak
        task = asyncio.create_task(session.run_turn("original-turn", "Add a burger.", 0))
        await asyncio.wait_for(tool_started.wait(), 1)
        await session.interrupt()
        finish_tool.set()
        await asyncio.wait_for(task, 1)
        saved = await domain.state_store.load(domain.session_id)
        assert len(saved.committed_operations) == 1
        assert saved.committed_operations[0].turn_id == "original-turn"
        assert saved.committed_operations[0].operation == "add_order_item"
        assert saved.status == "draft"
        assert not spoken
    asyncio.run(scenario())


async def batch_session(invoke):
    bridge = SimpleNamespace(invoke=invoke)
    domain = NativeVoiceAdapter(
        session_id="stream-batch-regression",
        transport=MemoryRealtimeTransport(),
        state_store=InMemoryOrderStateStore(),
        tool_bridge=bridge,
    )
    bridge.operation_id = lambda name, args, turn_id: domain._operation_id(name, args, turn_id)
    domain.turns.start("batch-turn")
    await domain._finalize_caller_turn("Please update my order.", generation=0)
    return session_for(domain)[0]


def batch_outcome(name, call_id, arguments, state_version, **overrides):
    values = dict(name=name, call_id=call_id, arguments=arguments,
                  result={"updated": True}, success=True, readback_verified=True,
                  state_version=state_version)
    values.update(overrides)
    return ToolOutcome(**values)


@pytest.mark.parametrize("pending", [False, True])
def test_batch_stops_at_first_failed_or_pending_action(pending):
    async def scenario():
        invoked = []

        async def invoke(**kw):
            name = kw["name"]
            invoked.append(name)
            if name == "set_order_fulfillment":
                return batch_outcome(name, kw["call_id"], kw["arguments"], kw["state_version"],
                    success=False, readback_verified=False, pending=pending,
                    error="" if pending else "fulfillment_time_unavailable",
                    confirmation_text="Please confirm this proposal." if pending else "")
            return batch_outcome(name, kw["call_id"], kw["arguments"], kw["state_version"])

        session = await batch_session(invoke)
        # Preserve explicit fixture envelopes; this test targets batch sequencing.
        session.domain._with_confirmation = lambda outcome: outcome
        result, candidate = await session.apply_order_changes(batch_plan(session, {"actions": [
            {"name": "set_order_notes", "arguments": {"order_notes": "birthday"}},
            {"name": "set_order_fulfillment", "arguments": {"fulfillment_type": "pickup"}},
            {"name": "set_order_notes", "arguments": {"order_notes": "must not execute"}},
        ]}), "batch", 0, 0)
        assert invoked == ["set_order_notes", "set_order_fulfillment"]
        assert result["status"] == ("pending_confirmation" if pending else "partial")
        assert result["completed"][0]["name"] == "set_order_notes"
        assert bool(candidate) is pending
        saved = await session.domain.state_store.load(session.domain.session_id)
        assert [op.operation for op in saved.committed_operations] == ["set_order_notes"]
    asyncio.run(scenario())


def test_batch_replay_does_not_reapply_committed_mutations():
    async def scenario():
        invoked = []

        async def invoke(**kw):
            invoked.append(kw["name"])
            return batch_outcome(kw["name"], kw["call_id"], kw["arguments"], kw["state_version"])

        session = await batch_session(invoke)
        actions = {"actions": [
            {"name": "set_order_notes", "arguments": {"order_notes": "quiet pickup"}},
            {"name": "set_order_fulfillment", "arguments": {"fulfillment_type": "pickup"}},
        ]}
        first, _ = await session.apply_order_changes(batch_plan(session, actions), "first-batch", 0, 0)
        second, _ = await session.apply_order_changes(batch_plan(session, actions), "retried-batch", 0, 0)
        assert first["status"] == second["status"] == "completed"
        assert invoked.count("set_order_notes") == 1
        assert invoked.count("set_order_fulfillment") == 1
        assert invoked.count("get_order_summary") == 2
        saved = await session.domain.state_store.load(session.domain.session_id)
        assert len(saved.committed_operations) == 2
    asyncio.run(scenario())


def test_batch_interruption_preserves_write_and_skips_remaining_actions():
    async def scenario():
        invoked = []
        started, release = asyncio.Event(), asyncio.Event()

        async def invoke(**kw):
            invoked.append(kw["name"])
            started.set()
            await release.wait()
            return batch_outcome(kw["name"], kw["call_id"], kw["arguments"], kw["state_version"],
                                 readback={"order_notes": "birthday", "status": "draft"})

        session = await batch_session(invoke)
        task = asyncio.create_task(session.apply_order_changes(batch_plan(session, {"actions": [
            {"name": "set_order_notes", "arguments": {"order_notes": "birthday"}},
            {"name": "set_order_fulfillment", "arguments": {"fulfillment_type": "pickup"}},
        ]}), "interrupted-batch", 0, 0))
        await asyncio.wait_for(started.wait(), 1)
        await session.interrupt()
        release.set()
        result, candidate = await asyncio.wait_for(task, 1)
        assert result["status"] == "interrupted"
        assert candidate is None
        assert invoked == ["set_order_notes"]
        saved = await session.domain.state_store.load(session.domain.session_id)
        assert saved.order_notes == "birthday"
        assert len(saved.committed_operations) == 1
        assert saved.committed_operations[0].turn_id == "batch-turn"
    asyncio.run(scenario())


def test_batch_schema_excludes_confirmation_and_standalone_order_mutations():
    from app.native_voice.streaming import model_tools, ORDER_CHANGES
    definitions = model_tools()
    names = {tool["name"] for tool in definitions}
    assert "apply_order_changes" in names and "confirm_order" in names
    assert not names.intersection(ORDER_CHANGES)
    batch = next(tool for tool in definitions if tool["name"] == "apply_order_changes")
    variants = batch["parameters"]["properties"]["actions"]["items"]["anyOf"]
    assert {v["properties"]["name"]["enum"][0] for v in variants} == ORDER_CHANGES


@pytest.mark.parametrize(("requested", "error"), [
    ("garbage", "invalid_fulfillment_time"),
    ("2026-09-23T19:30:00", "invalid_fulfillment_time"),
    ("2026-09-23T17:30:00-04:00", "fulfillment_time_too_soon"),
    ("2026-09-23T18:10:00-04:00", "fulfillment_time_too_soon"),
    ("2026-09-23T03:00:00-04:00", "fulfillment_time_too_soon"),
    ("2026-09-24T03:00:00-04:00", "fulfillment_time_unavailable"),
])
def test_requested_pickup_time_rejected_before_write(monkeypatch, requested, error):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import app.services.restaurant as restaurant
    monkeypatch.setattr(restaurant, "_restaurant_now",
        lambda: datetime(2026, 9, 23, 18, 0, tzinfo=ZoneInfo("America/New_York")))

    async def scenario():
        service = restaurant.RestaurantService()
        async def forbidden_write(**kwargs):
            raise AssertionError("Invalid schedule must not attempt a write")
        service._idempotent_write = forbidden_write
        with pytest.raises(restaurant.RestaurantServiceError) as caught:
            await service.set_order_fulfillment(call_id="schedule-test", idempotency_key="schedule-key",
                fulfillment_type="pickup", fulfillment_at=requested)
        assert caught.value.code == error
    asyncio.run(scenario())


def test_valid_requested_time_is_normalized_and_persisted(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import json
    import app.services.restaurant as restaurant
    monkeypatch.setattr(restaurant, "_restaurant_now",
        lambda: datetime(2026, 9, 23, 18, 0, tzinfo=ZoneInfo("America/New_York")))

    async def scenario():
        service = restaurant.RestaurantService()
        writes, checked = [], []
        class Connection:
            async def fetchrow(self, *args):
                return {"id": 7, "status": "pending"}
            async def execute(self, *args):
                writes.append(args)
        async def idempotent(**kw):
            assert kw["payload"]["fulfillment_at"] == "2026-09-23T19:30:00-04:00"
            return await kw["operation"](Connection()), False
        async def validate(conn, order_id, at, **kwargs):
            checked.append(at)
        async def summary(conn, order_id):
            return {"order_id": order_id, "status": "pending"}
        service._idempotent_write = idempotent
        service._ensure_order_items_valid_at = validate
        service._order_summary_with_conn = summary
        result = await service.set_order_fulfillment(call_id="schedule-test", idempotency_key="schedule-key",
            fulfillment_type="pickup", fulfillment_at="2026-09-23T23:30:00+00:00")
        assert result["updated"] is True
        assert len(writes) == 1
        assert json.loads(writes[0][3])["fulfillment_at"] == "2026-09-23T19:30:00-04:00"
        assert checked[0].isoformat() == "2026-09-23T19:30:00-04:00"
    asyncio.run(scenario())


@pytest.mark.parametrize(("stored", "verified"), [
    ("2026-09-23T19:30:00-04:00", True),
    ("2026-09-23T23:30:00+00:00", True),
    ("2026-09-23T18:30:00-04:00", False),
    ("2026-09-23T19:30:00", False),
    ("", False),
])
def test_fulfillment_readback_requires_requested_instant(stored, verified):
    from app.native_voice.tools import ToolBridge, _order_readback_hash
    readback = {
        "order_id": 7, "call_id": "schedule-test", "booking_id": 0,
        "status": "pending", "draft_version": 1, "total": 0.0,
        "items": [], "proposed_items": [], "fulfillment": "pickup",
        "fulfillment_type": "pickup", "fulfillment_details": {"fulfillment_at": stored},
        "order_notes": "", "allergy_notes": "", "unresolved_fields": [],
        "state_version": 1, "readback_committed": True,
    }
    readback["readback_hash"] = _order_readback_hash(readback)
    arguments = {"session_id": "schedule-test", "fulfillment_type": "pickup",
                 "fulfillment_at": "2026-09-23T19:30:00-04:00"}
    assert ToolBridge._verify_readback("set_order_fulfillment", arguments, readback, 1, {"order_id": 7}) is verified


@pytest.mark.parametrize("caption_failed", [False, True])
def test_speculative_tool_proposal_waits_for_successful_finalized_transcript(caption_failed):
    async def scenario():
        domain = NativeVoiceAdapter(
            session_id="stream-speculative-caption",
            transport=MemoryRealtimeTransport(),
            state_store=InMemoryOrderStateStore(),
        )
        session, _ = session_for(domain)
        proposal_ready = asyncio.Event()
        mutations, spoken = [], []
        response_count = 0

        async def response(epoch):
            nonlocal response_count
            response_count += 1
            if response_count == 1:
                proposal_ready.set()
                return {"status": "completed", "output": [
                    {"type": "function_call", "name": "set_order_notes", "call_id": "proposed",
                     "arguments": '{"order_notes":"birthday"}'},
                ]}
            return {"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Hello."}]},
            ]}

        async def run_tool(call_id, name, args, generation):
            assert domain._completed_turn is not None
            assert domain._completed_turn.transcript == "It is a birthday order."
            mutations.append((domain._completed_turn.turn_id, name))
            return ToolOutcome(name, call_id, args, {}, False, error="missing_order",
                               state_version=domain.state.version)

        async def speak(text, epoch, **kwargs):
            spoken.append(text)

        domain._run_tool = run_tool
        session.response, session.speak = response, speak
        worker = asyncio.create_task(session.work())
        await session.queue.put(("caption-turn", 0))
        await asyncio.wait_for(proposal_ready.wait(), 1)
        await asyncio.sleep(0)
        assert mutations == []
        assert domain._completed_turn is None
        if caption_failed:
            session.transcript_future("caption-turn").set_exception(RuntimeError("transcription_failed"))
        else:
            session.transcript_future("caption-turn").set_result("It is a birthday order.")
        await asyncio.wait_for(session.queue.join(), 1)
        assert mutations == ([] if caption_failed else [("caption-turn", "set_order_notes")])
        assert spoken
        await cancel_task(worker)
    asyncio.run(scenario())


def test_interruption_retires_speculative_response_before_next_turn():
    async def scenario():
        session, _ = session_for()
        handled = []
        handled_event = asyncio.Event()

        async def run_turn(item_id, transcript, epoch, *, initial_response=None):
            proposal = await initial_response
            handled.append((item_id, transcript, proposal["id"]))
            handled_event.set()

        session.run_turn = run_turn
        reader = asyncio.create_task(session.read_events())
        worker = asyncio.create_task(session.work())
        await session.queue.put(("old-turn", 0))
        await session.transport.sent_event.wait()
        await session.transport.feed({"type": "response.created", "response": {"id": "old-response"}})
        await session.interrupt()
        session.transcript_future("new-turn").set_result("Cancel that request.")
        await session.queue.put(("new-turn", session.epoch))
        for _ in range(10):
            await asyncio.sleep(0)
        assert len([e for e in session.transport.sent if e["type"] == "response.create"]) == 1
        assert handled == []
        assert {"type": "response.cancel", "response_id": "old-response"} in session.transport.sent
        await session.transport.feed({"type": "response.done", "response": {
            "id": "old-response", "status": "cancelled",
            "output": [{"type": "function_call", "name": "confirm_order"}],
        }})
        for _ in range(30):
            if len([e for e in session.transport.sent if e["type"] == "response.create"]) == 2:
                break
            await asyncio.sleep(0)
        assert len([e for e in session.transport.sent if e["type"] == "response.create"]) == 2
        await session.transport.feed(
            {"type": "response.created", "response": {"id": "new-response"}},
            {"type": "response.done", "response": {"id": "old-response", "status": "completed", "output": []}},
        )
        assert handled == []
        await session.transport.feed({"type": "response.done", "response": {
            "id": "new-response", "status": "completed", "output": [],
        }})
        await asyncio.wait_for(handled_event.wait(), 1)
        assert handled == [("new-turn", "Cancel that request.", "new-response")]
        await cancel_task(worker)
        await cancel_task(reader)
    asyncio.run(scenario())

def batch_plan(session, args):
    from app.native_voice.streaming import state_view
    changed = {a["arguments"].get("order_item_id") for a in args["actions"]
               if a["name"] in {"remove_order_item", "update_order_item"}}
    retained = [int(i.line_id) for i in session.domain.state.items
                if i.line_id.isdigit() and i.status != "removed" and int(i.line_id) not in changed]
    return {**args, "expected_order_revision": state_view(session.domain.state)["order_revision"],
            "retain_order_item_ids": retained}


@pytest.mark.parametrize(("success", "verified"), [(False, False), (True, False), (False, True)])
def test_menu_answer_never_uses_unverified_facts(success, verified):
    from app.native_voice.streaming import menu_answer
    outcome = ToolOutcome("check_menu_item_availability", "menu", {}, {}, success,
        readback_verified=verified, facts={"canonical_items": [
            {"id": "dish", "name": "Invented Dish", "price": 99, "ingredients": ["fiction"]},
        ]})
    answer = menu_answer(outcome, "details")
    assert "couldn't check" in answer
    assert "Invented" not in answer and "$99" not in answer and "fiction" not in answer


@pytest.mark.parametrize("items", [[], [
    {"id": "first", "name": "First Dish"}, {"id": "second", "name": "Second Dish"},
]])
def test_menu_answer_unknown_or_ambiguous_requires_clarification(items):
    from app.native_voice.streaming import menu_answer
    outcome = ToolOutcome("check_menu_item_availability", "menu", {}, {}, True,
        readback_verified=True, facts={"canonical_items": items})
    assert "couldn't find an exact match" in menu_answer(outcome, "details")


@pytest.mark.parametrize("allergens", [[], ["milk", "wheat"]])
def test_menu_allergy_answer_does_not_turn_ingredient_removal_into_safety(allergens):
    from app.native_voice.streaming import menu_answer
    outcome = ToolOutcome("check_menu_item_availability", "menu", {}, {}, True,
        readback_verified=True, facts={"canonical_items": [
            {"id": "burger", "name": "Hearth Burger", "allergens": allergens},
        ]})
    answer = menu_answer(outcome, "allergy")
    assert "does not establish" in answer and "cannot guarantee" in answer
    assert "restaurant staff" in answer
    if allergens:
        assert "milk, wheat" in answer
    else:
        assert "don't have a complete allergen assessment" in answer


def correction_state():
    return OrderState(items=(
        OrderItemState("burger", "Hearth Burger", 1, line_id="11", removals=("onion jam",)),
        OrderItemState("lemonade", "House Lemonade", 1, line_id="12"),
    ), fulfillment="pickup", fulfillment_details={"fulfillment_at": "2026-09-23T19:30:00-04:00"},
       allergy_notes="Peanut allergy")


@pytest.mark.parametrize("invalid", ["stale", "omitted", "overlap", "unknown", "duplicate-change"])
def test_invalid_preservation_plan_is_rejected_before_any_tool(invalid):
    async def scenario():
        async def forbidden(**kwargs):
            raise AssertionError("Invalid plan reached a tool")
        session = await batch_session(forbidden)
        session.domain.state = correction_state()
        plan = batch_plan(session, {"actions": [
            {"name": "remove_order_item", "arguments": {"order_item_id": 11}},
        ]})
        if invalid == "stale":
            plan["expected_order_revision"] = "old-state"
        elif invalid == "omitted":
            plan["retain_order_item_ids"] = []
        elif invalid == "overlap":
            plan["retain_order_item_ids"] = [11, 12]
        elif invalid == "unknown":
            plan["retain_order_item_ids"] = [13]
        else:
            plan["actions"].append(plan["actions"][0])
        result, candidate = await session.apply_order_changes(plan, "bad-plan", 0, 0)
        assert result["error"] == "order_preservation_plan_invalid"
        assert candidate is None
        assert session.domain._outcomes == []
    asyncio.run(scenario())


def test_order_revision_ignores_transcript_metadata_but_tracks_material_changes():
    from app.native_voice.streaming import state_view
    state = correction_state()
    original = state_view(state)
    metadata_only = replace(state, version=20, items=tuple(
        replace(item, source_turn_ids=("later-turn",)) for item in state.items))
    assert state_view(metadata_only)["order_revision"] == original["order_revision"]
    changed = replace(state, items=(replace(state.items[0], quantity=2), state.items[1]))
    assert state_view(changed)["order_revision"] != original["order_revision"]
    assert [i["order_item_id"] for i in original["items"]] == [11, 12]


def test_removed_line_is_absent_from_active_projection_and_revision():
    from app.native_voice.streaming import state_view
    state = correction_state()
    tombstone = OrderItemState("old", "Removed Dish", 1, line_id="13", status="removed")
    with_history = replace(state, items=state.items + (tombstone,))
    projected = state_view(with_history)
    assert [i["order_item_id"] for i in projected["items"]] == [11, 12]
    assert projected["order_revision"] == state_view(state)["order_revision"]


def test_preservation_plan_does_not_require_retaining_removed_database_lines():
    async def scenario():
        invoked = []
        async def invoke(**kwargs):
            invoked.append(kwargs["name"])
            return ToolOutcome(kwargs["name"], kwargs["call_id"], kwargs["arguments"],
                               {}, False, error="stop_after_plan_validation")
        session = await batch_session(invoke)
        state = correction_state()
        session.domain.state = replace(state, items=state.items + (
            OrderItemState("old", "Removed Dish", 1, line_id="13", status="removed"),))
        plan = batch_plan(session, {"actions": [
            {"name": "remove_order_item", "arguments": {"order_item_id": 11}},
        ]})
        result, _ = await session.apply_order_changes(plan, "valid-active-plan", 0, 0)
        assert result.get("error") != "order_preservation_plan_invalid"
        assert invoked == ["remove_order_item"]
    asyncio.run(scenario())


@pytest.mark.parametrize("affirmation", [
    "Yes.", "Yes, please confirm that.", "Yes, that is correct.", "Yes, that's correct.",
])
@pytest.mark.parametrize("mutation", ["add_order_item", "apply_order_changes"])
def test_plain_affirmative_cannot_rebuild_or_modify_order(affirmation, mutation):
    async def scenario():
        domain = NativeVoiceAdapter(session_id="affirmative-guard",
            transport=MemoryRealtimeTransport(), state_store=InMemoryOrderStateStore())
        session, _ = session_for(domain)
        mutations = []
        calls = 0

        async def response(epoch):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"status": "completed", "output": [{
                    "type": "function_call", "name": mutation,
                    "call_id": "wrong-mutation", "arguments": "{}",
                }]}
            return {"status": "completed", "output": [{
                "type": "message", "content": [{"type": "output_text", "text": "Could you clarify?"}],
            }]}

        async def run_tool(call_id, name, args, generation):
            mutations.append(name)
            return ToolOutcome(name, call_id, args, {}, False, error="unexpected_mutation")

        async def batch(args, call_id, generation, epoch):
            mutations.append("apply_order_changes")
            return {"status": "failed"}, None

        async def speak(text, epoch, **kwargs):
            pass

        domain._run_tool = run_tool
        session.apply_order_changes = batch
        session.response, session.speak = response, speak
        await session.run_turn("approval-turn", affirmation, 0)
        assert mutations == []
        outputs = [event["item"] for event in session.transport.sent
                   if event.get("item", {}).get("type") == "function_call_output"]
        assert "confirmation_turn_cannot_change_order" in outputs[0]["output"]
    asyncio.run(scenario())


@pytest.mark.parametrize("missing", ["customer_name", "customer_phone", "both"])
def test_first_order_without_contact_is_rejected_before_writes(missing):
    async def scenario():
        async def forbidden(**kwargs):
            raise AssertionError("Missing contact reached a write")
        session = await batch_session(forbidden)
        args = batch_plan(session, {"actions": [
            {"name": "add_order_item", "arguments": {"item_name": "House Lemonade"}},
        ], "customer_name": "Synthetic Guest", "customer_phone": "+15035550123"})
        for field in ("customer_name", "customer_phone"):
            if missing in (field, "both"):
                args[field] = " "
        result, candidate = await session.apply_order_changes(args, "missing-contact", 0, 0)
        assert result["error"] == "order_contact_required"
        assert candidate is None and session.domain._outcomes == []
    asyncio.run(scenario())


def test_batch_contact_and_session_are_propagated_to_every_add():
    async def scenario():
        captured = []
        async def invoke(**kwargs):
            captured.append((kwargs["name"], kwargs["arguments"]))
            return batch_outcome(kwargs["name"], kwargs["call_id"], kwargs["arguments"],
                                 kwargs["state_version"])
        session = await batch_session(invoke)
        args = batch_plan(session, {"actions": [
            {"name": "add_order_item", "arguments": {"item_name": "House Lemonade",
                "session_id": "foreign", "customer_name": "Wrong", "customer_phone": "Wrong"}},
            {"name": "add_order_item", "arguments": {"item_name": "Hearth Bread"}},
        ], "customer_name": "Synthetic Guest", "customer_phone": "+15035550123"})
        result, _ = await session.apply_order_changes(args, "contact-batch", 0, 0)
        assert result["status"] == "completed"
        additions = [a for name, a in captured if name == "add_order_item"]
        assert len(additions) == 2
        for arguments in additions:
            assert arguments["session_id"] == session.domain.session_id
            assert arguments["customer_name"] == "Synthetic Guest"
            assert arguments["customer_phone"] == "+15035550123"
        assert all(a["session_id"] == session.domain.session_id for _, a in captured)
    asyncio.run(scenario())


def test_whole_order_clear_is_rejected_even_with_valid_partition():
    async def scenario():
        async def forbidden(**kwargs):
            raise AssertionError("Generic correction cleared entire order")
        session = await batch_session(forbidden)
        session.domain.state = correction_state()
        args = batch_plan(session, {"actions": [
            {"name": "remove_order_item", "arguments": {"order_item_id": 11}},
            {"name": "remove_order_item", "arguments": {"order_item_id": 12}},
        ]})
        result, candidate = await session.apply_order_changes(args, "clear-all", 0, 0)
        assert result["error"] == "whole_order_clear_requires_separate_authorization"
        assert candidate is None
    asyncio.run(scenario())


def test_confirm_uses_fully_played_snapshot_version_and_server_session(monkeypatch):
    import json
    monkeypatch.setattr("app.native_voice.streaming.time.monotonic", lambda: 100.0)
    async def scenario():
        domain = NativeVoiceAdapter(session_id="heard-version-test",
            transport=MemoryRealtimeTransport(), state_store=InMemoryOrderStateStore())
        session, _ = session_for(domain)
        pending = ToolOutcome("get_order_summary", "readback", {}, {}, True,
            pending=True, readback_verified=True, state_version=0,
            confirmation_text="Full order readback. Shall I confirm?",
            facts={"draft_version": 17})
        session.utterance = utterance(session, text=pending.confirmation_text, outcomes=(pending,))
        assert session.heard_order is None
        assert await session.played("wrong-token") is False
        assert session.heard_order is None
        assert await session.played("spoken") is True
        assert session.heard_order["draft_version"] == 17
        captured = []
        calls = 0
        async def response(epoch):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"status": "completed", "output": [{
                    "type": "function_call", "name": "confirm_order", "call_id": "confirm",
                    "arguments": json.dumps({"session_id": "foreign", "expected_draft_version": 999,
                                             "caller_approved_full_readback": True}),
                }]}
            return {"status": "completed", "output": [{
                "type": "message", "content": [{"type": "output_text", "text": "Could you clarify?"}],
            }]}
        async def run_tool(call_id, name, args, generation):
            captured.append(dict(args))
            return ToolOutcome(name, call_id, args, {}, False, error="test_stop",
                               state_version=domain.state.version)
        async def speak(*args, **kwargs):
            pass
        domain._run_tool = run_tool
        session.response, session.speak = response, speak
        await session.run_turn("later-approval", "Yes, please confirm.", 0)
        assert captured[0]["expected_draft_version"] == 17
        assert captured[0]["session_id"] == "heard-version-test"
    asyncio.run(scenario())


def test_model_schema_does_not_request_server_owned_identifiers():
    from app.native_voice.streaming import model_tools
    definitions = model_tools()
    def visit(value):
        if isinstance(value, dict):
            if "properties" in value:
                assert "session_id" not in value["properties"]
                assert "expected_draft_version" not in value["properties"]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(definitions)
    batch = next(t for t in definitions if t["name"] == "apply_order_changes")
    assert {"customer_name", "customer_phone"} <= batch["parameters"]["properties"].keys()
    variants = batch["parameters"]["properties"]["actions"]["items"]["anyOf"]
    addition = next(v for v in variants if v["properties"]["name"]["enum"] == ["add_order_item"])
    assert "customer_name" not in addition["properties"]["arguments"]["properties"]
    assert "customer_phone" not in addition["properties"]["arguments"]["properties"]


def test_prompt_catalog_uses_actual_canonical_ids_and_modifier_references():
    import json
    from app.restaurant_knowledge import get_restaurant_knowledge
    prompt = streaming_config()["session"]["instructions"]
    marker = "\nCatalog directory (not live availability or evidence of a completed action): "
    entries = json.loads(prompt.split(marker, 1)[1])
    canonical = get_restaurant_knowledge().menu_items
    assert {item["item_id"] for item in entries} == {item["item_id"] for item in canonical}
    assert all(item["item_id"] and item["name"] for item in entries)
    for item in entries:
        options = item["modifier_options"]
        assert all(isinstance(option, str) and option.startswith("modifier.") for option in options)
        for group in item["required_modifier_groups"] or []:
            assert set(group["option_ids"]) <= set(options)
