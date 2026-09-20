"""Minimal development-only OpenAI Realtime adapter.

The adapter buffers model audio until the response is complete so the speech
gate can prevent unsupported claims from reaching the caller.  It is a
server-to-server WebSocket implementation for synthetic development audio;
production startup and Retell never import it.
"""

from __future__ import annotations

import base64
import asyncio
import hashlib
import json
import uuid
import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Mapping

from app.call_memory import reset_current_action_scope, reset_current_session_id, set_current_action_scope, set_current_session_id
from app.native_voice.contracts import OrderItemState, OrderPatch, OrderState
from app.native_voice.database_guard import (
    activate_native_voice_database,
    deactivate_native_voice_database,
)
from app.native_voice.protocol import EventRecorder, RealtimeTransport
from app.native_voice.speech import SpeechDecision, SpeechGate
from app.native_voice.state_store import CallSessionOrderStateStore, OrderStateStore
from app.native_voice.tools import (
    RestaurantToolExecutor,
    ToolBridge,
    ToolOutcome,
    MUTATING_TOOLS,
    realtime_tool_definitions,
)
from app.native_voice.turns import CompletedCallerTurn, TurnAssembler


@dataclass(frozen=True)
class RealtimeConfig:
    model: str = "gpt-realtime"
    voice: str = "marin"
    input_transcription_model: str = "gpt-4o-mini-transcribe"
    language: str = "en"
    sample_rate_hz: int = 24_000
    output_format: str = "audio/pcm"
    instructions: str = (
        "You are a restaurant host in a development-only synthetic test. "
        "Use the provided restaurant tools for every menu, price, availability, "
        "booking, and order fact. Never claim an action succeeded without a "
        "matching tool result and database readback. Ask one bounded clarification "
        "for ambiguity. Keep order memory in application state, not conversation history."
    )

    def session_update(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self.model,
                "output_modalities": ["audio"],
                "instructions": self.instructions,
                "audio": {
                    "input": {
                        "format": {"type": self.output_format, "rate": self.sample_rate_hz},
                        "transcription": {"model": self.input_transcription_model, "language": self.language},
                        "turn_detection": None,
                    },
                    "output": {
                        "format": {"type": self.output_format},
                        "voice": self.voice,
                    },
                },
                "tools": realtime_tool_definitions(),
                "tool_choice": "auto",
                "max_output_tokens": 256,
            },
        }


@dataclass(frozen=True)
class VoiceTurnResult:
    turn: CompletedCallerTurn | None
    audio: bytes
    transcript: str
    speech: SpeechDecision | None
    tool_outcomes: tuple[ToolOutcome, ...] = ()
    response_id: str = ""


@dataclass
class _ResponseBuffer:
    response_id: str = ""
    generation: int = 0
    audio: bytearray = field(default_factory=bytearray)
    transcript_parts: list[str] = field(default_factory=list)
    tool_calls: dict[str, tuple[str, dict[str, Any]]] = field(default_factory=dict)
    assistant_item_id: str = ""
    played_audio_bytes: int = 0
    assistant_transcript_seen: bool = False
    assistant_transcript_done: bool = False
    input_item_id: str = ""


class InterruptionController:
    """Generation-based cancellation guard for audio, events, and tools."""

    def __init__(self) -> None:
        self.generation = 0
        self.active_response_id = ""
        self.cancelled_response_ids: set[str] = set()
        self.terminal_generations: set[int] = set()

    def begin_response(self, response_id: str = "") -> int:
        self.active_response_id = response_id
        return self.generation

    def interrupt(self, response_id: str = "") -> int:
        if response_id:
            self.cancelled_response_ids.add(response_id)
        self.terminal_generations.add(self.generation)
        self.generation += 1
        self.active_response_id = ""
        return self.generation

    def accepts(self, *, generation: int, response_id: str = "", allow_new_response: bool = False) -> bool:
        if generation != self.generation or generation in self.terminal_generations:
            return False
        if response_id and response_id in self.cancelled_response_ids:
            return False
        if allow_new_response:
            return bool(response_id) and not self.active_response_id
        if response_id:
            return bool(self.active_response_id) and response_id == self.active_response_id
        return True


class NativeVoiceAdapter:
    """One synthetic development session over the OpenAI Realtime protocol."""

    def __init__(
        self,
        *,
        session_id: str,
        transport: RealtimeTransport,
        config: RealtimeConfig | None = None,
        state_store: OrderStateStore | None = None,
        tool_bridge: ToolBridge | None = None,
        recorder: EventRecorder | None = None,
        speech_gate: SpeechGate | None = None,
        facts_extractor: Callable[[CompletedCallerTurn, OrderState], OrderPatch | Awaitable[OrderPatch | None] | None] | None = None,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        self.session_id = session_id
        self.transport = transport
        self.config = config or RealtimeConfig()
        self.state_store = state_store or CallSessionOrderStateStore()
        self.tool_bridge = tool_bridge or ToolBridge(RestaurantToolExecutor())
        self.recorder = recorder or EventRecorder()
        self.speech_gate = speech_gate or SpeechGate()
        self.facts_extractor = facts_extractor
        self.turns = TurnAssembler()
        self.interruptions = InterruptionController()
        self.state = OrderState()
        self._started = False
        self._response: _ResponseBuffer | None = None
        self._completed_turn: CompletedCallerTurn | None = None
        self._last_result: VoiceTurnResult | None = None
        self._outcomes: list[ToolOutcome] = []
        self._seen_tool_calls: set[str] = set()
        self._replayed_finalized_turns: set[str] = set()
        self._cancelled_input_item_ids: set[str] = set()
        self._quarantined_input_item_ids: set[str] = set()
        self._expected_input_item_id = ""
        self._require_input_item_id = False
        self._input_transcript_quarantined = False
        self._memory_write_task: asyncio.Task[Any] | None = None
        self._turn_lock = asyncio.Lock()
        self._commit_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._uses_native_database = isinstance(self.state_store, CallSessionOrderStateStore) or isinstance(
            getattr(self.tool_bridge, "executor", None), RestaurantToolExecutor
        )
        if hasattr(self.tool_bridge, "bind_session"):
            self.tool_bridge.bind_session(session_id)

    async def start(self) -> None:
        if self._started:
            return
        token = activate_native_voice_database() if self._uses_native_database else None
        try:
            self.state = await self.state_store.load(self.session_id)
            event = self.config.session_update()
            await self._send(event)
            self.recorder.record(event)
            self._started = True
        finally:
            if token is not None:
                deactivate_native_voice_database(token)

    async def close(self) -> None:
        await self.transport.close()

    async def _send(self, event: Mapping[str, Any]) -> None:
        async with self._lifecycle_lock:
            await self.transport.send(event)

    async def apply_order_patch(self, patch: OrderPatch) -> OrderState:
        """Apply facts only after a completed caller turn exists."""
        token = activate_native_voice_database() if self._uses_native_database else None
        try:
            return await self._apply_order_patch(patch)
        finally:
            if token is not None:
                deactivate_native_voice_database(token)

    async def _apply_order_patch(
        self,
        patch: OrderPatch,
        *,
        generation: int | None = None,
    ) -> OrderState | None:
        if self._completed_turn is None or self._completed_turn.turn_id != patch.source_turn_id:
            raise RuntimeError("structured order mutation requires the matching finalized caller turn")
        if generation is not None and generation != self.interruptions.generation:
            return None
        current = await self.state_store.load(self.session_id)
        if generation is not None and generation != self.interruptions.generation:
            return None
        next_state = current.apply(patch)
        if not await self._save_state(next_state, expected_version=current.version, generation=generation):
            return None
        self.state = next_state
        self.recorder.record({"type": "facts_extracted", "turn_id": patch.source_turn_id, "state_version": next_state.version})
        return next_state

    async def _save_state(
        self,
        state: OrderState,
        *,
        expected_version: int,
        generation: int | None = None,
    ) -> bool:
        async with self._commit_lock:
            if generation is not None and generation != self.interruptions.generation:
                return False
            write_task = asyncio.create_task(
                self.state_store.save(self.session_id, state, expected_version=expected_version)
            )
            self._memory_write_task = write_task
            try:
                await write_task
            except asyncio.CancelledError:
                return False
            finally:
                if self._memory_write_task is write_task:
                    self._memory_write_task = None
            return generation is None or generation == self.interruptions.generation

    async def submit_audio(
        self,
        audio: bytes,
        *,
        turn_id: str | None = None,
        transcript: str | None = None,
    ) -> VoiceTurnResult:
        """Send one synthetic PCM16 turn and drain native output until done."""
        async with self._turn_lock:
            token = activate_native_voice_database() if self._uses_native_database else None
            try:
                return await self._submit_audio(audio, turn_id=turn_id, transcript=transcript)
            finally:
                if token is not None:
                    deactivate_native_voice_database(token)

    async def _submit_audio(
        self,
        audio: bytes,
        *,
        turn_id: str | None = None,
        transcript: str | None = None,
    ) -> VoiceTurnResult:
        await self.start()
        if self._response is not None:
            async with self._commit_lock:
                async with self._lifecycle_lock:
                    await self._interrupt_unlocked()
        resolved_turn_id = turn_id or f"turn-{uuid.uuid4().hex[:12]}"
        if self.turns.completed(resolved_turn_id) is not None:
            self._replayed_finalized_turns.add(resolved_turn_id)
        self._completed_turn = None
        self._expected_input_item_id = ""
        self.turns.start(resolved_turn_id)
        self.recorder.record_audio("audio_received", audio, turn_id=resolved_turn_id)
        append = {"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode("ascii")}
        await self._send(append)
        self.recorder.record_audio("input_audio_buffer.append", audio, turn_id=resolved_turn_id)
        await self._send({"type": "input_audio_buffer.commit"})
        self.recorder.record({"type": "input_audio_buffer.commit", "turn_id": resolved_turn_id})
        await self._send({"type": "response.create", "response": {"output_modalities": ["audio"]}})
        self.recorder.record({"type": "response.create", "turn_id": resolved_turn_id})
        return await self._drain_response(transcript=transcript)

    def mark_audio_played(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        if self._response is not None:
            self._response.played_audio_bytes = min(byte_count, len(self._response.audio))

    async def interrupt(self) -> None:
        token = activate_native_voice_database() if self._uses_native_database else None
        try:
            async with self._commit_lock:
                async with self._lifecycle_lock:
                    await self._interrupt_unlocked()
        finally:
            if token is not None:
                deactivate_native_voice_database(token)

    async def _interrupt_unlocked(self) -> None:
        response = self._response
        if self._memory_write_task is not None and not self._memory_write_task.done():
            self._memory_write_task.cancel()
        response_id = response.response_id if response is not None else self.interruptions.active_response_id
        if response is not None and response.input_item_id:
            self._cancelled_input_item_ids.add(response.input_item_id)
        if self._expected_input_item_id:
            self._cancelled_input_item_ids.add(self._expected_input_item_id)
        self._expected_input_item_id = ""
        self._require_input_item_id = True
        generation = self.interruptions.interrupt(response_id)
        self.turns.reset()
        await self.transport.send({"type": "response.cancel"})
        await self.transport.send({"type": "output_audio_buffer.clear"})
        if response is not None and response.assistant_item_id:
            audio_end_ms = round(response.played_audio_bytes * 1000 / (self.config.sample_rate_hz * 2))
            await self.transport.send(
                {
                    "type": "conversation.item.truncate",
                    "item_id": response.assistant_item_id,
                    "content_index": 0,
                    "audio_end_ms": audio_end_ms,
                }
            )
        self.recorder.record({"type": "interruption", "generation": generation})
        self._response = None
        self._outcomes.clear()

    async def _drain_response(self, *, transcript: str | None = None) -> VoiceTurnResult:
        generation = self.interruptions.generation
        self._response = _ResponseBuffer(generation=generation)
        self._outcomes = []
        self._seen_tool_calls = set()
        pending_response_id = ""
        pending_response_status = ""
        while True:
            event = await self.transport.receive()
            event_type = str(event.get("type") or "unknown")
            response = event.get("response")
            response = response if isinstance(response, Mapping) else {}
            response_id = str(event.get("response_id") or response.get("id") or "")
            if not self.interruptions.accepts(
                generation=generation,
                response_id=response_id,
                allow_new_response=event_type == "response.created",
            ):
                self.recorder.record({"type": "stale_event_ignored", "original_type": event_type, "response_id": response_id})
                if generation != self.interruptions.generation:
                    result = VoiceTurnResult(self._completed_turn, b"", "", None)
                    self._response = None
                    return result
                continue
            self.recorder.record(event)
            if event_type == "response.created":
                self._response.response_id = response_id
                self.interruptions.begin_response(self._response.response_id)
                continue
            if event_type == "input_audio_buffer.committed":
                item_id = str(event.get("item_id") or "")
                if item_id and item_id not in self._cancelled_input_item_ids and item_id not in self._quarantined_input_item_ids:
                    self._expected_input_item_id = item_id
                    self._response.input_item_id = item_id
                continue
            if event_type == "conversation.item.created":
                item = event.get("item") or {}
                if not self._require_input_item_id and item.get("role") == "user" and item.get("id"):
                    item_id = str(item["id"])
                    if item_id not in self._cancelled_input_item_ids and item_id not in self._quarantined_input_item_ids:
                        self._expected_input_item_id = item_id
                        self._response.input_item_id = item_id
                continue
            if event_type == "input_audio_buffer.speech_started":
                await self.interrupt()
                return VoiceTurnResult(self._completed_turn, b"", "", None)
            if event_type == "conversation.item.input_audio_transcription.delta":
                item_id = str(event.get("item_id") or "")
                if item_id in self._quarantined_input_item_ids or (
                    self._input_transcript_quarantined and not item_id
                ):
                    self.recorder.record({"type": "late_input_transcript_ignored", "item_id": item_id})
                    continue
                if self._require_input_item_id and (
                    not item_id or item_id != self._expected_input_item_id
                ):
                    self.recorder.record({"type": "late_input_transcript_ignored", "item_id": item_id})
                    continue
                if item_id:
                    if item_id in self._cancelled_input_item_ids:
                        self.recorder.record({"type": "late_input_transcript_ignored", "item_id": item_id})
                        continue
                    if self._expected_input_item_id and item_id != self._expected_input_item_id:
                        self.recorder.record({"type": "out_of_order_input_transcript_ignored", "item_id": item_id})
                        continue
                    self._expected_input_item_id = item_id
                    self._response.input_item_id = item_id
                self.turns.add_delta(str(event.get("delta") or ""))
                continue
            if event_type == "conversation.item.input_audio_transcription.completed":
                item_id = str(event.get("item_id") or "")
                if item_id in self._quarantined_input_item_ids or (
                    self._input_transcript_quarantined and not item_id
                ):
                    self.recorder.record({"type": "late_input_transcript_ignored", "item_id": item_id})
                    continue
                if self._require_input_item_id and (
                    not item_id or item_id != self._expected_input_item_id
                ):
                    self.recorder.record({"type": "late_input_transcript_ignored", "item_id": item_id})
                    continue
                if item_id and item_id in self._cancelled_input_item_ids:
                    self.recorder.record({"type": "late_input_transcript_ignored", "item_id": item_id})
                    continue
                if item_id and self._expected_input_item_id and item_id != self._expected_input_item_id:
                    self.recorder.record({"type": "out_of_order_input_transcript_ignored", "item_id": item_id})
                    continue
                if item_id:
                    self._expected_input_item_id = item_id
                    self._response.input_item_id = item_id
                self._require_input_item_id = False
                await self._finalize_caller_turn(
                    str(event.get("transcript") or transcript or ""),
                    generation=generation,
                )
                if pending_response_status:
                    result = await self._handle_completed_response(
                        response_id=pending_response_id,
                        status=pending_response_status,
                        transcript=transcript,
                        generation=generation,
                    )
                    pending_response_id = ""
                    pending_response_status = ""
                    if result is not None:
                        return result
                continue
            if event_type == "response.output_audio.delta":
                self._response.assistant_item_id = str(event.get("item_id") or self._response.assistant_item_id)
                try:
                    self._response.audio.extend(base64.b64decode(str(event.get("delta") or "")))
                except Exception:
                    self.recorder.record({"type": "audio_decode_error"})
                continue
            if event_type in {"response.output_item.added", "response.output_item.done"}:
                item = event.get("item") or {}
                if item.get("type") == "message" and item.get("role") == "assistant":
                    self._response.assistant_item_id = str(item.get("id") or self._response.assistant_item_id)
                if event_type == "response.output_item.done" and item.get("type") == "function_call":
                    self._remember_tool_call(item)
                continue
            if event_type == "response.output_audio_transcript.delta":
                if not self._response.assistant_transcript_done:
                    delta = str(event.get("delta") or "")
                    self._response.assistant_transcript_seen = self._response.assistant_transcript_seen or bool(delta.strip())
                    self._response.transcript_parts.append(delta)
                continue
            if event_type == "response.output_audio_transcript.done":
                if self._response.assistant_transcript_done:
                    self.recorder.record({"type": "duplicate_assistant_transcript_ignored"})
                else:
                    self._response.assistant_transcript_done = True
                    transcript_text = str(event.get("transcript") or "")
                    self._response.assistant_transcript_seen = self._response.assistant_transcript_seen or bool(transcript_text.strip())
                    if transcript_text:
                        self._response.transcript_parts = [transcript_text]
                continue
            if event_type == "response.function_call_arguments.done":
                self._remember_tool_call(event)
                continue
            if event_type in {
                "conversation.item.input_audio_transcription.failed",
                "conversation.item.input_audio_transcription.error",
            }:
                error = event.get("error") if isinstance(event.get("error"), Mapping) else {}
                return await self._terminal_response_failure(
                    code=str(error.get("code") or event_type),
                    response_id=response_id,
                    item_id=str(event.get("item_id") or ""),
                )
            if event_type == "error":
                error = event.get("error") if isinstance(event.get("error"), Mapping) else {}
                return await self._terminal_response_failure(
                    code=str(error.get("code") or "realtime_error"),
                    response_id=response_id,
                    item_id=str(event.get("item_id") or ""),
                )
            if event_type == "response.done":
                pending_response_id = response_id
                pending_response_status = str(response.get("status") or event.get("status") or "")
                if self._completed_turn is None and pending_response_status == "completed":
                    continue
                result = await self._handle_completed_response(
                    response_id=pending_response_id,
                    status=pending_response_status,
                    transcript=transcript,
                    generation=generation,
                )
                pending_response_id = ""
                pending_response_status = ""
                if result is not None:
                    return result
                continue

    async def _terminal_response_failure(
        self,
        *,
        code: str,
        response_id: str,
        item_id: str,
    ) -> VoiceTurnResult:
        response = self._response
        for candidate in (item_id, response.input_item_id if response is not None else "", self._expected_input_item_id):
            if candidate:
                self._quarantined_input_item_ids.add(candidate)
        self._input_transcript_quarantined = True
        self._require_input_item_id = True
        generation = self.interruptions.interrupt(response_id)
        self.turns.reset()
        self.recorder.record({"type": "tool_or_response_failure", "code": code, "generation": generation})
        self._response = None
        self._outcomes.clear()
        decision = SpeechDecision(
            allowed=False,
            text="",
            audio=b"",
            reasons=("realtime_response_failed",),
            replacement=self.speech_gate.replacement,
        )
        return VoiceTurnResult(None, b"", "", decision, response_id=response_id)

    async def _handle_completed_response(
        self,
        *,
        response_id: str,
        status: str,
        transcript: str | None,
        generation: int,
    ) -> VoiceTurnResult | None:
        if status != "completed":
            if self._response is not None:
                if self._response.input_item_id:
                    self._quarantined_input_item_ids.add(self._response.input_item_id)
                if self._expected_input_item_id:
                    self._quarantined_input_item_ids.add(self._expected_input_item_id)
            self._input_transcript_quarantined = True
            self._require_input_item_id = True
            self.interruptions.interrupt(response_id)
            self.recorder.record({
                "type": "incomplete_response_ignored",
                "response_id": response_id,
                "status": status,
            })
            self._response = None
            self._outcomes.clear()
            return VoiceTurnResult(self._completed_turn, b"", "", None, response_id=response_id)
        result = await self._finish_response(transcript=transcript)
        if generation != self.interruptions.generation:
            self._response = None
            self._outcomes.clear()
            return VoiceTurnResult(self._completed_turn, b"", "", None)
        if self._response is not None and self._response.tool_calls:
            for call_id, (name, args) in list(self._response.tool_calls.items()):
                if generation != self.interruptions.generation:
                    self._response = None
                    self._outcomes.clear()
                    return VoiceTurnResult(self._completed_turn, b"", "", None)
                outcome = await self._run_tool(call_id, name, args, generation=generation)
                if generation != self.interruptions.generation:
                    if outcome.success and outcome.readback_verified:
                        self._outcomes.append(outcome)
                        await self._sync_order_memory(generation=None)
                    self._response = None
                    self._outcomes.clear()
                    return VoiceTurnResult(self._completed_turn, b"", "", None)
                self._outcomes.append(outcome)
                output = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(self._model_tool_output(outcome), sort_keys=True),
                    },
                }
                await self._send(output)
                self.recorder.record({"type": "tool_result_sent", "call_id": call_id, "success": outcome.success})
            if generation != self.interruptions.generation:
                self._response = None
                self._outcomes.clear()
                return VoiceTurnResult(self._completed_turn, b"", "", None)
            synced = await self._sync_order_memory(generation=generation)
            if synced is not None and generation == self.interruptions.generation:
                order_mutations = {
                    "add_order_item",
                    "set_order_fulfillment",
                    "set_order_notes",
                    "update_order_item",
                    "remove_order_item",
                    "confirm_order",
                    "add_guest_note",
                }
                current_readback = next(
                    (
                        outcome.readback
                        for outcome in reversed(self._outcomes)
                        if outcome.name in order_mutations
                        and outcome.readback_verified
                        and isinstance(outcome.readback, Mapping)
                    ),
                    None,
                )
                self._outcomes = [
                    replace(outcome, state_version=synced.version)
                    if outcome.readback is current_readback
                    else outcome
                    for outcome in self._outcomes
                ]
            if generation != self.interruptions.generation:
                self._response = None
                self._outcomes.clear()
                return VoiceTurnResult(self._completed_turn, b"", "", None)
            self.interruptions.active_response_id = ""
            self._response = _ResponseBuffer(generation=generation)
            await self._send({"type": "response.create", "response": {"output_modalities": ["audio"]}})
            self.recorder.record({"type": "response.create", "reason": "after_tool"})
            return None
        self._last_result = result
        self._response = None
        return result

    @staticmethod
    def _model_tool_output(outcome: ToolOutcome) -> dict[str, Any]:
        verified = outcome.success and outcome.readback_verified
        clarification_required = outcome.error == "clarification_required"
        exact_item: Mapping[str, Any] | None = None
        if outcome.name == "add_order_item":
            result_item_id = outcome.result.get("order_item_id") if isinstance(outcome.result, Mapping) else None
            requested_name = str(outcome.arguments.get("item_name") or "").casefold()
            readback_items = outcome.readback.get("items") if isinstance(outcome.readback, Mapping) else ()
            for item in readback_items or ():
                if not isinstance(item, Mapping):
                    continue
                if result_item_id not in (None, "") and str(item.get("order_item_id")) == str(result_item_id):
                    exact_item = item
                    break
                if result_item_id in (None, "") and str(item.get("item_name") or item.get("name") or "").casefold() == requested_name:
                    exact_item = item
                    break
            if exact_item is None:
                verified = False
        if not verified:
            sentence = "I could not complete that request yet."
        elif outcome.name == "add_order_item":
            item_name = str(exact_item.get("item_name") or exact_item.get("name") or "the item")
            quantity = int(exact_item.get("quantity") or 1)
            sentence = f"I added {quantity} {item_name} to your order."
        elif outcome.name == "confirm_order":
            sentence = "Your order is confirmed."
        elif outcome.name == "create_booking":
            date_value = str(outcome.facts.get("date") or "")
            time_value = str(outcome.facts.get("time") or "")
            location = str(outcome.facts.get("location") or "")
            table_number = str(outcome.facts.get("table_number") or "")
            party_size = str(outcome.facts.get("party_size") or "")
            sentence = (
                f"Your reservation is confirmed for {date_value} at {time_value}"
                f"{f' at {location}' if location else ''}"
                f"{f' at table {table_number}' if table_number else ''}"
                f"{f' for {party_size} guests' if party_size else ''}."
                if date_value and time_value
                else "Your reservation is confirmed."
            )
        elif outcome.name == "cancel_booking":
            sentence = "Your reservation was cancelled."
        elif outcome.name == "update_confirmed_booking":
            sentence = "Your reservation was updated."
        elif outcome.name in {"lookup_booking", "lookup_order", "get_order_summary"}:
            sentence = "I found the requested record."
        elif outcome.name in {"get_full_menu", "check_menu_item_availability", "check_table_availability"}:
            sentence = "I checked the requested restaurant information."
        else:
            sentence = "Your request was applied."
        result_id = hashlib.sha256(
            f"{outcome.name}:{outcome.call_id}:{outcome.state_version}".encode("utf-8")
        ).hexdigest()[:24]
        return {
            "status": "completed" if verified else "failed",
            "result_id": result_id,
            "clarification_state": "required" if clarification_required else "none",
            "speech": sentence,
        }

    def _remember_tool_call(self, event: Mapping[str, Any]) -> None:
        call_id = str(event.get("call_id") or event.get("id") or "")
        if not call_id or call_id in self._seen_tool_calls:
            return
        raw_arguments = event.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = {}
        self._seen_tool_calls.add(call_id)
        self._response.tool_calls[call_id] = (str(event.get("name") or ""), arguments)
        self.recorder.record({"type": "tool_requested", "call_id": call_id, "name": event.get("name")})

    async def _finalize_caller_turn(self, transcript: str, *, generation: int) -> None:
        if generation != self.interruptions.generation:
            return
        completed = self.turns.finalize(transcript)
        if self._completed_turn is not None and self._completed_turn.turn_id == completed.turn_id:
            return
        self._completed_turn = completed
        self.recorder.record({"type": "turn_finalized", "turn_id": completed.turn_id, "version": completed.version})
        self.state = await self.state_store.load(self.session_id)
        if generation != self.interruptions.generation:
            return
        if completed.turn_id in self.state.finalized_turn_ids:
            self._replayed_finalized_turns.add(completed.turn_id)
            self.recorder.record({"type": "replayed_turn_rejected", "turn_id": completed.turn_id})
            return
        patch = None
        if self.facts_extractor is not None:
            extracted = self.facts_extractor(completed, self.state)
            patch = await extracted if inspect.isawaitable(extracted) else extracted
            if patch is not None and patch.source_turn_id != completed.turn_id:
                raise RuntimeError("facts extractor must return a patch for the finalized turn")
            if generation != self.interruptions.generation:
                return
        if patch is not None:
            if await self._apply_order_patch(patch, generation=generation) is None:
                return
        else:
            next_state = self.state.mark_turn_finalized(completed.turn_id)
            if not await self._save_state(
                next_state,
                expected_version=self.state.version,
                generation=generation,
            ):
                return
            self.state = next_state
            self.recorder.record({"type": "facts_extracted", "turn_id": completed.turn_id, "state_version": self.state.version})
        self.recorder.record({
            "type": "clarify_or_draft",
            "turn_id": completed.turn_id,
            "status": self.state.status,
            "unresolved_count": len(self.state.unresolved_fields),
        })

    async def _sync_order_memory(self, *, generation: int | None = None) -> OrderState | None:
        if generation is not None and generation != self.interruptions.generation:
            return None
        readbacks = [
            outcome.readback
            for outcome in self._outcomes
            if outcome.name in {
                "add_order_item",
                "set_order_fulfillment",
                "set_order_notes",
                "update_order_item",
                "remove_order_item",
                "confirm_order",
                "add_guest_note",
            }
            and outcome.success
            and outcome.readback_verified
            and isinstance(outcome.readback, Mapping)
        ]
        if not readbacks or self._completed_turn is None:
            return None
        readback = readbacks[-1]
        current = await self.state_store.load(self.session_id)
        if generation is not None and generation != self.interruptions.generation:
            return None
        turn_already_applied = self._completed_turn.turn_id in current.source_turn_ids
        if "items" in readback:
            items = tuple(
                OrderItemState(
                    canonical_item_id=str(value.get("item_id") or "order-item"),
                    item_name=str(value.get("item_name") or value.get("name") or "order item"),
                    quantity=int(value.get("quantity") or 1),
                    modifiers=tuple(str(item) for item in value.get("modifiers") or ()),
                    removals=tuple(str(item) for item in value.get("removals") or ()),
                    substitutions=tuple(str(item) for item in value.get("substitutions") or ()),
                    source_turn_ids=(self._completed_turn.turn_id,),
                    line_id=str(value.get("order_item_id") or ""),
                    notes=str(value.get("notes") or ""),
                )
                for value in readback.get("items") or ()
                if isinstance(value, Mapping)
            )
            incoming_line_ids = {item.line_id for item in items if item.line_id}
            remove_line_ids = tuple(
                item.line_id
                for item in current.items
                if item.line_id and item.status != "removed" and item.line_id not in incoming_line_ids
            )
        else:
            items = current.items
            remove_line_ids = ()
        order_notes = (
            str(readback.get("order_notes") or "")
            if "order_notes" in readback
            else current.order_notes
        )
        allergy_notes = (
            str(readback.get("allergy_notes") or "")
            if "allergy_notes" in readback
            else current.allergy_notes
        )
        guest_notes = (
            str(readback.get("guest_notes") or "")
            if "guest_notes" in readback
            else current.guest_notes
        )
        fulfillment = (
            str(readback.get("fulfillment") or "")
            if "fulfillment" in readback
            else current.fulfillment
        )
        fulfillment_details = (
            dict(readback.get("fulfillment_details") or {})
            if "fulfillment_details" in readback
            else dict(current.fulfillment_details)
        )
        status = (
            str(readback.get("status") or "")
            if "status" in readback
            else current.status
        )
        patch = OrderPatch(
            source_turn_id=self._completed_turn.turn_id,
            items=items,
            remove_line_ids=remove_line_ids,
            order_notes=order_notes,
            allergy_notes=allergy_notes,
            guest_notes=guest_notes,
            fulfillment=fulfillment,
            fulfillment_details=fulfillment_details,
            status=status,
        )
        if not items and not remove_line_ids and not patch.order_notes and not patch.allergy_notes and not patch.guest_notes and not patch.fulfillment:
            return None
        if turn_already_applied:
            next_state = replace(
                current,
                version=current.version + 1,
                items=items,
                order_notes=patch.order_notes if patch.order_notes is not None else current.order_notes,
                allergy_notes=patch.allergy_notes if patch.allergy_notes is not None else current.allergy_notes,
                guest_notes=patch.guest_notes if patch.guest_notes is not None else current.guest_notes,
                fulfillment=patch.fulfillment if patch.fulfillment is not None else current.fulfillment,
                fulfillment_details=(
                    dict(current.fulfillment_details)
                    if patch.fulfillment_details is None
                    else dict(patch.fulfillment_details)
                ),
                status=patch.status or current.status,
            )
        else:
            next_state = current.apply(patch)
        if generation is not None and generation != self.interruptions.generation:
            return None
        if not await self._save_state(
            next_state,
            expected_version=current.version,
            generation=generation,
        ):
            return None
        self.state = next_state
        self.recorder.record({"type": "order_memory_synced", "turn_id": patch.source_turn_id, "state_version": next_state.version})
        return next_state

    async def _run_tool(self, call_id: str, name: str, args: Mapping[str, Any], *, generation: int) -> ToolOutcome:
        if generation != self.interruptions.generation:
            return ToolOutcome(name=name, call_id=call_id, arguments=dict(args), result=None, success=False, error="stale_interrupted_tool_call", state_version=self.state.version)
        if self._completed_turn is not None and self._completed_turn.turn_id in self._replayed_finalized_turns:
            return ToolOutcome(name=name, call_id=call_id, arguments=dict(args), result=None, success=False, error="replayed_finalized_turn", state_version=self.state.version)
        if self.state.unresolved_fields:
            if name != "update_reservation_draft" or not self._reservation_correction_is_scoped(args):
                return ToolOutcome(
                    name=name,
                    call_id=call_id,
                    arguments=dict(args),
                    result=None,
                    success=False,
                    error="clarification_required",
                    state_version=self.state.version,
                )
        if name in MUTATING_TOOLS and self._completed_turn is None:
            return ToolOutcome(
                name=name,
                call_id=call_id,
                arguments=dict(args),
                result=None,
                success=False,
                error="caller_turn_not_finalized",
                state_version=self.state.version,
            )
        session_token = set_current_session_id(self.session_id)
        scope_token = set_current_action_scope(self._completed_turn.turn_id if self._completed_turn else call_id)
        try:
            outcome = await self.tool_bridge.invoke(
                call_id=call_id,
                name=name,
                arguments=args,
                turn_id=self._completed_turn.turn_id if self._completed_turn else "",
                state_version=self.state.version,
            )
        finally:
            reset_current_action_scope(scope_token)
            reset_current_session_id(session_token)
        if generation != self.interruptions.generation:
            return ToolOutcome(
                name=outcome.name,
                call_id=outcome.call_id,
                arguments=outcome.arguments,
                result=None,
                success=False,
                error="stale_interrupted_tool_call",
                state_version=outcome.state_version,
            )
        if name == "update_reservation_draft" and outcome.success and outcome.readback_verified:
            outcome = await self._apply_reservation_correction(outcome, generation=generation)
        self.recorder.record({"type": "tool_result_received", "call_id": call_id, "success": outcome.success, "readback_verified": outcome.readback_verified})
        if outcome.success and outcome.readback_verified:
            self.recorder.record({"type": "database_readback_verified", "call_id": call_id, "state_version": outcome.state_version})
        return outcome

    def _reservation_correction_is_scoped(self, arguments: Mapping[str, Any]) -> bool:
        aliases = {"customer_name": "name", "customer_phone": "phone"}
        unresolved = {
            aliases.get(field.rsplit(".", 1)[-1], field.rsplit(".", 1)[-1])
            for field in (item.field for item in self.state.unresolved_fields)
        }
        provided = {
            aliases.get(key, key)
            for key, value in arguments.items()
            if key != "session_id" and value is not None
        }
        allowed_fields = {
            "name", "phone", "date", "time", "party_size", "seating_preference",
            "seating_backup", "seating_avoid", "dietary", "occasion", "extra_notes",
            "require_approval_for_paid_items",
        }
        return bool(provided) and provided <= allowed_fields and provided <= unresolved

    async def _apply_reservation_correction(
        self,
        outcome: ToolOutcome,
        *,
        generation: int,
    ) -> ToolOutcome:
        aliases = {"customer_name": "name", "customer_phone": "phone"}
        resolved = {
            aliases.get(key, key)
            for key, value in outcome.arguments.items()
            if key != "session_id" and value is not None
        }
        current = await self.state_store.load(self.session_id)
        remaining = tuple(item for item in current.unresolved_fields if item.field.rsplit(".", 1)[-1] not in resolved)
        next_status = "needs_clarification" if remaining else ("draft" if current.items else "empty")
        next_state = replace(
            current,
            version=current.version + 1,
            unresolved_fields=remaining,
            status=next_status,
        )
        if not await self._save_state(next_state, expected_version=current.version, generation=generation):
            return replace(outcome, success=False, error="stale_interrupted_tool_call", readback_verified=False)
        self.state = next_state
        return replace(outcome, state_version=next_state.version)

    async def _finish_response(self, *, transcript: str | None) -> VoiceTurnResult:
        if self._response is None:
            return VoiceTurnResult(self._completed_turn, b"", "", None)
        text = "".join(self._response.transcript_parts)
        current = await self.state_store.load(self.session_id)
        self.state = current
        if self._response.audio and (
            not self._response.assistant_transcript_done
            or not self._response.assistant_transcript_seen
        ):
            decision = SpeechDecision(
                allowed=False,
                text="",
                audio=b"",
                reasons=("assistant_transcript_missing",),
                replacement=self.speech_gate.replacement,
            )
            self.recorder.record({"type": "speech_suppressed", "reasons": list(decision.reasons)})
            return VoiceTurnResult(
                turn=self._completed_turn,
                audio=b"",
                transcript="",
                speech=decision,
                tool_outcomes=tuple(self._outcomes),
                response_id=self._response.response_id,
            )
        decision = self.speech_gate.evaluate(
            text,
            bytes(self._response.audio),
            evidence=[outcome.as_evidence(turn_id=self._completed_turn.turn_id if self._completed_turn else "") for outcome in self._outcomes],
            current_state_version=current.version,
        )
        self.recorder.record({"type": "success_speakable" if decision.allowed else "speech_suppressed", "reasons": list(decision.reasons)})
        return VoiceTurnResult(
            turn=self._completed_turn,
            audio=decision.audio,
            transcript=text,
            speech=decision,
            tool_outcomes=tuple(self._outcomes),
            response_id=self._response.response_id,
        )


async def connect_development_adapter(
    *,
    session_id: str,
    config: RealtimeConfig | None = None,
    state_store: OrderStateStore | None = None,
) -> NativeVoiceAdapter:
    """Explicit development entry point; production settings fail closed."""
    from app.config import get_settings
    from app.native_voice.protocol import WebSocketRealtimeTransport

    settings = get_settings()
    if settings.is_production:
        raise RuntimeError("native voice is development-only and cannot start in production")
    if not getattr(settings, "native_voice_realtime_enabled", False):
        raise RuntimeError("set NATIVE_VOICE_REALTIME_ENABLED=true for the development adapter")
    resolved = config or RealtimeConfig(
        model=getattr(settings, "native_voice_realtime_model", "gpt-realtime"),
        voice=getattr(settings, "native_voice_realtime_voice", "marin"),
    )
    transport = await WebSocketRealtimeTransport.connect(
        api_key=settings.openai_api_key,
        model=resolved.model,
    )
    return NativeVoiceAdapter(
        session_id=session_id,
        transport=transport,
        config=resolved,
        state_store=state_store,
    )
