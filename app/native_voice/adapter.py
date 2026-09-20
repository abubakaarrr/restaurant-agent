"""Minimal development-only OpenAI Realtime adapter.

The adapter buffers model audio until the response is complete so the speech
gate can prevent unsupported claims from reaching the caller.  It is a
server-to-server WebSocket implementation for synthetic development audio;
production startup and Retell never import it.
"""

from __future__ import annotations

import base64
import json
import uuid
import inspect
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Mapping

from app.call_memory import reset_current_action_scope, reset_current_session_id, set_current_action_scope, set_current_session_id
from app.native_voice.contracts import OrderItemState, OrderPatch, OrderState
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


class InterruptionController:
    """Generation-based cancellation guard for audio, events, and tools."""

    def __init__(self) -> None:
        self.generation = 0
        self.active_response_id = ""

    def begin_response(self, response_id: str = "") -> int:
        self.active_response_id = response_id
        return self.generation

    def interrupt(self) -> int:
        self.generation += 1
        self.active_response_id = ""
        return self.generation

    def accepts(self, *, generation: int, response_id: str = "") -> bool:
        if generation != self.generation:
            return False
        return not response_id or not self.active_response_id or response_id == self.active_response_id


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
        if hasattr(self.tool_bridge, "bind_session"):
            self.tool_bridge.bind_session(session_id)

    async def start(self) -> None:
        if self._started:
            return
        self.state = await self.state_store.load(self.session_id)
        event = self.config.session_update()
        await self.transport.send(event)
        self.recorder.record(event)
        self._started = True

    async def close(self) -> None:
        await self.transport.close()

    async def apply_order_patch(self, patch: OrderPatch) -> OrderState:
        """Apply facts only after a completed caller turn exists."""
        if self._completed_turn is None or self._completed_turn.turn_id != patch.source_turn_id:
            raise RuntimeError("structured order mutation requires the matching finalized caller turn")
        current = await self.state_store.load(self.session_id)
        next_state = current.apply(patch)
        await self.state_store.save(self.session_id, next_state, expected_version=current.version)
        self.state = next_state
        self.recorder.record({"type": "facts_extracted", "turn_id": patch.source_turn_id, "state_version": next_state.version})
        return next_state

    async def submit_audio(
        self,
        audio: bytes,
        *,
        turn_id: str | None = None,
        transcript: str | None = None,
    ) -> VoiceTurnResult:
        """Send one synthetic PCM16 turn and drain native output until done."""
        await self.start()
        if self._response is not None:
            await self.interrupt()
        resolved_turn_id = turn_id or f"turn-{uuid.uuid4().hex[:12]}"
        if self.turns.completed(resolved_turn_id) is not None:
            self._replayed_finalized_turns.add(resolved_turn_id)
        self.turns.start(resolved_turn_id)
        self.recorder.record_audio("audio_received", audio, turn_id=resolved_turn_id)
        append = {"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode("ascii")}
        await self.transport.send(append)
        self.recorder.record_audio("input_audio_buffer.append", audio, turn_id=resolved_turn_id)
        await self.transport.send({"type": "input_audio_buffer.commit"})
        self.recorder.record({"type": "input_audio_buffer.commit", "turn_id": resolved_turn_id})
        await self.transport.send({"type": "response.create", "response": {"output_modalities": ["audio"]}})
        self.recorder.record({"type": "response.create", "turn_id": resolved_turn_id})
        return await self._drain_response(transcript=transcript)

    def mark_audio_played(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        if self._response is not None:
            self._response.played_audio_bytes = min(byte_count, len(self._response.audio))

    async def interrupt(self) -> None:
        response = self._response
        generation = self.interruptions.interrupt()
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
        while True:
            event = await self.transport.receive()
            event_type = str(event.get("type") or "unknown")
            response_id = str(event.get("response_id") or "")
            if not self.interruptions.accepts(generation=generation, response_id=response_id):
                self.recorder.record({"type": "stale_event_ignored", "original_type": event_type, "response_id": response_id})
                if generation != self.interruptions.generation:
                    result = VoiceTurnResult(self._completed_turn, b"", "", None)
                    self._response = None
                    return result
                continue
            self.recorder.record(event)
            if event_type == "response.created":
                self._response.response_id = response_id or str((event.get("response") or {}).get("id") or "")
                self.interruptions.begin_response(self._response.response_id)
                continue
            if event_type == "input_audio_buffer.speech_started":
                await self.interrupt()
                return VoiceTurnResult(self._completed_turn, b"", "", None)
            if event_type == "conversation.item.input_audio_transcription.delta":
                self.turns.add_delta(str(event.get("delta") or ""))
                continue
            if event_type == "conversation.item.input_audio_transcription.completed":
                await self._finalize_caller_turn(str(event.get("transcript") or transcript or ""))
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
                self._response.transcript_parts.append(str(event.get("delta") or ""))
                continue
            if event_type == "response.output_audio_transcript.done":
                if event.get("transcript"):
                    self._response.transcript_parts = [str(event["transcript"])]
                continue
            if event_type == "response.function_call_arguments.done":
                self._remember_tool_call(event)
                continue
            if event_type == "error":
                self.recorder.record({"type": "tool_or_response_failure", "code": (event.get("error") or {}).get("code", "realtime_error")})
                continue
            if event_type == "response.done":
                result = await self._finish_response(transcript=transcript)
                if self._response is not None and self._response.tool_calls:
                    for call_id, (name, args) in list(self._response.tool_calls.items()):
                        outcome = await self._run_tool(call_id, name, args, generation=generation)
                        self._outcomes.append(outcome)
                        output = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps({
                                    "ok": outcome.success,
                                    "error": outcome.error or None,
                                    "result": outcome.result,
                                    "readback_verified": outcome.readback_verified,
                                    "state_version": outcome.state_version,
                                }, default=str),
                            },
                        }
                        await self.transport.send(output)
                        self.recorder.record({"type": "tool_result_sent", "call_id": call_id, "success": outcome.success})
                    synced = await self._sync_order_memory()
                    if synced is not None:
                        self._outcomes = [replace(outcome, state_version=synced.version) for outcome in self._outcomes]
                    # The next response has a new server response id.  Clear
                    # the old id before accepting its ``response.created``.
                    self.interruptions.active_response_id = ""
                    self._response = _ResponseBuffer(generation=generation)
                    await self.transport.send({"type": "response.create", "response": {"output_modalities": ["audio"]}})
                    self.recorder.record({"type": "response.create", "reason": "after_tool"})
                    continue
                self._last_result = result
                self._response = None
                return result

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

    async def _finalize_caller_turn(self, transcript: str) -> None:
        completed = self.turns.finalize(transcript)
        if self._completed_turn is not None and self._completed_turn.turn_id == completed.turn_id:
            return
        self._completed_turn = completed
        self.recorder.record({"type": "turn_finalized", "turn_id": completed.turn_id, "version": completed.version})
        self.state = await self.state_store.load(self.session_id)
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
        if patch is not None:
            await self.apply_order_patch(patch)
        else:
            next_state = self.state.mark_turn_finalized(completed.turn_id)
            await self.state_store.save(self.session_id, next_state, expected_version=self.state.version)
            self.state = next_state
            self.recorder.record({"type": "facts_extracted", "turn_id": completed.turn_id, "state_version": self.state.version})
        self.recorder.record({
            "type": "clarify_or_draft",
            "turn_id": completed.turn_id,
            "status": self.state.status,
            "unresolved_count": len(self.state.unresolved_fields),
        })

    async def _sync_order_memory(self) -> OrderState | None:
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
            }
            and outcome.success
            and outcome.readback_verified
            and isinstance(outcome.readback, Mapping)
        ]
        if not readbacks or self._completed_turn is None:
            return None
        readback = readbacks[-1]
        current = await self.state_store.load(self.session_id)
        if self._completed_turn.turn_id in current.source_turn_ids:
            return None
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
        patch = OrderPatch(
            source_turn_id=self._completed_turn.turn_id,
            items=items,
            remove_line_ids=remove_line_ids,
            order_notes=str(readback.get("order_notes") or ""),
            allergy_notes=str(readback.get("allergy_notes") or ""),
            fulfillment=str(readback.get("fulfillment") or ""),
            fulfillment_details=readback.get("fulfillment_details") or {},
            status=str(readback.get("status") or "draft"),
        )
        if not items and not remove_line_ids and not patch.order_notes and not patch.allergy_notes and not patch.fulfillment:
            return None
        next_state = current.apply(patch)
        await self.state_store.save(self.session_id, next_state, expected_version=current.version)
        self.state = next_state
        self.recorder.record({"type": "order_memory_synced", "turn_id": patch.source_turn_id, "state_version": next_state.version})
        return next_state

    async def _run_tool(self, call_id: str, name: str, args: Mapping[str, Any], *, generation: int) -> ToolOutcome:
        if generation != self.interruptions.generation:
            return ToolOutcome(name=name, call_id=call_id, arguments=dict(args), result=None, success=False, error="stale_interrupted_tool_call", state_version=self.state.version)
        if self._completed_turn is not None and self._completed_turn.turn_id in self._replayed_finalized_turns:
            return ToolOutcome(name=name, call_id=call_id, arguments=dict(args), result=None, success=False, error="replayed_finalized_turn", state_version=self.state.version)
        if name in MUTATING_TOOLS and self.state.unresolved_fields:
            return ToolOutcome(
                name=name,
                call_id=call_id,
                arguments=dict(args),
                result=None,
                success=False,
                error="clarification_required",
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
        self.recorder.record({"type": "tool_result_received", "call_id": call_id, "success": outcome.success, "readback_verified": outcome.readback_verified})
        if outcome.success and outcome.readback_verified:
            self.recorder.record({"type": "database_readback_verified", "call_id": call_id, "state_version": outcome.state_version})
        return outcome

    async def _finish_response(self, *, transcript: str | None) -> VoiceTurnResult:
        if self._response is None:
            return VoiceTurnResult(self._completed_turn, b"", "", None)
        text = "".join(self._response.transcript_parts) or (transcript or "")
        current = await self.state_store.load(self.session_id)
        self.state = current
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
