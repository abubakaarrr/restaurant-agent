"""Protocol-safe Retell custom-LLM rollback adapter.

The production pilot uses Retell managed Conversation Flow. This adapter remains
available for one rollback release and therefore prioritizes cancellation,
idempotency, transfer safety, and measurable first-response latency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.runner import stream_agent_tokens
from app.behavior import (
    BehaviorControl,
    BehaviorReduction,
    BehaviorState,
    TurnObservation,
    reduce_behavior,
)
from app.behavior_store import load_behavior_state, save_behavior_state
from app.call_analytics import record_call_event
from app.call_flags import (
    clear_call_control,
    consume_call_control,
    reset_call_control_scope,
    set_call_control_scope,
)
from app.caller_turn import process_caller_turn
from app.config import settings
from app.restaurant_settings import load_restaurant_settings
from app.spoken_delivery import (
    INCOMPLETE_INPUT_REPLY,
    TRANSFER_UNAVAILABLE_REPLY,
    ResponseGenerationGate,
    SpokenTextBuffer,
    sanitize_spoken_text,
)
from app.transfer_availability import current_staff_transfer_number


logger = logging.getLogger(__name__)


def _locale_supported(requested: str | None) -> bool:
    if not requested:
        return True
    requested_base = requested.casefold().split("-", 1)[0]
    return any(
        requested.casefold() == allowed.casefold()
        or requested_base == allowed.casefold().split("-", 1)[0]
        for allowed in settings.locale_list
    )


def _latest_user_turn(transcript: list[dict[str, Any]] | None) -> dict[str, Any]:
    for turn in reversed(transcript or []):
        if turn.get("role") == "user":
            return turn
    return {}


def _response_event(
    response_id: int,
    content: str,
    *,
    complete: bool,
    end_call: bool = False,
    transfer_number: str = "",
    no_interruption_allowed: bool = False,
) -> str:
    payload: dict[str, Any] = {
        "response_type": "response",
        "response_id": response_id,
        "content": sanitize_spoken_text(content),
        "content_complete": complete,
    }
    if end_call:
        payload["end_call"] = True
    if transfer_number:
        payload["transfer_number"] = transfer_number
        payload["transfer_caller_id"] = True
    if no_interruption_allowed:
        payload["no_interruption_allowed"] = True
    return json.dumps(payload)


def _agent_update_event(
    *,
    responsiveness: float | None = None,
    interruption_sensitivity: float | None = None,
    reminder_trigger_ms: int | None = None,
    reminder_max_count: int | None = None,
) -> str:
    config: dict[str, Any] = {}
    if responsiveness is not None:
        config["responsiveness"] = max(0.0, min(1.0, responsiveness))
    if interruption_sensitivity is not None:
        config["interruption_sensitivity"] = max(
            0.0, min(1.0, interruption_sensitivity)
        )
    if reminder_trigger_ms is not None:
        config["reminder_trigger_ms"] = max(1, reminder_trigger_ms)
    if reminder_max_count is not None:
        config["reminder_max_count"] = max(0, reminder_max_count)
    return json.dumps({"response_type": "update_agent", "agent_config": config})


async def _record_safely(call_id: str, event_type: str, **kwargs: Any) -> None:
    try:
        await record_call_event(call_id, event_type, **kwargs)
    except Exception:
        logger.warning("Could not persist call metric %s", event_type, exc_info=True)


def _record_background(call_id: str, event_type: str, **kwargs: Any) -> None:
    asyncio.create_task(_record_safely(call_id, event_type, **kwargs))


async def handle_retell_connection(websocket: WebSocket, call_id: str) -> None:
    """Drive one authenticated custom-LLM WebSocket until disconnect."""
    send_lock = asyncio.Lock()
    behavior_state: BehaviorState = await load_behavior_state(call_id)
    last_agent_controls: dict[str, float | int] | None = None

    async def send(payload: str) -> None:
        async with send_lock:
            await websocket.send_text(payload)

    await send(
        json.dumps(
            {
                "response_type": "config",
                "config": {"auto_reconnect": True, "call_details": True},
            }
        )
    )

    caller_number = ""
    current_task: asyncio.Task[None] | None = None
    current_response_id: int | None = None
    response_gate = ResponseGenerationGate()
    reminder_count = 0
    latest_transcript: list[dict[str, Any]] = []
    greeting_sent = False

    def stage_behavior(
        turn: dict[str, Any],
        *,
        reminder: bool = False,
        interrupted: bool = False,
    ) -> BehaviorReduction:
        words = tuple(turn.get("words") or ())
        return reduce_behavior(
            behavior_state,
            TurnObservation(
                text=str(turn.get("content") or ""),
                timed_words=words,
                reminder=reminder,
                interrupted=interrupted,
            ),
        )

    async def commit_behavior(
        response_id: int, reduction: BehaviorReduction
    ) -> bool:
        nonlocal behavior_state, last_agent_controls
        if not response_allowed(response_id, "behavior_commit"):
            return False
        behavior_state = reduction.state
        asyncio.create_task(save_behavior_state(call_id, behavior_state))
        controls = reduction.directive.to_retell_controls()
        if (
            controls != last_agent_controls
            and reduction.directive.control is BehaviorControl.CONTINUE
            and response_gate.allows(response_id)
        ):
            await send(
                _agent_update_event(
                    responsiveness=float(controls["responsiveness"]),
                    interruption_sensitivity=float(
                        controls["interruption_sensitivity"]
                    ),
                    reminder_trigger_ms=int(controls["reminder_trigger_ms"]),
                )
            )
            if response_gate.allows(response_id):
                last_agent_controls = controls
        return True

    async def cancel_current(reason: str) -> None:
        nonlocal current_task, current_response_id
        if not current_task or current_task.done():
            current_task = None
            current_response_id = None
            return
        started = time.monotonic()
        cancelled_response_id = current_response_id
        current_task.cancel()
        with suppress(asyncio.CancelledError):
            await current_task
        if cancelled_response_id is not None:
            clear_call_control(call_id, str(cancelled_response_id))
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _record_background(
            call_id,
            "generation_cancelled",
            duration_ms=elapsed_ms,
            payload={"reason": reason},
        )
        current_task = None
        current_response_id = None

    def response_allowed(response_id: int, stage: str) -> bool:
        if response_gate.allows(response_id):
            return True
        _record_background(
            call_id,
            "stale_response_suppressed",
            response_id=response_id,
            payload={
                "active_response_id": response_gate.active_response_id,
                "stage": stage,
            },
        )
        return False

    async def send_response(
        response_id: int,
        content: str,
        *,
        complete: bool,
        end_call: bool = False,
        transfer_number: str = "",
        no_interruption_allowed: bool = False,
        stage: str,
    ) -> bool:
        async with send_lock:
            if not response_allowed(response_id, stage):
                return False
            await websocket.send_text(
                _response_event(
                    response_id,
                    content,
                    complete=complete,
                    end_call=end_call,
                    transfer_number=transfer_number,
                    no_interruption_allowed=no_interruption_allowed,
                )
            )
        return response_allowed(response_id, f"{stage}_post_send")

    async def run_turn(
        response_id: int,
        user_text: str,
        behavior_directive: str = "",
        caller_turn: dict[str, Any] | None = None,
    ) -> bool:
        started = time.monotonic()
        first_chunk_sent = False
        completed = False
        spoken = SpokenTextBuffer()
        try:
            async for token in stream_agent_tokens(
                call_id,
                user_text,
                caller_number,
                behavior_directive,
                prepared_caller_turn=caller_turn,
            ):
                if not token:
                    continue
                for spoken_token in spoken.feed(token):
                    delivered = await send_response(
                        response_id,
                        spoken_token,
                        complete=False,
                        stage="stream_chunk",
                    )
                    if not delivered:
                        return False
                    if not first_chunk_sent:
                        first_chunk_sent = True
                        elapsed_ms = int((time.monotonic() - started) * 1000)
                        _record_background(
                            call_id,
                            "first_response_chunk",
                            response_id=response_id,
                            duration_ms=elapsed_ms,
                        )

            for spoken_token in spoken.flush():
                delivered = await send_response(
                    response_id,
                    spoken_token,
                    complete=False,
                    stage="stream_flush",
                )
                if not delivered:
                    return False
                if not first_chunk_sent:
                    first_chunk_sent = True
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    _record_background(
                        call_id,
                        "first_response_chunk",
                        response_id=response_id,
                        duration_ms=elapsed_ms,
                    )

            if not response_allowed(response_id, "completion"):
                return False
            control = consume_call_control(call_id)
            end_call = bool(control and control.action == "end")
            transfer_number = ""
            final_content = ""
            no_interruption = False
            if control and control.action == "transfer":
                transfer_number = control.transfer_number
                if transfer_number:
                    no_interruption = True
                else:
                    final_content = TRANSFER_UNAVAILABLE_REPLY
            completed = await send_response(
                response_id,
                final_content,
                complete=True,
                end_call=end_call,
                transfer_number=transfer_number,
                no_interruption_allowed=no_interruption,
                stage="completion",
            )
            if not completed:
                return False
            _record_background(
                call_id,
                "response_complete",
                response_id=response_id,
                duration_ms=int((time.monotonic() - started) * 1000),
                payload={
                    "ended": end_call,
                    "transferred": bool(transfer_number),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Retell agent stream error", exc_info=True)
            if response_allowed(response_id, "error_fallback"):
                with suppress(Exception):
                    transfer_number = current_staff_transfer_number()
                    completed = await send_response(
                        response_id,
                        (
                            "I'm sorry, I had a technical issue. "
                            + (
                                "I can connect you with the restaurant team."
                                if transfer_number
                                else "I can't transfer right now, but I can take a callback message."
                            )
                        ),
                        complete=True,
                        transfer_number=transfer_number,
                        stage="error_fallback",
                    )
            _record_background(
                call_id,
                "generation_error",
                response_id=response_id,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        finally:
            # If generation failed without a newer response superseding it,
            # close the current response so Retell never waits forever.
            if (
                not completed
                and response_gate.allows(response_id)
                and not asyncio.current_task().cancelled()
            ):
                with suppress(Exception):
                    await send_response(
                        response_id,
                        "",
                        complete=True,
                        stage="finalizer",
                    )
        return completed

    async def process_interaction(
        message: dict[str, Any],
        *,
        response_id: int,
        was_interrupted: bool,
        transcript_snapshot: list[dict[str, Any]],
    ) -> None:
        nonlocal reminder_count, greeting_sent
        scope_token = set_call_control_scope(str(response_id))
        try:
            interaction = message.get("interaction_type")
            if interaction == "reminder_required":
                next_reminder_count = reminder_count + 1
                reduction = stage_behavior(
                    {"content": "", "words": []},
                    reminder=True,
                    interrupted=was_interrupted,
                )
                directive = reduction.directive
                delivered = await send_response(
                    response_id,
                    directive.direct_reply or "Are you still there?",
                    complete=True,
                    end_call=directive.control is BehaviorControl.END_CALL,
                    stage="reminder",
                )
                if not delivered or not await commit_behavior(response_id, reduction):
                    return
                reminder_count = next_reminder_count
                _record_background(
                    call_id,
                    "silence_reminder",
                    response_id=response_id,
                    payload={"count": next_reminder_count},
                )
                return

            reminder_count = 0
            user_turn = _latest_user_turn(transcript_snapshot)
            user_text = str(user_turn.get("content") or "").strip()

            if not user_text and not greeting_sent:
                runtime = load_restaurant_settings()
                restaurant = runtime["restaurant_name"]
                agent_name = runtime.get("ai_agent_name") or settings.ai_agent_name
                delivered = await send_response(
                    response_id,
                    (
                        f"Hi, you've reached {restaurant}. This is {agent_name}. "
                        "How can I help you today?"
                    ),
                    complete=True,
                    stage="greeting",
                )
                if delivered:
                    greeting_sent = True
                return

            if not user_text:
                await send_response(
                    response_id,
                    INCOMPLETE_INPUT_REPLY,
                    complete=True,
                    stage="incomplete_input",
                )
                return

            greeting_sent = True
            caller_turn = await process_caller_turn(call_id, user_text)
            if not response_allowed(response_id, "caller_turn"):
                return
            if caller_turn.get("handled"):
                await send_response(
                    response_id,
                    str(caller_turn.get("message") or ""),
                    complete=True,
                    stage="caller_turn_reply",
                )
                return

            reduction = stage_behavior(
                user_turn,
                interrupted=was_interrupted,
            )
            directive = reduction.directive
            if not response_allowed(response_id, "behavior"):
                return
            if directive.locale and not _locale_supported(directive.locale):
                transfer_number = current_staff_transfer_number()
                can_transfer = bool(transfer_number)
                delivered = await send_response(
                    response_id,
                    (
                        "I'm sorry, I don't support that language reliably yet. "
                        + (
                            "I'll connect you with the restaurant team."
                            if can_transfer
                            else "I can't transfer right now, but I can take a callback message."
                        )
                    ),
                    complete=True,
                    transfer_number=transfer_number,
                    no_interruption_allowed=can_transfer,
                    stage="locale_reply",
                )
                if delivered:
                    await commit_behavior(response_id, reduction)
                return

            if directive.direct_reply is not None:
                transfer_number = directive.transfer_number
                delivered = await send_response(
                    response_id,
                    directive.direct_reply,
                    complete=True,
                    end_call=directive.control is BehaviorControl.END_CALL,
                    transfer_number=transfer_number,
                    no_interruption_allowed=bool(transfer_number),
                    stage="behavior_reply",
                )
                if delivered:
                    await commit_behavior(response_id, reduction)
                return

            _record_background(
                call_id,
                "response_requested",
                response_id=response_id,
            )
            completed = await run_turn(
                response_id,
                user_text,
                directive.prompt_instruction,
                caller_turn,
            )
            if completed:
                await commit_behavior(response_id, reduction)
        finally:
            if not response_gate.allows(response_id):
                clear_call_control(call_id, str(response_id))
            reset_call_control_scope(scope_token)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Retell sent a non-JSON frame")
                continue

            interaction = message.get("interaction_type")
            if interaction == "ping_pong":
                await send(
                    json.dumps(
                        {
                            "response_type": "ping_pong",
                            "timestamp": message.get("timestamp"),
                        }
                    )
                )
                continue

            if interaction == "call_details":
                call = message.get("call") if isinstance(message.get("call"), dict) else {}
                caller_number = str(
                    call.get("from_number") or call.get("from") or ""
                )
                _record_background(call_id, "websocket_connected")
                continue

            if interaction == "update_only":
                transcript = message.get("transcript")
                if isinstance(transcript, list):
                    latest_transcript = transcript
                continue

            if interaction not in {"response_required", "reminder_required"}:
                continue

            was_interrupted = bool(current_task and not current_task.done())
            response_id = int(message.get("response_id", 0))
            response_gate.begin(response_id)
            await cancel_current("new_response")
            transcript = message.get("transcript")
            if isinstance(transcript, list):
                latest_transcript = transcript
            current_task = asyncio.create_task(
                process_interaction(
                    message,
                    response_id=response_id,
                    was_interrupted=was_interrupted,
                    transcript_snapshot=list(latest_transcript),
                )
            )
            current_response_id = response_id
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info("Retell call %s disconnected", call_id)
    except Exception:
        logger.error("Retell WebSocket error on call %s", call_id, exc_info=True)
    finally:
        await cancel_current("socket_closed")
        # Do not clear call state here: Retell may reconnect. The signed
        # call_ended webhook and retention policy own final cleanup.
