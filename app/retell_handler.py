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
    BehaviorState,
    TurnObservation,
    reduce_behavior,
)
from app.behavior_store import load_behavior_state, save_behavior_state
from app.call_analytics import record_call_event
from app.call_flags import consume_call_control
from app.config import settings
from app.restaurant_settings import load_restaurant_settings
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
        "content": content,
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
    active_response_id = -1
    reminder_count = 0
    latest_transcript: list[dict[str, Any]] = []

    async def apply_behavior(
        turn: dict[str, Any],
        *,
        reminder: bool = False,
        interrupted: bool = False,
    ):
        nonlocal behavior_state, last_agent_controls
        words = tuple(turn.get("words") or ())
        reduction = reduce_behavior(
            behavior_state,
            TurnObservation(
                text=str(turn.get("content") or ""),
                timed_words=words,
                reminder=reminder,
                interrupted=interrupted,
            ),
        )
        behavior_state = reduction.state
        asyncio.create_task(save_behavior_state(call_id, behavior_state))
        controls = reduction.directive.to_retell_controls()
        if controls != last_agent_controls and reduction.directive.control is BehaviorControl.CONTINUE:
            await send(
                _agent_update_event(
                    responsiveness=float(controls["responsiveness"]),
                    interruption_sensitivity=float(
                        controls["interruption_sensitivity"]
                    ),
                    reminder_trigger_ms=int(controls["reminder_trigger_ms"]),
                )
            )
            last_agent_controls = controls
        return reduction.directive

    async def cancel_current(reason: str) -> None:
        nonlocal current_task
        if not current_task or current_task.done():
            return
        started = time.monotonic()
        current_task.cancel()
        with suppress(asyncio.CancelledError):
            await current_task
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _record_background(
            call_id,
            "generation_cancelled",
            duration_ms=elapsed_ms,
            payload={"reason": reason},
        )
        current_task = None

    async def run_turn(
        response_id: int,
        user_text: str,
        behavior_directive: str = "",
    ) -> None:
        nonlocal current_task
        started = time.monotonic()
        first_chunk_sent = False
        completed = False
        try:
            async for token in stream_agent_tokens(
                call_id,
                user_text,
                caller_number,
                behavior_directive,
            ):
                if response_id != active_response_id:
                    return
                if not token:
                    continue
                await send(_response_event(response_id, token, complete=False))
                if not first_chunk_sent:
                    first_chunk_sent = True
                    elapsed_ms = int((time.monotonic() - started) * 1000)
                    _record_background(
                        call_id,
                        "first_response_chunk",
                        response_id=response_id,
                        duration_ms=elapsed_ms,
                    )

            if response_id != active_response_id:
                return
            control = consume_call_control(call_id)
            end_call = bool(control and control.action == "end")
            transfer_number = ""
            final_content = ""
            no_interruption = False
            if control and control.action == "transfer":
                transfer_number = current_staff_transfer_number()
                if transfer_number:
                    no_interruption = True
                else:
                    final_content = (
                        "I'm sorry, I can't transfer the call right now. "
                        "I can take a message and callback details for the restaurant team."
                    )
            await send(
                _response_event(
                    response_id,
                    final_content,
                    complete=True,
                    end_call=end_call,
                    transfer_number=transfer_number,
                    no_interruption_allowed=no_interruption,
                )
            )
            completed = True
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
            if response_id == active_response_id:
                with suppress(Exception):
                    transfer_number = current_staff_transfer_number()
                    await send(
                        _response_event(
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
                        )
                    )
                    completed = True
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
                and response_id == active_response_id
                and not asyncio.current_task().cancelled()
            ):
                with suppress(Exception):
                    await send(_response_event(response_id, "", complete=True))

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
            await cancel_current("new_response")
            active_response_id = int(message.get("response_id", 0))

            if interaction == "reminder_required":
                # A reminder is never a replay of the previous user action.
                reminder_count += 1
                directive = await apply_behavior(
                    {"content": "", "words": []},
                    reminder=True,
                    interrupted=was_interrupted,
                )
                content = directive.direct_reply or "Are you still there?"
                end_call = directive.control is BehaviorControl.END_CALL
                await send(
                    _response_event(
                        active_response_id,
                        content,
                        complete=True,
                        end_call=end_call,
                    )
                )
                _record_background(
                    call_id,
                    "silence_reminder",
                    response_id=active_response_id,
                    payload={"count": reminder_count},
                )
                continue

            reminder_count = 0
            transcript = message.get("transcript")
            if isinstance(transcript, list):
                latest_transcript = transcript
            user_turn = _latest_user_turn(latest_transcript)
            user_text = str(user_turn.get("content") or "").strip()

            if not user_text:
                runtime = load_restaurant_settings()
                restaurant = runtime.get("restaurant_name") or settings.restaurant_name
                agent_name = runtime.get("ai_agent_name") or settings.ai_agent_name
                greeting = (
                    f"Hi, you've reached {restaurant}. This is {agent_name}. "
                    "How can I help you today?"
                )
                await send(
                    _response_event(active_response_id, greeting, complete=True)
                )
                continue

            directive = await apply_behavior(
                user_turn,
                interrupted=was_interrupted,
            )
            if directive.locale and not _locale_supported(directive.locale):
                transfer_number = current_staff_transfer_number()
                can_transfer = bool(transfer_number)
                await send(
                    _response_event(
                        active_response_id,
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
                    )
                )
                continue

            if directive.direct_reply is not None:
                transfer_number = (
                    current_staff_transfer_number()
                    if directive.control is BehaviorControl.HANDOFF
                    else ""
                )
                await send(
                    _response_event(
                        active_response_id,
                        directive.direct_reply,
                        complete=True,
                        end_call=directive.control is BehaviorControl.END_CALL,
                        transfer_number=transfer_number,
                        no_interruption_allowed=bool(transfer_number),
                    )
                )
                continue

            _record_background(
                call_id,
                "response_requested",
                response_id=active_response_id,
            )
            current_task = asyncio.create_task(
                run_turn(
                    active_response_id,
                    user_text,
                    directive.prompt_instruction,
                )
            )
    except WebSocketDisconnect:
        logger.info("Retell call %s disconnected", call_id)
    except Exception:
        logger.error("Retell WebSocket error on call %s", call_id, exc_info=True)
    finally:
        await cancel_current("socket_closed")
        # Do not clear call state here: Retell may reconnect. The signed
        # call_ended webhook and retention policy own final cleanup.
