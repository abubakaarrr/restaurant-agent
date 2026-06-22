"""Retell custom-LLM WebSocket handler.

Retell owns the audio layer: telephony + ASR (their own transcriber) +
turn-detection + TTS. It opens a WebSocket to this server, streams us live
transcripts, and asks for responses. We run the existing LangGraph agent and
stream text tokens back; Retell converts our text to speech.

This intentionally reuses `stream_agent_tokens()` from app.agent.runner, so the
agent, tools, prompt, and database logic are identical to the Vapi path.

Protocol reference: https://docs.retellai.com/api-references/llm-websocket
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket, WebSocketDisconnect

from app.agent.runner import clear_session, stream_agent_tokens
from app.config import settings

logger = logging.getLogger(__name__)


def _latest_user_utterance(transcript: list[dict] | None) -> str:
    """Pull the most recent user turn from Retell's transcript array."""
    for turn in reversed(transcript or []):
        if turn.get("role") == "user":
            return (turn.get("content") or "").strip()
    return ""


def _response_event(
    response_id: int,
    content: str,
    *,
    complete: bool,
    end_call: bool = False,
) -> str:
    """Build a Retell `response` event (one streamed chunk of agent speech)."""
    return json.dumps(
        {
            "response_type": "response",
            "response_id": response_id,
            "content": content,
            "content_complete": complete,
            "end_call": end_call,
        }
    )


async def handle_retell_connection(websocket: WebSocket, call_id: str) -> None:
    """Drive one Retell call over the LLM WebSocket until it disconnects.

    The main loop ONLY reads messages and dispatches — it never blocks on agent
    generation. Each turn runs as a cancellable background task so we can keep
    answering keepalive pings and, crucially, cancel a stale response when the
    caller keeps talking or interrupts (barge-in). All sends go through a lock
    so the reader loop and the generation task never interleave WebSocket frames.
    """
    send_lock = asyncio.Lock()

    async def send(payload: str) -> None:
        async with send_lock:
            await websocket.send_text(payload)

    # Optional config event — tell Retell we want call details + auto-reconnect.
    await send(
        json.dumps(
            {
                "response_type": "config",
                "config": {"auto_reconnect": True, "call_details": True},
            }
        )
    )

    caller_number = ""
    current_task: asyncio.Task | None = None

    async def run_turn(response_id: int, user_text: str) -> None:
        """Stream one agent reply. Cancellable if a newer turn supersedes it."""
        try:
            async for token in stream_agent_tokens(call_id, user_text, caller_number):
                if token:
                    await send(_response_event(response_id, token, complete=False))
            await send(_response_event(response_id, "", complete=True))
        except asyncio.CancelledError:
            # Caller interrupted / a newer turn arrived — drop this reply silently.
            raise
        except Exception:
            logger.error("Retell agent stream error", exc_info=True)
            try:
                await send(
                    _response_event(
                        response_id,
                        "I'm sorry, I had a technical issue. Could you please repeat that?",
                        complete=True,
                    )
                )
            except Exception:
                pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Retell sent non-JSON message; ignoring")
                continue

            interaction = msg.get("interaction_type")

            # Keepalive — must be answered even while a turn is generating.
            if interaction == "ping_pong":
                await send(
                    json.dumps(
                        {"response_type": "ping_pong", "timestamp": msg.get("timestamp")}
                    )
                )
                continue

            # Call metadata sent once at the start.
            if interaction == "call_details":
                call = msg.get("call", {}) or {}
                caller_number = call.get("from_number") or call.get("from") or ""
                continue

            # Live transcript update — no response expected.
            if interaction == "update_only":
                continue

            if interaction in ("response_required", "reminder_required"):
                # A newer turn supersedes whatever we were saying (barge-in or the
                # caller simply kept talking past an earlier pause).
                if current_task and not current_task.done():
                    current_task.cancel()

                response_id = msg.get("response_id", 0)
                user_text = _latest_user_utterance(msg.get("transcript"))

                # Nudge a silent caller.
                if interaction == "reminder_required" and not user_text:
                    await send(
                        _response_event(
                            response_id,
                            "Are you still there? How can I help you?",
                            complete=True,
                        )
                    )
                    continue

                # First turn with no caller speech yet → greet.
                if not user_text:
                    greeting = (
                        f"Hello! Thank you for calling {settings.restaurant_name}. "
                        "How can I help you today?"
                    )
                    await send(_response_event(response_id, greeting, complete=True))
                    continue

                # Generate in the background so the loop keeps reading messages.
                current_task = asyncio.create_task(run_turn(response_id, user_text))

    except WebSocketDisconnect:
        logger.info("Retell call %s disconnected", call_id)
    except Exception:
        logger.error("Retell websocket error on call %s", call_id, exc_info=True)
    finally:
        if current_task and not current_task.done():
            current_task.cancel()
        clear_session(call_id)
