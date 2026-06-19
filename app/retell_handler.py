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
    """Drive one Retell call over the LLM WebSocket until it disconnects."""
    # Optional config event — tell Retell we want call details + auto-reconnect.
    await websocket.send_text(
        json.dumps(
            {
                "response_type": "config",
                "config": {"auto_reconnect": True, "call_details": True},
            }
        )
    )

    caller_number = ""

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Retell sent non-JSON message; ignoring")
                continue

            interaction = msg.get("interaction_type")

            # Keepalive.
            if interaction == "ping_pong":
                await websocket.send_text(
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
                response_id = msg.get("response_id", 0)
                user_text = _latest_user_utterance(msg.get("transcript"))

                # Nudge a silent caller.
                if interaction == "reminder_required" and not user_text:
                    await websocket.send_text(
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
                    await websocket.send_text(
                        _response_event(response_id, greeting, complete=True)
                    )
                    continue

                # Normal turn — stream the LangGraph agent's reply token by token.
                try:
                    async for token in stream_agent_tokens(call_id, user_text, caller_number):
                        if token:
                            await websocket.send_text(
                                _response_event(response_id, token, complete=False)
                            )
                except Exception:
                    logger.error("Retell agent stream error", exc_info=True)
                    await websocket.send_text(
                        _response_event(
                            response_id,
                            "I'm sorry, I had a technical issue. Could you please repeat that?",
                            complete=True,
                        )
                    )
                    continue

                # Close out this turn.
                await websocket.send_text(
                    _response_event(response_id, "", complete=True)
                )

    except WebSocketDisconnect:
        logger.info("Retell call %s disconnected", call_id)
    except Exception:
        logger.error("Retell websocket error on call %s", call_id, exc_info=True)
    finally:
        clear_session(call_id)
