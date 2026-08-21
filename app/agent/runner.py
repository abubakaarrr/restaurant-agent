"""Agent runners — blocking invoke + token streaming for voice."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncIterator

from app.agent.graph import restaurant_agent
from app.call_memory import (
    clear_call_memory,
    hydrate_call_memory,
    reset_current_action_scope,
    reset_current_session_id,
    set_current_action_scope,
    set_current_session_id,
)
from app.pending_confirmation import begin_caller_turn
from app.turn_evidence import audit_assistant_speech, begin_turn, end_turn
from app.reply_guard import is_clerk_inventory, is_repeated_reply
from app.restaurant_settings import load_restaurant_settings
from app.config import settings
from app.call_flags import clear_call_control

logger = logging.getLogger(__name__)

_sessions: dict[str, list[dict]] = {}

# Re-export so callers can do: from app.agent.runner import consume_end_call
from app.call_flags import consume_end_call as consume_end_call  # noqa: E402


def get_session_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
    clear_call_memory(session_id)
    clear_call_control(session_id)


def _text_from_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _text_from_chunk(chunk: object) -> str:
    content = getattr(chunk, "content", None)
    return _text_from_message_content(content) if content else ""


def _chunk_has_tool_calls(chunk: object) -> bool:
    if not chunk:
        return False
    if getattr(chunk, "tool_call_chunks", None):
        return True
    if getattr(chunk, "tool_calls", None):
        return True
    additional = getattr(chunk, "additional_kwargs", None) or {}
    return bool(additional.get("tool_calls") or additional.get("function_call"))


def _message_role(msg: object) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role") or msg.get("type") or "")
    msg_type = getattr(msg, "type", None)
    if msg_type:
        return str(msg_type)
    return ""


def _extract_reply(messages: list) -> str:
    """Return speakable text from *this* turn only — never a prior assistant turn.

    LangGraph returns the full transcript. Walking every AIMessage can echo an
    older reply when the latest model step emitted tool_calls with empty content.
    """
    last_human_idx = -1
    for index, msg in enumerate(messages):
        role = _message_role(msg).casefold()
        if role in {"human", "user"}:
            last_human_idx = index
    candidates = messages[last_human_idx + 1 :] if last_human_idx >= 0 else messages
    for msg in reversed(candidates):
        role = _message_role(msg).casefold()
        if role != "ai" and role != "assistant":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        reply = _text_from_message_content(content).strip()
        if reply:
            return reply
    return ""


def _history_digest(history: list[dict]) -> list[dict[str, str]]:
    digest = []
    for message in history[-12:]:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "")
        digest.append(
            {
                "role": role,
                "content": content if len(content) <= 240 else content[:237] + "...",
            }
        )
    return digest


def _save_turn(session_id: str, history: list[dict], user_message: str, reply: str) -> None:
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    # Keep more turns so booking → pre-order context isn't dropped mid-call
    _sessions[session_id] = history[-40:]


def opening_greeting() -> str:
    runtime = load_restaurant_settings()
    restaurant = str(runtime.get("restaurant_name") or settings.restaurant_name)
    agent = str(runtime.get("ai_agent_name") or settings.ai_agent_name)
    return (
        f"Hi, you've reached {restaurant}. This is {agent}. "
        "How can I help you today?"
    )


def seed_opening_history(history: list[dict]) -> list[dict]:
    """The chat UI already shows the greeting; put it in history so the model does not replay it."""
    if history:
        return history
    return [{"role": "assistant", "content": opening_greeting()}]


def _action_scope(session_id: str, history_length: int, user_message: str) -> str:
    digest = hashlib.sha256(user_message.encode("utf-8")).hexdigest()[:16]
    return f"{session_id}:{history_length}:{digest}"


async def run_agent(session_id: str, user_message: str, caller_phone: str = "") -> str:
    """Run one agent turn and return the full text reply."""
    await hydrate_call_memory(session_id)
    # Server-owned affirmation fact — tools must not invent caller_confirmed.
    begin_caller_turn(session_id, user_message)
    history = seed_opening_history(list(_sessions.get(session_id, [])))
    history.append({"role": "user", "content": user_message})
    previous_reply = ""
    previous_user_message = ""
    for message in reversed(history[:-1]):
        role = message.get("role")
        if role == "assistant" and not previous_reply:
            previous_reply = str(message.get("content") or "")
        elif role == "user" and not previous_user_message:
            previous_user_message = str(message.get("content") or "")
        if previous_reply and previous_user_message:
            break

    token = set_current_session_id(session_id)
    scope = _action_scope(session_id, len(history), user_message)
    action_token = set_current_action_scope(scope)
    begin_turn(session_id, scope)
    logger.info(
        "run_agent start session=%s scope=%s user=%r history=%s",
        session_id,
        scope,
        user_message,
        _history_digest(history),
    )
    try:
        payload = {
            "messages": history,
            "session_id": session_id,
            "caller_phone": caller_phone,
            "turn_count": len(history) // 2,
            "tool_iterations": 0,
            "behavior_directive": "",
        }
        result = await restaurant_agent.ainvoke(
            payload,
            config={"configurable": {"restaurant_name": settings.restaurant_name}},
        )
        raw_messages = result.get("messages", [])
        reply = _extract_reply(raw_messages) or "I'm sorry, could you repeat that?"
        logger.info(
            "run_agent model_output session=%s scope=%s reply=%r "
            "result_msg_count=%s extracted_after_last_human=%s",
            session_id,
            scope,
            reply,
            len(raw_messages),
            bool(_extract_reply(raw_messages)),
        )
        retry_reason = ""
        if is_repeated_reply(
            user_message,
            previous_reply,
            reply,
            previous_user_message=previous_user_message,
        ):
            retry_reason = (
                "BUG SIGNAL: your reply was identical or nearly identical to your "
                "previous turn, but the caller said something different. "
                "Answer ONLY the latest user message in fresh words. Use tools if needed."
            )
        elif is_clerk_inventory(reply):
            retry_reason = (
                "That reply listed a record. Speak like a host: "
                "'You're down as Hamza, five this Friday at seven on the patio.' "
                "Do not start with 'I have [name]' and do not say a note 'is saved'."
            )
        if retry_reason:
            end_turn()
            begin_turn(session_id, scope + ":retry")
            payload["behavior_directive"] = retry_reason
            result = await restaurant_agent.ainvoke(
                payload,
                config={"configurable": {"restaurant_name": settings.restaurant_name}},
            )
            raw_messages = result.get("messages", [])
            retried = _extract_reply(raw_messages)
            logger.info(
                "run_agent retry session=%s scope=%s reason=%r reply=%r",
                session_id,
                scope,
                retry_reason[:80],
                retried,
            )
            reply = retried or "I'm sorry, could you repeat that?"
            if is_repeated_reply(
                user_message,
                previous_reply,
                reply,
                previous_user_message=previous_user_message,
            ):
                logger.error(
                    "run_agent still_duplicate_after_retry session=%s scope=%s "
                    "forcing_soft_fallback user=%r",
                    session_id,
                    scope,
                    user_message,
                )
                reply = (
                    "Sorry — I repeated myself there. "
                    "What did you need me to do just now?"
                )
        audit_assistant_speech(reply)
        history.append({"role": "assistant", "content": reply})
        _sessions[session_id] = history[-40:]
        return reply
    finally:
        end_turn()
        reset_current_action_scope(action_token)
        reset_current_session_id(token)


async def stream_agent_tokens(
    session_id: str,
    user_message: str,
    caller_phone: str = "",
    behavior_directive: str = "",
) -> AsyncIterator[str]:
    """Stream speakable tokens from the agent for Vapi / Retell.

    Strategy:
    - 1st LLM call may choose tools → buffer text, discard if tool_calls.
    - 2nd+ LLM call (after tools) → stream tokens live to the caller.

    Barge-in safety: the user message is written to _sessions *before* we
    start generation so that a CancelledError mid-stream never erases it from
    conversation history.  The assistant reply is appended only on success.
    """
    await hydrate_call_memory(session_id)
    # Server-owned affirmation fact — tools must not invent caller_confirmed.
    begin_caller_turn(session_id, user_message)
    history = seed_opening_history(list(_sessions.get(session_id, [])))
    history.append({"role": "user", "content": user_message})
    # Persist the user turn immediately — if this coroutine is cancelled
    # (Retell barge-in), the caller's utterance survives in history.
    _sessions[session_id] = history[-40:]

    config = {"configurable": {"restaurant_name": settings.restaurant_name}}
    input_state = {
        "messages": history,
        "session_id": session_id,
        "caller_phone": caller_phone,
        "turn_count": len(history) // 2,
        "tool_iterations": 0,
        "behavior_directive": behavior_directive,
    }

    agent_llm_invocation = 0
    first_invocation_is_tools = False
    streamed_parts: list[str] = []
    final_messages: list | None = None

    token = set_current_session_id(session_id)
    scope = _action_scope(session_id, len(history), user_message)
    action_token = set_current_action_scope(scope)
    begin_turn(session_id, scope)
    try:
        async for event in restaurant_agent.astream_events(
            input_state,
            config=config,
            version="v2",
        ):
            event_type = event.get("event")
            metadata = event.get("metadata", {})
            node = metadata.get("langgraph_node")

            if event_type == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and output.get("messages"):
                    final_messages = output["messages"]

            if node != "agent":
                continue

            if event_type == "on_chat_model_start":
                agent_llm_invocation += 1
                if agent_llm_invocation == 1:
                    first_invocation_is_tools = False
                continue

            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if agent_llm_invocation <= 1 and _chunk_has_tool_calls(chunk):
                    first_invocation_is_tools = True
                    continue
                if agent_llm_invocation <= 1 and first_invocation_is_tools:
                    continue
                text = _text_from_chunk(chunk)
                if not text:
                    continue
                streamed_parts.append(text)
                yield text
                continue
    finally:
        reset_current_action_scope(action_token)
        reset_current_session_id(token)

    reply = "".join(streamed_parts).strip()
    if not reply and final_messages:
        reply = _extract_reply(final_messages)
    if not reply:
        reply = "I'm sorry, could you repeat that?"

    logger.info(
        "stream_agent_tokens done session=%s reply=%r streamed=%s",
        session_id,
        reply,
        bool(streamed_parts),
    )
    audit_assistant_speech(reply)
    end_turn()

    # Append assistant reply to the history we already persisted above.
    current = list(_sessions.get(session_id, []))
    current.append({"role": "assistant", "content": reply})
    _sessions[session_id] = current[-40:]
