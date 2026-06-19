"""Agent runners — blocking invoke + token streaming for voice."""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage

from app.agent.graph import restaurant_agent
from app.config import settings

_sessions: dict[str, list[dict]] = {}


def get_session_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


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


def _extract_reply(messages: list) -> str:
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "ai":
            reply = _text_from_message_content(msg.content).strip()
            if reply:
                return reply
    return ""


def _save_turn(session_id: str, history: list[dict], user_message: str, reply: str) -> None:
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history[-20:]


async def run_agent(session_id: str, user_message: str, caller_phone: str = "") -> str:
    """Run one agent turn and return the full text reply."""
    history = list(_sessions.get(session_id, []))
    history.append({"role": "user", "content": user_message})

    result = await restaurant_agent.ainvoke(
        {
            "messages": history,
            "session_id": session_id,
            "caller_phone": caller_phone,
            "turn_count": len(history) // 2,
        },
        config={"configurable": {"restaurant_name": settings.restaurant_name}},
    )

    reply = _extract_reply(result.get("messages", [])) or "I'm sorry, could you repeat that?"
    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history[-20:]
    return reply


async def stream_agent_tokens(
    session_id: str,
    user_message: str,
    caller_phone: str = "",
) -> AsyncIterator[str]:
    """Stream speakable tokens from the agent for Vapi.

    Strategy:
    - 1st LLM call may choose tools → buffer text, discard if tool_calls.
    - 2nd+ LLM call (after tools) → stream tokens live to the caller.
  """
    history = list(_sessions.get(session_id, []))
    history.append({"role": "user", "content": user_message})

    config = {"configurable": {"restaurant_name": settings.restaurant_name}}
    input_state = {
        "messages": history,
        "session_id": session_id,
        "caller_phone": caller_phone,
        "turn_count": len(history) // 2,
    }

    agent_llm_invocation = 0
    first_invocation_buffer: list[str] = []
    streamed_parts: list[str] = []
    final_messages: list | None = None

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
            continue

        if event_type == "on_chat_model_stream":
            text = _text_from_chunk(event.get("data", {}).get("chunk"))
            if not text:
                continue
            if agent_llm_invocation <= 1:
                first_invocation_buffer.append(text)
            else:
                streamed_parts.append(text)
                yield text
            continue

        if event_type == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            tool_calls = []
            if isinstance(output, AIMessage):
                tool_calls = output.tool_calls or []
            if agent_llm_invocation == 1 and not tool_calls:
                for piece in first_invocation_buffer:
                    streamed_parts.append(piece)
                    yield piece
            first_invocation_buffer = []

    reply = "".join(streamed_parts).strip()
    if not reply and final_messages:
        reply = _extract_reply(final_messages)
    if not reply:
        reply = "I'm sorry, could you repeat that?"

    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history[-20:]
