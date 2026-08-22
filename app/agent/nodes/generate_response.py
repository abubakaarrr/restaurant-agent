"""Main agent node — calls the configured OpenAI model with tools bound."""

from __future__ import annotations

import datetime
import json
import zoneinfo
from functools import lru_cache
from pathlib import Path
from typing import cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig

from app.agent.configuration import AgentConfiguration
from app.agent.state import RestaurantAgentState
from app.call_memory import format_memory_for_prompt
from app.config import settings
from app.local_calendar import upcoming_named_dates
from app.restaurant_settings import load_restaurant_settings
from app.services.restaurant import restaurant_service
from app.tools import ALL_TOOLS

SYSTEM_PROMPT_TEMPLATE = (Path(__file__).parent.parent.parent / "prompts" / "system.md").read_text(encoding="utf-8")


def _approx_message_tokens(messages) -> int:
    """Local length estimate so we never wait on the chat model to count tokens."""
    if isinstance(messages, str):
        return max(1, len(messages) // 4)
    if isinstance(messages, list):
        return max(1, sum(_approx_message_tokens(item) for item in messages))
    content = getattr(messages, "content", None)
    if content is None:
        return max(1, len(str(messages)) // 4)
    return _approx_message_tokens(content)


@lru_cache(maxsize=8)
def _configured_model(
    model_name: str,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
):
    kwargs: dict = {
        "model": model_name,
        "api_key": settings.openai_api_key,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "streaming": True,
        "stream_usage": False,
        "timeout": 20,
    }
    if model_name.lower().startswith("gpt-5") and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    model = ChatOpenAI(**kwargs)
    return model, model.bind_tools(ALL_TOOLS)


async def generate_response(
    state: RestaurantAgentState,
    config: RunnableConfig,
) -> dict:
    """Core agent node: builds system prompt, trims history, calls the LLM."""
    agent_config = AgentConfiguration.from_runnable_config(config)
    model, model_with_tools = _configured_model(
        agent_config.model,
        agent_config.max_response_tokens,
        agent_config.temperature,
        agent_config.reasoning_effort,
    )

    runtime = load_restaurant_settings()
    timezone_name = str(runtime.get("timezone") or settings.restaurant_timezone)
    restaurant_name = str(
        runtime.get("restaurant_name") or agent_config.restaurant_name
    )
    agent_name = str(runtime.get("ai_agent_name") or settings.ai_agent_name)
    try:
        tz = zoneinfo.ZoneInfo(timezone_name)
    except zoneinfo.ZoneInfoNotFoundError:
        tz = datetime.timezone.utc

    now_local = datetime.datetime.now(tz)
    session_id = state.get("session_id", "unknown")
    try:
        seating = await restaurant_service.seating_limits()
    except Exception:
        seating = {"max_party_phone": 12, "max_seats_by_location": {}, "largest_table": 0}
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        restaurant_name=restaurant_name,
        agent_name=agent_name,
        today_datetime=now_local.strftime("%A, %B %d %Y at %I:%M %p"),
        timezone=timezone_name,
        session_id=session_id,
        call_memory=format_memory_for_prompt(session_id),
        restaurant_context=json.dumps(
            {
                "phone_number": runtime.get("phone_number", ""),
                "address": " ".join(
                    value
                    for value in (
                        runtime.get("street_address", ""),
                        runtime.get("city", ""),
                    )
                    if value
                ),
                "opening_hours": runtime.get("opening_hours") or {},
                "hours_unconfirmed": bool(runtime.get("hours_unconfirmed")),
                "hours_note": runtime.get("hours_note") or "",
                "tagline": runtime.get("tagline") or "",
                "languages": runtime.get("languages", []),
                "upcoming_dates": upcoming_named_dates(now_local),
                "seating": seating,
            },
            ensure_ascii=False,
        ),
        behavior_directive=state.get("behavior_directive")
        or "Talk like a host. Do the latest request; do not announce a change without making it.",
    )
    if any(
        (getattr(message, "type", "") == "ai")
        or (isinstance(message, dict) and message.get("role") == "assistant")
        for message in state.get("messages") or []
    ):
        system_prompt += (
            "\n\nThis is not the first turn. Never repeat the opening greeting. "
            "Handle only the latest user message. Do not finish a previous unanswered offer. "
            "Never offer to connect the team for a name change or for water."
        )

    trimmed = trim_messages(
        state["messages"],
        max_tokens=3500,
        strategy="last",
        token_counter=_approx_message_tokens,
        include_system=False,
        allow_partial=True,
    )

    response = cast(
        AIMessage,
        await model_with_tools.ainvoke(
            [{"role": "system", "content": system_prompt}, *trimmed],
            config,
        ),
    )

    return {
        "messages": [response],
        "turn_count": state.get("turn_count", 0) + 1,
    }
