"""Main agent node — calls GPT-4o with all tools bound."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, trim_messages
from langchain_core.runnables import RunnableConfig

from app.agent.configuration import AgentConfiguration
from app.agent.state import RestaurantAgentState
from app.config import settings
from app.tools import ALL_TOOLS

SYSTEM_PROMPT_TEMPLATE = (Path(__file__).parent.parent.parent / "prompts" / "system.md").read_text(encoding="utf-8")

model = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=settings.openai_api_key,
    max_tokens=120,
    temperature=0.3,
    streaming=True,
)
model_with_tools = model.bind_tools(ALL_TOOLS)


async def generate_response(
    state: RestaurantAgentState,
    config: RunnableConfig,
) -> dict:
    """Core agent node: builds system prompt, trims history, calls Claude."""
    agent_config = AgentConfiguration.from_runnable_config(config)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        restaurant_name=agent_config.restaurant_name,
        today_datetime=datetime.datetime.now().strftime("%A, %B %d %Y at %I:%M %p"),
        timezone=settings.restaurant_timezone,
        session_id=state.get("session_id", "unknown"),
    )

    trimmed = trim_messages(
        state["messages"],
        max_tokens=8000,
        strategy="last",
        token_counter=len,  # approximate — good enough for POC
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
