"""Tone / duplicate-reply backstop tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.runner import clear_session, run_agent
from app.reply_guard import is_repeated_reply


def test_byte_identical_reply_flagged_when_users_differ() -> None:
    previous = "Got it, 7:00 PM it is."
    assert is_repeated_reply(
        "yes",
        previous,
        previous,
        previous_user_message="Make that 7:00 PM instead of 7:30.",
    )


def test_thanks_ack_still_allowed_short_reply() -> None:
    assert not is_repeated_reply(
        "thanks",
        "We serve vegetarian pizza.",
        "You got it.",
        previous_user_message="do you have vegetarian options?",
    )


@pytest.mark.asyncio
async def test_different_user_inputs_never_return_identical_consecutive_replies() -> None:
    session_id = "tone-no-dup"
    clear_session(session_id)
    canned = "Got it, 7:00 PM it is."

    async def fake_ainvoke(payload, config=None):
        history = list(payload["messages"])
        converted = []
        for message in history:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content") or ""
                if role == "user":
                    converted.append(HumanMessage(content=content))
                else:
                    converted.append(AIMessage(content=content))
            else:
                converted.append(message)
        directive = str(payload.get("behavior_directive") or "")
        if "BUG SIGNAL" in directive or "identical" in directive.casefold():
            converted.append(AIMessage(content="What would you like me to do next?"))
        else:
            converted.append(AIMessage(content=canned))
        return {"messages": converted}

    with patch(
        "app.agent.runner.restaurant_agent.ainvoke",
        new=AsyncMock(side_effect=fake_ainvoke),
    ):
        first = await run_agent(
            session_id, "Wait, make that 7:00 PM instead of 7:30."
        )
        second = await run_agent(session_id, "yes")

    assert first == canned
    assert second != first
    assert second != canned
    clear_session(session_id)
