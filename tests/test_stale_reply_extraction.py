"""Stale/echoed assistant replies must not come from prior-turn extraction."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.runner import _extract_reply, clear_session, run_agent
from app.reply_guard import is_repeated_reply


def test_extract_reply_ignores_assistant_text_before_latest_human() -> None:
    """Reproduce the transcript bug: empty this-turn AI → old reply echoed."""
    messages = [
        HumanMessage(content="Make that 7:00 PM instead of 7:30."),
        AIMessage(content="Got it, 7:00 PM it is."),
        HumanMessage(content="yes"),
        # Tool-only step with no speakable content — old extractor walked past this
        # and returned the previous turn's "Got it, 7:00 PM it is."
        AIMessage(content="", tool_calls=[{"name": "get_reservation_draft", "args": {}, "id": "1"}]),
        ToolMessage(content="draft facts", tool_call_id="1"),
        AIMessage(content=""),
    ]
    assert _extract_reply(messages) == ""


def test_extract_reply_uses_final_text_after_tools() -> None:
    messages = [
        AIMessage(content="I don't have the cancellation policy confirmed."),
        HumanMessage(content="Can you read the reservation details back to me?"),
        AIMessage(content="", tool_calls=[{"name": "get_reservation_draft", "args": {}, "id": "1"}]),
        ToolMessage(content="Hamza, party 5, 19:00, patio", tool_call_id="1"),
        AIMessage(content="You're down for five at seven on the patio under Hamza."),
    ]
    assert "five" in _extract_reply(messages).casefold()
    assert "cancellation" not in _extract_reply(messages).casefold()


@pytest.mark.asyncio
async def test_three_identical_asks_do_not_echo_unrelated_prior_reply() -> None:
    """Three identical readback asks must not return a prior unrelated answer."""
    session_id = "stale-echo-readback"
    clear_session(session_id)

    prior_unrelated = (
        "I don't have the cancellation policy confirmed with the restaurant, "
        "so I can't quote it. I can take a message or note your question."
    )
    readback = (
        "You're reserved for five this Friday at 7:00 PM on the patio under Hamza. "
        "Reference 6."
    )

    # Seed history as if a prior turn already answered something else.
    from app.agent import runner as runner_mod

    runner_mod._sessions[session_id] = [
        {"role": "user", "content": "What's your cancellation policy?"},
        {"role": "assistant", "content": prior_unrelated},
    ]

    call_count = {"n": 0}

    async def fake_ainvoke(payload, config=None):
        call_count["n"] += 1
        history = list(payload["messages"])
        # Simulate LangGraph returning full history + a tool-only AI for early calls,
        # then a real answer — the bug was returning prior_unrelated on early calls.
        user = ""
        for message in reversed(history):
            if isinstance(message, dict) and message.get("role") == "user":
                user = str(message.get("content") or "")
                break
            if getattr(message, "type", None) == "human":
                user = str(getattr(message, "content", "") or "")
                break
        assert "read the reservation" in user.casefold()
        # Convert history dicts into message-like objects for the extractor path.
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
        if call_count["n"] == 1:
            # Tool-only turn with empty final content — must NOT echo prior_unrelated.
            converted.extend(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "get_reservation_draft",
                                "args": {},
                                "id": "t1",
                            }
                        ],
                    ),
                    ToolMessage(content="facts", tool_call_id="t1"),
                    AIMessage(content=""),
                ]
            )
        else:
            converted.append(AIMessage(content=readback))
        return {"messages": converted}

    with patch("app.agent.runner.restaurant_agent.ainvoke", new=AsyncMock(side_effect=fake_ainvoke)):
        first = await run_agent(session_id, "Can you read the reservation details back to me?")
        second = await run_agent(session_id, "Can you read the reservation details back to me?")
        third = await run_agent(session_id, "Can you read the reservation details back to me?")

    assert first != prior_unrelated
    assert "cancellation" not in first.casefold()
    # First may be the soft fallback when the model returned no text this turn.
    assert first in {
        "I'm sorry, could you repeat that?",
        readback,
    }
    assert second == readback
    assert third == readback
    assert first != prior_unrelated and second != prior_unrelated
    clear_session(session_id)


def test_yes_after_time_change_is_not_treated_as_harmless_ack_for_repeat_gate() -> None:
    previous = "Got it, 7:00 PM it is. You're all set for Friday."
    # "yes" is not in the ack list — repeated prior reply should still be flagged.
    assert is_repeated_reply(
        "yes",
        previous,
        previous,
        previous_user_message="Make that 7:00 PM instead of 7:30.",
    )
