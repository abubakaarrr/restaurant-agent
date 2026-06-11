"""
Demo simulation — run this to test the full agent without a phone call.

Usage:
    python tests/simulate_call.py

Runs 3 demo scenarios:
  1. Full booking flow (name, party, date, time → confirmed)
  2. Menu query with dietary filter
  3. General info (hours + parking)
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from app.agent.graph import restaurant_agent
from app.config import settings


DIVIDER = "-" * 60


async def run_scenario(title: str, turns: list[str]) -> None:
    print(f"\n{DIVIDER}")
    print(f"  SCENARIO: {title}")
    print(DIVIDER)

    messages = []
    session_id = f"demo-{title[:10].lower().replace(' ', '-')}"

    for user_msg in turns:
        messages.append(HumanMessage(content=user_msg))
        print(f"\nCaller : {user_msg}")

        try:
            result = await restaurant_agent.ainvoke(
                {
                    "messages": messages,
                    "session_id": session_id,
                    "caller_phone": "+1555000001",
                    "turn_count": len(messages) // 2,
                },
                config={"configurable": {"restaurant_name": settings.restaurant_name}},
            )
        except Exception as e:
            print(f"  [ERROR]: {e}")
            break

        # Pull the last AI message
        result_msgs = result.get("messages", [])
        reply = ""
        for msg in reversed(result_msgs):
            if hasattr(msg, "type") and msg.type == "ai":
                reply = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if not reply:
            reply = "[No response generated]"

        print(f"Sana   : {reply}")
        # Keep the full updated message list for the next turn
        messages = list(result.get("messages", messages))


async def main() -> None:
    print(f"\nRestaurant AI Receptionist - Demo Simulation")
    print(f"Restaurant: {settings.restaurant_name}")
    print(f"Model: Claude claude-sonnet-4-20250514 via LangGraph")

    # ── Scenario 1: Full booking flow ─────────────────────────
    await run_scenario(
        "Full Booking Flow",
        [
            "Hi, I'd like to book a table for Saturday night",
            "There will be 4 of us",
            "This Saturday — June 7th",
            "Around 7 PM please",
            "My name is Ahmed",
            "Yes, please confirm the booking",
        ],
    )

    # ── Scenario 2: Menu query ─────────────────────────────────
    await run_scenario(
        "Menu Query — Vegetarian Options",
        [
            "Hi, do you have any vegetarian options?",
            "What about something without gluten too?",
            "How much is the mushroom risotto?",
        ],
    )

    # ── Scenario 3: General info ───────────────────────────────
    await run_scenario(
        "General Info — Hours and Parking",
        [
            "What time do you open on Sundays?",
            "Is there parking nearby?",
            "Do you have wheelchair access?",
        ],
    )

    print(f"\n{DIVIDER}")
    print("  Demo complete.")
    print(DIVIDER)


if __name__ == "__main__":
    asyncio.run(main())
