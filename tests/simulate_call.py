"""
Demo simulation — run this to test the full agent without a phone call.

Usage:
    python tests/simulate_call.py

Runs 3 demo scenarios:
  1. Full booking flow (name, party, date, time → confirmed)
  2. Menu query with dietary filter
  3. General info (hours + parking)

Also contains the Old Fashioned pre-order cascade regression (DB-state asserts),
collectible via: pytest tests/simulate_call.py -k old_fashioned
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from app.agent.graph import restaurant_agent
from app.config import settings
from app.restaurant_settings import load_restaurant_settings


DIVIDER = "-" * 60


async def run_scenario(title: str, turns: list[str]) -> list[str]:
    print(f"\n{DIVIDER}")
    print(f"  SCENARIO: {title}")
    print(DIVIDER)

    messages = []
    replies: list[str] = []
    session_id = f"demo-{title[:10].lower().replace(' ', '-')}"

    for user_msg in turns:
        messages.append(HumanMessage(content=user_msg))
        print(f"\nCaller : {user_msg}")

        result = await restaurant_agent.ainvoke(
            {
                "messages": messages,
                "session_id": session_id,
                "caller_phone": "+1555000001",
                "turn_count": len(messages) // 2,
                "tool_iterations": 0,
                "behavior_directive": "Use standard concise phone style.",
            },
            config={"configurable": {}},
        )

        result_msgs = result.get("messages", [])
        reply = ""
        for msg in reversed(result_msgs):
            if hasattr(msg, "type") and msg.type == "ai":
                reply = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        assert reply.strip(), f"{title}: no response for caller turn {user_msg!r}"
        assert "traceback" not in reply.casefold()

        print(f"{settings.ai_agent_name}: {reply}")
        replies.append(reply)
        messages = list(result.get("messages", messages))
    assert len(replies) == len(turns), f"{title}: scenario stopped early"
    return replies


async def main() -> None:
    runtime = load_restaurant_settings()
    print(f"\nRestaurant AI Receptionist - Demo Simulation")
    print(f"Restaurant: {runtime['restaurant_name']}")
    print(f"Model: {settings.llm_model} via legacy LangGraph rollback adapter")
    booking_date = date.today() + timedelta(days=30)

    await run_scenario(
        "Full Booking Flow",
        [
            "Hi, I'd like to book a table for Saturday night",
            "There will be 4 of us",
            f"The date is {booking_date.isoformat()}",
            "Around 7 PM please",
            "My name is Ahmed",
            "My callback number is +1 555 000 0001",
            "Yes, please confirm the booking",
        ],
    )

    await run_scenario(
        "Menu Query — Vegetarian Options",
        [
            "Hi, do you have any vegetarian options?",
            "What about something without gluten too?",
            "How much is the mushroom risotto?",
        ],
    )

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


async def run_old_fashioned_preorder_regression(
    *,
    call_id: str = "sim-old-fashioned-cascade",
) -> dict:
    """Replay the failure transcript's order section; assert final DB state.

    Scenario: reservation for five → five Old Fashioneds as a dine-in pre-order →
    attempted confirm without a real affirmative → correction without re-adding.
    Passes once Tasks 1–2 gates are in; fails on the old cascade path.
    """
    from app.call_memory import clear_call_memory
    from app.db_pool import close_pool, get_pool
    from app.pending_confirmation import (
        ACTION_CREATE_BOOKING,
        begin_caller_turn,
        booking_confirmation_payload,
        register_pending_confirmation,
    )
    from app.services.restaurant import RestaurantServiceError, restaurant_service

    clear_call_memory(call_id)
    await close_pool()
    booking_date = (date.today() + timedelta(days=30)).isoformat()
    booking_time = "19:00"

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (951, 6, 'main')
            ON CONFLICT (table_number) DO UPDATE
            SET capacity = EXCLUDED.capacity, location = EXCLUDED.location
            """
        )
        await conn.execute(
            """
            INSERT INTO menu_items
                (name, category, price, description, dietary, available)
            VALUES
                ('Old Fashioned', 'drink', 15.00, 'Confirmed real item', '{}', TRUE)
            ON CONFLICT DO NOTHING
            """
        )
        await conn.execute(
            "DELETE FROM order_items WHERE order_id IN (SELECT id FROM orders WHERE session_id = $1)",
            call_id,
        )
        await conn.execute("DELETE FROM orders WHERE session_id = $1", call_id)
        await conn.execute(
            "DELETE FROM bookings WHERE customer_name = 'Hamza Cascade'"
        )

    book_payload = booking_confirmation_payload(
        customer_name="Hamza Cascade",
        customer_phone="+14155550951",
        date=booking_date,
        time=booking_time,
        party_size=5,
        notes="",
    )
    begin_caller_turn(call_id, "read the reservation back")
    register_pending_confirmation(call_id, ACTION_CREATE_BOOKING, book_payload)
    begin_caller_turn(call_id, "yes")
    booked = await restaurant_service.create_booking(
        call_id=call_id,
        idempotency_key=f"{call_id}-book",
        customer_name="Hamza Cascade",
        customer_phone="+14155550951",
        date=booking_date,
        time=booking_time,
        party_size=5,
        notes="",
        confirmed=True,
    )
    assert booked["created"] is True
    booking_id = int(booked["booking_id"])

    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key=f"{call_id}-old-fashioned",
        item_name="Old Fashioned",
        quantity=5,
        customer_name="Hamza Cascade",
        customer_phone="+14155550951",
    )
    assert added["added"] is True
    assert added["fulfillment"] == "dine_in"
    assert added["booking_id"] == booking_id

    switched = await restaurant_service.set_order_fulfillment(
        call_id=call_id,
        idempotency_key=f"{call_id}-fulfill-dinein",
        fulfillment_type="dine_in",
        booking_id=booking_id,
    )
    assert switched["order_id"] == added["order_id"]
    assert len(switched["items"]) == 1
    assert switched["items"][0]["quantity"] == 5

    begin_caller_turn(call_id, "that's everything")
    summary = await restaurant_service.get_order_summary(call_id=call_id)
    begin_caller_turn(call_id, "what else is on the menu?")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.confirm_order(
            call_id=call_id,
            idempotency_key=f"{call_id}-confirm-early",
            expected_draft_version=summary["draft_version"],
            approved=True,
        )
    assert exc.value.status == 409

    pool = await get_pool()
    async with pool.acquire() as conn:
        orders = await conn.fetch(
            "SELECT id, status, fulfillment_type, booking_id FROM orders WHERE session_id = $1",
            call_id,
        )
        assert len(orders) == 1
        assert orders[0]["status"] == "pending"
        assert orders[0]["fulfillment_type"] == "dine_in"
        assert orders[0]["booking_id"] == booking_id
        items = await conn.fetch(
            """
            SELECT item_name, quantity FROM order_items
            WHERE order_id = $1 AND COALESCE(proposed, FALSE) IS FALSE
            ORDER BY id
            """,
            orders[0]["id"],
        )
        old_fashioned = [
            row for row in items if "old fashioned" in row["item_name"].casefold()
        ]
        total_qty = sum(int(row["quantity"]) for row in old_fashioned)
        assert total_qty == 5, (
            f"expected 5 Old Fashioneds, got {total_qty} across {old_fashioned}"
        )
        assert len(old_fashioned) <= 5
        assert total_qty not in {10, 15}

    summary2 = await restaurant_service.get_order_summary(call_id=call_id)
    begin_caller_turn(call_id, "yes")
    confirmed = await restaurant_service.confirm_order(
        call_id=call_id,
        idempotency_key=f"{call_id}-confirm-ok",
        expected_draft_version=summary2["draft_version"],
        approved=True,
    )
    assert confirmed["confirmed"] is True
    assert confirmed["fulfillment"] == "dine_in"

    async with pool.acquire() as conn:
        final = await conn.fetchrow(
            "SELECT status, fulfillment_type, booking_id FROM orders WHERE session_id = $1",
            call_id,
        )
        assert final["status"] == "confirmed"
        assert final["fulfillment_type"] == "dine_in"
        assert final["booking_id"] == booking_id

    clear_call_memory(call_id)
    await close_pool()
    return {
        "booking_id": booking_id,
        "order_id": added["order_id"],
        "old_fashioned_qty": 5,
    }


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1" or not os.getenv("TEST_DATABASE_URL"),
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


@pytest.mark.asyncio
async def test_old_fashioned_preorder_no_cascade(monkeypatch) -> None:
    from app.db_pool import close_pool

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    result = await run_old_fashioned_preorder_regression(
        call_id=f"sim-old-fashioned-{os.getpid()}"
    )
    assert result["old_fashioned_qty"] == 5


if __name__ == "__main__":
    asyncio.run(main())
