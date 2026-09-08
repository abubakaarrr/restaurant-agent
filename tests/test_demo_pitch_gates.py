from __future__ import annotations

from datetime import datetime
import os

import pytest

from app.local_calendar import upcoming_named_dates
from app.reply_guard import is_clerk_inventory, is_repeated_reply
from app.reservation_draft import normalize_preferred_location, speak_draft
from app.services.restaurant import format_availability_speech


def test_this_friday_is_a_calendar_fact() -> None:
    names = upcoming_named_dates(datetime(2026, 8, 20, 12, 0))
    assert names["today"] == "2026-08-20"
    assert names["this_friday"] == "2026-08-21"
    assert names["this_saturday"] == "2026-08-22"


def test_any_location_does_not_keep_patio() -> None:
    assert normalize_preferred_location("any") == ""
    assert normalize_preferred_location("anywhere") == ""
    assert normalize_preferred_location("patio") == "patio"
    assert normalize_preferred_location("outdoor") == "patio"
    assert normalize_preferred_location("indoor") == "main"
    assert normalize_preferred_location("a high-top by the bar") == "bar"


def test_impossible_availability_is_explicit() -> None:
    text = format_availability_speech(
        {
            "available": False,
            "preferred_location": "patio",
            "party_size": 5,
            "impossible_at_location": True,
            "max_seats_at_location": 4,
            "availability_nonce": "abc",
            "alternatives": [
                {
                    "kind": "location",
                    "location": "main",
                    "date": "2026-08-22",
                    "display_time": "7:00 PM",
                    "table_number": 6,
                    "capacity": 6,
                }
            ],
        }
    )
    assert "IMPOSSIBLE" in text
    assert "Largest patio table seats 4" in text
    assert "Do not offer other patio times" in text
    assert "main" in text


def test_repeated_reply_is_flagged_except_thanks() -> None:
    previous = "We have vegetarian options including the Eval Pizza and Margherita Pizza, both $18."
    repeated = "We have vegetarian options including the Eval Pizza and Margherita Pizza, both $18. Would you like details?"
    assert is_repeated_reply("do you serve water when we arrive", previous, repeated)
    assert not is_repeated_reply("thanks", previous, "You got it.")


def test_draft_readback_sounds_like_a_host() -> None:
    spoken = speak_draft(
        {
            "customer_name": "Hamza",
            "customer_phone": "123654789",
            "party_size": 5,
            "date": "2026-08-21",
            "time": "19:00",
            "seating_preference": "patio",
            "dietary": "vegetarian",
        }
    )
    assert spoken.startswith("You're down as Hamza")
    assert "five of you" in spoken
    assert "Friday, August 21" in spoken
    assert "7:00 PM" in spoken
    assert "callback 123 654 789" in spoken
    assert "I have Hamza" not in spoken
    assert "status" not in spoken
    assert is_clerk_inventory(
        "I have Hamza, 123 654 789, for five people this Friday, August 21 at 7:00 PM, with patio seating."
    )
    assert not is_clerk_inventory(spoken)


@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1" or not os.getenv("TEST_DATABASE_URL"),
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)
@pytest.mark.asyncio
async def test_five_on_patio_is_impossible(monkeypatch) -> None:
    """When patio's largest table is 4, a party of 5 must be impossible there."""
    from datetime import timedelta

    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import restaurant_service

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    restaurant_service._seating_limits = None
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            TRUNCATE voice_action_idempotency, order_items, orders, bookings,
                     call_sessions, tables
            RESTART IDENTITY CASCADE
            """
        )
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (931, 4, 'patio'), (932, 8, 'main')
            """
        )
    finally:
        await connection.close()
    when = datetime.now() + timedelta(days=21)
    date = when.date().isoformat()
    first = await restaurant_service.check_availability(
        date, "19:00", 5, preferred_location="patio"
    )
    assert first["available"] is False
    assert first["impossible_at_location"] is True
    assert first["max_seats_at_location"] == 4
    assert not any(row.get("kind") == "time" for row in first["alternatives"])
    await close_pool()
