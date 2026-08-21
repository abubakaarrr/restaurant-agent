from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from app.call_memory import (
    clear_call_memory,
    get_reservation_draft,
    set_current_session_id,
    update_reservation_draft,
)
from app.agent.runner import seed_opening_history
from app.pending_confirmation import (
    ACTION_CREATE_BOOKING,
    begin_caller_turn,
    booking_confirmation_payload,
    register_pending_confirmation,
)
from app.reservation_draft import compose_notes, patch_draft
from app.services.restaurant import RestaurantServiceError, restaurant_service
from app.tools.db import check_table_availability


def test_seed_opening_history_only_when_empty() -> None:
    seeded = seed_opening_history([])
    assert seeded[0]["role"] == "assistant"
    assert "How can I help you today?" in seeded[0]["content"]
    existing = [{"role": "user", "content": "hi"}]
    assert seed_opening_history(existing) == existing
    draft = patch_draft(
        {"party_size": 5, "date": "2026-09-01", "time": "19:00", "customer_name": "Sam"},
        {"time": "20:00"},
    )
    assert draft["party_size"] == 5
    assert draft["time"] == "20:00"
    assert draft["customer_name"] == "Sam"


def test_clear_dietary_does_not_reset_party() -> None:
    clear_call_memory("draft-clear")
    update_reservation_draft(
        "draft-clear",
        party_size=5,
        dietary="vegetarian",
        date="2026-09-01",
        time="19:00",
        customer_name="Sam",
        customer_phone="03098121804",
    )
    update_reservation_draft("draft-clear", dietary="")
    draft = get_reservation_draft("draft-clear")
    assert draft["party_size"] == 5
    assert draft["dietary"] == ""
    assert "vegetarian" not in compose_notes(draft)


async def test_availability_hypothetical_does_not_write_draft(monkeypatch) -> None:
    clear_call_memory("draft-hypo")
    token = set_current_session_id("draft-hypo")
    try:
        update_reservation_draft(
            "draft-hypo",
            party_size=4,
            date="2026-09-01",
            time="19:00",
        )

        async def fake_check(date, time, party_size, **kwargs):
            return {
                "available": True,
                "date": date,
                "time": time,
                "party_size": party_size,
                "tables": [{"table_number": 3, "capacity": 6, "location": "main"}],
                "alternatives": [],
            }

        monkeypatch.setattr(restaurant_service, "check_availability", fake_check)
        result = await check_table_availability.ainvoke(
            {
                "date": "2026-09-01",
                "time": "19:00",
                "party_size": 6,
                "session_id": "draft-hypo",
            }
        )
        assert "Available" in result
        draft = get_reservation_draft("draft-hypo")
        assert draft["party_size"] == 4
    finally:
        from app.call_memory import reset_current_session_id

        reset_current_session_id(token)
        clear_call_memory("draft-hypo")


async def test_cancel_without_confirm_registers_pending() -> None:
    clear_call_memory("cancel-no")
    result = await restaurant_service.cancel_booking(
        call_id="cancel-no",
        idempotency_key="cancel-key-1",
        booking_id=12,
        customer_name="Sam",
        confirmed=False,
    )
    assert result.get("pending") is True
    assert result.get("readback_required") is True
    assert result.get("cancelled") is False


async def test_update_booking_without_confirm_registers_pending() -> None:
    clear_call_memory("upd-no")
    result = await restaurant_service.update_confirmed_booking(
        call_id="upd-no",
        idempotency_key="update-key-1",
        booking_id=12,
        confirmed=False,
        time="20:00",
    )
    assert result.get("pending") is True
    assert result.get("readback_required") is True
    assert result.get("updated") is False


async def test_update_booking_confirmed_without_pending_is_rejected() -> None:
    from app.pending_confirmation import begin_caller_turn

    clear_call_memory("upd-no-pending")
    begin_caller_turn("upd-no-pending", "yes")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.update_confirmed_booking(
            call_id="upd-no-pending",
            idempotency_key="update-key-missing-pending",
            booking_id=12,
            confirmed=True,
            time="20:00",
        )
    assert exc.value.status == 409
    assert exc.value.code in {
        "pending_confirmation_missing",
        "affirmation_required",
        "confirmation_hash_mismatch",
    }


pytestmark_db = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


@pytestmark_db
@pytest.mark.asyncio
async def test_confirmed_booking_updates_time_notes_and_food(monkeypatch) -> None:
    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            TRUNCATE voice_action_idempotency, order_items, orders, bookings,
                     call_sessions, menu_items, tables
            RESTART IDENTITY CASCADE
            """
        )
        await connection.execute(
            "INSERT INTO tables (table_number, capacity, location) VALUES (1, 4, 'main'), (10, 4, 'patio')"
        )
        await connection.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ('Margherita Pizza', 'main', 18.00, 'Tomato and mozzarella', ARRAY['vegetarian'], TRUE)
            """
        )
    finally:
        await connection.close()

    date = (datetime.now() + timedelta(days=20)).date().isoformat()
    clear_call_memory("res-update")
    book_payload = booking_confirmation_payload(
        customer_name="Sam",
        customer_phone="+14155550123",
        date=date,
        time="19:00",
        party_size=2,
        notes="dietary: vegetarian",
    )
    begin_caller_turn("res-update", "read it back")
    register_pending_confirmation("res-update", ACTION_CREATE_BOOKING, book_payload)
    begin_caller_turn("res-update", "yes")
    created = await restaurant_service.create_booking(
        call_id="res-update",
        idempotency_key="create-then-update",
        customer_name="Sam",
        customer_phone="+14155550123",
        date=date,
        time="19:00",
        party_size=2,
        notes="dietary: vegetarian",
        confirmed=True,
    )
    booking_id = created["booking_id"]

    from app.pending_confirmation import (
        ACTION_UPDATE_CONFIRMED_BOOKING,
        update_booking_confirmation_payload,
    )

    update_payload = update_booking_confirmation_payload(
        booking_id=booking_id,
        time="20:00",
        dietary="",
        extra_notes="window table",
    )
    begin_caller_turn("res-update", "read the change back")
    register_pending_confirmation(
        "res-update", ACTION_UPDATE_CONFIRMED_BOOKING, update_payload
    )
    begin_caller_turn("res-update", "yes")
    updated = await restaurant_service.update_confirmed_booking(
        call_id="res-update",
        idempotency_key="change-time-notes",
        booking_id=booking_id,
        confirmed=True,
        time="20:00",
        dietary="",
        extra_notes="window table",
    )
    assert updated["updated"] is True
    assert updated["time"] == "20:00"
    assert updated["booking_id"] == booking_id
    assert "vegetarian" not in (updated["notes"] or "").casefold()
    assert "window table" in updated["notes"]

    added = await restaurant_service.add_order_item(
        call_id="res-update",
        idempotency_key="preorder-pizza",
        item_name="Margherita Pizza",
        quantity=1,
        booking_id=booking_id,
        customer_name="Sam",
        customer_phone="+14155550123",
    )
    assert added["added"] is True
    begin_caller_turn("res-update", "that's everything")
    summary = await restaurant_service.get_order_summary(call_id="res-update")
    begin_caller_turn("res-update", "yes")
    confirmed_order = await restaurant_service.confirm_order(
        call_id="res-update",
        idempotency_key="confirm-preorder",
        expected_draft_version=summary["draft_version"],
        approved=True,
    )
    assert confirmed_order["confirmed"] is True

    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.add_order_item(
            call_id="res-update",
            idempotency_key="need-confirm-food",
            item_name="Margherita Pizza",
            quantity=1,
            booking_id=booking_id,
            caller_confirmed=False,
        )
    assert exc.value.code == "confirmation_required"

    second = await restaurant_service.add_order_item(
        call_id="res-update",
        idempotency_key="add-after-confirm",
        item_name="Margherita Pizza",
        quantity=1,
        booking_id=booking_id,
        caller_confirmed=True,
    )
    assert second["added"] is True
    assert len(second["items"]) == 2
    assert second["status"] == "confirmed"
    connection = await asyncpg.connect(database_url)
    try:
        count = await connection.fetchval("SELECT COUNT(*) FROM bookings WHERE id = $1", booking_id)
        assert count == 1
    finally:
        await connection.close()
        await close_pool()
