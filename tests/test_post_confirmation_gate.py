"""Post-confirmation edits must use update_confirmed_booking + pending gate."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from app.call_memory import (
    clear_call_memory,
    get_reservation_draft,
    set_active_booking,
    update_reservation_draft,
)
from app.pending_confirmation import (
    ACTION_CREATE_BOOKING,
    ACTION_UPDATE_CONFIRMED_BOOKING,
    begin_caller_turn,
    booking_confirmation_payload,
    register_pending_confirmation,
    update_booking_confirmation_payload,
)
from app.reservation_draft import DRAFT_STATUS_CONFIRMED
from app.services.restaurant import RestaurantServiceError, restaurant_service


def test_update_reservation_draft_refuses_confirmed_booking() -> None:
    clear_call_memory("draft-refuse")
    set_active_booking(
        "draft-refuse",
        booking_id=6,
        customer_name="Hamza",
        customer_phone="+14155550100",
        party_size=4,
        date="2026-09-12",
        time="19:00",
        table_number=10,
        table_location="patio",
        notes="",
    )
    draft = get_reservation_draft("draft-refuse")
    assert int(draft.get("booking_id") or 0) == 6
    assert draft.get("status") == DRAFT_STATUS_CONFIRMED
    with pytest.raises(ValueError) as exc:
        update_reservation_draft("draft-refuse", time="19:00")
    assert "update_confirmed_booking" in str(exc.value)
    clear_call_memory("draft-refuse")


@pytest.mark.asyncio
async def test_update_confirmed_true_without_pending_is_409() -> None:
    clear_call_memory("upd-gate-409")
    begin_caller_turn("upd-gate-409", "yes")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.update_confirmed_booking(
            call_id="upd-gate-409",
            idempotency_key="upd-gate-missing",
            booking_id=99,
            confirmed=True,
            time="19:00",
        )
    assert exc.value.status == 409
    clear_call_memory("upd-gate-409")


@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)
@pytest.mark.asyncio
async def test_post_confirm_change_requires_readback_then_yes(monkeypatch) -> None:
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
    finally:
        await connection.close()

    date = (datetime.now() + timedelta(days=21)).date().isoformat()
    call_id = "post-confirm-gate"
    clear_call_memory(call_id)
    book_payload = booking_confirmation_payload(
        customer_name="Hamza",
        customer_phone="+14155550111",
        date=date,
        time="19:30",
        party_size=4,
    )
    begin_caller_turn(call_id, "readback")
    register_pending_confirmation(call_id, ACTION_CREATE_BOOKING, book_payload)
    begin_caller_turn(call_id, "yes")
    created = await restaurant_service.create_booking(
        call_id=call_id,
        idempotency_key="pcg-create",
        customer_name="Hamza",
        customer_phone="+14155550111",
        date=date,
        time="19:30",
        party_size=4,
        confirmed=True,
    )
    booking_id = int(created["booking_id"])

    # Bare confirmed=True with no pending must fail.
    begin_caller_turn(call_id, "yes")
    with pytest.raises(RestaurantServiceError) as bare:
        await restaurant_service.update_confirmed_booking(
            call_id=call_id,
            idempotency_key="pcg-bare",
            booking_id=booking_id,
            confirmed=True,
            time="19:00",
        )
    assert bare.value.status == 409

    pending = await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="pcg-propose",
        booking_id=booking_id,
        confirmed=False,
        time="19:00",
    )
    assert pending.get("pending") is True
    assert pending.get("updated") is False

    payload = update_booking_confirmation_payload(booking_id=booking_id, time="19:00")
    begin_caller_turn(call_id, "yes")
    applied = await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="pcg-apply",
        booking_id=booking_id,
        confirmed=True,
        time="19:00",
    )
    assert applied.get("updated") is True
    assert applied.get("time") == "19:00"
    clear_call_memory(call_id)
