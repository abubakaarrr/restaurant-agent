"""Explicit order fulfillment_type defaults and set_order_fulfillment."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import uuid

import pytest

from app.call_memory import clear_call_memory
from app.pending_confirmation import (
    ACTION_CREATE_BOOKING,
    begin_caller_turn,
    booking_confirmation_payload,
    register_pending_confirmation,
)


pytestmark_db = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1" or not os.getenv("TEST_DATABASE_URL"),
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


def _future() -> tuple[str, str]:
    when = datetime.now() + timedelta(days=25)
    return when.date().isoformat(), "19:00"


@pytestmark_db
@pytest.mark.asyncio
async def test_add_order_item_defaults_dine_in_when_booking_in_session(
    monkeypatch,
) -> None:
    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import restaurant_service

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)

    suffix = uuid.uuid4().hex[:10]
    call_id = f"ff-dinein-{suffix}"
    clear_call_memory(call_id)
    date, time = _future()

    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (921, 4, 'main')
            ON CONFLICT (table_number) DO NOTHING
            """
        )
        await connection.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ('Fulfill Salad', 'starter', 9.00, 'Test', ARRAY['vegan'], TRUE)
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        await connection.close()

    payload = booking_confirmation_payload(
        customer_name="Alex",
        customer_phone="+14155550921",
        date=date,
        time=time,
        party_size=2,
        notes="",
    )
    begin_caller_turn(call_id, "read it back")
    register_pending_confirmation(call_id, ACTION_CREATE_BOOKING, payload)
    begin_caller_turn(call_id, "yes")
    booked = await restaurant_service.create_booking(
        call_id=call_id,
        idempotency_key=f"{call_id}-book",
        customer_name="Alex",
        customer_phone="+14155550921",
        date=date,
        time=time,
        party_size=2,
        notes="",
        confirmed=True,
    )
    assert booked["created"] is True

    # No booking_id passed — service should pull active booking from session state.
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key=f"{call_id}-add",
        item_name="Fulfill Salad",
        quantity=1,
        customer_name="Alex",
        customer_phone="+14155550921",
    )
    assert added["added"] is True
    assert added["fulfillment"] == "dine_in"
    assert added["fulfillment_type"] == "dine_in"
    assert added["booking_id"] == booked["booking_id"]

    connection = await asyncpg.connect(database_url)
    try:
        row = await connection.fetchrow(
            "SELECT fulfillment_type, booking_id FROM orders WHERE session_id = $1",
            call_id,
        )
        assert row["fulfillment_type"] == "dine_in"
        assert row["booking_id"] == booked["booking_id"]
    finally:
        await connection.close()
        await close_pool()


@pytestmark_db
@pytest.mark.asyncio
async def test_set_order_fulfillment_switches_without_duplicating_items(
    monkeypatch,
) -> None:
    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import restaurant_service

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)

    suffix = uuid.uuid4().hex[:10]
    call_id = f"ff-switch-{suffix}"
    clear_call_memory(call_id)
    date, time = _future()

    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (922, 4, 'main')
            ON CONFLICT (table_number) DO NOTHING
            """
        )
        await connection.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ('Fulfill Soup', 'starter', 8.00, 'Test', ARRAY['vegan'], TRUE)
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        await connection.close()

    # Start as pickup with five lines.
    for index in range(5):
        added = await restaurant_service.add_order_item(
            call_id=call_id,
            idempotency_key=f"{call_id}-add-{index}",
            item_name="Fulfill Soup",
            quantity=1,
            customer_name="Blair",
            customer_phone="+14155550922",
        )
        assert added["added"] is True
    assert added["fulfillment"] == "pickup"
    assert len(added["items"]) == 5
    order_id = added["order_id"]

    # Book on a sibling call so create_booking does not auto-attach this order.
    book_call = f"{call_id}-book"
    clear_call_memory(book_call)
    payload = booking_confirmation_payload(
        customer_name="Blair",
        customer_phone="+14155550922",
        date=date,
        time=time,
        party_size=2,
        notes="",
    )
    begin_caller_turn(book_call, "read it back")
    register_pending_confirmation(book_call, ACTION_CREATE_BOOKING, payload)
    begin_caller_turn(book_call, "yes")
    booked = await restaurant_service.create_booking(
        call_id=book_call,
        idempotency_key=f"{call_id}-book",
        customer_name="Blair",
        customer_phone="+14155550922",
        date=date,
        time=time,
        party_size=2,
        notes="",
        confirmed=True,
    )

    switched = await restaurant_service.set_order_fulfillment(
        call_id=call_id,
        idempotency_key=f"{call_id}-fulfill",
        fulfillment_type="dine_in",
        booking_id=booked["booking_id"],
    )
    assert switched["updated"] is True
    assert switched["fulfillment"] == "dine_in"
    assert switched["booking_id"] == booked["booking_id"]
    assert switched["order_id"] == order_id
    assert len(switched["items"]) == 5

    connection = await asyncpg.connect(database_url)
    try:
        order_count = await connection.fetchval(
            "SELECT COUNT(*) FROM orders WHERE session_id = $1",
            call_id,
        )
        item_count = await connection.fetchval(
            "SELECT COUNT(*) FROM order_items WHERE order_id = $1",
            order_id,
        )
        assert order_count == 1
        assert item_count == 5
    finally:
        await connection.close()
        await close_pool()
