"""Table selection: preferred location is a filter; chosen tables are validated."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import uuid

import pytest

from app.call_memory import clear_call_memory, set_current_session_id
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
    when = datetime.now() + timedelta(days=28)
    return when.date().isoformat(), "19:00"


@pytestmark_db
@pytest.mark.asyncio
async def test_patio_too_small_returns_unavailable_with_alternatives(
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
    date, time = _future()

    connection = await asyncpg.connect(database_url)
    try:
        # Patio max 2; main seats 8 — party of 6 must not silently land on main.
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (931, 2, 'patio'), (932, 8, 'main')
            ON CONFLICT (table_number) DO UPDATE
            SET capacity = EXCLUDED.capacity, location = EXCLUDED.location
            """
        )
    finally:
        await connection.close()

    result = await restaurant_service.check_availability(
        date, time, 6, preferred_location="patio"
    )
    assert result["available"] is False
    assert result["tables"] == []
    assert result.get("impossible_at_location") is True
    assert result["alternatives"]
    assert any(
        str(row.get("location") or "").casefold() == "main"
        for row in result["alternatives"]
        if row.get("kind") == "location"
    )
    assert all(
        str(row.get("location") or "").casefold() != "patio"
        or row.get("kind") == "time"
        for row in result["alternatives"]
    )
    await close_pool()


@pytestmark_db
@pytest.mark.asyncio
async def test_create_booking_honors_offered_table_and_rejects_stale(
    monkeypatch,
) -> None:
    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import RestaurantServiceError, restaurant_service

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)

    suffix = uuid.uuid4().hex[:10]
    call_id = f"tbl-pick-{suffix}"
    clear_call_memory(call_id)
    date, time = _future()

    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (941, 4, 'main'), (942, 4, 'main')
            ON CONFLICT (table_number) DO UPDATE
            SET capacity = EXCLUDED.capacity, location = EXCLUDED.location
            """
        )
    finally:
        await connection.close()

    token = set_current_session_id(call_id)
    try:
        offer = await restaurant_service.check_availability(
            date, time, 2, preferred_location="main", call_id=call_id
        )
    finally:
        from app.call_memory import reset_current_session_id

        reset_current_session_id(token)

    assert offer["available"] is True
    assert offer["tables"]
    chosen = int(offer["tables"][0]["table_number"])

    payload = booking_confirmation_payload(
        customer_name="Casey",
        customer_phone="+14155550941",
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
        customer_name="Casey",
        customer_phone="+14155550941",
        date=date,
        time=time,
        party_size=2,
        notes="",
        confirmed=True,
        preferred_location="main",
        table_number=chosen,
    )
    assert booked["created"] is True
    assert booked["table_number"] == chosen

    # Stale / never-offered table is rejected.
    call_id2 = f"tbl-stale-{suffix}"
    clear_call_memory(call_id2)
    payload2 = booking_confirmation_payload(
        customer_name="Dana",
        customer_phone="+14155550942",
        date=date,
        time=time,
        party_size=2,
        notes="",
    )
    begin_caller_turn(call_id2, "read it back")
    register_pending_confirmation(call_id2, ACTION_CREATE_BOOKING, payload2)
    begin_caller_turn(call_id2, "yes")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.create_booking(
            call_id=call_id2,
            idempotency_key=f"{call_id2}-book",
            customer_name="Dana",
            customer_phone="+14155550942",
            date=date,
            time=time,
            party_size=2,
            notes="",
            confirmed=True,
            table_number=9999,
        )
    assert exc.value.status == 409
    assert exc.value.code in {
        "availability_offer_missing",
        "table_not_offered",
        "availability_offer_mismatch",
    }
    await close_pool()
