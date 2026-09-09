"""Pending-confirmation gate: readback hash + server-side affirmation."""

from __future__ import annotations

from datetime import datetime, timedelta
import os

import pytest

from app.call_memory import clear_call_memory, get_call_memory, hydrate_call_memory
from app.pending_confirmation import (
    ACTION_CANCEL_BOOKING,
    ACTION_CONFIRM_ORDER,
    ACTION_CREATE_BOOKING,
    begin_caller_turn,
    booking_confirmation_payload,
    classify_affirmation,
    clear_pending_confirmation,
    order_confirmation_payload,
    pending_state_patch,
    register_pending_confirmation,
    require_pending_confirmation,
)


def test_classify_affirmation_keywords() -> None:
    assert classify_affirmation("yes") == "affirmative"
    assert classify_affirmation("Yeah, that's right.") == "affirmative"
    assert classify_affirmation("sounds good") == "affirmative"
    assert classify_affirmation("go ahead") == "affirmative"
    assert classify_affirmation("no") == "negative"
    assert classify_affirmation("wait, change the time") == "negative"
    assert classify_affirmation("actually make it six") == "negative"
    assert classify_affirmation("move it to seven, not cancel it") == "negative"
    assert classify_affirmation("what time do you close?") == "unclear"
    assert classify_affirmation("yes, but change the name") == "negative"


def test_require_pending_rejects_without_record() -> None:
    from app.services.restaurant import RestaurantServiceError

    clear_call_memory("pc-missing")
    begin_caller_turn("pc-missing", "yes")
    with pytest.raises(RestaurantServiceError) as exc:
        require_pending_confirmation(
            "pc-missing",
            ACTION_CONFIRM_ORDER,
            order_confirmation_payload(
                {
                    "order_id": 1,
                    "draft_version": 1,
                    "booking_id": 0,
                    "fulfillment": "pickup",
                    "total": 10.0,
                    "items": [{"item_name": "X", "quantity": 1, "notes": ""}],
                }
            ),
        )
    assert exc.value.status == 409
    assert exc.value.code == "pending_confirmation_missing"


def test_require_pending_rejects_unrelated_utterance() -> None:
    from app.services.restaurant import RestaurantServiceError

    clear_call_memory("pc-unrelated")
    payload = order_confirmation_payload(
        {
            "order_id": 7,
            "draft_version": 2,
            "booking_id": 0,
            "fulfillment": "pickup",
            "total": 18.0,
            "items": [{"item_name": "Pizza", "quantity": 1, "notes": ""}],
        }
    )
    register_pending_confirmation("pc-unrelated", ACTION_CONFIRM_ORDER, payload)
    begin_caller_turn("pc-unrelated", "what time do you close?")
    with pytest.raises(RestaurantServiceError) as exc:
        require_pending_confirmation("pc-unrelated", ACTION_CONFIRM_ORDER, payload)
    assert exc.value.status == 409
    assert exc.value.code == "affirmation_required"


def test_require_pending_rejects_same_turn_read_and_confirm() -> None:
    from app.services.restaurant import RestaurantServiceError

    clear_call_memory("pc-same-turn")
    begin_caller_turn("pc-same-turn", "that's everything")
    payload = booking_confirmation_payload(
        customer_name="Sam",
        customer_phone="+14155550100",
        date="2026-09-01",
        time="19:00",
        party_size=2,
        notes="",
    )
    register_pending_confirmation("pc-same-turn", ACTION_CREATE_BOOKING, payload)
    # Affirmative on the *same* turn as registration must still fail.
    from app.call_memory import update_call_memory

    update_call_memory("pc-same-turn", last_turn_affirmation="affirmative")
    with pytest.raises(RestaurantServiceError) as exc:
        require_pending_confirmation("pc-same-turn", ACTION_CREATE_BOOKING, payload)
    assert exc.value.code == "same_turn_confirmation"


def test_require_pending_accepts_after_yes_on_later_turn() -> None:
    clear_call_memory("pc-ok")
    begin_caller_turn("pc-ok", "please read it back")
    payload = booking_confirmation_payload(
        customer_name="Sam",
        customer_phone="+14155550100",
        date="2026-09-01",
        time="19:00",
        party_size=2,
        notes="",
    )
    register_pending_confirmation("pc-ok", ACTION_CREATE_BOOKING, payload)
    begin_caller_turn("pc-ok", "yes")
    require_pending_confirmation("pc-ok", ACTION_CREATE_BOOKING, payload)
    assert get_call_memory("pc-ok")["last_turn_affirmation"] == "affirmative"


def test_cancellation_confirmation_is_not_durable() -> None:
    session_id = "pc-cancel-nondurable"
    clear_call_memory(session_id)
    register_pending_confirmation(
        session_id,
        ACTION_CANCEL_BOOKING,
        {"booking_id": 42},
    )
    assert ACTION_CANCEL_BOOKING not in pending_state_patch(session_id)[
        "pending_confirmations"
    ]


@pytest.mark.asyncio
async def test_hydration_discards_legacy_pending_cancellation(monkeypatch) -> None:
    session_id = "pc-cancel-legacy-hydration"
    clear_call_memory(session_id)

    class Connection:
        async def fetchrow(self, query: str, *args: object):
            if "FROM call_sessions" in query:
                return {
                    "caller_phone": "",
                    "state": {
                        "pending_confirmations": {
                            ACTION_CANCEL_BOOKING: {"payload": {"booking_id": 42}}
                        }
                    },
                }
            return None

    class Acquire:
        async def __aenter__(self):
            return Connection()

        async def __aexit__(self, *args: object) -> None:
            return None

    class Pool:
        def acquire(self):
            return Acquire()

    async def fake_pool():
        return Pool()

    monkeypatch.setattr("app.call_memory.get_pool", fake_pool)
    restored = await hydrate_call_memory(session_id)
    assert restored["pending_confirmations"] == {}


pytestmark_db = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1" or not os.getenv("TEST_DATABASE_URL"),
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


def _future() -> tuple[str, str]:
    when = datetime.now() + timedelta(days=21)
    return when.date().isoformat(), "19:00"


@pytestmark_db
@pytest.mark.asyncio
async def test_confirm_order_gate_unrelated_then_yes(monkeypatch) -> None:
    """get_summary → unrelated → 409; get_summary → yes → success."""
    import asyncpg
    import uuid

    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import RestaurantServiceError, restaurant_service

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)

    suffix = uuid.uuid4().hex[:12]
    call_id = f"pc-confirm-{suffix}"
    clear_call_memory(call_id)

    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ('Gate Pizza', 'main', 18.00, 'Test', ARRAY['vegetarian'], TRUE)
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        await connection.close()

    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key=f"pc-confirm-add-{suffix}",
        item_name="Gate Pizza",
        quantity=1,
        customer_name="Sam",
        customer_phone="+14155550911",
    )
    assert added["added"] is True

    begin_caller_turn(call_id, "that's everything")
    summary = await restaurant_service.get_order_summary(call_id=call_id)
    assert summary.get("readback_required") is True
    assert summary.get("pending_confirmation_hash")

    begin_caller_turn(call_id, "what time do you close?")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.confirm_order(
            call_id=call_id,
            idempotency_key=f"pc-confirm-bad-{suffix}",
            expected_draft_version=summary["draft_version"],
            approved=True,
        )
    assert exc.value.status == 409
    assert exc.value.code == "affirmation_required"

    # Re-register after another summary readback on this turn, then affirm later.
    summary2 = await restaurant_service.get_order_summary(call_id=call_id)
    begin_caller_turn(call_id, "yes")
    confirmed = await restaurant_service.confirm_order(
        call_id=call_id,
        idempotency_key=f"pc-confirm-ok-{suffix}",
        expected_draft_version=summary2["draft_version"],
        approved=True,
    )
    assert confirmed["confirmed"] is True
    assert confirmed["status"] == "confirmed"
    await close_pool()


@pytestmark_db
@pytest.mark.asyncio
async def test_create_booking_requires_pending_and_affirmation(monkeypatch) -> None:
    import asyncpg
    import uuid

    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import RestaurantServiceError, restaurant_service

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)

    suffix = uuid.uuid4().hex[:12]
    call_id = f"pc-book-{suffix}"
    clear_call_memory(call_id)
    date, time = _future()

    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (911, 4, 'main')
            ON CONFLICT (table_number) DO NOTHING
            """
        )
    finally:
        await connection.close()

    payload = booking_confirmation_payload(
        customer_name="Riley",
        customer_phone="+14155550912",
        date=date,
        time=time,
        party_size=2,
        notes="",
    )

    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.create_booking(
            call_id=call_id,
            idempotency_key=f"pc-book-no-pending-{suffix}",
            customer_name="Riley",
            customer_phone="+14155550912",
            date=date,
            time=time,
            party_size=2,
            notes="",
            confirmed=True,
        )
    assert exc.value.status == 409
    assert exc.value.code == "pending_confirmation_missing"

    begin_caller_turn(call_id, "read it back")
    register_pending_confirmation(call_id, ACTION_CREATE_BOOKING, payload)
    begin_caller_turn(call_id, "yes")
    booked = await restaurant_service.create_booking(
        call_id=call_id,
        idempotency_key=f"pc-book-ok-{suffix}",
        customer_name="Riley",
        customer_phone="+14155550912",
        date=date,
        time=time,
        party_size=2,
        notes="",
        confirmed=True,
    )
    assert booked["created"] is True
    await close_pool()
