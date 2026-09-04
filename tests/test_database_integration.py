from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
import os

import asyncpg
import pytest
import pytest_asyncio

from app.config import settings
from app.call_analytics import ingest_retell_webhook
from app.behavior import TurnObservation, reduce_behavior
from app.behavior_store import load_behavior_state, save_behavior_state
from app.call_memory import clear_call_memory, hydrate_call_memory
from app.db_pool import close_pool
from app.pending_confirmation import (
    ACTION_CONFIRM_ORDER,
    ACTION_CREATE_BOOKING,
    begin_caller_turn,
    booking_confirmation_payload,
    order_confirmation_payload,
    register_pending_confirmation,
)
from app.services.restaurant import RestaurantServiceError, restaurant_service
from db.seed import MENU_ITEMS, TABLES, seed


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        os.getenv("RUN_DB_INTEGRATION") != "1",
        reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
    ),
]


def _arm(session_id: str, action: str, payload: dict) -> None:
    begin_caller_turn(session_id, "read it back")
    register_pending_confirmation(session_id, action, payload)
    begin_caller_turn(session_id, "yes")


@pytest_asyncio.fixture(autouse=True)
async def isolated_database(monkeypatch: pytest.MonkeyPatch):
    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    connection = await asyncpg.connect(database_url)
    await connection.execute(
        """
        TRUNCATE voice_action_idempotency, call_events, provider_webhook_events,
                 order_items, orders, bookings, call_sessions, menu_items, tables,
                 knowledge_gaps, operator_knowledge
        RESTART IDENTITY CASCADE
        """
    )
    await connection.execute(
        """
        INSERT INTO tables (table_number, capacity, location)
        VALUES (1, 4, 'main')
        """
    )
    await connection.execute(
        """
        INSERT INTO menu_items (name, category, price, description, dietary, available)
        VALUES
            ('Margherita Pizza', 'main', 18.00, 'Tomato and mozzarella', ARRAY['vegetarian'], TRUE),
            ('Garden Salad', 'starter', 9.00, 'Seasonal vegetables', ARRAY['vegan'], TRUE)
        """
    )
    await connection.close()
    yield
    await close_pool()


def _future_date() -> str:
    return (datetime.now() + timedelta(days=30)).date().isoformat()


async def test_booking_race_and_idempotent_replay() -> None:
    date = _future_date()

    async def create(call_id: str, key: str):
        clear_call_memory(call_id)
        _arm(
            call_id,
            ACTION_CREATE_BOOKING,
            booking_confirmation_payload(
                customer_name="Taylor",
                customer_phone="+14155550123",
                date=date,
                time="19:00",
                party_size=4,
                notes="",
            ),
        )
        return await restaurant_service.create_booking(
            call_id=call_id,
            idempotency_key=key,
            customer_name="Taylor",
            customer_phone="+14155550123",
            date=date,
            time="19:00",
            party_size=4,
            confirmed=True,
        )

    results = await asyncio.gather(
        create("call-a", "booking-key-a"),
        create("call-b", "booking-key-b"),
        return_exceptions=True,
    )
    successes = [result for result in results if isinstance(result, dict)]
    conflicts = [
        result
        for result in results
        if isinstance(result, RestaurantServiceError)
    ]
    assert len(successes) == 1, repr(results)
    assert len(conflicts) == 1
    assert conflicts[0].code == "slot_unavailable"

    winner = successes[0]
    winner_index = 0 if results[0] is winner else 1
    winner_call = "call-a" if winner_index == 0 else "call-b"
    winner_key = "booking-key-a" if winner_index == 0 else "booking-key-b"
    replay = await create(
        winner_call,
        winner_key,
    )
    assert replay["booking_id"] == winner["booking_id"]
    assert replay["idempotent_replay"] is True
    clear_call_memory(winner_call)
    restored = await hydrate_call_memory(winner_call)
    assert restored["booking_id"] == winner["booking_id"]
    assert restored["customer_name"] == "Taylor"


async def test_order_corrections_confirmation_and_duplicate_delivery() -> None:
    ambiguous = await restaurant_service.add_order_item(
        call_id="order-call",
        idempotency_key="must-not-be-consumed",
        item_name="pizza",
        quantity=1,
        customer_name="Jordan",
        customer_phone="+14155550124",
    )
    assert ambiguous["added"] is False
    assert ambiguous["needs_confirmation"] is True
    assert ambiguous["candidates"][0]["name"] == "Margherita Pizza"
    connection = await asyncpg.connect(settings.database_url)
    try:
        assert await connection.fetchval("SELECT COUNT(*) FROM orders") == 0
    finally:
        await connection.close()

    first = await restaurant_service.add_order_item(
        call_id="order-call",
        idempotency_key="add-pizza-1",
        item_name="Margherita Pizza",
        quantity=1,
        customer_name="Jordan",
        customer_phone="+14155550124",
    )
    replay = await restaurant_service.add_order_item(
        call_id="order-call",
        idempotency_key="add-pizza-1",
        item_name="Margherita Pizza",
        quantity=1,
        customer_name="Jordan",
        customer_phone="+14155550124",
    )
    assert replay["order_item_id"] == first["order_item_id"]
    assert replay["idempotent_replay"] is True
    assert len(replay["items"]) == 1

    updated = await restaurant_service.update_order_item(
        call_id="order-call",
        idempotency_key="update-pizza-1",
        order_item_id=first["order_item_id"],
        quantity=2,
        notes="one without basil",
    )
    assert updated["items"][0]["quantity"] == 2
    version = updated["draft_version"]

    with pytest.raises(RestaurantServiceError, match="changed after the readback"):
        await restaurant_service.confirm_order(
            call_id="order-call",
            idempotency_key="confirm-wrong-version",
            expected_draft_version=version - 1,
            approved=True,
        )

    summary = await restaurant_service.get_order_summary(call_id="order-call")
    begin_caller_turn("order-call", "yes")
    confirmed = await restaurant_service.confirm_order(
        call_id="order-call",
        idempotency_key="confirm-correct-version",
        expected_draft_version=version,
        approved=True,
    )
    confirmed_replay = await restaurant_service.confirm_order(
        call_id="order-call",
        idempotency_key="confirm-correct-version",
        expected_draft_version=version,
        approved=True,
    )
    assert confirmed["confirmed"] is True
    assert confirmed["total"] == 36.0
    assert confirmed_replay["order_id"] == confirmed["order_id"]
    assert confirmed_replay["idempotent_replay"] is True
    assert summary["draft_version"] == version


async def test_webhook_deduplication_and_behavior_reconnect_state() -> None:
    payload = {
        "event": "call_ended",
        "call": {
            "call_id": "retell-call-1",
            "call_status": "ended",
            "duration_ms": 42000,
            "from_number": "+14155559999",
            "transcript": "must not be retained by default",
            "call_analysis": {
                "user_sentiment": "Positive",
                "call_successful": True,
            },
        },
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    assert await ingest_retell_webhook(payload, raw) is True
    assert await ingest_retell_webhook(payload, raw) is False

    connection = await asyncpg.connect(settings.database_url)
    try:
        stored = await connection.fetchrow(
            """
            SELECT payload
            FROM provider_webhook_events
            WHERE provider = 'retell' AND call_id = 'retell-call-1'
            """
        )
        serialized = str(stored["payload"])
        assert "+14155559999" not in serialized
        assert "must not be retained" not in serialized
    finally:
        await connection.close()

    reduction = reduce_behavior(
        None,
        TurnObservation(text="Please speak slower."),
    )
    await save_behavior_state("retell-call-1", reduction.state)
    restored = await load_behavior_state("retell-call-1")
    next_reduction = reduce_behavior(
        restored,
        TurnObservation(text="I need a table."),
    )
    assert next_reduction.state.explicit_pace.value == "slower"


async def test_live_menu_seed_is_idempotent_without_embeddings() -> None:
    connection = await asyncpg.connect(settings.database_url)
    try:
        await connection.execute(
            "TRUNCATE menu_items, tables RESTART IDENTITY CASCADE"
        )
        await seed(connection, with_embeddings=False)
        await seed(connection, with_embeddings=False)
        assert await connection.fetchval("SELECT COUNT(*) FROM tables") == len(TABLES)
        assert await connection.fetchval("SELECT COUNT(*) FROM menu_items") == len(
            MENU_ITEMS
        )
    finally:
        await connection.close()
