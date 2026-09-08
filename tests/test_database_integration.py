from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
import json
import os
from zoneinfo import ZoneInfo

import asyncpg
import pytest
import pytest_asyncio

from app.config import settings
from app.agent.runner import clear_session, stream_agent_tokens
from app.caller_turn import process_caller_turn
from app.call_analytics import ingest_retell_webhook
from app.behavior import TurnObservation, reduce_behavior
from app.behavior_store import load_behavior_state, save_behavior_state
from app.call_memory import clear_call_memory, hydrate_call_memory
from app.db_pool import close_pool
from app.pending_confirmation import (
    ACTION_CANCEL_BOOKING,
    ACTION_CONFIRM_ORDER,
    ACTION_CREATE_BOOKING,
    begin_caller_turn,
    booking_confirmation_payload,
    order_confirmation_payload,
    get_pending_confirmation,
    register_pending_confirmation,
)
from app.services.restaurant import RestaurantServiceError, restaurant_service
from app.tools.rag import search_menu
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
    timezone_info = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 18, 0, tzinfo=timezone_info),
    )
    connection = await asyncpg.connect(database_url)
    await connection.execute(
        """
        TRUNCATE voice_action_idempotency, call_events, provider_webhook_events,
                 order_items, orders, bookings, call_sessions, menu_items, tables,
                 knowledge_gaps, operator_knowledge, restaurant_knowledge_records
        RESTART IDENTITY CASCADE
        """
    )
    await seed(connection, with_embeddings=False)
    await connection.execute("TRUNCATE tables RESTART IDENTITY CASCADE")
    await connection.execute(
        "INSERT INTO tables (table_number, capacity, location) VALUES (1, 4, 'main')"
    )
    await connection.close()
    yield
    await close_pool()


def _future_date() -> str:
    candidate = (datetime.now() + timedelta(days=30)).date()
    excluded = {"2026-10-18", "2026-11-26", "2026-12-24"}
    while candidate.weekday() == 0 or candidate.isoformat() in excluded:
        candidate += timedelta(days=1)
    return candidate.isoformat()


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


async def test_public_agent_cancellation_reversal_preserves_booking() -> None:
    call_id = "reservation-reversal"
    clear_session(call_id)
    booked_at = datetime.fromisoformat(f"{_future_date()}T19:00")
    connection = await asyncpg.connect(settings.database_url)
    try:
        table_id = await connection.fetchval(
            "SELECT id FROM tables WHERE table_number = 1"
        )
        booking_id = await connection.fetchval(
            """
            INSERT INTO bookings
                (customer_name, customer_phone, table_id, booked_at, party_size, status)
            VALUES ('Taylor', '+14155550123', $1, $2, 2, 'confirmed')
            RETURNING id
            """,
            table_id,
            booked_at,
        )
        await connection.execute(
            """
            INSERT INTO call_sessions (session_id, state)
            VALUES ($1, $2::jsonb)
            """,
            call_id,
            json.dumps({"booking_id": booking_id}),
        )
    finally:
        await connection.close()

    pending = await restaurant_service.cancel_booking(
        call_id=call_id,
        idempotency_key="pending-cancel-reversal",
        booking_id=booking_id,
        customer_name="Taylor",
        customer_phone="+14155550123",
        confirmed=False,
    )
    assert pending["pending"] is True
    assert get_pending_confirmation(call_id, ACTION_CANCEL_BOOKING) is not None
    clear_call_memory(call_id)
    assert get_pending_confirmation(call_id, ACTION_CANCEL_BOOKING) is None

    processed = await process_caller_turn(
        call_id,
        "Don't cancel it; I'm checking whether I can move it to seven.",
    )
    assert processed["handled"] is False
    assert processed["kind"] == "cancellation_reversal_with_remaining_intent"
    assert get_pending_confirmation(call_id, ACTION_CANCEL_BOOKING) is None

    connection = await asyncpg.connect(settings.database_url)
    try:
        status = await connection.fetchval(
            "SELECT status FROM bookings WHERE id = $1", booking_id
        )
        assert status == "confirmed"
        persisted = await connection.fetchval(
            "SELECT state FROM call_sessions WHERE session_id = $1", call_id
        )
        if isinstance(persisted, str):
            persisted = json.loads(persisted)
        assert persisted.get("pending_confirmations") == {}
    finally:
        await connection.close()
        clear_session(call_id)


async def test_order_corrections_confirmation_and_duplicate_delivery() -> None:
    ambiguous = await restaurant_service.add_order_item(
        call_id="order-call",
        idempotency_key="must-not-be-consumed",
        item_name="market",
        quantity=1,
        customer_name="Jordan",
        customer_phone="+14155550124",
    )
    assert ambiguous["added"] is False
    assert ambiguous["needs_confirmation"] is True
    assert ambiguous["candidates"][0]["name"] == "Market Greens"
    connection = await asyncpg.connect(settings.database_url)
    try:
        assert await connection.fetchval("SELECT COUNT(*) FROM orders") == 0
    finally:
        await connection.close()
    first = await restaurant_service.add_order_item(
        call_id="order-call",
        idempotency_key="add-greens-1",
        item_name="Market Greens",
        quantity=1,
        customer_name="Jordan",
        customer_phone="+14155550124",
    )
    replay = await restaurant_service.add_order_item(
        call_id="order-call",
        idempotency_key="add-greens-1",
        item_name="Market Greens",
        quantity=1,
        customer_name="Jordan",
        customer_phone="+14155550124",
    )
    assert replay["order_item_id"] == first["order_item_id"]
    assert replay["idempotent_replay"] is True
    assert len(replay["items"]) == 1

    updated = await restaurant_service.update_order_item(
        call_id="order-call",
        idempotency_key="update-greens-1",
        order_item_id=first["order_item_id"],
        quantity=2,
        notes="one without basil",
    )
    assert updated["items"][0]["quantity"] == 2
    assert updated["items"][0]["notes"] == "one without basil"

    preserved = await restaurant_service.update_order_item(
        call_id="order-call",
        idempotency_key="update-greens-quantity-only",
        order_item_id=first["order_item_id"],
        quantity=3,
    )
    assert preserved["items"][0]["notes"] == "one without basil"

    updated = await restaurant_service.update_order_item(
        call_id="order-call",
        idempotency_key="update-greens-clear-note",
        order_item_id=first["order_item_id"],
        quantity=2,
        notes="",
    )
    assert updated["items"][0]["notes"] == ""
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
    assert confirmed["total"] == 26.0
    assert confirmed_replay["order_id"] == confirmed["order_id"]
    assert confirmed_replay["idempotent_replay"] is True
    assert summary["draft_version"] == version


async def test_order_confirmation_rejects_closed_fulfillment_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = "closed-pickup-confirmation"
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="closed-pickup-add",
        item_name="Market Greens",
        quantity=1,
        customer_name="Jordan",
        customer_phone="+14155550124",
    )
    summary = await restaurant_service.get_order_summary(call_id)
    begin_caller_turn(call_id, "yes")
    timezone_info = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 14, 12, 0, tzinfo=timezone_info),
    )

    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.confirm_order(
            call_id=call_id,
            idempotency_key="closed-pickup-confirm",
            expected_draft_version=summary["draft_version"],
            approved=True,
        )
    assert exc.value.code == "fulfillment_unavailable"

    connection = await asyncpg.connect(settings.database_url)
    try:
        status = await connection.fetchval(
            "SELECT status FROM orders WHERE id = $1", added["order_id"]
        )
        assert status == "pending"
    finally:
        await connection.close()


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
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM tables WHERE location = 'bar'"
        ) == 4
        assert await connection.fetchval(
            "SELECT COALESCE(SUM(capacity), 0) FROM tables WHERE location = 'bar'"
        ) == 16
        assert await connection.fetchval("SELECT COUNT(*) FROM menu_items") == len(
            MENU_ITEMS
        )
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM restaurant_knowledge_records"
        ) >= 50
        assert await connection.fetchval(
            "SELECT COUNT(*) FROM restaurant_knowledge_records WHERE synthetic IS NOT TRUE"
        ) == 0
    finally:
        await connection.close()


async def test_menu_seed_renames_by_canonical_id_without_breaking_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import db.seed as seed_module

    connection = await asyncpg.connect(settings.database_url)
    try:
        original_id = await connection.fetchval(
            "SELECT id FROM menu_items WHERE canonical_id = $1",
            "menu.salad.market-greens",
        )
        order_id = await connection.fetchval(
            """
            INSERT INTO orders (session_id, customer_name, customer_phone)
            VALUES ('seed-rename-history', 'Morgan', '+15035550101')
            RETURNING id
            """
        )
        await connection.execute(
            """
            INSERT INTO order_items
                (order_id, menu_item_id, item_name, quantity, unit_price)
            VALUES ($1, $2, 'Old Market Greens', 1, 13)
            """,
            order_id,
            original_id,
        )
        await connection.execute(
            "UPDATE menu_items SET name = 'Old Market Greens' WHERE id = $1",
            original_id,
        )

        renamed_fixture = deepcopy(seed_module.MENU_ITEMS)
        renamed = next(
            item
            for item in renamed_fixture
            if item["item_id"] == "menu.salad.market-greens"
        )
        renamed["name"] = "Market Greens Renamed"
        monkeypatch.setattr(seed_module, "MENU_ITEMS", renamed_fixture)
        await seed_module.seed(connection, with_embeddings=False)

        row = await connection.fetchrow(
            "SELECT id, name FROM menu_items WHERE canonical_id = $1",
            "menu.salad.market-greens",
        )
        assert row["id"] == original_id
        assert row["name"] == "Market Greens Renamed"
        assert await connection.fetchval(
            "SELECT menu_item_id FROM order_items WHERE order_id = $1", order_id
        ) == original_id
    finally:
        await connection.close()


async def test_canonical_modifiers_delivery_and_order_notes_survive_confirmation() -> None:
    connection = await asyncpg.connect(settings.database_url)
    try:
        await seed(connection, with_embeddings=False)
    finally:
        await connection.close()

    ingredient_answer = await search_menu.ainvoke(
        {"query": "What ingredients and allergens are in the Market Greens?"}
    )
    assert "pear" in ingredient_answer.casefold()
    assert "hazelnut" in ingredient_answer.casefold()
    assert "tree_nut" in ingredient_answer.casefold()
    assert "cross-contact" in ingredient_answer.casefold()

    unavailable = await restaurant_service.add_order_item(
        call_id="knowledge-order-unavailable",
        idempotency_key="knowledge-unavailable-1",
        item_name="Smoked Salmon Dip",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert unavailable["unavailable"] is True
    assert {item["item_id"] for item in unavailable["candidates"]} == {
        "menu.main.cedar-salmon", "menu.starter.hearth-bread"
    }

    ambiguous = await restaurant_service.add_order_item(
        call_id="knowledge-order-ambiguous",
        idempotency_key="knowledge-ambiguous-1",
        item_name="chicken",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert ambiguous["added"] is False
    assert ambiguous["needs_confirmation"] is True

    incompatible = await restaurant_service.add_order_item(
        call_id="knowledge-order-incompatible",
        idempotency_key="knowledge-incompatible-1",
        item_name="Hearth Burger",
        modifier_ids=[
            "modifier.extra-cheddar", "modifier.remove-cheese", "modifier.side-fries"
        ],
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert incompatible["customization_status"] == "clarification_required"

    with pytest.raises(RestaurantServiceError) as outside_zone:
        await restaurant_service.set_order_fulfillment(
            call_id="knowledge-order-ambiguous",
            idempotency_key="knowledge-delivery-outside-1",
            fulfillment_type="delivery",
            delivery_address="55 Example Road, Portland, OR 99999",
        )
    assert outside_zone.value.code == "delivery_outside_zone"

    call_id = "knowledge-order-complete"
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="knowledge-add-complete-1",
        item_name="Hearth Burger",
        quantity=2,
        notes="Cut both in half",
        modifier_ids=["modifier.extra-cheddar", "modifier.side-fries"],
        removals=["onion jam"],
        order_notes="No utensils",
        allergy_notes="Severe sesame allergy; no safety guarantee requested",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert added["added"] is True
    assert added["items"][0]["unit_price"] == 23
    assert added["items"][0]["subtotal"] == 46
    assert added["items"][0]["removals"] == ["onion jam"]
    assert added["order_notes"] == "No utensils"
    assert "sesame" in added["allergy_notes"]

    delivered = await restaurant_service.set_order_fulfillment(
        call_id=call_id,
        idempotency_key="knowledge-delivery-1",
        fulfillment_type="delivery",
        delivery_address="101 Test Avenue, Portland, OR 97205",
        delivery_instructions="Leave with recipient only",
    )
    assert delivered["fulfillment"] == "delivery"
    assert delivered["item_total"] == 46
    assert delivered["total"] == 51
    assert delivered["fulfillment_details"]["live_integration"] is False

    first_summary = await restaurant_service.get_order_summary(call_id=call_id)
    first_hash = first_summary["pending_confirmation_hash"]
    first_version = first_summary["draft_version"]
    noted = await restaurant_service.add_guest_note(
        call_id=call_id,
        idempotency_key="knowledge-guest-note-1",
        note="Pack sauces separately",
    )
    assert noted["saved"] is True
    noted_summary = await restaurant_service.get_order_summary(call_id=call_id)
    assert noted_summary["draft_version"] == first_version + 1
    assert noted_summary["pending_confirmation_hash"] != first_hash
    first_hash = noted_summary["pending_confirmation_hash"]
    first_version = noted_summary["draft_version"]
    changed = await restaurant_service.set_order_notes(
        call_id=call_id,
        idempotency_key="knowledge-notes-change-1",
        allergy_notes="Severe dairy and sesame allergies; shared kitchen acknowledged",
    )
    assert changed["draft_version"] == first_version + 1
    second_summary = await restaurant_service.get_order_summary(call_id=call_id)
    assert second_summary["pending_confirmation_hash"] != first_hash

    clear_call_memory(call_id)
    await hydrate_call_memory(call_id)
    restarted = await restaurant_service.get_order_summary(call_id=call_id)
    assert restarted["order_notes"] == "No utensils; Pack sauces separately"
    assert "dairy and sesame" in restarted["allergy_notes"]
    assert restarted["items"][0]["notes"] == "Cut both in half"
    assert restarted["items"][0]["modifiers"][0]["option_id"] == "modifier.extra-cheddar"
    assert restarted["fulfillment_details"]["address"].startswith("101 Test Avenue")

    begin_caller_turn(call_id, "yes")
    confirmed = await restaurant_service.confirm_order(
        call_id=call_id,
        idempotency_key="knowledge-confirm-complete-1",
        expected_draft_version=restarted["draft_version"],
        approved=True,
    )
    assert confirmed["confirmed"] is True
    assert confirmed["total"] == 51
    assert confirmed["order_notes"] == "No utensils; Pack sauces separately"
    assert "dairy and sesame" in confirmed["allergy_notes"]

    readback = await restaurant_service.lookup_order(
        order_id=confirmed["order_id"], customer_name="Morgan"
    )
    assert readback["allergy_notes"] == confirmed["allergy_notes"]
    assert readback["fulfillment"] == "delivery"


async def test_seeded_alcohol_item_cannot_be_confirmed_as_transaction() -> None:
    call_id = "alcohol-confirmation-rejected"
    connection = await asyncpg.connect(settings.database_url)
    try:
        await seed(connection, with_embeddings=False)
        menu_item = await connection.fetchrow(
            "SELECT id, name, price FROM menu_items WHERE canonical_id = $1",
            "menu.alcohol.lager",
        )
        order_id = await connection.fetchval(
            """
            INSERT INTO orders
                (session_id, customer_name, customer_phone, fulfillment_type)
            VALUES ($1, 'Morgan', '+15035550101', 'pickup')
            RETURNING id
            """,
            call_id,
        )
        await connection.execute(
            """
            INSERT INTO order_items
                (order_id, menu_item_id, item_name, quantity, unit_price)
            VALUES ($1, $2, $3, 1, $4)
            """,
            order_id,
            menu_item["id"],
            menu_item["name"],
            menu_item["price"],
        )
    finally:
        await connection.close()

    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.confirm_order(
            call_id=call_id,
            idempotency_key="alcohol-confirm-1",
            expected_draft_version=1,
            approved=True,
        )
    assert exc.value.code == "alcohol_transaction_unsupported"

    connection = await asyncpg.connect(settings.database_url)
    try:
        assert await connection.fetchval(
            "SELECT status FROM orders WHERE id = $1", order_id
        ) == "pending"
    finally:
        await connection.close()


async def test_booking_notes_do_not_modify_order_owned_notes() -> None:
    call_id = "booking-order-note-ownership"
    connection = await asyncpg.connect(settings.database_url)
    try:
        booking_id = await connection.fetchval(
            """
            INSERT INTO bookings
                (customer_name, customer_phone, table_id, booked_at, party_size, notes)
            VALUES ('Morgan', '+15035550101', 1, '2026-09-12 19:00', 2, '')
            RETURNING id
            """
        )
        await connection.execute(
            "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
            call_id,
            json.dumps({"booking_id": booking_id}),
        )
        order_id = await connection.fetchval(
            """
            INSERT INTO orders
                (session_id, booking_id, customer_name, customer_phone,
                 fulfillment_type, notes)
            VALUES ($1, $2, 'Morgan', '+15035550101', 'dine_in', 'No utensils')
            RETURNING id
            """,
            call_id,
            booking_id,
        )
    finally:
        await connection.close()

    noted = await restaurant_service.add_guest_note(
        call_id=call_id,
        idempotency_key="booking-note-sync-1",
        note="Quiet table if possible",
        booking_id=booking_id,
    )
    assert noted["saved"] is True

    await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="booking-note-update-1",
        booking_id=booking_id,
        confirmed=False,
        occasion="Birthday",
    )
    begin_caller_turn(call_id, "yes")
    await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="booking-note-update-2",
        booking_id=booking_id,
        confirmed=True,
        occasion="Birthday",
    )

    connection = await asyncpg.connect(settings.database_url)
    try:
        assert await connection.fetchval(
            "SELECT notes FROM orders WHERE id = $1", order_id
        ) == "No utensils"
    finally:
        await connection.close()


async def test_pickup_items_use_one_persisted_fulfillment_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timezone_info = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 14, 45, tzinfo=timezone_info),
    )
    unavailable = await restaurant_service.add_order_item(
        call_id="pickup-between-services",
        idempotency_key="pickup-between-services-add",
        item_name="Market Greens",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert unavailable["added"] is False
    assert unavailable["unavailable"] is True

    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 16, 45, tzinfo=timezone_info),
    )
    added = await restaurant_service.add_order_item(
        call_id="pickup-dinner-service",
        idempotency_key="pickup-dinner-service-add",
        item_name="Market Greens",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert added["added"] is True
    assert added["fulfillment_details"]["fulfillment_at"] == (
        "2026-09-08T17:15:00-07:00"
    )

    summary = await restaurant_service.get_order_summary(
        call_id="pickup-dinner-service"
    )
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 16, 55, tzinfo=timezone_info),
    )
    begin_caller_turn("pickup-dinner-service", "yes")
    confirmed = await restaurant_service.confirm_order(
        call_id="pickup-dinner-service",
        idempotency_key="pickup-dinner-confirm",
        expected_draft_version=summary["draft_version"],
        approved=True,
    )
    assert confirmed["confirmed"] is True
    assert confirmed["fulfillment_details"]["fulfillment_at"] == (
        "2026-09-08T17:15:00-07:00"
    )
    assert confirmed["timing"] == "ready for pickup in about 20 minutes"


async def test_existing_pickup_time_wins_over_active_booking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timezone_info = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 14, 45, tzinfo=timezone_info),
    )
    call_id = "pickup-with-active-booking"
    connection = await asyncpg.connect(settings.database_url)
    try:
        booking_id = await connection.fetchval(
            """
            INSERT INTO bookings
                (customer_name, customer_phone, table_id, booked_at, party_size)
            VALUES ('Morgan', '+15035550101', 1, '2026-09-08 19:00', 2)
            RETURNING id
            """
        )
        await connection.execute(
            "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
            call_id,
            json.dumps({"booking_id": booking_id}),
        )
        await connection.execute(
            """
            INSERT INTO orders
                (session_id, customer_name, customer_phone, fulfillment_type,
                 fulfillment_details)
            VALUES ($1, 'Morgan', '+15035550101', 'pickup', $2::jsonb)
            """,
            call_id,
            json.dumps({"fulfillment_at": "2026-09-08T15:15:00-07:00"}),
        )
    finally:
        await connection.close()

    result = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="pickup-active-booking-add",
        item_name="Market Greens",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert result["added"] is False
    assert result["unavailable"] is True


async def test_fulfillment_retry_replays_original_generated_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timezone_info = ZoneInfo("America/Los_Angeles")
    call_id = "fulfillment-time-replay"
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 17, 0, tzinfo=timezone_info),
    )
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="fulfillment-replay-add",
        item_name="Market Greens",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert added["added"] is True
    first = await restaurant_service.set_order_fulfillment(
        call_id=call_id,
        idempotency_key="fulfillment-replay-set",
        fulfillment_type="pickup",
    )

    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 17, 5, tzinfo=timezone_info),
    )
    replay = await restaurant_service.set_order_fulfillment(
        call_id=call_id,
        idempotency_key="fulfillment-replay-set",
        fulfillment_type="pickup",
    )
    assert replay["idempotent_replay"] is True
    assert replay["fulfillment_details"]["fulfillment_at"] == first[
        "fulfillment_details"
    ]["fulfillment_at"]


async def test_expired_fulfillment_cannot_create_confirmation_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timezone_info = ZoneInfo("America/Los_Angeles")
    call_id = "expired-fulfillment-readback"
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 17, 0, tzinfo=timezone_info),
    )
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="expired-fulfillment-add",
        item_name="Market Greens",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert added["added"] is True

    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 8, 18, 0, tzinfo=timezone_info),
    )
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.get_order_summary(call_id=call_id)
    assert exc.value.code == "fulfillment_time_expired"
    assert get_pending_confirmation(call_id, ACTION_CONFIRM_ORDER) is None


async def test_confirmed_delivery_total_uses_persisted_fee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = "persisted-delivery-fee"
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="persisted-delivery-add",
        item_name="Market Greens",
        quantity=2,
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert added["added"] is True
    delivered = await restaurant_service.set_order_fulfillment(
        call_id=call_id,
        idempotency_key="persisted-delivery-fulfillment",
        fulfillment_type="delivery",
        delivery_address="101 Test Avenue, Portland, OR 97205",
    )
    assert delivered["fees"][0]["amount"] == 5

    summary = await restaurant_service.get_order_summary(call_id=call_id)
    begin_caller_turn(call_id, "yes")
    confirmed = await restaurant_service.confirm_order(
        call_id=call_id,
        idempotency_key="persisted-delivery-confirm",
        expected_draft_version=summary["draft_version"],
        approved=True,
    )
    monkeypatch.setattr(
        "app.services.restaurant._delivery_rule",
        lambda: {
            "delivery_fee": 7,
            "delivery_minimum": 20,
            "delivery_eta_minutes": [45, 60],
        },
    )
    restarted = await restaurant_service.lookup_order(
        order_id=confirmed["order_id"], customer_name="Morgan"
    )
    assert restarted["fees"][0]["amount"] == 5
    assert restarted["total"] == confirmed["total"]


async def test_booking_creation_does_not_rewrite_confirmed_delivery() -> None:
    call_id = "confirmed-delivery-before-booking"
    connection = await asyncpg.connect(settings.database_url)
    try:
        order_id = await connection.fetchval(
            """
            INSERT INTO orders
                (session_id, customer_name, customer_phone, status, total_amount,
                 fulfillment_type, fulfillment_details)
            VALUES ($1, 'Morgan', '+15035550101', 'confirmed', 25,
                    'delivery', $2::jsonb)
            RETURNING id
            """,
            call_id,
            json.dumps(
                {
                    "address": "101 Test Avenue, Portland, OR 97205",
                    "delivery_fee": 5,
                    "fulfillment_at": "2026-09-08T18:45:00-07:00",
                    "zone_status": "eligible",
                }
            ),
        )
    finally:
        await connection.close()

    booking_date = _future_date()
    confirmation = booking_confirmation_payload(
        customer_name="Morgan",
        customer_phone="+15035550101",
        date=booking_date,
        time="19:00",
        party_size=2,
        notes="",
    )
    _arm(call_id, ACTION_CREATE_BOOKING, confirmation)
    booked = await restaurant_service.create_booking(
        call_id=call_id,
        idempotency_key="confirmed-delivery-booking-create",
        customer_name="Morgan",
        customer_phone="+15035550101",
        date=booking_date,
        time="19:00",
        party_size=2,
        notes="",
        confirmed=True,
    )
    assert booked["created"] is True
    assert "order" not in booked

    connection = await asyncpg.connect(settings.database_url)
    try:
        order = await connection.fetchrow(
            """
            SELECT booking_id, fulfillment_type, fulfillment_details
            FROM orders WHERE id = $1
            """,
            order_id,
        )
    finally:
        await connection.close()
    assert order["booking_id"] is None
    assert order["fulfillment_type"] == "delivery"
    assert dict(order["fulfillment_details"])["delivery_fee"] == 5


async def test_booking_creation_attaches_only_unselected_draft_order() -> None:
    call_id = "unselected-draft-before-booking"
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="unselected-draft-add",
        item_name="Market Greens",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert added["fulfillment"] == "pickup"
    connection = await asyncpg.connect(settings.database_url)
    try:
        assert await connection.fetchval(
            "SELECT fulfillment_type FROM orders WHERE session_id = $1", call_id
        ) is None
    finally:
        await connection.close()

    booking_date = _future_date()
    confirmation = booking_confirmation_payload(
        customer_name="Morgan",
        customer_phone="+15035550101",
        date=booking_date,
        time="19:00",
        party_size=2,
        notes="",
    )
    _arm(call_id, ACTION_CREATE_BOOKING, confirmation)
    booked = await restaurant_service.create_booking(
        call_id=call_id,
        idempotency_key="unselected-draft-booking",
        customer_name="Morgan",
        customer_phone="+15035550101",
        date=booking_date,
        time="19:00",
        party_size=2,
        notes="",
        confirmed=True,
    )
    assert booked["order"]["fulfillment"] == "dine_in"
    assert booked["order"]["booking_id"] == booked["booking_id"]
    assert booked["order"]["fulfillment_details"] == {}


async def test_confirmed_pickup_change_requires_staff_status_check() -> None:
    call_id = "confirmed-pickup-change"
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="confirmed-pickup-add",
        item_name="Market Greens",
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    summary = await restaurant_service.get_order_summary(call_id=call_id)
    begin_caller_turn(call_id, "yes")
    await restaurant_service.confirm_order(
        call_id=call_id,
        idempotency_key="confirmed-pickup-confirm",
        expected_draft_version=summary["draft_version"],
        approved=True,
    )

    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.update_order_item(
            call_id=call_id,
            idempotency_key="confirmed-pickup-update",
            order_item_id=added["order_item_id"],
            quantity=2,
            caller_confirmed=True,
        )
    assert exc.value.code == "confirmed_fulfillment_change_requires_staff"

    connection = await asyncpg.connect(settings.database_url)
    try:
        quantity = await connection.fetchval(
            "SELECT quantity FROM order_items WHERE id = $1",
            added["order_item_id"],
        )
    finally:
        await connection.close()
    assert quantity == 1


async def test_legacy_confirmed_null_fulfillment_requires_staff_check() -> None:
    call_id = "legacy-null-confirmed-fulfillment"
    connection = await asyncpg.connect(settings.database_url)
    try:
        menu_item = await connection.fetchrow(
            "SELECT id, name, price FROM menu_items WHERE canonical_id = $1",
            "menu.salad.market-greens",
        )
        order_id = await connection.fetchval(
            """
            INSERT INTO orders
                (session_id, customer_name, customer_phone, status, fulfillment_type)
            VALUES ($1, 'Morgan', '+15035550101', 'confirmed', NULL)
            RETURNING id
            """,
            call_id,
        )
        order_item_id = await connection.fetchval(
            """
            INSERT INTO order_items
                (order_id, menu_item_id, item_name, quantity, unit_price)
            VALUES ($1, $2, $3, 1, $4)
            RETURNING id
            """,
            order_id,
            menu_item["id"],
            menu_item["name"],
            menu_item["price"],
        )
    finally:
        await connection.close()

    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.update_order_item(
            call_id=call_id,
            idempotency_key="legacy-null-confirmed-update",
            order_item_id=order_item_id,
            quantity=2,
            caller_confirmed=True,
        )
    assert exc.value.code == "confirmed_fulfillment_change_requires_staff"


async def test_dine_in_menu_period_uses_booking_time_and_rechecks_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timezone_info = ZoneInfo("America/Los_Angeles")
    monkeypatch.setattr(
        "app.services.restaurant._restaurant_now",
        lambda: datetime(2026, 9, 12, 10, 0, tzinfo=timezone_info),
    )
    call_id = "future-dinner-menu-period"
    connection = await asyncpg.connect(settings.database_url)
    try:
        booking_id = await connection.fetchval(
            """
            INSERT INTO bookings
                (customer_name, customer_phone, table_id, booked_at, party_size)
            VALUES ('Morgan', '+15035550101', 1, '2026-09-12 19:00', 2)
            RETURNING id
            """
        )
        await connection.execute(
            "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
            call_id,
            json.dumps({"booking_id": booking_id}),
        )
    finally:
        await connection.close()

    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key="future-dinner-add-1",
        item_name="Market Greens",
        booking_id=booking_id,
        customer_name="Morgan",
        customer_phone="+15035550101",
    )
    assert added["added"] is True

    summary = await restaurant_service.get_order_summary(call_id=call_id)
    connection = await asyncpg.connect(settings.database_url)
    try:
        await connection.execute(
            "UPDATE bookings SET booked_at = '2026-09-12 10:00' WHERE id = $1",
            booking_id,
        )
    finally:
        await connection.close()
    begin_caller_turn(call_id, "yes")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.confirm_order(
            call_id=call_id,
            idempotency_key="future-dinner-confirm-1",
            expected_draft_version=summary["draft_version"],
            approved=True,
        )
    assert exc.value.code == "service_period_unavailable"


async def test_booking_time_change_rejects_invalid_confirmed_dine_in_order_period() -> None:
    call_id = "confirmed-order-booking-reschedule"
    connection = await asyncpg.connect(settings.database_url)
    try:
        menu_item = await connection.fetchrow(
            "SELECT id, name, price FROM menu_items WHERE canonical_id = $1",
            "menu.salad.market-greens",
        )
        booking_id = await connection.fetchval(
            """
            INSERT INTO bookings
                (customer_name, customer_phone, table_id, booked_at, party_size)
            VALUES ('Morgan', '+15035550101', 1, '2026-09-12 19:00', 2)
            RETURNING id
            """
        )
        order_id = await connection.fetchval(
            """
            INSERT INTO orders
                (session_id, booking_id, customer_name, customer_phone,
                 status, fulfillment_type)
            VALUES ($1, $2, 'Morgan', '+15035550101', 'confirmed', 'dine_in')
            RETURNING id
            """,
            call_id,
            booking_id,
        )
        await connection.execute(
            """
            INSERT INTO order_items
                (order_id, menu_item_id, item_name, quantity, unit_price)
            VALUES ($1, $2, $3, 1, $4)
            """,
            order_id,
            menu_item["id"],
            menu_item["name"],
            menu_item["price"],
        )
        await connection.execute(
            "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
            call_id,
            json.dumps({"booking_id": booking_id}),
        )
    finally:
        await connection.close()

    await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="reschedule-period-pending",
        booking_id=booking_id,
        confirmed=False,
        time="10:00",
    )
    begin_caller_turn(call_id, "yes")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.update_confirmed_booking(
            call_id=call_id,
            idempotency_key="reschedule-period-confirmed",
            booking_id=booking_id,
            confirmed=True,
            time="10:00",
        )
    assert exc.value.code == "service_period_unavailable"

    connection = await asyncpg.connect(settings.database_url)
    try:
        assert await connection.fetchval(
            "SELECT booked_at FROM bookings WHERE id = $1", booking_id
        ) == datetime(2026, 9, 12, 19, 0)
    finally:
        await connection.close()


async def test_booking_time_change_rejects_expired_seasonal_order() -> None:
    call_id = "seasonal-order-booking-reschedule"
    connection = await asyncpg.connect(settings.database_url)
    try:
        menu_item = await connection.fetchrow(
            "SELECT id, name, price FROM menu_items WHERE canonical_id = $1",
            "menu.seasonal.corn-ravioli",
        )
        booking_id = await connection.fetchval(
            """
            INSERT INTO bookings
                (customer_name, customer_phone, table_id, booked_at, party_size)
            VALUES ('Morgan', '+15035550101', 1, '2026-09-30 19:00', 2)
            RETURNING id
            """
        )
        order_id = await connection.fetchval(
            """
            INSERT INTO orders
                (session_id, booking_id, customer_name, customer_phone,
                 status, fulfillment_type)
            VALUES ($1, $2, 'Morgan', '+15035550101', 'confirmed', 'dine_in')
            RETURNING id
            """,
            call_id,
            booking_id,
        )
        await connection.execute(
            """
            INSERT INTO order_items
                (order_id, menu_item_id, item_name, quantity, unit_price)
            VALUES ($1, $2, $3, 1, $4)
            """,
            order_id,
            menu_item["id"],
            menu_item["name"],
            menu_item["price"],
        )
        await connection.execute(
            "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
            call_id,
            json.dumps({"booking_id": booking_id}),
        )
    finally:
        await connection.close()

    await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="seasonal-reschedule-pending",
        booking_id=booking_id,
        confirmed=False,
        date="2026-10-01",
    )
    begin_caller_turn(call_id, "yes")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.update_confirmed_booking(
            call_id=call_id,
            idempotency_key="seasonal-reschedule-confirmed",
            booking_id=booking_id,
            confirmed=True,
            date="2026-10-01",
        )
    assert exc.value.code == "menu_item_unavailable"

    connection = await asyncpg.connect(settings.database_url)
    try:
        booked_at = await connection.fetchval(
            "SELECT booked_at FROM bookings WHERE id = $1",
            booking_id,
        )
    finally:
        await connection.close()
    assert booked_at == datetime(2026, 9, 30, 19, 0)


async def test_reschedule_ignores_later_inventory_for_confirmed_item() -> None:
    call_id = "sold-out-order-booking-reschedule"
    connection = await asyncpg.connect(settings.database_url)
    try:
        menu_item = await connection.fetchrow(
            "SELECT id, name, price FROM menu_items WHERE canonical_id = $1",
            "menu.salad.market-greens",
        )
        booking_id = await connection.fetchval(
            """
            INSERT INTO bookings
                (customer_name, customer_phone, table_id, booked_at, party_size)
            VALUES ('Morgan', '+15035550101', 1, '2026-09-12 19:00', 2)
            RETURNING id
            """
        )
        order_id = await connection.fetchval(
            """
            INSERT INTO orders
                (session_id, booking_id, customer_name, customer_phone,
                 status, fulfillment_type)
            VALUES ($1, $2, 'Morgan', '+15035550101', 'confirmed', 'dine_in')
            RETURNING id
            """,
            call_id,
            booking_id,
        )
        await connection.execute(
            """
            INSERT INTO order_items
                (order_id, menu_item_id, item_name, quantity, unit_price)
            VALUES ($1, $2, $3, 1, $4)
            """,
            order_id,
            menu_item["id"],
            menu_item["name"],
            menu_item["price"],
        )
        await connection.execute(
            """
            UPDATE menu_items
            SET available = FALSE, availability_status = 'sold_out'
            WHERE id = $1
            """,
            menu_item["id"],
        )
        await connection.execute(
            "INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)",
            call_id,
            json.dumps({"booking_id": booking_id}),
        )
    finally:
        await connection.close()

    await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="sold-out-reschedule-pending",
        booking_id=booking_id,
        confirmed=False,
        date="2026-09-13",
    )
    begin_caller_turn(call_id, "yes")
    updated = await restaurant_service.update_confirmed_booking(
        call_id=call_id,
        idempotency_key="sold-out-reschedule-confirmed",
        booking_id=booking_id,
        confirmed=True,
        date="2026-09-13",
    )
    assert updated["updated"] is True
    assert updated["date"] == "2026-09-13"
