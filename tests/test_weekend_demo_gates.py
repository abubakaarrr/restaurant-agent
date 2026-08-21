"""Evals for weekend-demo gates.

These fail on the pre-fix behavior (patio treated as a sort hint, notes
overwritten, paid items attached without a yes, confirm without a summary)
and pass on the environment gates added in this change.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os

import pytest

from app.call_memory import (
    clear_call_memory,
    format_memory_for_prompt,
    get_call_memory,
    update_call_memory,
    update_reservation_draft,
)
from app.reservation_draft import patch_draft
from app.turn_evidence import (
    audit_assistant_speech,
    begin_turn,
    end_turn,
    record_availability,
    record_order_summary,
    speech_contains_total,
)


def test_repeat_availability_claim_without_fresh_tool_is_flagged() -> None:
    begin_turn("eval-avail", "turn-1")
    record_availability(
        {
            "availability_nonce": "aaa111",
            "available": False,
            "preferred_location": "patio",
            "tables": [],
            "alternatives": [{"kind": "location", "location": "main"}],
        }
    )
    first = audit_assistant_speech("The patio is full tonight.")
    assert not any(flag["code"] == "ungrounded_availability" for flag in first)
    end_turn()

    begin_turn("eval-avail", "turn-2")
    flags = audit_assistant_speech("Yes, the patio is available at seven.")
    end_turn()
    assert any(flag["code"] == "ungrounded_availability" for flag in flags)


def test_patio_claim_against_negative_nonce_is_flagged() -> None:
    begin_turn("eval-conflict", "turn-1")
    record_availability(
        {
            "availability_nonce": "bbb222",
            "available": False,
            "preferred_location": "patio",
            "tables": [],
            "alternatives": [],
        }
    )
    flags = audit_assistant_speech("The patio is available at 7.")
    end_turn()
    assert any(flag["code"] == "conflicting_availability" for flag in flags)


def test_fresh_nonce_allows_true_patio_claim() -> None:
    begin_turn("eval-ok", "turn-1")
    record_availability(
        {
            "availability_nonce": "ccc333",
            "available": True,
            "preferred_location": "patio",
            "tables": [{"table_number": 10, "location": "patio", "capacity": 4}],
            "alternatives": [],
        }
    )
    flags = audit_assistant_speech("The patio is available at 7.")
    end_turn()
    assert flags == []


def test_guest_notes_survive_structured_flatten() -> None:
    clear_call_memory("eval-notes")
    update_reservation_draft(
        "eval-notes",
        party_size=4,
        seating_preference="window",
        customer_name="Sam",
        customer_phone="03098121804",
        date="2026-09-01",
        time="19:00",
    )
    update_call_memory(
        "eval-notes",
        notes="don't add anything with an extra charge without asking me first",
    )
    update_reservation_draft("eval-notes", dietary="vegetarian")
    memory = get_call_memory("eval-notes")
    prompt = format_memory_for_prompt("eval-notes")
    assert "don't add anything with an extra charge" in (memory.get("guest_notes") or "")
    assert "don't add anything with an extra charge" in prompt
    assert "vegetarian" in prompt
    assert "window" in prompt
    clear_call_memory("eval-notes")


def test_paid_approval_flag_is_a_draft_field() -> None:
    draft = patch_draft({}, {"require_approval_for_paid_items": True, "party_size": 2})
    assert draft["require_approval_for_paid_items"] is True
    assert draft["party_size"] == 2
    cleared = patch_draft(draft, {"require_approval_for_paid_items": False})
    assert cleared["require_approval_for_paid_items"] is False


def test_speech_contains_matching_order_total() -> None:
    assert speech_contains_total("That's $18.00 all in. Is that correct?", 18.0)
    assert not speech_contains_total("Booking for four at seven.", 18.0)


def test_missing_total_in_confirmation_speech_is_flagged() -> None:
    begin_turn("eval-total", "turn-1")
    record_order_summary(
        {
            "summary_nonce": "sum1",
            "total": 18.0,
            "draft_version": 1,
            "order_id": 9,
            "booking_id": 4,
            "fulfillment": "dine_in",
        }
    )
    flags = audit_assistant_speech("You're all set for Saturday at seven.")
    end_turn()
    assert any(flag["code"] == "missing_order_total" for flag in flags)

    begin_turn("eval-total", "turn-2")
    record_order_summary(
        {
            "summary_nonce": "sum2",
            "total": 18.0,
            "draft_version": 1,
            "order_id": 9,
            "booking_id": 4,
            "fulfillment": "dine_in",
        }
    )
    ok = audit_assistant_speech("Table for two and the pizza, $18.00 total. You're booked.")
    end_turn()
    assert not any(flag["code"] == "missing_order_total" for flag in ok)


def test_prompt_keeps_greeting_and_stated_edit_rules() -> None:
    from pathlib import Path

    prompt = (
        Path(__file__).resolve().parent.parent / "app" / "prompts" / "system.md"
    ).read_text(encoding="utf-8")
    assert "Never repeat that script later" in prompt
    assert "If they already stated a change, that is confirmation" in prompt
    assert "When an occasion is mentioned, acknowledge it warmly" in prompt


pytestmark_db = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


def _future() -> tuple[str, str]:
    when = datetime.now() + timedelta(days=21)
    return when.date().isoformat(), "19:00"


@pytestmark_db
@pytest.mark.asyncio
async def test_patio_full_is_not_available_when_main_is_free(monkeypatch) -> None:
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
        await connection.execute(
            "DELETE FROM bookings WHERE customer_name LIKE 'PatioBlocker%'"
        )
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (901, 4, 'main'), (910, 4, 'patio')
            ON CONFLICT (table_number) DO UPDATE
            SET capacity = EXCLUDED.capacity, location = EXCLUDED.location
            """
        )
        patio_ids = [
            row["id"]
            for row in await connection.fetch(
                "SELECT id FROM tables WHERE location = 'patio'"
            )
        ]
        booked_at = datetime.fromisoformat(f"{date}T{time}")
        for index, patio_id in enumerate(patio_ids):
            await connection.execute(
                """
                INSERT INTO bookings
                    (customer_name, customer_phone, table_id, booked_at, party_size, status)
                VALUES ($1, '+14155550901', $2, $3, 2, 'confirmed')
                """,
                f"PatioBlocker{index}",
                patio_id,
                booked_at,
            )
    finally:
        await connection.close()

    begin_turn("eval-patio", "t1")
    try:
        first = await restaurant_service.check_availability(
            date, time, 2, preferred_location="patio"
        )
        assert first["available"] is False
        assert first["tables"] == []
        assert first["availability_nonce"]
        assert first["alternatives"]
        assert all(
            row.get("kind") in {"time", "location"} for row in first["alternatives"]
        )
        assert any(row.get("kind") == "location" for row in first["alternatives"])
        assert all(
            str(row.get("location") or "").casefold() != "patio"
            for row in first["alternatives"]
            if row.get("kind") == "location"
        )
        assert all(
            row.get("location") == "patio"
            for row in first["alternatives"]
            if row.get("kind") == "time"
        )
        first_nonce = first["availability_nonce"]

        second = await restaurant_service.check_availability(
            date, time, 2, preferred_location="patio"
        )
        assert second["available"] is False
        assert second["availability_nonce"] != first_nonce
        flags = audit_assistant_speech(
            "Yes the patio is available, same as before."
        )
        assert any(
            flag["code"] in {"conflicting_availability", "ungrounded_availability"}
            for flag in flags
        )
    finally:
        end_turn()
        cleanup = await asyncpg.connect(database_url)
        try:
            await cleanup.execute(
                "DELETE FROM bookings WHERE customer_name LIKE 'PatioBlocker%'"
            )
        finally:
            await cleanup.close()
        await close_pool()


@pytestmark_db
@pytest.mark.asyncio
async def test_paid_item_lands_proposed_until_explicit_yes(monkeypatch) -> None:
    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import restaurant_service

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ('Eval Steak', 'main', 32.00, 'Test steak', ARRAY[]::text[], TRUE)
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        await connection.close()

    from app.call_memory import clear_call_memory, get_reservation_draft, update_reservation_draft
    from app.reservation_draft import flatten_draft

    clear_call_memory("eval-paid")
    update_reservation_draft(
        "eval-paid",
        require_approval_for_paid_items=True,
        customer_name="Sam",
        customer_phone="+14155550902",
    )
    await restaurant_service.persist_call_state(
        "eval-paid",
        flatten_draft(get_reservation_draft("eval-paid")),
        caller_phone="+14155550902",
    )
    proposed = await restaurant_service.add_order_item(
        call_id="eval-paid",
        idempotency_key="eval-paid-add-1",
        item_name="Eval Steak",
        quantity=1,
        caller_confirmed=False,
    )
    assert proposed["proposed"] is True
    assert proposed["added"] is False
    assert proposed["items"] == []
    assert proposed["proposed_items"]
    assert proposed["total"] == 0

    confirmed = await restaurant_service.add_order_item(
        call_id="eval-paid",
        idempotency_key="eval-paid-add-2",
        item_name="Eval Steak",
        quantity=1,
        caller_confirmed=True,
    )
    assert confirmed["added"] is True
    assert confirmed["proposed"] is False
    assert len(confirmed["items"]) == 1
    await close_pool()
    clear_call_memory("eval-paid")


@pytestmark_db
@pytest.mark.asyncio
async def test_confirm_order_requires_pending_readback_then_affirmation(
    monkeypatch,
) -> None:
    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool
    from app.pending_confirmation import begin_caller_turn
    from app.services.restaurant import RestaurantServiceError, restaurant_service
    from app.turn_evidence import begin_turn, end_turn

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    date, time = _future()
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES (902, 4, 'main')
            ON CONFLICT (table_number) DO NOTHING
            """
        )
        await connection.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ('Eval Pizza', 'main', 18.00, 'Test pizza', ARRAY['vegetarian'], TRUE)
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        await connection.close()

    call_id = f"eval-dinein-{__import__('uuid').uuid4().hex[:10]}"
    clear_call_memory(call_id)
    added = await restaurant_service.add_order_item(
        call_id=call_id,
        idempotency_key=f"{call_id}-food",
        item_name="Eval Pizza",
        quantity=1,
        customer_name="Sam",
        customer_phone="+14155550903",
    )
    assert added["added"] is True
    assert added["fulfillment"] == "pickup"
    assert not added["booking_id"]

    begin_turn(call_id, "confirm-turn")
    try:
        with pytest.raises(RestaurantServiceError) as exc:
            await restaurant_service.confirm_order(
                call_id=call_id,
                idempotency_key=f"{call_id}-confirm-early",
                expected_draft_version=added["draft_version"],
                approved=True,
            )
        assert exc.value.code in {"pending_confirmation_missing", "affirmation_required"}

        begin_caller_turn(call_id, "read the order")
        summary = await restaurant_service.get_order_summary(call_id=call_id)
        assert summary["total"] == 18.0

        # Booking also needs its own pending confirmation + later affirmation.
        from app.pending_confirmation import (
            ACTION_CREATE_BOOKING,
            booking_confirmation_payload,
            register_pending_confirmation,
        )

        book_payload = booking_confirmation_payload(
            customer_name="Sam",
            customer_phone="+14155550903",
            date=date,
            time=time,
            party_size=2,
            notes="",
        )
        register_pending_confirmation(call_id, ACTION_CREATE_BOOKING, book_payload)
        begin_caller_turn(call_id, "yes")
        booked = await restaurant_service.create_booking(
            call_id=call_id,
            idempotency_key=f"{call_id}-book",
            customer_name="Sam",
            customer_phone="+14155550903",
            date=date,
            time=time,
            party_size=2,
            notes="",
            confirmed=True,
        )
        assert booked["created"] is True
        assert booked["order"]["booking_id"] == booked["booking_id"]
        assert booked["order"]["fulfillment"] == "dine_in"
        assert booked["order"]["total"] == 18.0

        # Order hash changed when booking_id attached — need a fresh summary + yes.
        begin_caller_turn(call_id, "confirm the food too")
        summary2 = await restaurant_service.get_order_summary(call_id=call_id)
        begin_caller_turn(call_id, "yes")
        confirmed = await restaurant_service.confirm_order(
            call_id=call_id,
            idempotency_key=f"{call_id}-confirm",
            expected_draft_version=summary2["draft_version"],
            approved=True,
        )
        assert confirmed["confirmed"] is True
        assert confirmed["fulfillment"] == "dine_in"
        assert confirmed["total"] == 18.0
        speech = (
            f"You're booked at table {booked['table_number']}, "
            f"pizza pre-order ${confirmed['total']:.2f}."
        )
        flags = audit_assistant_speech(speech)
        assert not any(flag["code"] == "missing_order_total" for flag in flags)
    finally:
        end_turn()
        await close_pool()
