"""Party-size changes must cite a fresh check_table_availability offer."""

from __future__ import annotations

import pytest

from app.availability_offer import remember_availability_offer
from app.call_memory import clear_call_memory, update_reservation_draft
from app.pending_confirmation import begin_caller_turn
from app.services.restaurant import RestaurantServiceError, restaurant_service


def test_draft_party_change_rejected_without_availability_check() -> None:
    clear_call_memory("party-no-check")
    begin_caller_turn("party-no-check", "book for four")
    update_reservation_draft(
        "party-no-check",
        customer_name="Hamza",
        customer_phone="+14155550100",
        date="2026-09-12",
        time="19:00",
        party_size=4,
    )
    begin_caller_turn("party-no-check", "make it five")
    with pytest.raises(RestaurantServiceError) as exc:
        update_reservation_draft("party-no-check", party_size=5)
    assert exc.value.code == "availability_offer_missing"
    clear_call_memory("party-no-check")


def test_draft_party_change_succeeds_with_matching_available_offer() -> None:
    clear_call_memory("party-ok")
    begin_caller_turn("party-ok", "four please")
    update_reservation_draft(
        "party-ok",
        customer_name="Hamza",
        customer_phone="+14155550100",
        date="2026-09-12",
        time="19:00",
        party_size=4,
    )
    begin_caller_turn("party-ok", "could five fit?")
    remember_availability_offer(
        "party-ok",
        {
            "available": True,
            "date": "2026-09-12",
            "time": "19:00",
            "party_size": 5,
            "preferred_location": "",
            "availability_nonce": "abc",
            "tables": [{"table_number": 3, "capacity": 6, "location": "main"}],
        },
    )
    begin_caller_turn("party-ok", "yes make it five")
    draft = update_reservation_draft("party-ok", party_size=5)
    assert int(draft["party_size"]) == 5
    clear_call_memory("party-ok")


def test_draft_party_change_rejects_when_offer_says_unavailable() -> None:
    clear_call_memory("party-no")
    begin_caller_turn("party-no", "four")
    update_reservation_draft(
        "party-no",
        customer_name="Hamza",
        customer_phone="+14155550100",
        date="2026-09-12",
        time="19:00",
        party_size=4,
    )
    begin_caller_turn("party-no", "five?")
    remember_availability_offer(
        "party-no",
        {
            "available": False,
            "date": "2026-09-12",
            "time": "19:00",
            "party_size": 5,
            "preferred_location": "",
            "availability_nonce": "def",
            "tables": [],
        },
    )
    begin_caller_turn("party-no", "make it five")
    with pytest.raises(RestaurantServiceError) as exc:
        update_reservation_draft("party-no", party_size=5)
    assert exc.value.code == "capacity_unavailable"
    clear_call_memory("party-no")


@pytest.mark.asyncio
async def test_confirmed_party_change_requires_matching_check() -> None:
    clear_call_memory("party-upd")
    begin_caller_turn("party-upd", "change to five")
    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.update_confirmed_booking(
            call_id="party-upd",
            idempotency_key="party-upd-1",
            booking_id=6,
            confirmed=False,
            party_size=5,
            date="2026-09-12",
            time="19:00",
        )
    assert exc.value.code == "availability_offer_missing"
    clear_call_memory("party-upd")
