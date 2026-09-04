"""Confirmed booking readback must match the live bookings row."""

from __future__ import annotations

from app.call_memory import (
    apply_live_booking_to_memory,
    clear_call_memory,
    get_call_memory,
    get_reservation_draft,
    set_active_booking,
)


def test_apply_live_booking_refreshes_seating_and_table() -> None:
    clear_call_memory("sot-1")
    set_active_booking(
        "sot-1",
        booking_id=6,
        customer_name="Hamza",
        customer_phone="+14155550100",
        party_size=4,
        date="2026-09-12",
        time="19:00",
        table_number=10,
        table_location="patio",
        notes="seating: patio",
        seating_preference="patio",
    )
    # Simulate a confirmed update that the session flatten missed.
    first = apply_live_booking_to_memory(
        "sot-1",
        {
            "booking_id": 6,
            "customer_name": "Hamza",
            "customer_phone": "+14155550100",
            "date": "2026-09-12",
            "time": "19:00",
            "party_size": 4,
            "status": "confirmed",
            "table_number": 951,
            "location": "main",
            "notes": "seating: main area",
        },
    )
    second = apply_live_booking_to_memory(
        "sot-1",
        {
            "booking_id": 6,
            "customer_name": "Hamza",
            "customer_phone": "+14155550100",
            "date": "2026-09-12",
            "time": "19:00",
            "party_size": 4,
            "status": "confirmed",
            "table_number": 951,
            "location": "main",
            "notes": "seating: main area",
        },
    )
    assert first["seating_preference"] == "main area"
    assert second["seating_preference"] == first["seating_preference"]
    assert get_call_memory("sot-1").get("table_number") == 951
    assert get_call_memory("sot-1").get("table_location") == "main"
    assert get_reservation_draft("sot-1")["seating_preference"] == "main area"
    clear_call_memory("sot-1")


def test_apply_live_booking_after_party_reassignment() -> None:
    clear_call_memory("sot-2")
    set_active_booking(
        "sot-2",
        booking_id=7,
        customer_name="Sam",
        customer_phone="+14155550101",
        party_size=4,
        date="2026-09-13",
        time="20:00",
        table_number=10,
        table_location="patio",
        seating_preference="patio",
    )
    a = apply_live_booking_to_memory(
        "sot-2",
        {
            "booking_id": 7,
            "customer_name": "Sam",
            "customer_phone": "+14155550101",
            "date": "2026-09-13",
            "time": "20:00",
            "party_size": 6,
            "status": "confirmed",
            "table_number": 3,
            "location": "main",
            "notes": "seating: main",
        },
    )
    b = get_reservation_draft("sot-2")
    assert a["party_size"] == 6
    assert b["party_size"] == 6
    assert get_call_memory("sot-2")["table_number"] == 3
    assert a["seating_preference"] == b["seating_preference"]
    clear_call_memory("sot-2")
