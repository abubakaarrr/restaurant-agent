"""Remember the latest availability offer so create_booking can honor a chosen table."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.call_memory import get_call_memory, resolve_session_id, update_call_memory

OFFER_TTL = timedelta(minutes=5)


def remember_availability_offer(session_id: str, result: dict[str, Any]) -> None:
    sid = resolve_session_id(session_id)
    if not sid or not isinstance(result, dict):
        return
    now = datetime.now(timezone.utc)
    tables = result.get("tables") or []
    update_call_memory(
        sid,
        last_availability_offer={
            "date": str(result.get("date") or ""),
            "time": str(result.get("time") or ""),
            "party_size": int(result.get("party_size") or 0),
            "preferred_location": str(result.get("preferred_location") or ""),
            "table_numbers": [
                int(row.get("table_number"))
                for row in tables
                if row.get("table_number") is not None
            ],
            "tables": list(tables),
            "nonce": str(result.get("availability_nonce") or ""),
            "created_at": now.isoformat(),
            "expires_at": (now + OFFER_TTL).isoformat(),
        },
    )


def get_availability_offer(session_id: str) -> dict[str, Any] | None:
    mem = get_call_memory(session_id)
    raw = mem.get("last_availability_offer")
    return dict(raw) if isinstance(raw, dict) else None


def require_offered_table(
    session_id: str,
    *,
    table_number: int,
    date: str,
    time: str,
    party_size: int,
) -> None:
    """Reject a table_number that was not in a fresh availability offer for this slot."""
    from app.services.restaurant import RestaurantServiceError

    offer = get_availability_offer(session_id)
    if not offer:
        raise RestaurantServiceError(
            "No recent availability check for that table. Check availability again, "
            "then book one of the offered tables.",
            code="availability_offer_missing",
            status=409,
        )
    expires_raw = str(offer.get("expires_at") or "")
    try:
        expires_at = datetime.fromisoformat(expires_raw)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except ValueError:
        expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    if datetime.now(timezone.utc) > expires_at:
        raise RestaurantServiceError(
            "The availability offer expired. Check availability again before booking.",
            code="availability_offer_expired",
            status=409,
        )
    if (
        str(offer.get("date") or "") != str(date)
        or str(offer.get("time") or "") != str(time)
        or int(offer.get("party_size") or 0) != int(party_size)
    ):
        raise RestaurantServiceError(
            "That table was offered for a different date, time, or party size. "
            "Check availability again.",
            code="availability_offer_mismatch",
            status=409,
        )
    offered = {int(n) for n in (offer.get("table_numbers") or [])}
    if int(table_number) not in offered:
        raise RestaurantServiceError(
            f"Table {table_number} was not in the latest availability offer. "
            "Offer the returned tables or check availability again.",
            code="table_not_offered",
            status=409,
        )
