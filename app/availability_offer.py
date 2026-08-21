"""Remember the latest availability offer so create_booking can honor a chosen table."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.call_memory import get_call_memory, resolve_session_id, update_call_memory

OFFER_TTL = timedelta(minutes=5)
# Party-size edits must cite a check from this many confirmation turns ago or fewer.
PARTY_CHANGE_MAX_TURN_AGE = 2


def remember_availability_offer(session_id: str, result: dict[str, Any]) -> None:
    sid = resolve_session_id(session_id)
    if not sid or not isinstance(result, dict):
        return
    now = datetime.now(timezone.utc)
    tables = result.get("tables") or []
    from app.pending_confirmation import current_confirmation_turn

    update_call_memory(
        sid,
        last_availability_offer={
            "date": str(result.get("date") or ""),
            "time": str(result.get("time") or ""),
            "party_size": int(result.get("party_size") or 0),
            "preferred_location": str(result.get("preferred_location") or ""),
            "available": bool(result.get("available")),
            "table_numbers": [
                int(row.get("table_number"))
                for row in tables
                if row.get("table_number") is not None
            ],
            "tables": list(tables),
            "nonce": str(result.get("availability_nonce") or ""),
            "created_at": now.isoformat(),
            "expires_at": (now + OFFER_TTL).isoformat(),
            "created_turn": current_confirmation_turn(sid),
        },
    )


def get_availability_offer(session_id: str) -> dict[str, Any] | None:
    mem = get_call_memory(session_id)
    raw = mem.get("last_availability_offer")
    return dict(raw) if isinstance(raw, dict) else None


def _offer_still_fresh(
    offer: dict[str, Any],
    *,
    session_id: str = "",
    max_turn_age: int | None = None,
) -> None:
    from app.pending_confirmation import current_confirmation_turn
    from app.services.restaurant import RestaurantServiceError

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
    if max_turn_age is not None:
        try:
            created_turn = int(offer.get("created_turn") or 0)
        except (TypeError, ValueError):
            created_turn = 0
        sid = resolve_session_id(session_id) if session_id else resolve_session_id()
        current = current_confirmation_turn(sid) if sid else 0
        if created_turn <= 0 or current - created_turn > max_turn_age:
            raise RestaurantServiceError(
                "No recent availability check for that party size. "
                "Call check_table_availability for the new party size first.",
                code="availability_offer_stale_turn",
                status=409,
            )


def require_fresh_availability_for_party_change(
    session_id: str,
    *,
    date: str,
    time: str,
    party_size: int,
    preferred_location: str = "",
) -> dict[str, Any]:
    """Require a recent check_table_availability result for this exact slot/party.

    Used whenever party_size changes on a draft or confirmed booking so capacity
    claims cannot be invented without a tool result.
    """
    from app.reservation_draft import normalize_preferred_location
    from app.services.restaurant import RestaurantServiceError

    offer = get_availability_offer(session_id)
    if not offer:
        raise RestaurantServiceError(
            "No recent availability check for that party size. "
            "Call check_table_availability for the new date, time, and party size first.",
            code="availability_offer_missing",
            status=409,
        )
    _offer_still_fresh(
        offer, session_id=session_id, max_turn_age=PARTY_CHANGE_MAX_TURN_AGE
    )
    pref = normalize_preferred_location(preferred_location)
    offer_pref = normalize_preferred_location(str(offer.get("preferred_location") or ""))
    if (
        str(offer.get("date") or "") != str(date)
        or str(offer.get("time") or "") != str(time)
        or int(offer.get("party_size") or 0) != int(party_size)
        or (pref and offer_pref and pref != offer_pref)
    ):
        raise RestaurantServiceError(
            "The latest availability check does not match this date, time, location, "
            "or party size. Check availability again for the exact party size first.",
            code="availability_offer_mismatch",
            status=409,
        )
    if not offer.get("available"):
        raise RestaurantServiceError(
            f"A party of {party_size} is not available at {time} on {date} "
            "according to the latest availability check. Offer alternatives; "
            "do not change the party size.",
            code="capacity_unavailable",
            status=409,
        )
    return offer


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
    _offer_still_fresh(offer)
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
