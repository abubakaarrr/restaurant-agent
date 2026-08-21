"""Server-owned confirmation gate: pending readback + caller affirmation.

The LLM may set caller_confirmed / approved on tool calls, but writes only
succeed when this module has a live matching pending record and the latest
user utterance was classified affirmative on a later turn than the readback.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from app.call_memory import get_call_memory, resolve_session_id, update_call_memory

Affirmation = Literal["affirmative", "negative", "unclear"]

ACTION_CREATE_BOOKING = "create_booking"
ACTION_CONFIRM_ORDER = "confirm_order"
VALID_ACTIONS = frozenset({ACTION_CREATE_BOOKING, ACTION_CONFIRM_ORDER})

PENDING_TTL = timedelta(minutes=15)

_AFFIRMATIVE_RE = re.compile(
    r"(?:"
    r"\byes\b|\byeah\b|\byep\b|\byup\b|"
    r"\bcorrect\b|that'?s right|sounds good|go ahead|\bconfirmed\b|"
    r"all (?:of )?that(?:'s| is)? correct|that(?:'s| is) (?:all )?correct|"
    r"\bplease do\b|\babsolutely\b"
    r")",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"(?:"
    r"\bno\b|\bnope\b|not correct|\bwait\b|\bactually\b|\bchange\b|"
    r"hold on|never ?mind|don'?t|do not|wrong|instead"
    r")",
    re.IGNORECASE,
)


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def booking_confirmation_payload(
    *,
    customer_name: str,
    customer_phone: str,
    date: str,
    time: str,
    party_size: int,
    notes: str = "",
) -> dict[str, Any]:
    from app.config import settings
    from app.security import normalize_caller_phone

    name = " ".join(str(customer_name or "").split())
    raw_phone = str(customer_phone or "").strip()
    phone = (
        normalize_caller_phone(
            raw_phone,
            default_country_code=settings.default_caller_country_code,
        )
        or raw_phone
    )
    return {
        "customer_name": name,
        "customer_phone": phone,
        "date": str(date or "").strip(),
        "time": str(time or "").strip(),
        "party_size": int(party_size or 0),
        "notes": str(notes or "").strip(),
    }


def order_confirmation_payload(summary: dict[str, Any]) -> dict[str, Any]:
    items = []
    for item in summary.get("items") or []:
        items.append(
            {
                "item_name": str(item.get("item_name") or item.get("name") or ""),
                "quantity": int(item.get("quantity") or 0),
                "notes": str(item.get("notes") or ""),
            }
        )
    items.sort(key=lambda row: (row["item_name"], row["quantity"], row["notes"]))
    return {
        "order_id": int(summary.get("order_id") or 0),
        "draft_version": int(summary.get("draft_version") or 0),
        "booking_id": int(summary.get("booking_id") or 0),
        "fulfillment": str(summary.get("fulfillment") or ""),
        "total": round(float(summary.get("total") or 0), 2),
        "items": items,
    }


def classify_affirmation(utterance: str) -> Affirmation:
    text = " ".join((utterance or "").split())
    if not text:
        return "unclear"
    has_neg = bool(_NEGATIVE_RE.search(text))
    has_aff = bool(_AFFIRMATIVE_RE.search(text))
    if has_neg and not has_aff:
        return "negative"
    if has_neg and has_aff:
        return "negative"
    if has_aff:
        return "affirmative"
    return "unclear"


def current_confirmation_turn(session_id: str) -> int:
    mem = get_call_memory(session_id)
    try:
        return int(mem.get("confirmation_turn") or 0)
    except (TypeError, ValueError):
        return 0


def begin_caller_turn(session_id: str, utterance: str) -> Affirmation:
    """Advance the confirmation turn and store the utterance classification."""
    sid = resolve_session_id(session_id)
    if not sid:
        return "unclear"
    turn = current_confirmation_turn(sid) + 1
    affirmation = classify_affirmation(utterance)
    update_call_memory(
        sid,
        confirmation_turn=turn,
        last_turn_affirmation=affirmation,
    )
    return affirmation


def _pending_map(session_id: str) -> dict[str, Any]:
    mem = get_call_memory(session_id)
    raw = mem.get("pending_confirmations") or {}
    return dict(raw) if isinstance(raw, dict) else {}


def register_pending_confirmation(
    session_id: str,
    action_type: str,
    payload: dict[str, Any],
) -> str:
    """Record a readback pending confirmation; overwrites prior for this action."""
    sid = resolve_session_id(session_id)
    if not sid:
        raise ValueError("session_id is required to register a pending confirmation")
    if action_type not in VALID_ACTIONS:
        raise ValueError(f"Unknown action_type: {action_type}")
    digest = payload_hash(payload)
    now = datetime.now(timezone.utc)
    pending = _pending_map(sid)
    pending[action_type] = {
        "payload_hash": digest,
        "created_at": now.isoformat(),
        "expires_at": (now + PENDING_TTL).isoformat(),
        "created_turn": current_confirmation_turn(sid),
        "payload": payload,
    }
    update_call_memory(sid, pending_confirmations=pending)
    return digest


def clear_pending_confirmation(session_id: str, action_type: str) -> None:
    sid = resolve_session_id(session_id)
    if not sid:
        return
    pending = _pending_map(sid)
    if action_type in pending:
        pending.pop(action_type, None)
        update_call_memory(sid, pending_confirmations=pending)


def get_pending_confirmation(
    session_id: str, action_type: str
) -> dict[str, Any] | None:
    pending = _pending_map(session_id)
    record = pending.get(action_type)
    return dict(record) if isinstance(record, dict) else None


def pending_state_patch(session_id: str) -> dict[str, Any]:
    """Fields to merge into call_sessions.state for cross-request durability."""
    mem = get_call_memory(session_id)
    return {
        "pending_confirmations": mem.get("pending_confirmations") or {},
        "confirmation_turn": int(mem.get("confirmation_turn") or 0),
        "last_turn_affirmation": str(mem.get("last_turn_affirmation") or "unclear"),
    }


def require_pending_confirmation(
    session_id: str,
    action_type: str,
    payload: dict[str, Any],
) -> None:
    """Raise RestaurantServiceError(409) unless the server-owned gate passes."""
    from app.services.restaurant import RestaurantServiceError

    sid = resolve_session_id(session_id) or session_id
    if action_type not in VALID_ACTIONS:
        raise RestaurantServiceError(
            f"Unknown confirmation action: {action_type}",
            code="confirmation_required",
            status=409,
        )

    record = get_pending_confirmation(sid, action_type)
    if not record:
        raise RestaurantServiceError(
            "No matching readback is pending. Call get_reservation_draft or "
            "get_order_summary, read it back, then confirm after the caller says yes.",
            code="pending_confirmation_missing",
            status=409,
        )

    expires_raw = str(record.get("expires_at") or "")
    try:
        expires_at = datetime.fromisoformat(expires_raw)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
    except ValueError:
        expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    if datetime.now(timezone.utc) > expires_at:
        clear_pending_confirmation(sid, action_type)
        raise RestaurantServiceError(
            "The pending confirmation expired. Read the details back again.",
            code="pending_confirmation_expired",
            status=409,
        )

    expected = str(record.get("payload_hash") or "")
    actual = payload_hash(payload)
    if expected != actual:
        raise RestaurantServiceError(
            "The details changed since the last readback. Get a fresh summary "
            "and confirm again after the caller approves it.",
            code="pending_confirmation_mismatch",
            status=409,
        )

    mem = get_call_memory(sid)
    affirmation = str(mem.get("last_turn_affirmation") or "unclear")
    if affirmation != "affirmative":
        raise RestaurantServiceError(
            "The caller has not affirmatively confirmed the readback on this turn.",
            code="affirmation_required",
            status=409,
        )

    try:
        created_turn = int(record.get("created_turn") or 0)
    except (TypeError, ValueError):
        created_turn = 0
    current_turn = current_confirmation_turn(sid)
    if current_turn <= created_turn:
        raise RestaurantServiceError(
            "Same-turn readback and confirm is not allowed. The caller must "
            "hear the full readback, then affirm on a later turn.",
            code="same_turn_confirmation",
            status=409,
        )
