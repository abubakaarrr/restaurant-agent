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
ACTION_UPDATE_CONFIRMED_BOOKING = "update_confirmed_booking"
ACTION_CANCEL_BOOKING = "cancel_booking"
VALID_ACTIONS = frozenset(
    {
        ACTION_CREATE_BOOKING,
        ACTION_CONFIRM_ORDER,
        ACTION_UPDATE_CONFIRMED_BOOKING,
        ACTION_CANCEL_BOOKING,
    }
)

PENDING_TTL = timedelta(minutes=15)

_AFFIRMATIVE_RE = re.compile(
    r"^\s*(?:eh[,. ]+)?(?:"
    r"yes\b|yeah\b|yep\b|yup\b|yap\b|ya(?:[\s,]+ya)?\b|"
    r"sure\b|okay\b|ok\b|correct\b|"
    r"that'?s right|sounds good|go ahead|"
    r"all (?:of )?that(?:'s| is)? correct|that(?:'s| is) (?:all )?correct|"
    r"please do\b|absolutely\b|book that\b|make that change\b"
    r")",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(
    r"(?:"
    r"\bno\b|\bnope\b|not correct|\bwait\b|\bactually\b|"
    r"hold on|never ?mind|don'?t|do not|\bnot\b|wrong|instead|"
    r"\bbut\b.{0,40}\b(?:change|make|move|use)\b|"
    r"\bchange\s+(?:the|my|our|it\s+to)\b|"
    r"\bcorrect\s+(?:the|my|our)\b|"
    r"\b(?:but|and|also|plus)\b.{0,40}\b(?:add|remove|swap|change|increase|decrease)\b"
    r")",
    re.IGNORECASE,
)
_ORDER_ABANDONMENT_RE = re.compile(
    r"\b(?:leave|skip|cancel|forget|drop)\b.{0,50}"
    r"\b(?:pre[ -]?order|food order|the order|that order)\b",
    re.IGNORECASE,
)


def requests_order_abandonment(utterance: str) -> bool:
    """Recognize an explicit request to leave an unconfirmed food order."""
    text = " ".join((utterance or "").casefold().split())
    if re.search(r"\b(?:you|you have|you've)\s+(?:forget|forgot|forgotten|dropped|skipped|cancelled)\b|\b(?:don't|do not|never)\s+(?:forget|drop|skip|cancel|leave)\b", text):
        return False
    return bool(re.fullmatch(r"(?:okay[,. ]+|ok[,. ]+|yes[,. ]+|please\s+|can you\s+|could you\s+|i (?:want|would like) (?:you )?to\s+)*(?:leave|skip|cancel|forget|drop)\s+(?:about\s+)?(?:the|my|our|that)?\s*(?:pre[ -]?order|food order|order)(?:\s+(?:please|for now|instead))?[.!\s]*", text))


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
                "unit_price": round(float(item.get("unit_price") or 0), 2),
                "subtotal": round(float(item.get("subtotal") or 0), 2),
                "notes": str(item.get("notes") or ""),
                "modifiers": item.get("modifiers") or [],
                "removals": item.get("removals") or [],
                "substitutions": item.get("substitutions") or [],
            }
        )
    items.sort(
        key=lambda row: (
            row["item_name"],
            row["quantity"],
            row["notes"],
            json.dumps(row["modifiers"], sort_keys=True, default=str),
        )
    )
    return {
        "order_id": int(summary.get("order_id") or 0),
        "draft_version": int(summary.get("draft_version") or 0),
        "booking_id": int(summary.get("booking_id") or 0),
        "fulfillment": str(summary.get("fulfillment") or ""),
        "fulfillment_details": summary.get("fulfillment_details") or {},
        "order_notes": str(summary.get("order_notes") or ""),
        "allergy_notes": str(summary.get("allergy_notes") or ""),
        "fees": summary.get("fees") or [],
        "total": round(float(summary.get("total") or 0), 2),
        "items": items,
    }


def update_booking_confirmation_payload(
    *,
    booking_id: int,
    date: str = "",
    time: str = "",
    party_size: int = 0,
    preferred_location: str = "",
    seating_preference: str | None = None,
    seating_backup: str | None = None,
    seating_avoid: str | None = None,
    dietary: str | None = None,
    occasion: str | None = None,
    extra_notes: str | None = None,
    notes: str | None = None,
    customer_name: str = "",
    customer_phone: str | None = None,
    require_approval_for_paid_items: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "booking_id": int(booking_id or 0),
        "date": str(date or "").strip(),
        "time": str(time or "").strip(),
        "party_size": int(party_size or 0),
        "preferred_location": str(preferred_location or "").strip(),
        "customer_name": " ".join(str(customer_name or "").split()),
    }
    optional = {
        "seating_preference": seating_preference,
        "seating_backup": seating_backup,
        "seating_avoid": seating_avoid,
        "dietary": dietary,
        "occasion": occasion,
        "extra_notes": extra_notes,
        "notes": notes,
        "customer_phone": customer_phone,
        "require_approval_for_paid_items": require_approval_for_paid_items,
    }
    for key, value in optional.items():
        if value is not None:
            payload[key] = value
    return payload


def cancel_booking_confirmation_payload(
    *,
    booking_id: int,
    customer_name: str = "",
    customer_phone: str = "",
    reason: str = "",
) -> dict[str, Any]:
    from app.config import settings
    from app.security import normalize_caller_phone

    raw_phone = str(customer_phone or "").strip()
    phone = (
        normalize_caller_phone(
            raw_phone,
            default_country_code=settings.default_caller_country_code,
        )
        or raw_phone
    )
    return {
        "booking_id": int(booking_id or 0),
        "customer_name": " ".join(str(customer_name or "").split()),
        "customer_phone": phone,
        "reason": str(reason or "").strip(),
    }


def classify_affirmation(utterance: str) -> Affirmation:
    text = " ".join((utterance or "").split())
    if not text:
        return "unclear"
    has_neg = bool(_NEGATIVE_RE.search(text)) or requests_order_abandonment(text)
    # Consume the whole single-intent answer; leading "yes" is insufficient.
    clean = re.sub(r"[.,!]+", " ", text.casefold()).strip()
    clean = re.sub(r"\s+", " ", clean)
    closed = r"(?:eh )?(?:yes|yeah|yep|yup|yap|ya(?: ya)?|sure|okay|ok|correct|that's right|that is right|sounds good|go ahead|absolutely|please do|book that|make that change|all (?:of )?that(?:'s| is)? correct|that(?:'s| is) (?:all )?correct)(?: (?:please|confirm(?: it| that)?|that's right|that is right|that is correct|that's correct|those details are correct|book it|book that|go ahead|you can make that change|make that change|thank you|thanks|motherfucker))*"
    positive = r"(?:yes|yeah|yep|yup|yap|sure|okay|ok|absolutely|correct)"
    repeated = rf"{positive}(?: {positive}){{1,4}}"
    directed = rf"(?:{positive} )?(?:(?:please )?(?:you can |you may |go ahead and )?)(?:finalize|confirm|book|proceed with) (?:my |the |that )?(?:booking|reservation|table|order)(?: please)?"
    has_aff = '?' not in text and bool(re.fullmatch(closed, clean) or re.fullmatch(repeated, clean) or re.fullmatch(directed, clean))
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


def release_pending_confirmation(
    session_id: str,
    action_type: str,
    confirmation_hash: str,
    *,
    response_id: str = "",
) -> bool:
    sid = resolve_session_id(session_id)
    if not sid or not confirmation_hash:
        return False
    pending = _pending_map(sid)
    record = pending.get(action_type)
    if not isinstance(record, dict) or str(record.get("payload_hash") or "") != confirmation_hash:
        return False
    record = dict(record)
    # A caller's next approval must refer to one audible proposal, never to
    # whichever of several old actions the model happens to choose.
    for other_action, other in list(pending.items()):
        if other_action == action_type or not isinstance(other, dict):
            continue
        other = dict(other)
        other.pop("readback_released", None)
        other.pop("released_turn", None)
        other.pop("released_response_id", None)
        pending[other_action] = other
    record["readback_released"] = True
    record["released_turn"] = current_confirmation_turn(sid)
    if response_id:
        record["released_response_id"] = response_id
    pending[action_type] = record
    update_call_memory(sid, pending_confirmations=pending)
    return True


def active_released_confirmation(session_id: str) -> tuple[str, dict[str, Any]] | None:
    """Return the sole heard proposal eligible for a later caller turn.

    This is the server-owned action pointer used by voice transports. The
    service still verifies the payload hash, turn, scope, and write outcome.
    """
    current_turn = current_confirmation_turn(session_id)
    eligible: list[tuple[str, dict[str, Any]]] = []
    for action, record in _pending_map(session_id).items():
        if action not in VALID_ACTIONS or not isinstance(record, dict):
            continue
        if not record.get("readback_released") or not isinstance(record.get("payload"), dict):
            continue
        try:
            released_turn = int(record.get("released_turn") or 0)
            expires_at = datetime.fromisoformat(str(record.get("expires_at") or ""))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            continue
        if current_turn > released_turn and datetime.now(timezone.utc) <= expires_at:
            eligible.append((action, record))
    return eligible[0] if len(eligible) == 1 else None


def revoke_released_confirmations(session_id: str) -> bool:
    sid = resolve_session_id(session_id)
    if not sid:
        return False
    pending = _pending_map(sid)
    changed = False
    for action, value in list(pending.items()):
        if not isinstance(value, dict) or not value.get("readback_released"):
            continue
        record = dict(value)
        record.pop("readback_released", None)
        record.pop("released_turn", None)
        record.pop("released_response_id", None)
        pending[action] = record
        changed = True
    if changed:
        update_call_memory(sid, pending_confirmations=pending)
    return changed


def pending_state_patch(session_id: str) -> dict[str, Any]:
    """Fields to merge into call_sessions.state for cross-request durability."""
    mem = get_call_memory(session_id)
    pending = _pending_map(session_id)
    pending.pop(ACTION_CANCEL_BOOKING, None)
    return {
        "pending_confirmations": pending,
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
