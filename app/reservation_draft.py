"""Structured reservation draft stored in per-call memory.

Chat history is not a source of truth. Patch named fields, and treat empty
string as an explicit clear so one correction cannot wipe unrelated details.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

DRAFT_STATUS_COLLECTING = "collecting"
DRAFT_STATUS_READY = "ready"
DRAFT_STATUS_CONFIRMED = "confirmed"
DRAFT_STATUS_CANCELLED = "cancelled"

TEXT_FIELDS = (
    "customer_name",
    "customer_phone",
    "date",
    "time",
    "seating_preference",
    "seating_backup",
    "seating_avoid",
    "dietary",
    "occasion",
    "extra_notes",
)

INT_FIELDS = ("party_size", "booking_id")
BOOL_FIELDS = ("require_approval_for_paid_items",)
ALL_PATCH_FIELDS = TEXT_FIELDS + INT_FIELDS + BOOL_FIELDS + ("status",)

_OUTDOOR_TOKENS = ("patio", "outdoor", "outside", "al fresco", "alfresco")
_ANY_LOCATION = {"any", "all", "anywhere", "either", "none", "*"}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"", "0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    return bool(text)


def merge_note_text(existing: str, incoming: str, *, limit: int = 500) -> str:
    """Append a guest note without duplicating or replacing structured notes."""
    existing = " ".join((existing or "").split())
    incoming = " ".join((incoming or "").split())
    if not incoming:
        return existing[:limit]
    if incoming.casefold() in existing.casefold():
        return existing[:limit]
    if not existing:
        return incoming[:limit]
    return f"{existing}; {incoming}"[:limit]


def empty_draft() -> dict[str, Any]:
    return {
        "customer_name": "",
        "customer_phone": "",
        "date": "",
        "time": "",
        "party_size": 0,
        "seating_preference": "",
        "seating_backup": "",
        "seating_avoid": "",
        "dietary": "",
        "occasion": "",
        "extra_notes": "",
        "status": DRAFT_STATUS_COLLECTING,
        "booking_id": 0,
        "require_approval_for_paid_items": False,
    }


def coerce_draft(value: Any) -> dict[str, Any]:
    draft = empty_draft()
    if not isinstance(value, dict):
        return draft
    return patch_draft(draft, value, ignore_unknown=True)


def patch_draft(
    current: dict[str, Any],
    updates: dict[str, Any],
    *,
    ignore_unknown: bool = False,
) -> dict[str, Any]:
    """Apply only provided keys. None means leave unchanged; '' / 0 clears."""
    draft = {**empty_draft(), **(current or {})}
    for key, value in updates.items():
        if value is None:
            continue
        if key not in ALL_PATCH_FIELDS:
            if ignore_unknown:
                continue
            raise ValueError(f"Unknown reservation draft field: {key}")
        if key in INT_FIELDS:
            try:
                draft[key] = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer.") from exc
            if key == "party_size" and draft[key] and not 1 <= draft[key] <= 24:
                raise ValueError("Party size must be between 1 and 24.")
            if draft[key] < 0:
                draft[key] = 0
        elif key in BOOL_FIELDS:
            draft[key] = _coerce_bool(value)
        elif key == "status":
            status = str(value).strip() or DRAFT_STATUS_COLLECTING
            draft[key] = status
        else:
            draft[key] = " ".join(str(value).split())[:300]
    if (
        draft.get("status") != DRAFT_STATUS_CONFIRMED
        and draft.get("status") != DRAFT_STATUS_CANCELLED
        and _has_required_booking_fields(draft)
    ):
        draft["status"] = DRAFT_STATUS_READY
    return draft


def _has_required_booking_fields(draft: dict[str, Any]) -> bool:
    return bool(
        draft.get("customer_name")
        and draft.get("customer_phone")
        and draft.get("date")
        and draft.get("time")
        and int(draft.get("party_size") or 0) >= 1
    )


def compose_notes(draft: dict[str, Any], *, limit: int = 500) -> str:
    parts: list[str] = []
    if draft.get("seating_preference"):
        parts.append(f"seating: {draft['seating_preference']}")
    if draft.get("seating_backup"):
        parts.append(f"backup seating: {draft['seating_backup']}")
    if draft.get("occasion"):
        parts.append(f"occasion: {draft['occasion']}")
    if draft.get("dietary"):
        parts.append(f"dietary: {draft['dietary']}")
    if draft.get("seating_avoid"):
        parts.append(f"avoid: {draft['seating_avoid']}")
    if draft.get("extra_notes"):
        parts.append(str(draft["extra_notes"]))
    return "; ".join(parts)[:limit]


def preferred_location(draft: dict[str, Any] | str) -> str:
    text = draft if isinstance(draft, str) else str(
        (draft or {}).get("seating_preference") or ""
    )
    lowered = text.casefold()
    if any(token in lowered for token in _OUTDOOR_TOKENS):
        return "patio"
    if "private" in lowered:
        return "private"
    if "bar" in lowered or "high top" in lowered or "high-top" in lowered:
        return "bar"
    if "indoor" in lowered or "main" in lowered:
        return "main"
    return ""


def normalize_preferred_location(value: str) -> str:
    """Canonical dining-room filter. 'any' means all rooms, not the draft patio."""
    text = (value or "").strip().casefold()
    if not text or text in _ANY_LOCATION:
        return ""
    if any(token in text for token in _OUTDOOR_TOKENS):
        return "patio"
    if "private" in text:
        return "private"
    if "bar" in text or "high top" in text or "high-top" in text:
        return "bar"
    if text in {"indoor", "inside", "main", "dining", "dining room"}:
        return "main"
    return text


def flatten_draft(draft: dict[str, Any], *, guest_notes: str = "") -> dict[str, Any]:
    """Keys persisted on call_sessions.state and mirrored in call memory."""
    guest = " ".join((guest_notes or draft.get("guest_notes") or "").split())
    structured = compose_notes({k: v for k, v in (draft or {}).items() if k != "guest_notes"})
    notes = merge_note_text(structured, guest)
    payload = {
        "reservation_draft": dict(draft),
        "customer_name": draft.get("customer_name") or "",
        "customer_phone": draft.get("customer_phone") or "",
        "booking_date": draft.get("date") or "",
        "booking_time": draft.get("time") or "",
        "party_size": int(draft.get("party_size") or 0),
        "booking_id": int(draft.get("booking_id") or 0),
        "notes": notes,
        "draft_status": draft.get("status") or DRAFT_STATUS_COLLECTING,
        "require_approval_for_paid_items": bool(
            draft.get("require_approval_for_paid_items")
        ),
    }
    if guest:
        payload["guest_notes"] = guest
    return payload


def draft_from_memory(memory: dict[str, Any]) -> dict[str, Any]:
    nested = memory.get("reservation_draft")
    if isinstance(nested, dict) and nested:
        return coerce_draft(nested)
    return coerce_draft(
        {
            "customer_name": memory.get("customer_name") or "",
            "customer_phone": memory.get("customer_phone") or "",
            "date": memory.get("booking_date") or memory.get("date") or "",
            "time": memory.get("booking_time") or memory.get("time") or "",
            "party_size": memory.get("party_size") or 0,
            "booking_id": memory.get("booking_id") or 0,
            "extra_notes": memory.get("extra_notes") or "",
            "require_approval_for_paid_items": memory.get(
                "require_approval_for_paid_items"
            )
            or False,
            "status": memory.get("draft_status")
            or (
                DRAFT_STATUS_CONFIRMED
                if memory.get("booking_id")
                else DRAFT_STATUS_COLLECTING
            ),
        }
    )


def format_draft_lines(draft: dict[str, Any]) -> list[str]:
    lines = ["Reservation draft (AUTHORITATIVE — do not re-ask these):"]
    if draft.get("booking_id"):
        lines.append(f"- booking_id: {draft['booking_id']}")
    lines.append(f"- status: {draft.get('status') or DRAFT_STATUS_COLLECTING}")
    if draft.get("customer_name"):
        lines.append(f"- name: {draft['customer_name']}")
    if draft.get("customer_phone"):
        lines.append(f"- phone: {draft['customer_phone']}")
    if draft.get("party_size"):
        lines.append(f"- party_size: {draft['party_size']}")
    if draft.get("date") or draft.get("time"):
        lines.append(f"- when: {draft.get('date') or '?'} at {draft.get('time') or '?'}")
    notes = compose_notes(draft)
    if notes:
        lines.append(f"- notes: {notes}")
    if draft.get("require_approval_for_paid_items"):
        lines.append(
            "- require_approval_for_paid_items: true. "
            "Do not attach a priced item until the caller explicitly says yes."
        )
    if draft.get("booking_id"):
        lines.append(
            "- To change time, party size, food, or notes, update this booking. "
            "Do not cancel and recreate it."
        )
    return lines


_NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}


def speak_phone(raw: str) -> str:
    digits = re.sub(r"\D", "", raw or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    if len(digits) == 9:
        return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    return " ".join((raw or "").split())


def speak_time(raw: str) -> str:
    try:
        value = datetime.strptime((raw or "").strip(), "%H:%M")
    except ValueError:
        return raw or ""
    return value.strftime("%I:%M %p").lstrip("0")


def speak_date(raw: str) -> str:
    try:
        value = datetime.strptime((raw or "").strip(), "%Y-%m-%d")
    except ValueError:
        return raw or ""
    return value.strftime("%A, %B ") + str(value.day)


def speak_draft(draft: dict[str, Any]) -> str:
    """Host-stand readback of the same draft facts. Not a form dump."""
    draft = draft or {}
    bits: list[str] = []
    name = str(draft.get("customer_name") or "").strip()
    if name:
        bits.append(name)
    party = int(draft.get("party_size") or 0)
    if party == 1:
        bits.append("just you")
    elif party > 1:
        bits.append(f"{_NUMBER_WORDS.get(party, str(party))} of you")
    date_text = speak_date(str(draft.get("date") or ""))
    time_text = speak_time(str(draft.get("time") or ""))
    if date_text and time_text:
        bits.append(f"{date_text} at {time_text}")
    elif date_text or time_text:
        bits.append(date_text or time_text)
    seating = str(draft.get("seating_preference") or "").strip()
    if seating:
        bits.append(seating)
    dietary = str(draft.get("dietary") or "").strip()
    if dietary:
        bits.append(f"one of you {dietary}")
    occasion = str(draft.get("occasion") or "").strip()
    if occasion:
        bits.append(occasion)
    extra = str(draft.get("extra_notes") or "").strip()
    if extra:
        bits.append(extra)
    phone = speak_phone(str(draft.get("customer_phone") or ""))
    if phone:
        bits.append(f"callback {phone}")
    picture = ", ".join(bit for bit in bits if bit)
    if not picture:
        return "Nothing saved yet."
    if draft.get("booking_id"):
        return (
            f"You're booked: {picture}. "
            f"Reference {draft['booking_id']}."
        )
    return f"You're down as {picture}. Not booked yet."
