"""Per-call structured memory — survives across LLM turns.

Tool results (booking IDs, names, etc.) are not kept in chat history between
turns. This module is the source of truth for "what already happened on this call"
so pre-orders can reuse the reservation without re-asking.
"""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from typing import Any

from app.db_pool import get_pool
from app.reservation_draft import (
    DRAFT_STATUS_CANCELLED,
    DRAFT_STATUS_CONFIRMED,
    compose_notes,
    draft_from_memory,
    empty_draft,
    flatten_draft,
    format_draft_lines,
    merge_note_text,
    patch_draft,
)

# session_id → memory dict
_memory: dict[str, dict[str, Any]] = {}

# Set by runner for the duration of a turn so tools can fall back if the
# model forgets to pass session_id.
_current_session_id: ContextVar[str] = ContextVar("current_session_id", default="")
_current_action_scope: ContextVar[str] = ContextVar("current_action_scope", default="")


def set_current_session_id(session_id: str):
    """Bind this turn's session id; returns a token for reset."""
    return _current_session_id.set(session_id or "")


def reset_current_session_id(token) -> None:
    _current_session_id.reset(token)


def set_current_action_scope(scope: str):
    """Bind a stable identifier for all retries of one caller turn."""
    return _current_action_scope.set(scope or "")


def reset_current_action_scope(token) -> None:
    _current_action_scope.reset(token)


def make_idempotency_key(action: str, payload: dict[str, Any]) -> str:
    """Derive a stable rollback-tool key without trusting the language model."""
    scope = _current_action_scope.get() or resolve_session_id() or "unknown"
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(f"{scope}:{action}:{encoded}".encode("utf-8")).hexdigest()
    return f"legacy-{digest}"


def resolve_session_id(session_id: str = "") -> str:
    """Prefer explicit tool arg; else the turn's contextvar."""
    if session_id and session_id != "unknown":
        return session_id
    ctx = _current_session_id.get()
    return ctx if ctx and ctx != "unknown" else ""


def get_call_memory(session_id: str) -> dict[str, Any]:
    sid = resolve_session_id(session_id)
    return dict(_memory.get(sid) or {})


def get_reservation_draft(session_id: str) -> dict[str, Any]:
    return draft_from_memory(get_call_memory(session_id))


def _store_flattened(sid: str, draft: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    current = _memory.setdefault(sid, {})
    extra = extra or {}
    if extra.get("guest_notes") is not None:
        guest = str(extra.get("guest_notes") or "")
    else:
        guest = str(current.get("guest_notes") or "")
    current.update(flatten_draft(draft, guest_notes=guest))
    for key, value in extra.items():
        if value is None or key in {"guest_notes", "reservation_draft", "notes"}:
            continue
        current[key] = value
    return dict(current)


def update_reservation_draft(
    session_id: str,
    updates: dict[str, Any] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Patch reservation draft fields. Empty string clears a text field.

    Refuses once a confirmed booking exists — use update_confirmed_booking instead.
    """
    sid = resolve_session_id(session_id)
    if not sid:
        return empty_draft()
    current = get_reservation_draft(sid)
    if (
        int(current.get("booking_id") or 0) > 0
        and str(current.get("status") or "") == DRAFT_STATUS_CONFIRMED
    ):
        raise ValueError(
            "This reservation is already confirmed (booking_id set). "
            "Use update_confirmed_booking with a full read-back and explicit yes — "
            "do not call update_reservation_draft."
        )
    payload = {**(updates or {}), **fields}
    if "party_size" in payload and payload["party_size"] is not None:
        try:
            new_party = int(payload["party_size"])
        except (TypeError, ValueError):
            new_party = 0
        current_party = int(current.get("party_size") or 0)
        if new_party > 0 and current_party > 0 and new_party != current_party:
            from app.availability_offer import require_fresh_availability_for_party_change

            require_fresh_availability_for_party_change(
                sid,
                date=str(payload.get("date") or current.get("date") or ""),
                time=str(payload.get("time") or current.get("time") or ""),
                party_size=new_party,
                preferred_location=str(current.get("seating_preference") or ""),
            )
    draft = patch_draft(current, payload)
    _store_flattened(sid, draft)
    return dict(draft)


async def hydrate_call_memory(session_id: str) -> dict[str, Any]:
    """Restore structured transaction context without persisting transcripts."""
    sid = resolve_session_id(session_id)
    if not sid or _memory.get(sid):
        return get_call_memory(sid)
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            session = await conn.fetchrow(
                "SELECT caller_phone, state FROM call_sessions WHERE session_id = $1",
                sid,
            )
            order = await conn.fetchrow(
                """
                SELECT booking_id, customer_name, customer_phone
                FROM orders
                WHERE session_id = $1 AND status IN ('pending', 'confirmed')
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC
                LIMIT 1
                """,
                sid,
            )
    except Exception:
        return {}

    restored: dict[str, Any] = {}
    if session:
        state = session["state"]
        if isinstance(state, str):
            try:
                state = json.loads(state)
            except json.JSONDecodeError:
                state = {}
        if isinstance(state, dict):
            restored.update(state)
        if session["caller_phone"]:
            restored.setdefault("customer_phone", session["caller_phone"])
    if order:
        if order["booking_id"]:
            restored.setdefault("booking_id", order["booking_id"])
        if order["customer_name"]:
            restored.setdefault("customer_name", order["customer_name"])
        if order["customer_phone"]:
            restored.setdefault("customer_phone", order["customer_phone"])
    if restored:
        draft = draft_from_memory(restored)
        guest = str(restored.get("guest_notes") or "")
        if not guest:
            combined = str(restored.get("notes") or "")
            structured = compose_notes(draft)
            if combined and structured and combined.casefold().startswith(structured.casefold()):
                guest = combined[len(structured) :].lstrip(" ;")
            elif combined and combined.casefold() != structured.casefold():
                guest = combined
        extra = {
            key: value
            for key, value in restored.items()
            if key not in flatten_draft(empty_draft())
        }
        extra["guest_notes"] = guest
        _store_flattened(sid, draft, extra)
        if restored.get("table_number"):
            _memory[sid]["table_number"] = restored["table_number"]
        if restored.get("table_location"):
            _memory[sid]["table_location"] = restored["table_location"]
    return get_call_memory(sid)


def update_call_memory(session_id: str, **fields: Any) -> dict[str, Any]:
    """Merge fields into this call's memory. Ignores empty session_id / None values.

    Empty strings and 0 are skipped so accidental blanks do not wipe known
    facts. Use update_reservation_draft to clear a named draft field.
    """
    sid = resolve_session_id(session_id)
    if not sid:
        return {}
    current = _memory.setdefault(sid, {})
    incoming_notes = fields.get("notes")
    incoming_guest = fields.get("guest_notes")
    if incoming_guest:
        current["guest_notes"] = merge_note_text(
            str(current.get("guest_notes") or ""),
            str(incoming_guest),
        )
    elif incoming_notes:
        current["guest_notes"] = merge_note_text(
            str(current.get("guest_notes") or ""),
            str(incoming_notes),
        )
    for key, value in fields.items():
        if key in {"notes", "guest_notes"}:
            continue
        if value is None or value == "" or value == 0:
            continue
        current[key] = value
    if any(
        key in fields
        for key in (
            "customer_name",
            "customer_phone",
            "booking_date",
            "booking_time",
            "party_size",
            "booking_id",
            "notes",
            "guest_notes",
        )
    ):
        draft = draft_from_memory(current)
        current.update(
            flatten_draft(draft, guest_notes=str(current.get("guest_notes") or ""))
        )
    return dict(current)


def set_active_booking(
    session_id: str,
    *,
    booking_id: int,
    customer_name: str,
    customer_phone: str = "",
    party_size: int = 0,
    date: str = "",
    time: str = "",
    table_number: int | None = None,
    table_location: str = "",
    notes: str = "",
) -> None:
    sid = resolve_session_id(session_id)
    if not sid:
        return
    draft = patch_draft(
        get_reservation_draft(sid),
        {
            "booking_id": booking_id,
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "party_size": party_size,
            "date": date,
            "time": time,
            "status": "confirmed",
        },
    )
    if notes and not compose_notes(draft):
        draft["extra_notes"] = notes
    extra = {}
    if table_number:
        extra["table_number"] = table_number
    if table_location:
        extra["table_location"] = table_location
    _store_flattened(sid, draft, extra)


def clear_call_memory(session_id: str) -> None:
    sid = resolve_session_id(session_id)
    if sid:
        _memory.pop(sid, None)


def clear_active_booking(session_id: str) -> None:
    """Remove reservation fields but keep name/phone if present."""
    sid = resolve_session_id(session_id)
    mem = _memory.get(sid)
    if not mem:
        return
    draft = get_reservation_draft(sid)
    name = draft.get("customer_name") or mem.get("customer_name") or ""
    phone = draft.get("customer_phone") or mem.get("customer_phone") or ""
    for key in (
        "booking_id",
        "booking_date",
        "booking_time",
        "table_number",
        "table_location",
        "party_size",
        "reservation_draft",
        "draft_status",
        "notes",
    ):
        mem.pop(key, None)
    if name or phone:
        _store_flattened(
            sid,
            patch_draft(
                empty_draft(),
                {
                    "customer_name": name,
                    "customer_phone": phone,
                    "status": DRAFT_STATUS_CANCELLED,
                },
            ),
        )


def format_memory_for_prompt(session_id: str) -> str:
    """Human-readable block injected into the system prompt every turn."""
    mem = get_call_memory(session_id)
    if not mem:
        return (
            "Active call memory: none yet.\n"
            "If the caller has not booked this call, treat orders as pickup "
            "(booking_id = 0) and ask for a name when ordering."
        )

    draft = draft_from_memory(mem)
    lines = format_draft_lines(draft)
    guest = str(mem.get("guest_notes") or "")
    if guest:
        lines.append(f"- guest_notes: {guest}")
    if mem.get("table_number"):
        loc = mem.get("table_location") or ""
        lines.append(f"- table: {mem['table_number']}" + (f" ({loc})" if loc else ""))

    if draft.get("booking_id"):
        lines.insert(
            0,
            "- BOOKING ALREADY CONFIRMED. Do not call create_booking again. "
            "If they ask to confirm, read back the existing reference, table, and time.",
        )
        lines.append(
            "- Pre-order / order on this call: DO NOT ask for name again. "
            f"Always pass booking_id={draft['booking_id']} and "
            f"customer_name=\"{draft.get('customer_name', '')}\" to add_order_item. "
            "Changing food on a confirmed order requires caller_confirmed=true."
        )
    elif draft.get("customer_name"):
        lines.append(
            f"- Name already known: use customer_name=\"{draft['customer_name']}\" "
            "for orders; do not ask again."
        )

    return "\n".join(lines)
