"""Typed restaurant business operations for every voice transport.

The managed Retell flow and the legacy LangGraph adapter both call this module.
All externally visible write operations are guarded by an idempotency ledger and
the rollout write flag.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.db_pool import get_pool
from app.knowledge_search import (
    format_knowledge_hits,
    normalize_question,
    search_faq_rows,
    search_static_knowledge,
)
from app.reservation_draft import (
    DRAFT_STATUS_CANCELLED,
    DRAFT_STATUS_CONFIRMED,
    compose_notes,
    coerce_draft,
    flatten_draft,
    merge_note_text,
    normalize_preferred_location,
    patch_draft,
    preferred_location as draft_preferred_location,
)
from app.restaurant_settings import HOURS_UNCONFIRMED_NOTE, load_restaurant_settings
from app.security import canonical_request_hash, normalize_caller_phone
from app.pending_confirmation import (
    ACTION_CONFIRM_ORDER,
    ACTION_CREATE_BOOKING,
    booking_confirmation_payload,
    clear_pending_confirmation,
    order_confirmation_payload,
    pending_state_patch,
    register_pending_confirmation,
    require_pending_confirmation,
)
from app.turn_evidence import (
    current_turn,
    record_availability,
    record_order_summary,
)


JsonDict = dict[str, Any]
T = TypeVar("T", bound=JsonDict)


class RestaurantServiceError(Exception):
    """Expected domain error safe to return as a short API message."""

    def __init__(self, message: str, *, code: str = "invalid_request", status: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class WritesDisabledError(RestaurantServiceError):
    def __init__(self) -> None:
        super().__init__(
            "Live booking and order writes are disabled during pilot validation.",
            code="writes_disabled",
            status=503,
        )


def _normalized_text(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _combine_notes(existing: str, incoming: str, *, limit: int = 500) -> str:
    return merge_note_text(existing, incoming, limit=limit)


def format_menu_price(item: Mapping[str, Any]) -> str:
    """Spoken/tool price string. Estimated rows must not look confirmed."""
    price = float(item.get("price") or 0)
    if item.get("price_estimated"):
        return f"about ${price:.2f} (estimated, pending venue confirmation)"
    return f"${price:.2f}"


def format_availability_speech(result: Mapping[str, Any]) -> str:
    nonce = result.get("availability_nonce") or ""
    if result.get("available"):
        tables = ", ".join(
            f"table {row['table_number']} ({row['capacity']} seats, {row['location']})"
            for row in result.get("tables") or []
        )
        return f"Available: {tables}. availability_nonce={nonce}."
    alternatives = result.get("alternatives") or []
    alt_text = "; ".join(
        (
            f"{row.get('kind') or 'option'}: "
            f"{row.get('location') or 'table'} {row.get('date')} at {row['display_time']}"
            + (
                f" (table {row['table_number']}, {row.get('capacity')} seats)"
                if row.get("table_number")
                else ""
            )
        )
        for row in alternatives
    )
    preferred = result.get("preferred_location") or "requested"
    if result.get("impossible_at_location"):
        max_seats = result.get("max_seats_at_location") or 0
        party = result.get("party_size")
        text = (
            f"IMPOSSIBLE: no {preferred} table seats {party}. "
            f"Largest {preferred} table seats {max_seats}. "
            f"Do not offer other {preferred} times."
        )
        if alt_text:
            text += f" Alternatives (not substituted): {alt_text}."
        return f"{text} availability_nonce={nonce}."
    if alt_text:
        return (
            f"Requested {preferred} slot unavailable. "
            f"Do not tell the caller this time is reserved. "
            f"Alternatives (not substituted): {alt_text}. "
            f"availability_nonce={nonce}."
        )
    return (
        f"Requested slot unavailable and no nearby alternative was found. "
        f"Do not tell the caller this time is reserved. "
        f"availability_nonce={nonce}."
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class RestaurantService:
    """Database-backed, provider-neutral restaurant operations."""

    def __init__(self) -> None:
        self._seating_limits: JsonDict | None = None

    @staticmethod
    def _require_call_id(call_id: str) -> str:
        call_id = call_id.strip()
        if not call_id or len(call_id) > 200:
            raise RestaurantServiceError("A valid call_id is required.", code="invalid_call_id")
        return call_id

    @staticmethod
    def _require_idempotency_key(value: str) -> str:
        value = value.strip()
        if not value or len(value) < 8 or len(value) > 200:
            raise RestaurantServiceError(
                "A unique Idempotency-Key (8-200 characters) is required.",
                code="invalid_idempotency_key",
            )
        return value

    @staticmethod
    def _validate_name(value: str, field: str = "customer_name") -> str:
        value = " ".join(value.split())
        if not 1 <= len(value) <= 100:
            raise RestaurantServiceError(
                f"{field} must contain 1-100 characters.",
                code=f"invalid_{field}",
            )
        return value

    @staticmethod
    def _validate_phone(value: str, *, required: bool = False) -> str:
        value = value.strip()
        if not value and not required:
            return ""
        normalized = normalize_caller_phone(
            value,
            default_country_code=settings.default_caller_country_code,
        )
        if not normalized:
            raise RestaurantServiceError(
                "I need the full callback number the caller already uses, for example 03098121804 or 415 555 0123.",
                code="invalid_phone",
            )
        return normalized

    @staticmethod
    def _validate_party_size(value: int) -> int:
        if not 1 <= value <= 12:
            raise RestaurantServiceError(
                "Party size must be between 1 and 12.",
                code="invalid_party_size",
            )
        return value

    @staticmethod
    def _parse_booking_datetime(date: str, time: str) -> datetime:
        try:
            value = datetime.fromisoformat(f"{date}T{time}")
        except ValueError as exc:
            raise RestaurantServiceError(
                "Date and time must use YYYY-MM-DD and HH:MM.",
                code="invalid_datetime",
            ) from exc
        try:
            timezone_info = ZoneInfo(settings.restaurant_timezone)
        except ZoneInfoNotFoundError:
            timezone_info = timezone.utc
        now_local = datetime.now(timezone_info).replace(tzinfo=None)
        if value < now_local:
            raise RestaurantServiceError(
                "The requested reservation time is in the past.",
                code="past_booking",
            )
        if value > now_local + timedelta(days=366):
            raise RestaurantServiceError(
                "Reservations can only be made up to one year ahead.",
                code="booking_too_far_ahead",
            )
        return value

    @staticmethod
    def _ensure_writes_enabled() -> None:
        if not settings.voice_live_writes_enabled:
            raise WritesDisabledError()

    @staticmethod
    async def _merge_session_state(
        conn: Any,
        call_id: str,
        patch: Mapping[str, Any],
        *,
        caller_phone: str = "",
    ) -> None:
        payload = {key: value for key, value in dict(patch).items() if value is not None}
        await conn.execute(
            """
            INSERT INTO call_sessions (session_id, caller_phone, state)
            VALUES ($1, $2, $3::jsonb)
            ON CONFLICT (session_id) DO UPDATE
            SET caller_phone = CASE
                    WHEN EXCLUDED.caller_phone <> '' THEN EXCLUDED.caller_phone
                    ELSE call_sessions.caller_phone
                END,
                state = call_sessions.state || EXCLUDED.state,
                updated_at = NOW()
            """,
            call_id,
            caller_phone,
            json.dumps(payload, default=str),
        )

    @staticmethod
    def _coerce_state(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        return dict(value) if isinstance(value, dict) else {}

    async def persist_call_state(
        self,
        call_id: str,
        patch: Mapping[str, Any],
        *,
        caller_phone: str = "",
    ) -> None:
        call_id = self._require_call_id(call_id)
        pool = await get_pool()
        async with pool.acquire() as conn:
            await self._merge_session_state(
                conn, call_id, patch, caller_phone=caller_phone
            )

    def _table_select_sql(self, *, for_update: bool, require_location_match: bool) -> str:
        lock = "FOR UPDATE OF t SKIP LOCKED" if for_update else ""
        location_filter = "AND ($5 = '' OR t.location = $5)" if require_location_match else ""
        return f"""
            SELECT t.id, t.table_number, t.capacity, t.location
            FROM tables t
            WHERE t.capacity >= $1
              {location_filter}
              AND NOT EXISTS (
                SELECT 1
                FROM bookings b
                WHERE b.table_id = t.id
                  AND b.status = 'confirmed'
                  AND ($4::int = 0 OR b.id <> $4)
                  AND b.booked_at < $3
                  AND b.booked_at + (b.duration_mins * interval '1 minute') > $2
              )
            ORDER BY
              CASE WHEN $5 <> '' AND t.location = $5 THEN 0 ELSE 1 END,
              t.capacity ASC,
              t.table_number ASC
            {lock}
            LIMIT $6
        """

    async def _idempotent_write(
        self,
        *,
        action: str,
        idempotency_key: str,
        call_id: str,
        payload: Mapping[str, Any],
        operation: Callable[[Any], Awaitable[T]],
    ) -> tuple[T, bool]:
        """Run a DB mutation and its ledger update in one transaction."""
        self._ensure_writes_enabled()
        key = self._require_idempotency_key(idempotency_key)
        call_id = self._require_call_id(call_id)
        request_hash = canonical_request_hash(payload)

        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                inserted = await conn.fetchrow(
                    """
                    INSERT INTO voice_action_idempotency
                        (action, idempotency_key, call_id, request_hash, status)
                    VALUES ($1, $2, $3, $4, 'processing')
                    ON CONFLICT (action, idempotency_key) DO NOTHING
                    RETURNING id
                    """,
                    action,
                    key,
                    call_id,
                    request_hash,
                )
                if not inserted:
                    existing = await conn.fetchrow(
                        """
                        SELECT request_hash, status, response
                        FROM voice_action_idempotency
                        WHERE action = $1 AND idempotency_key = $2
                        """,
                        action,
                        key,
                    )
                    if not existing:
                        raise RestaurantServiceError(
                            "The previous action is still resolving; retry shortly.",
                            code="idempotency_in_progress",
                            status=409,
                        )
                    if existing["request_hash"] != request_hash:
                        raise RestaurantServiceError(
                            "That idempotency key was already used for different inputs.",
                            code="idempotency_conflict",
                            status=409,
                        )
                    if existing["status"] == "completed" and existing["response"] is not None:
                        response = _json_value(existing["response"])
                        if not isinstance(response, dict):
                            raise RestaurantServiceError(
                                "Stored action result is invalid.",
                                code="idempotency_corrupt",
                                status=500,
                            )
                        return response, True
                    raise RestaurantServiceError(
                        "The previous action is still processing; retry shortly.",
                        code="idempotency_in_progress",
                        status=409,
                    )

                result = await operation(conn)
                await conn.execute(
                    """
                    UPDATE voice_action_idempotency
                    SET status = 'completed', response = $3::jsonb, completed_at = NOW()
                    WHERE action = $1 AND idempotency_key = $2
                    """,
                    action,
                    key,
                    json.dumps(result, default=str),
                )
                return result, False

    async def get_available_tables(
        self,
        date: str,
        time: str,
        party_size: int,
        *,
        limit: int = 5,
        preferred_location: str = "",
        exclude_booking_id: int = 0,
        conn: Any | None = None,
        require_location_match: bool | None = None,
    ) -> list[JsonDict]:
        dt = self._parse_booking_datetime(date, time)
        party_size = self._validate_party_size(party_size)
        window_start = dt - timedelta(minutes=30)
        window_end = dt + timedelta(minutes=90)
        location = (preferred_location or "").strip().casefold()
        if require_location_match is None:
            require_location_match = bool(location)
        sql = self._table_select_sql(
            for_update=conn is not None,
            require_location_match=require_location_match,
        )
        args = (
            party_size,
            window_start,
            window_end,
            int(exclude_booking_id or 0),
            location,
            limit,
        )
        if conn is not None:
            rows = await conn.fetch(sql, *args)
            return [dict(row) for row in rows]
        pool = await get_pool()
        async with pool.acquire() as acquired:
            rows = await acquired.fetch(sql, *args)
        return [dict(row) for row in rows]

    async def seating_limits(self, *, conn: Any | None = None) -> JsonDict:
        if self._seating_limits is not None and conn is None:
            return dict(self._seating_limits)
        sql = """
            SELECT location, MAX(capacity) AS max_seats, COUNT(*) AS tables
            FROM tables
            GROUP BY location
        """
        if conn is not None:
            rows = await conn.fetch(sql)
        else:
            pool = await get_pool()
            async with pool.acquire() as acquired:
                rows = await acquired.fetch(sql)
        by_location = {
            str(row["location"] or "").casefold(): int(row["max_seats"] or 0)
            for row in rows
            if row["location"]
        }
        payload = {
            "max_party_phone": 12,
            "max_seats_by_location": by_location,
            "largest_table": max(by_location.values()) if by_location else 0,
        }
        if conn is None:
            self._seating_limits = payload
        return dict(payload)

    async def check_availability(
        self,
        date: str,
        time: str,
        party_size: int,
        *,
        preferred_location: str = "",
        exclude_booking_id: int = 0,
    ) -> JsonDict:
        preferred = normalize_preferred_location(preferred_location)
        limits = await self.seating_limits()
        max_at_location = int((limits.get("max_seats_by_location") or {}).get(preferred) or 0)
        impossible_at_location = bool(preferred and max_at_location and party_size > max_at_location)
        tables = []
        if not impossible_at_location:
            tables = await self.get_available_tables(
                date,
                time,
                party_size,
                preferred_location=preferred,
                exclude_booking_id=exclude_booking_id,
                require_location_match=bool(preferred),
            )
        alternatives: list[JsonDict] = []
        if not tables:
            alternatives = await self._availability_alternatives(
                date,
                time,
                party_size,
                preferred=preferred,
                exclude_booking_id=exclude_booking_id,
                skip_preferred_times=impossible_at_location,
            )
        result = {
            "available": bool(tables),
            "date": date,
            "time": time,
            "party_size": party_size,
            "preferred_location": preferred,
            "availability_nonce": secrets.token_hex(8),
            "impossible_at_location": impossible_at_location,
            "max_seats_at_location": max_at_location if preferred else int(limits.get("largest_table") or 0),
            "tables": [
                {
                    "table_number": row["table_number"],
                    "capacity": row["capacity"],
                    "location": row["location"],
                }
                for row in tables
            ],
            "alternatives": alternatives,
        }
        record_availability(result)
        return result

    async def _availability_alternatives(
        self,
        date: str,
        time: str,
        party_size: int,
        *,
        preferred: str,
        exclude_booking_id: int,
        skip_preferred_times: bool = False,
    ) -> list[JsonDict]:
        """At most two explicit alternatives. Never silent substitutes.

        Nearby times at the requested location can fill both slots and hide
        other dining rooms. When a preferred location is full, keep one
        location alternative if another room is free at the requested time.
        If the party cannot physically fit that location, skip time hunting.
        """
        requested = self._parse_booking_datetime(date, time)
        time_alts: list[JsonDict] = []
        if preferred and not skip_preferred_times:
            for minutes in (-60, 60, -120, 120, -30, 30):
                candidate = requested + timedelta(minutes=minutes)
                candidate_tables = await self.get_available_tables(
                    candidate.date().isoformat(),
                    candidate.strftime("%H:%M"),
                    party_size,
                    preferred_location=preferred,
                    exclude_booking_id=exclude_booking_id,
                    require_location_match=bool(preferred),
                )
                if candidate_tables:
                    first = candidate_tables[0]
                    time_alts.append(
                        {
                            "kind": "time",
                            "date": candidate.date().isoformat(),
                            "time": candidate.strftime("%H:%M"),
                            "display_time": candidate.strftime("%I:%M %p").lstrip("0"),
                            "location": first.get("location") or preferred,
                            "table_number": first.get("table_number"),
                            "capacity": first.get("capacity"),
                        }
                    )
                if len(time_alts) == 2:
                    break

        location_alts: list[JsonDict] = []
        if preferred:
            other_tables = await self.get_available_tables(
                date,
                time,
                party_size,
                preferred_location="",
                exclude_booking_id=exclude_booking_id,
                require_location_match=False,
            )
            display_time = requested.strftime("%I:%M %p").lstrip("0")
            seen: set[str] = set()
            for row in other_tables:
                location = str(row.get("location") or "").casefold()
                if not location or location == preferred or location in seen:
                    continue
                seen.add(location)
                location_alts.append(
                    {
                        "kind": "location",
                        "date": date,
                        "time": time,
                        "display_time": display_time,
                        "location": row["location"],
                        "table_number": row["table_number"],
                        "capacity": row["capacity"],
                    }
                )
                if len(location_alts) == 2:
                    break

        if time_alts and location_alts:
            return [time_alts[0], location_alts[0]]
        return (time_alts or location_alts)[:2]

    async def create_booking(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        customer_name: str,
        customer_phone: str,
        date: str,
        time: str,
        party_size: int,
        notes: str = "",
        confirmed: bool,
        preferred_location: str = "",
    ) -> JsonDict:
        if confirmed is not True:
            raise RestaurantServiceError(
                "The caller must explicitly confirm every booking field first.",
                code="confirmation_required",
                status=409,
            )
        call_id = self._require_call_id(call_id)
        name = self._validate_name(customer_name)
        phone = self._validate_phone(customer_phone, required=True)
        party_size = self._validate_party_size(party_size)
        dt = self._parse_booking_datetime(date, time)
        notes = notes.strip()[:500]
        confirmation_payload = booking_confirmation_payload(
            customer_name=name,
            customer_phone=phone,
            date=date,
            time=time,
            party_size=party_size,
            notes=notes,
        )
        payload = {
            "call_id": call_id,
            "customer_name": name,
            "customer_phone": phone,
            "date": date,
            "time": time,
            "party_size": party_size,
            "notes": notes,
            "preferred_location": preferred_location or "",
        }

        async def operation(conn: Any) -> JsonDict:
            session = await conn.fetchrow(
                "SELECT state FROM call_sessions WHERE session_id = $1",
                call_id,
            )
            session_state = self._coerce_state(session["state"] if session else {})
            existing_draft = coerce_draft(
                session_state.get("reservation_draft") or session_state
            )
            existing_id = 0
            try:
                existing_id = int(
                    existing_draft.get("booking_id") or session_state.get("booking_id") or 0
                )
            except (TypeError, ValueError):
                existing_id = 0
            if existing_id:
                existing = await conn.fetchrow(
                    """
                    SELECT b.id, b.customer_name, b.customer_phone, b.party_size,
                           b.notes, b.booked_at, t.table_number, t.location
                    FROM bookings b
                    JOIN tables t ON t.id = b.table_id
                    WHERE b.id = $1 AND b.status = 'confirmed'
                    """,
                    existing_id,
                )
                if existing:
                    booked_at = existing["booked_at"]
                    return {
                        "created": False,
                        "already_confirmed": True,
                        "booking_id": existing["id"],
                        "customer_name": existing["customer_name"],
                        "customer_phone": existing["customer_phone"],
                        "party_size": existing["party_size"],
                        "date": booked_at.date().isoformat(),
                        "time": booked_at.strftime("%H:%M"),
                        "table_number": existing["table_number"],
                        "location": existing["location"],
                        "notes": existing["notes"] or "",
                    }
            require_pending_confirmation(
                call_id, ACTION_CREATE_BOOKING, confirmation_payload
            )
            location_pref = normalize_preferred_location(preferred_location)
            if location_pref:
                limits = await self.seating_limits(conn=conn)
                max_here = int(
                    (limits.get("max_seats_by_location") or {}).get(location_pref) or 0
                )
                if max_here and party_size > max_here:
                    raise RestaurantServiceError(
                        f"No {location_pref} table seats {party_size}. "
                        f"Largest {location_pref} table seats {max_here}. "
                        "Offer another room or a smaller party.",
                        code="capacity_unavailable",
                        status=409,
                    )
            tables = await self.get_available_tables(
                date,
                time,
                party_size,
                limit=1,
                preferred_location=location_pref,
                conn=conn,
                require_location_match=bool(location_pref),
            )
            table = tables[0] if tables else None
            if not table:
                raise RestaurantServiceError(
                    "That slot is no longer available. Offer a new time.",
                    code="slot_unavailable",
                    status=409,
                )
            approval_required = bool(
                existing_draft.get("require_approval_for_paid_items")
            )
            pending = await conn.fetchrow(
                """
                SELECT id FROM orders
                WHERE session_id = $1 AND status = 'pending'
                ORDER BY id DESC LIMIT 1
                """,
                call_id,
            )
            if pending and current_turn() is not None and not current_turn().last_order_summary:
                raise RestaurantServiceError(
                    "Read the pending order total before confirming the reservation.",
                    code="readback_required",
                    status=409,
                )
            row = await conn.fetchrow(
                """
                INSERT INTO bookings
                    (customer_name, customer_phone, table_id, booked_at, party_size,
                     notes, require_approval_for_paid_items)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                name,
                phone,
                table["id"],
                dt,
                party_size,
                notes,
                approval_required,
            )
            draft = patch_draft(
                existing_draft,
                {
                    "customer_name": name,
                    "customer_phone": phone,
                    "date": date,
                    "time": time,
                    "party_size": party_size,
                    "booking_id": row["id"],
                    "status": DRAFT_STATUS_CONFIRMED,
                    "extra_notes": notes or existing_draft.get("extra_notes") or "",
                    "require_approval_for_paid_items": approval_required,
                },
            )
            guest_notes = str(session_state.get("guest_notes") or "")
            state = flatten_draft(draft, guest_notes=guest_notes)
            state["table_number"] = table["table_number"]
            state["table_location"] = table["location"]
            clear_pending_confirmation(call_id, ACTION_CREATE_BOOKING)
            state.update(pending_state_patch(call_id))
            await self._merge_session_state(conn, call_id, state, caller_phone=phone)
            await conn.execute(
                """
                UPDATE orders
                SET booking_id = $1
                WHERE session_id = $2
                  AND booking_id IS NULL
                  AND status IN ('pending', 'confirmed')
                """,
                row["id"],
                call_id,
            )
            attached_order = None
            order_row = await conn.fetchrow(
                """
                SELECT id FROM orders
                WHERE session_id = $1 AND status IN ('pending', 'confirmed')
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC
                LIMIT 1
                """,
                call_id,
            )
            if order_row:
                attached_order = await self._order_summary_with_conn(conn, order_row["id"])
            payload_out = {
                "created": True,
                "booking_id": row["id"],
                "customer_name": name,
                "customer_phone": phone,
                "party_size": party_size,
                "date": date,
                "time": time,
                "table_number": table["table_number"],
                "location": table["location"],
                "notes": notes,
                "require_approval_for_paid_items": approval_required,
            }
            if attached_order:
                payload_out["order"] = attached_order
                payload_out["fulfillment"] = attached_order.get("fulfillment")
            return payload_out

        result, replayed = await self._idempotent_write(
            action="create_booking",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def update_confirmed_booking(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        booking_id: int,
        confirmed: bool,
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
        require_approval_for_paid_items: bool | None = None,
    ) -> JsonDict:
        """Change time, party size, name, or notes on an existing booking. Never creates a new row."""
        if confirmed is not True:
            raise RestaurantServiceError(
                "The caller must explicitly confirm the reservation change first.",
                code="confirmation_required",
                status=409,
            )
        call_id = self._require_call_id(call_id)
        if booking_id <= 0:
            raise RestaurantServiceError("A valid booking_id is required.")
        note_updates = {
            key: value
            for key, value in {
                "seating_preference": seating_preference,
                "seating_backup": seating_backup,
                "seating_avoid": seating_avoid,
                "dietary": dietary,
                "occasion": occasion,
                "extra_notes": extra_notes if extra_notes is not None else notes,
            }.items()
            if value is not None
        }
        changing_slot = bool(date or time or party_size)
        new_name = customer_name.strip()
        if (
            not changing_slot
            and not note_updates
            and not new_name
            and require_approval_for_paid_items is None
        ):
            raise RestaurantServiceError(
                "Provide a new date, time, party size, name, or note field to update.",
                code="empty_update",
            )
        payload = {
            "call_id": call_id,
            "booking_id": booking_id,
            "date": date,
            "time": time,
            "party_size": party_size,
            "preferred_location": preferred_location or "",
            "customer_name": new_name,
            "require_approval_for_paid_items": require_approval_for_paid_items,
            **note_updates,
        }

        async def operation(conn: Any) -> JsonDict:
            row = await conn.fetchrow(
                """
                SELECT b.id, b.customer_name, b.customer_phone, b.booked_at,
                       b.party_size, b.status, b.notes, b.table_id,
                       t.table_number, t.location
                FROM bookings b
                LEFT JOIN tables t ON t.id = b.table_id
                WHERE b.id = $1
                FOR UPDATE OF b
                """,
                booking_id,
            )
            if not row:
                raise RestaurantServiceError(
                    "The booking was not found.", code="booking_not_found", status=404
                )
            if row["status"] != "confirmed":
                raise RestaurantServiceError(
                    "Only a confirmed booking can be updated.",
                    code="booking_not_updatable",
                    status=409,
                )
            updated_name = (
                self._validate_name(new_name) if new_name else row["customer_name"]
            )

            session = await conn.fetchrow(
                "SELECT state FROM call_sessions WHERE session_id = $1",
                call_id,
            )
            session_state: dict[str, Any] = {}
            if session and session["state"]:
                raw = session["state"]
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        raw = {}
                if isinstance(raw, dict):
                    session_state = dict(raw)
            draft = coerce_draft(session_state.get("reservation_draft") or session_state)
            booked_at = row["booked_at"]
            new_date = date or booked_at.date().isoformat()
            new_time = time or booked_at.strftime("%H:%M")
            new_party = party_size or int(row["party_size"])
            location_pref = preferred_location or draft_preferred_location(
                seating_preference if seating_preference is not None else draft
            )
            table_number = row["table_number"]
            location = row["location"]
            table_id = row["table_id"]

            if changing_slot:
                tables = await self.get_available_tables(
                    new_date,
                    new_time,
                    new_party,
                    limit=1,
                    preferred_location=location_pref,
                    exclude_booking_id=booking_id,
                    conn=conn,
                    require_location_match=bool(location_pref),
                )
                if not tables:
                    availability = await self.check_availability(
                        new_date,
                        new_time,
                        new_party,
                        preferred_location=location_pref,
                        exclude_booking_id=booking_id,
                    )
                    return {
                        "updated": False,
                        "booking_id": booking_id,
                        "slot_unavailable": True,
                        "alternatives": availability["alternatives"],
                        "customer_name": row["customer_name"],
                        "date": booked_at.date().isoformat(),
                        "time": booked_at.strftime("%H:%M"),
                        "party_size": row["party_size"],
                        "table_number": table_number,
                        "location": location,
                        "notes": row["notes"] or "",
                    }
                chosen = tables[0]
                table_id = chosen["id"]
                table_number = chosen["table_number"]
                location = chosen["location"]
                dt = self._parse_booking_datetime(new_date, new_time)
                await conn.execute(
                    """
                    UPDATE bookings
                    SET booked_at = $1, party_size = $2, table_id = $3
                    WHERE id = $4
                    """,
                    dt,
                    new_party,
                    table_id,
                    booking_id,
                )

            draft = patch_draft(
                draft,
                {
                    "customer_name": updated_name,
                    "customer_phone": row["customer_phone"] or "",
                    "date": new_date,
                    "time": new_time,
                    "party_size": new_party,
                    "booking_id": booking_id,
                    "status": DRAFT_STATUS_CONFIRMED,
                    **note_updates,
                },
            )
            rebuilt_notes = compose_notes(draft)
            if require_approval_for_paid_items is not None:
                draft = patch_draft(
                    draft,
                    {"require_approval_for_paid_items": require_approval_for_paid_items},
                )
            await conn.execute(
                """
                UPDATE bookings SET notes = $1, customer_name = $2,
                    require_approval_for_paid_items = COALESCE($4, require_approval_for_paid_items)
                WHERE id = $3
                """,
                rebuilt_notes,
                updated_name,
                booking_id,
                require_approval_for_paid_items,
            )
            await conn.execute(
                """
                UPDATE orders SET notes = $1, customer_name = $2 WHERE booking_id = $3
                """,
                rebuilt_notes,
                updated_name,
                booking_id,
            )
            state = flatten_draft(
                draft,
                guest_notes=str(session_state.get("guest_notes") or ""),
            )
            state["table_number"] = table_number
            state["table_location"] = location or ""
            await self._merge_session_state(
                conn, call_id, state, caller_phone=row["customer_phone"] or ""
            )
            return {
                "updated": True,
                "booking_id": booking_id,
                "customer_name": updated_name,
                "customer_phone": row["customer_phone"] or "",
                "party_size": new_party,
                "date": new_date,
                "time": new_time,
                "table_number": table_number,
                "location": location or "",
                "notes": rebuilt_notes,
            }

        result, replayed = await self._idempotent_write(
            action="update_confirmed_booking",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def add_guest_note(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        note: str,
        booking_id: int = 0,
    ) -> JsonDict:
        """Save a caller instruction on the booking, order, and call session."""
        call_id = self._require_call_id(call_id)
        note = " ".join(note.split())
        if not 2 <= len(note) <= 500:
            raise RestaurantServiceError(
                "A guest note must be 2-500 characters.",
                code="invalid_note",
            )
        payload = {
            "call_id": call_id,
            "booking_id": booking_id,
            "note": note,
        }

        async def operation(conn: Any) -> JsonDict:
            target_booking = booking_id if booking_id > 0 else 0
            session = await conn.fetchrow(
                "SELECT state FROM call_sessions WHERE session_id = $1",
                call_id,
            )
            session_state: dict[str, Any] = {}
            if session and session["state"]:
                raw = session["state"]
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        raw = {}
                if isinstance(raw, dict):
                    session_state = dict(raw)
            if not target_booking:
                try:
                    target_booking = int(session_state.get("booking_id") or 0)
                except (TypeError, ValueError):
                    target_booking = 0

            saved_on_booking = False
            combined = note
            if target_booking:
                row = await conn.fetchrow(
                    """
                    SELECT id, notes FROM bookings
                    WHERE id = $1 AND status = 'confirmed'
                    FOR UPDATE
                    """,
                    target_booking,
                )
                if not row:
                    raise RestaurantServiceError(
                        "No confirmed booking was found for that note.",
                        code="booking_not_found",
                        status=404,
                    )
                combined = _combine_notes(row["notes"] or "", note)
                await conn.execute(
                    "UPDATE bookings SET notes = $1 WHERE id = $2",
                    combined,
                    row["id"],
                )
                await conn.execute(
                    "UPDATE orders SET notes = $1 WHERE booking_id = $2",
                    combined,
                    row["id"],
                )
                saved_on_booking = True
                target_booking = int(row["id"])
            else:
                order = await conn.fetchrow(
                    """
                    SELECT id, notes FROM orders
                    WHERE session_id = $1 AND status IN ('pending', 'confirmed')
                    ORDER BY id DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    call_id,
                )
                if order:
                    combined = _combine_notes(order["notes"] or "", note)
                    await conn.execute(
                        "UPDATE orders SET notes = $1 WHERE id = $2",
                        combined,
                        order["id"],
                    )
                else:
                    combined = _combine_notes(str(session_state.get("notes") or ""), note)

            session_state["notes"] = combined
            guest_notes = merge_note_text(str(session_state.get("guest_notes") or ""), note)
            session_state["guest_notes"] = guest_notes
            if target_booking:
                session_state["booking_id"] = target_booking
            state_patch = {"notes": combined, "guest_notes": guest_notes}
            if target_booking:
                state_patch["booking_id"] = target_booking
            await conn.execute(
                """
                INSERT INTO call_sessions (session_id, state)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (session_id) DO UPDATE
                SET state = call_sessions.state || EXCLUDED.state,
                    updated_at = NOW()
                """,
                call_id,
                json.dumps(state_patch),
            )
            return {
                "saved": True,
                "booking_id": target_booking or 0,
                "saved_on_booking": saved_on_booking,
                "notes": combined,
                "guest_notes": guest_notes,
            }

        result, replayed = await self._idempotent_write(
            action="add_guest_note",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def lookup_booking(
        self,
        *,
        booking_id: int = 0,
        customer_name: str = "",
        customer_phone: str = "",
    ) -> JsonDict:
        phone = self._validate_phone(customer_phone) if customer_phone else ""
        pool = await get_pool()
        async with pool.acquire() as conn:
            if booking_id:
                row = await conn.fetchrow(
                    """
                    SELECT b.id, b.customer_name, b.customer_phone, b.booked_at,
                           b.party_size, b.status, b.notes, t.table_number, t.location
                    FROM bookings b
                    LEFT JOIN tables t ON t.id = b.table_id
                    WHERE b.id = $1
                    """,
                    booking_id,
                )
            elif customer_name and phone:
                name = self._validate_name(customer_name)
                row = await conn.fetchrow(
                    """
                    SELECT b.id, b.customer_name, b.customer_phone, b.booked_at,
                           b.party_size, b.status, b.notes, t.table_number, t.location
                    FROM bookings b
                    LEFT JOIN tables t ON t.id = b.table_id
                    WHERE LOWER(b.customer_name) = LOWER($1)
                      AND b.customer_phone = $2
                    ORDER BY b.booked_at DESC
                    LIMIT 1
                    """,
                    name,
                    phone,
                )
            else:
                raise RestaurantServiceError(
                    "Provide the booking ID, or both the exact name and phone number.",
                    code="verification_required",
                )
        if not row:
            raise RestaurantServiceError(
                "No booking matched those verified details.",
                code="booking_not_found",
                status=404,
            )
        return {
            "booking_id": row["id"],
            "customer_name": row["customer_name"],
            "booked_at": row["booked_at"].isoformat(),
            "party_size": row["party_size"],
            "status": row["status"],
            "table_number": row["table_number"],
            "location": row["location"],
            "notes": row["notes"] or "",
        }

    async def cancel_booking(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        booking_id: int,
        customer_name: str = "",
        customer_phone: str = "",
        reason: str = "",
        confirmed: bool,
    ) -> JsonDict:
        if confirmed is not True:
            raise RestaurantServiceError(
                "The caller must explicitly confirm the cancellation.",
                code="confirmation_required",
                status=409,
            )
        call_id = self._require_call_id(call_id)
        if booking_id <= 0:
            raise RestaurantServiceError("A valid booking_id is required.")
        name = self._validate_name(customer_name) if customer_name else ""
        phone = self._validate_phone(customer_phone) if customer_phone else ""
        if not name and not phone:
            raise RestaurantServiceError(
                "Verify either the booking name or phone number before cancellation.",
                code="verification_required",
            )
        reason = reason.strip()[:300]
        payload = {
            "call_id": call_id,
            "booking_id": booking_id,
            "customer_name": name,
            "customer_phone": phone,
            "reason": reason,
        }

        async def operation(conn: Any) -> JsonDict:
            row = await conn.fetchrow(
                """
                SELECT id, customer_name, customer_phone, status
                FROM bookings
                WHERE id = $1
                FOR UPDATE
                """,
                booking_id,
            )
            if not row:
                raise RestaurantServiceError(
                    "The booking was not found.", code="booking_not_found", status=404
                )
            name_matches = name and _normalized_text(name) == _normalized_text(row["customer_name"])
            phone_matches = phone and phone == (row["customer_phone"] or "")
            if not name_matches and not phone_matches:
                raise RestaurantServiceError(
                    "The verification details do not match the booking.",
                    code="verification_failed",
                    status=403,
                )
            if row["status"] == "cancelled":
                return {
                    "cancelled": True,
                    "already_cancelled": True,
                    "booking_id": booking_id,
                    "customer_name": row["customer_name"],
                }
            if row["status"] == "completed":
                raise RestaurantServiceError(
                    "A completed booking cannot be cancelled.",
                    code="booking_completed",
                    status=409,
                )
            await conn.execute(
                """
                UPDATE bookings
                SET status = 'cancelled', cancellation_reason = $1
                WHERE id = $2
                """,
                reason or "No reason provided",
                booking_id,
            )
            await self._merge_session_state(
                conn,
                call_id,
                {
                    "booking_id": booking_id,
                    "draft_status": DRAFT_STATUS_CANCELLED,
                    "reservation_draft": patch_draft(
                        coerce_draft({"booking_id": booking_id}),
                        {"status": DRAFT_STATUS_CANCELLED, "booking_id": booking_id},
                    ),
                },
            )
            return {
                "cancelled": True,
                "already_cancelled": False,
                "booking_id": booking_id,
                "customer_name": row["customer_name"],
            }

        result, replayed = await self._idempotent_write(
            action="cancel_booking",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def list_menu(self, *, available_only: bool = True) -> JsonDict:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, category, price, description, dietary, available,
                       COALESCE(price_estimated, FALSE) AS price_estimated
                FROM menu_items
                WHERE ($1::boolean = FALSE OR available = TRUE)
                ORDER BY category, name
                """,
                available_only,
            )
        return {
            "items": [
                {
                    "id": row["id"],
                    "name": row["name"],
                    "category": row["category"],
                    "price": float(row["price"]),
                    "price_estimated": bool(row["price_estimated"]),
                    "description": row["description"] or "",
                    "dietary": list(row["dietary"] or []),
                    "available": row["available"],
                }
                for row in rows
            ],
            "allergen_notice": (
                "Menu descriptions cannot guarantee an allergen-free preparation or prevent "
                "cross-contact. Transfer severe allergy questions to restaurant staff."
            ),
        }

    async def find_menu_item(self, item_name: str) -> JsonDict:
        requested = _normalized_text(item_name)
        if not requested:
            raise RestaurantServiceError("An item_name is required.")
        menu = await self.list_menu(available_only=False)
        items: list[JsonDict] = menu["items"]
        exact = [item for item in items if _normalized_text(item["name"]) == requested]
        if len(exact) == 1:
            return {"match": exact[0], "needs_confirmation": False, "candidates": []}

        contained = [
            item
            for item in items
            if requested in _normalized_text(item["name"])
            or _normalized_text(item["name"]) in requested
        ]
        if len(contained) == 1:
            return {
                "match": None,
                "needs_confirmation": True,
                "candidates": contained,
            }

        ranked = sorted(
            (
                (
                    SequenceMatcher(None, requested, _normalized_text(item["name"])).ratio(),
                    item,
                )
                for item in items
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        candidates = [item for score, item in ranked[:3] if score >= 0.45]
        return {
            "match": None,
            "needs_confirmation": bool(candidates),
            "candidates": candidates,
        }

    async def _booking_id_from_session(self, conn: Any, call_id: str) -> int:
        session = await conn.fetchrow(
            "SELECT state FROM call_sessions WHERE session_id = $1",
            call_id,
        )
        state = self._coerce_state(session["state"] if session else {})
        try:
            return int(state.get("booking_id") or 0)
        except (TypeError, ValueError):
            return 0

    async def _paid_item_approval_required(
        self,
        conn: Any,
        call_id: str,
        booking_id: int,
    ) -> bool:
        if booking_id:
            flag = await conn.fetchval(
                """
                SELECT require_approval_for_paid_items FROM bookings WHERE id = $1
                """,
                booking_id,
            )
            if flag:
                return True
        session = await conn.fetchrow(
            "SELECT state FROM call_sessions WHERE session_id = $1",
            call_id,
        )
        state = self._coerce_state(session["state"] if session else {})
        draft = coerce_draft(state.get("reservation_draft") or state)
        return bool(
            draft.get("require_approval_for_paid_items")
            or state.get("require_approval_for_paid_items")
        )

    async def add_order_item(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        item_name: str,
        quantity: int = 1,
        notes: str = "",
        booking_id: int = 0,
        customer_name: str = "",
        customer_phone: str = "",
        caller_confirmed: bool = False,
    ) -> JsonDict:
        call_id = self._require_call_id(call_id)
        if not 1 <= quantity <= 20:
            raise RestaurantServiceError(
                "Quantity must be between 1 and 20.", code="invalid_quantity"
            )
        match = await self.find_menu_item(item_name)
        if not match["match"]:
            return {
                "added": False,
                "needs_confirmation": match["needs_confirmation"],
                "candidates": match["candidates"],
            }
        menu_item = match["match"]
        if not menu_item["available"]:
            return {
                "added": False,
                "unavailable": True,
                "item": menu_item,
                "candidates": [],
            }
        name = self._validate_name(customer_name) if customer_name else ""
        phone = self._validate_phone(customer_phone) if customer_phone else ""
        notes = notes.strip()[:300]
        payload = {
            "call_id": call_id,
            "menu_item_id": menu_item["id"],
            "quantity": quantity,
            "notes": notes,
            "booking_id": booking_id,
            "customer_name": name,
            "customer_phone": phone,
            "caller_confirmed": bool(caller_confirmed),
        }

        async def operation(conn: Any) -> JsonDict:
            resolved_booking = booking_id or await self._booking_id_from_session(
                conn, call_id
            )
            order = await self._lock_order_for_mutation(
                conn,
                call_id=call_id,
                booking_id=resolved_booking,
                caller_confirmed=caller_confirmed,
                create_if_missing=True,
                customer_name=name,
                customer_phone=phone,
            )
            order_id = order["id"]
            if order.get("existing"):
                await conn.execute(
                    """
                    UPDATE orders
                    SET booking_id = COALESCE(booking_id, $1),
                        customer_name = CASE WHEN customer_name = '' THEN $2 ELSE customer_name END,
                        customer_phone = CASE WHEN customer_phone = '' THEN $3 ELSE customer_phone END,
                        draft_version = draft_version + 1
                    WHERE id = $4
                    """,
                    resolved_booking or None,
                    name,
                    phone,
                    order_id,
                )
            approval_required = await self._paid_item_approval_required(
                conn,
                call_id,
                resolved_booking or int(order.get("booking_id") or 0),
            )
            unit_price = float(menu_item["price"])
            propose = (
                approval_required
                and unit_price > 0
                and caller_confirmed is not True
            )
            if caller_confirmed is True and approval_required:
                existing_proposed = await conn.fetchrow(
                    """
                    SELECT id FROM order_items
                    WHERE order_id = $1 AND menu_item_id = $2 AND proposed IS TRUE
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    order_id,
                    menu_item["id"],
                )
                if existing_proposed:
                    await conn.execute(
                        """
                        UPDATE order_items
                        SET proposed = FALSE, quantity = $1, notes = $2
                        WHERE id = $3
                        """,
                        quantity,
                        notes,
                        existing_proposed["id"],
                    )
                    summary = await self._order_summary_with_conn(conn, order_id)
                    if summary["status"] == "confirmed":
                        await conn.execute(
                            "UPDATE orders SET total_amount = $1 WHERE id = $2",
                            summary["total"],
                            order_id,
                        )
                    return {
                        "added": True,
                        "proposed": False,
                        "order_item_id": existing_proposed["id"],
                        **summary,
                    }
            item = await conn.fetchrow(
                """
                INSERT INTO order_items
                    (order_id, menu_item_id, item_name, quantity, unit_price, notes, proposed)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
                """,
                order_id,
                menu_item["id"],
                menu_item["name"],
                quantity,
                menu_item["price"],
                notes,
                propose,
            )
            summary = await self._order_summary_with_conn(conn, order_id)
            if summary["status"] == "confirmed" and not propose:
                await conn.execute(
                    "UPDATE orders SET total_amount = $1 WHERE id = $2",
                    summary["total"],
                    order_id,
                )
            if propose:
                return {
                    "added": False,
                    "proposed": True,
                    "needs_caller_yes": True,
                    "order_item_id": item["id"],
                    **summary,
                }
            return {
                "added": True,
                "proposed": False,
                "order_item_id": item["id"],
                **summary,
            }

        result, replayed = await self._idempotent_write(
            action="add_order_item",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def _lock_order_for_mutation(
        self,
        conn: Any,
        *,
        call_id: str,
        booking_id: int = 0,
        caller_confirmed: bool = False,
        create_if_missing: bool = False,
        customer_name: str = "",
        customer_phone: str = "",
        order_item_id: int = 0,
    ) -> JsonDict:
        order = None
        if order_item_id:
            order = await conn.fetchrow(
                """
                SELECT o.id, o.booking_id, o.customer_name, o.customer_phone,
                       o.draft_version, o.status, TRUE AS existing
                FROM order_items oi
                JOIN orders o ON o.id = oi.order_id
                WHERE oi.id = $1
                  AND o.session_id = $2
                  AND o.status IN ('pending', 'confirmed')
                FOR UPDATE OF oi, o
                """,
                order_item_id,
                call_id,
            )
        if not order:
            order = await conn.fetchrow(
                """
                SELECT id, booking_id, customer_name, customer_phone,
                       draft_version, status, TRUE AS existing
                FROM orders
                WHERE session_id = $1
                  AND status IN ('pending', 'confirmed')
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC
                LIMIT 1
                FOR UPDATE
                """,
                call_id,
            )
        if not order and booking_id:
            order = await conn.fetchrow(
                """
                SELECT id, booking_id, customer_name, customer_phone,
                       draft_version, status, TRUE AS existing
                FROM orders
                WHERE booking_id = $1
                  AND status IN ('pending', 'confirmed')
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC
                LIMIT 1
                FOR UPDATE
                """,
                booking_id,
            )
        if not order and create_if_missing:
            created = await conn.fetchrow(
                """
                INSERT INTO orders
                    (session_id, booking_id, customer_name, customer_phone, status, draft_version)
                VALUES ($1, $2, $3, $4, 'pending', 1)
                RETURNING id, booking_id, customer_name, customer_phone,
                          draft_version, status, FALSE AS existing
                """,
                call_id,
                booking_id or None,
                customer_name,
                customer_phone,
            )
            return dict(created)
        if not order:
            raise RestaurantServiceError(
                "That draft order item was not found."
                if order_item_id
                else "No order exists for this call.",
                code="order_item_not_found" if order_item_id else "order_not_found",
                status=404,
            )
        if order["status"] == "confirmed" and caller_confirmed is not True:
            raise RestaurantServiceError(
                "The caller must explicitly confirm changing the existing order.",
                code="confirmation_required",
                status=409,
            )
        return dict(order)

    async def _order_summary_with_conn(self, conn: Any, order_id: int) -> JsonDict:
        order = await conn.fetchrow(
            """
            SELECT id, session_id, booking_id, customer_name, customer_phone,
                   status, total_amount, draft_version, created_at
            FROM orders WHERE id = $1
            """,
            order_id,
        )
        if not order:
            raise RestaurantServiceError("Order not found.", code="order_not_found", status=404)
        items = await conn.fetch(
            """
            SELECT id, item_name, quantity, unit_price, subtotal, notes,
                   COALESCE(proposed, FALSE) AS proposed
            FROM order_items WHERE order_id = $1 ORDER BY id
            """,
            order_id,
        )
        committed = [item for item in items if not item["proposed"]]
        proposed = [item for item in items if item["proposed"]]
        calculated_total = sum(float(item["subtotal"]) for item in committed)

        def _item_payload(item: Any) -> JsonDict:
            return {
                "order_item_id": item["id"],
                "item_name": item["item_name"],
                "quantity": item["quantity"],
                "unit_price": float(item["unit_price"]),
                "subtotal": float(item["subtotal"]),
                "notes": item["notes"] or "",
                "proposed": bool(item["proposed"]),
            }

        return {
            "order_id": order["id"],
            "call_id": order["session_id"],
            "booking_id": order["booking_id"],
            "customer_name": order["customer_name"] or "",
            "status": order["status"],
            "draft_version": order["draft_version"],
            "total": calculated_total,
            "items": [_item_payload(item) for item in committed],
            "proposed_items": [_item_payload(item) for item in proposed],
            "fulfillment": "dine_in" if order["booking_id"] else "pickup",
        }

    async def get_order_summary(self, *, call_id: str) -> JsonDict:
        call_id = self._require_call_id(call_id)
        pool = await get_pool()
        async with pool.acquire() as conn:
            order = await conn.fetchrow(
                """
                SELECT id FROM orders
                WHERE session_id = $1 AND status IN ('pending', 'confirmed')
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC
                LIMIT 1
                """,
                call_id,
            )
            if not order:
                raise RestaurantServiceError(
                    "No order exists for this call.",
                    code="order_not_found",
                    status=404,
                )
            result = await self._order_summary_with_conn(conn, order["id"])
            if result.get("status") == "pending" and result.get("items"):
                digest = register_pending_confirmation(
                    call_id,
                    ACTION_CONFIRM_ORDER,
                    order_confirmation_payload(result),
                )
                result["pending_confirmation_hash"] = digest
                result["readback_required"] = True
                await self._merge_session_state(
                    conn, call_id, pending_state_patch(call_id)
                )
        result["summary_nonce"] = secrets.token_hex(8)
        record_order_summary(result)
        return result

    async def update_order_item(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        order_item_id: int,
        quantity: int,
        notes: str = "",
        caller_confirmed: bool = False,
    ) -> JsonDict:
        call_id = self._require_call_id(call_id)
        if not 1 <= quantity <= 20:
            raise RestaurantServiceError("Quantity must be between 1 and 20.")
        notes = notes.strip()[:300]
        payload = {
            "call_id": call_id,
            "order_item_id": order_item_id,
            "quantity": quantity,
            "notes": notes,
            "caller_confirmed": bool(caller_confirmed),
        }

        async def operation(conn: Any) -> JsonDict:
            order = await self._lock_order_for_mutation(
                conn,
                call_id=call_id,
                caller_confirmed=caller_confirmed,
                order_item_id=order_item_id,
            )
            await conn.execute(
                "UPDATE order_items SET quantity = $1, notes = $2 WHERE id = $3",
                quantity,
                notes,
                order_item_id,
            )
            await conn.execute(
                "UPDATE orders SET draft_version = draft_version + 1 WHERE id = $1",
                order["id"],
            )
            summary = await self._order_summary_with_conn(conn, order["id"])
            if summary["status"] == "confirmed":
                await conn.execute(
                    "UPDATE orders SET total_amount = $1 WHERE id = $2",
                    summary["total"],
                    order["id"],
                )
            return {"updated": True, **summary}

        result, replayed = await self._idempotent_write(
            action="update_order_item",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def remove_order_item(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        order_item_id: int,
        caller_confirmed: bool = False,
    ) -> JsonDict:
        call_id = self._require_call_id(call_id)
        payload = {
            "call_id": call_id,
            "order_item_id": order_item_id,
            "caller_confirmed": bool(caller_confirmed),
        }

        async def operation(conn: Any) -> JsonDict:
            order = await self._lock_order_for_mutation(
                conn,
                call_id=call_id,
                caller_confirmed=caller_confirmed,
                order_item_id=order_item_id,
            )
            await conn.execute("DELETE FROM order_items WHERE id = $1", order_item_id)
            await conn.execute(
                "UPDATE orders SET draft_version = draft_version + 1 WHERE id = $1",
                order["id"],
            )
            summary = await self._order_summary_with_conn(conn, order["id"])
            if summary["status"] == "confirmed":
                await conn.execute(
                    "UPDATE orders SET total_amount = $1 WHERE id = $2",
                    summary["total"],
                    order["id"],
                )
            return {"removed": True, **summary}

        result, replayed = await self._idempotent_write(
            action="remove_order_item",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def confirm_order(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        expected_draft_version: int,
        approved: bool,
    ) -> JsonDict:
        call_id = self._require_call_id(call_id)
        if approved is not True:
            raise RestaurantServiceError(
                "The caller must approve the complete itemized readback.",
                code="confirmation_required",
                status=409,
            )
        payload = {
            "call_id": call_id,
            "expected_draft_version": expected_draft_version,
            "approved": True,
        }

        async def operation(conn: Any) -> JsonDict:
            order = await conn.fetchrow(
                """
                SELECT id, draft_version, booking_id, customer_name, customer_phone
                FROM orders
                WHERE session_id = $1 AND status = 'pending'
                FOR UPDATE
                """,
                call_id,
            )
            if not order:
                raise RestaurantServiceError(
                    "No draft order exists for this call.",
                    code="order_not_found",
                    status=404,
                )
            if order["draft_version"] != expected_draft_version:
                raise RestaurantServiceError(
                    "The order changed after the readback. Read the updated order and confirm again.",
                    code="draft_version_conflict",
                    status=409,
                )
            if not order["booking_id"] and (
                not (order["customer_name"] or "").strip()
                or not (order["customer_phone"] or "").strip()
            ):
                raise RestaurantServiceError(
                    "Pickup orders require a verified name and callback phone before confirmation.",
                    code="pickup_contact_required",
                    status=409,
                )
            summary = await self._order_summary_with_conn(conn, order["id"])
            if not summary["items"]:
                raise RestaurantServiceError(
                    "The draft order is empty.", code="empty_order", status=409
                )
            require_pending_confirmation(
                call_id,
                ACTION_CONFIRM_ORDER,
                order_confirmation_payload(summary),
            )
            await conn.execute(
                """
                UPDATE orders
                SET status = 'confirmed', total_amount = $1, confirmed_at = NOW()
                WHERE id = $2
                """,
                summary["total"],
                order["id"],
            )
            clear_pending_confirmation(call_id, ACTION_CONFIRM_ORDER)
            await self._merge_session_state(conn, call_id, pending_state_patch(call_id))
            return {
                **summary,
                "confirmed": True,
                "status": "confirmed",
                "timing": (
                    "served at the reserved table on arrival"
                    if summary["fulfillment"] == "dine_in"
                    else "ready for pickup in about 30 minutes"
                ),
            }

        result, replayed = await self._idempotent_write(
            action="confirm_order",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def lookup_order(
        self,
        *,
        order_id: int,
        customer_name: str,
    ) -> JsonDict:
        if order_id <= 0:
            raise RestaurantServiceError("A valid order_id is required.")
        name = self._validate_name(customer_name)
        pool = await get_pool()
        async with pool.acquire() as conn:
            order = await conn.fetchrow(
                "SELECT id, customer_name FROM orders WHERE id = $1",
                order_id,
            )
            if not order:
                raise RestaurantServiceError(
                    "Order not found.", code="order_not_found", status=404
                )
            if _normalized_text(name) != _normalized_text(order["customer_name"] or ""):
                raise RestaurantServiceError(
                    "The name does not match that order.",
                    code="verification_failed",
                    status=403,
                )
            return await self._order_summary_with_conn(conn, order_id)

    def _load_public_settings(self) -> JsonDict:
        loaded = load_restaurant_settings()
        return {
            key: loaded.get(key)
            for key in (
                "restaurant_name",
                "tagline",
                "phone_number",
                "timezone",
                "street_address",
                "city",
                "languages",
                "opening_hours",
                "hours_unconfirmed",
                "hours_note",
            )
        }

    def _hours_info(self, data: JsonDict) -> JsonDict:
        unconfirmed = bool(data.get("hours_unconfirmed")) or not data.get("opening_hours")
        note = str(data.get("hours_note") or HOURS_UNCONFIRMED_NOTE)
        if unconfirmed:
            return {
                "matched": True,
                "answers": [],
                "formatted": note,
                "log_unknown": False,
                "restaurant_name": data.get("restaurant_name") or "",
                "timezone": data.get("timezone") or "",
                "opening_hours": {},
                "hours_unconfirmed": True,
                "hours_note": note,
            }
        return {
            "matched": True,
            "answers": [],
            "formatted": "",
            "log_unknown": False,
            "restaurant_name": data.get("restaurant_name") or "",
            "timezone": data.get("timezone") or "",
            "opening_hours": data.get("opening_hours") or {},
            "hours_unconfirmed": False,
            "hours_note": "",
        }

    async def restaurant_info(self, topic: str = "") -> JsonDict:
        """Search markdown, operator FAQ, then structured settings. No embeddings."""
        data = self._load_public_settings()
        query = (topic or "").strip()
        normalized = _normalized_text(query)
        hour_tokens = {
            "hour",
            "hours",
            "open",
            "opens",
            "opened",
            "opening",
            "close",
            "closes",
            "closed",
            "closing",
            "schedule",
        }
        if query and hour_tokens & set(normalized.split()):
            return self._hours_info(data)

        capacity_tokens = {"capacity", "capacities", "seats", "largest"}
        if query and capacity_tokens & set(normalized.split()):
            limits = await self.seating_limits()
            by_loc = limits.get("max_seats_by_location") or {}
            parts = [
                f"Phone bookings are for 1 to {limits.get('max_party_phone', 12)} guests."
            ]
            if by_loc:
                loc_text = ", ".join(
                    f"{name} up to {seats}" for name, seats in sorted(by_loc.items())
                )
                parts.append(f"Largest tables by room: {loc_text}.")
            parts.append(
                f"Largest table overall seats {limits.get('largest_table') or 0}."
            )
            return {
                "matched": True,
                "answers": [],
                "formatted": " ".join(parts),
                "log_unknown": False,
                "restaurant_name": data.get("restaurant_name") or "",
                **limits,
            }

        faq_rows: list[JsonDict] = []
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT question, answer
                    FROM operator_knowledge
                    WHERE active = TRUE
                    ORDER BY updated_at DESC, id DESC
                    """
                )
            faq_rows = [dict(row) for row in rows]
        except Exception:
            faq_rows = []

        hits = search_faq_rows(query, faq_rows) + search_static_knowledge(query)
        hits.sort(key=lambda item: (-int(item.get("score") or 0), item.get("heading") or ""))
        hits = hits[:4]
        if hits:
            return {
                "matched": True,
                "answers": [
                    {
                        "kind": hit.get("kind"),
                        "heading": hit.get("heading") or "",
                        "content": hit.get("content") or "",
                        "source": hit.get("source") or "",
                    }
                    for hit in hits
                ],
                "formatted": format_knowledge_hits(hits),
                "restaurant_name": data.get("restaurant_name") or "",
            }

        normalized = _normalized_text(query)
        settings_hit: JsonDict = {}
        if not query:
            settings_hit = dict(data)
        elif hour_tokens & set(normalized.split()):
            return self._hours_info(data)
        elif any(word in normalized for word in ("where", "address", "location", "phone")):
            settings_hit = {
                "restaurant_name": data["restaurant_name"],
                "street_address": data["street_address"],
                "city": data["city"],
                "phone_number": data["phone_number"],
            }
        if settings_hit:
            return {
                "matched": True,
                "answers": [],
                "formatted": "",
                "log_unknown": False,
                **settings_hit,
            }
        return {
            "matched": False,
            "answers": [],
            "formatted": "",
            "log_unknown": True,
            "restaurant_name": data.get("restaurant_name") or "",
        }

    async def log_unknown_question(
        self,
        *,
        call_id: str,
        question: str,
        context_excerpt: str = "",
        agent_response: str = "",
    ) -> JsonDict:
        call_id = self._require_call_id(call_id)
        question = " ".join(question.split())
        if not 3 <= len(question) <= 500:
            raise RestaurantServiceError(
                "A question of 3-500 characters is required.",
                code="invalid_question",
            )
        normalized = normalize_question(question)
        excerpt = " ".join(context_excerpt.split())[:500]
        reply = " ".join(agent_response.split())[:500]
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO knowledge_gaps
                    (session_id, question, question_normalized, context_excerpt, agent_response)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (session_id, question_normalized) DO UPDATE
                SET context_excerpt = CASE
                        WHEN knowledge_gaps.context_excerpt = '' THEN EXCLUDED.context_excerpt
                        ELSE knowledge_gaps.context_excerpt
                    END
                RETURNING id, status, created_at
                """,
                call_id,
                question,
                normalized,
                excerpt,
                reply,
            )
        return {
            "logged": True,
            "gap_id": row["id"],
            "status": row["status"],
            "already_logged": row["status"] == "resolved",
            "duplicate": False,
        }

    async def list_knowledge_gaps(self) -> JsonDict:
        pool = await get_pool()
        async with pool.acquire() as conn:
            gaps = await conn.fetch(
                """
                SELECT id, session_id, question, context_excerpt, agent_response,
                       status, resolved_answer, resolved_by, created_at, resolved_at
                FROM knowledge_gaps
                ORDER BY CASE status WHEN 'unresolved' THEN 0 ELSE 1 END,
                         created_at DESC
                """
            )
            faq = await conn.fetch(
                """
                SELECT id, question, answer, source_gap_id, active, created_at, updated_at
                FROM operator_knowledge
                ORDER BY active DESC, updated_at DESC, id DESC
                """
            )
        return {
            "gaps": [
                {
                    "id": row["id"],
                    "session_id": row["session_id"],
                    "question": row["question"],
                    "context_excerpt": row["context_excerpt"] or "",
                    "agent_response": row["agent_response"] or "",
                    "status": row["status"],
                    "resolved_answer": row["resolved_answer"] or "",
                    "resolved_by": row["resolved_by"] or "",
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "resolved_at": row["resolved_at"].isoformat() if row["resolved_at"] else None,
                }
                for row in gaps
            ],
            "knowledge": [
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "source_gap_id": row["source_gap_id"],
                    "active": row["active"],
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
                for row in faq
            ],
        }

    async def resolve_knowledge_gap(
        self,
        gap_id: int,
        *,
        answer: str,
        resolved_by: str = "admin",
    ) -> JsonDict:
        answer = " ".join(answer.split())
        if not 2 <= len(answer) <= 2000:
            raise RestaurantServiceError(
                "An answer of 2-2000 characters is required.",
                code="invalid_answer",
            )
        resolved_by = " ".join(resolved_by.split())[:80] or "admin"
        pool = await get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                gap = await conn.fetchrow(
                    """
                    SELECT id, question FROM knowledge_gaps
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    gap_id,
                )
                if not gap:
                    raise RestaurantServiceError(
                        "That knowledge gap was not found.",
                        code="gap_not_found",
                        status=404,
                    )
                existing = await conn.fetchrow(
                    """
                    SELECT id FROM operator_knowledge
                    WHERE LOWER(question) = LOWER($1)
                    FOR UPDATE
                    """,
                    gap["question"],
                )
                if existing:
                    faq = await conn.fetchrow(
                        """
                        UPDATE operator_knowledge
                        SET answer = $2,
                            source_gap_id = $3,
                            active = TRUE,
                            updated_at = NOW()
                        WHERE id = $1
                        RETURNING id, question, answer
                        """,
                        existing["id"],
                        answer,
                        gap["id"],
                    )
                else:
                    faq = await conn.fetchrow(
                        """
                        INSERT INTO operator_knowledge (question, answer, source_gap_id, active)
                        VALUES ($1, $2, $3, TRUE)
                        RETURNING id, question, answer
                        """,
                        gap["question"],
                        answer,
                        gap["id"],
                    )
                await conn.execute(
                    """
                    UPDATE knowledge_gaps
                    SET status = 'resolved',
                        resolved_answer = $2,
                        resolved_by = $3,
                        resolved_at = NOW()
                    WHERE id = $1
                    """,
                    gap["id"],
                    answer,
                    resolved_by,
                )
        return {
            "resolved": True,
            "gap_id": gap_id,
            "knowledge_id": faq["id"],
            "question": faq["question"],
            "answer": faq["answer"],
        }


restaurant_service = RestaurantService()
