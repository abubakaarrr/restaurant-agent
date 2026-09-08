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
from copy import deepcopy
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
from app.restaurant_knowledge import (
    KnowledgeFixtureError,
    get_restaurant_knowledge,
    normalize_text as normalize_knowledge_text,
    text_tokens,
)
from app.security import canonical_request_hash, normalize_caller_phone
from app.pending_confirmation import (
    ACTION_CANCEL_BOOKING,
    ACTION_CONFIRM_ORDER,
    ACTION_CREATE_BOOKING,
    ACTION_UPDATE_CONFIRMED_BOOKING,
    booking_confirmation_payload,
    cancel_booking_confirmation_payload,
    clear_pending_confirmation,
    get_pending_confirmation,
    order_confirmation_payload,
    pending_state_patch,
    register_pending_confirmation,
    require_pending_confirmation,
    update_booking_confirmation_payload,
)
from app.turn_evidence import (
    current_turn,
    record_availability,
    record_order_summary,
)
from app.transfer_availability import current_staff_transfer_number


JsonDict = dict[str, Any]
T = TypeVar("T", bound=JsonDict)
RESERVATION_DURATION_MINUTES = 90


def _restaurant_now() -> datetime:
    try:
        timezone_info = ZoneInfo(get_restaurant_knowledge().identity["timezone"])
    except (KnowledgeFixtureError, KeyError, ZoneInfoNotFoundError):
        timezone_info = timezone.utc
    return datetime.now(timezone_info)


def _topic_rule(topic_id: str) -> JsonDict:
    knowledge = get_restaurant_knowledge()
    for topic in knowledge.topics:
        if topic.get("topic_id") == topic_id:
            return deepcopy(topic.get("rule") or {})
    return {}


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
    if result.get("restaurant_closed"):
        return (
            f"Unavailable: {result.get('message') or 'The restaurant is closed at that time.'} "
            f"availability_nonce={nonce}."
        )
    if result.get("available"):
        tables = ", ".join(
            f"table {row['table_number']} ({row['capacity']} seats, {row['location']})"
            for row in result.get("tables") or []
        )
        sections = result.get("available_sections") or []
        section_text = ""
        if sections:
            section_text = (
                " Offer the caller these open sections and ask which they prefer: "
                + ", ".join(sections)
                + ". Keep individual table numbers internal unless they ask."
            )
        return f"Available: {tables}.{section_text} availability_nonce={nonce}."
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


def _delivery_rule() -> JsonDict:
    match = get_restaurant_knowledge().find_topic("delivery")
    if match.status != "known":
        return {}
    return deepcopy(match.records[0].get("rule") or {})


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
        rule = _topic_rule("topic.reservations")
        maximum = int(rule.get("max_phone_party") or 10)
        large_party_min = int(rule.get("large_party_min") or maximum + 1)
        if large_party_min <= value <= 24:
            raise RestaurantServiceError(
                "Parties of 11 to 24 require the private-dining reservations route; phone table inventory was not checked or booked.",
                code="large_party_route_required",
                status=409,
            )
        if not 1 <= value <= maximum:
            raise RestaurantServiceError(
                f"Phone table reservations support parties of 1 to {maximum}.",
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
        now_local = _restaurant_now().astimezone(timezone_info).replace(tzinfo=None)
        if value < now_local:
            raise RestaurantServiceError(
                "The requested reservation time is in the past.",
                code="past_booking",
            )
        window_days = int(_topic_rule("topic.reservations").get("window_days") or 30)
        released_days = window_days if now_local.hour >= 9 else window_days - 1
        if value.date() > now_local.date() + timedelta(days=released_days):
            raise RestaurantServiceError(
                f"Reservations open {window_days} days ahead at 9:00 AM Pacific.",
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
    async def _update_order_notes_with_conn(
        conn: Any,
        *,
        order_notes: str | None = None,
        allergy_notes: str | None = None,
        order_id: int = 0,
        booking_id: int = 0,
    ) -> None:
        if bool(order_id) == bool(booking_id):
            raise ValueError("Specify exactly one order note owner")
        owner_column = "id" if order_id else "booking_id"
        owner_value = order_id or booking_id
        await conn.execute(
            f"""
            UPDATE orders
            SET notes = CASE WHEN $1::text IS NULL THEN notes ELSE $1 END,
                allergy_notes = CASE WHEN $2::text IS NULL THEN allergy_notes ELSE $2 END,
                draft_version = draft_version + 1
            WHERE {owner_column} = $3
              AND (
                  ($1::text IS NOT NULL AND notes IS DISTINCT FROM $1)
                  OR ($2::text IS NOT NULL AND allergy_notes IS DISTINCT FROM $2)
              )
            """,
            order_notes,
            allergy_notes,
            owner_value,
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
              AND ($7::int = 0 OR t.table_number = $7)
              AND NOT (LOWER(t.location) = ANY($8::text[]))
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
        table_number: int = 0,
        conn: Any | None = None,
        require_location_match: bool | None = None,
    ) -> list[JsonDict]:
        dt = self._parse_booking_datetime(date, time)
        party_size = self._validate_party_size(party_size)
        operating_status = self._reservation_operating_status(dt)
        if not operating_status["available"]:
            raise RestaurantServiceError(
                operating_status["customer_message"],
                code="restaurant_closed",
                status=409,
            )
        window_start = dt - timedelta(minutes=30)
        window_end = dt + timedelta(minutes=RESERVATION_DURATION_MINUTES)
        location = (preferred_location or "").strip().casefold()
        knowledge = get_restaurant_knowledge()
        closed_locations: list[str] = []
        for area in knowledge.raw["dining_areas"]:
            area_location = str(area["area_id"]).removeprefix("area.").casefold()
            if party_size > int(area.get("max_phone_party") or 10):
                closed_locations.append(area_location)
                continue
            if area_location == "patio" and not knowledge.schedule_status(
                "patio", dt, duration_minutes=RESERVATION_DURATION_MINUTES
            )["available"]:
                closed_locations.append(area_location)
        if location and location in closed_locations:
            return []
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
            int(table_number or 0),
            closed_locations,
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
            "max_party_phone": int(
                _topic_rule("topic.reservations").get("max_phone_party") or 10
            ),
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
        call_id: str = "",
    ) -> JsonDict:
        preferred = normalize_preferred_location(preferred_location)
        requested = self._parse_booking_datetime(date, time)
        party_size = self._validate_party_size(party_size)
        operating_status = self._reservation_operating_status(requested)
        if not operating_status["available"]:
            result = {
                "available": False,
                "restaurant_closed": True,
                "date": date,
                "time": time,
                "party_size": party_size,
                "preferred_location": preferred,
                "availability_nonce": secrets.token_hex(8),
                "impossible_at_location": False,
                "max_seats_at_location": 0,
                "tables": [],
                "available_sections": [],
                "alternatives": [],
                "message": operating_status["customer_message"],
                "hours_kind": operating_status["kind"],
            }
            self._record_availability_result(result, call_id)
            return result
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
            "available_sections": sorted(
                {
                    str(row["location"]).casefold()
                    for row in tables
                    if row.get("location")
                }
            ),
            "alternatives": alternatives,
        }
        self._record_availability_result(result, call_id)
        return result

    @staticmethod
    def _record_availability_result(result: JsonDict, call_id: str) -> None:
        record_availability(result)
        from app.availability_offer import remember_availability_offer
        from app.call_memory import resolve_session_id

        sid = resolve_session_id(call_id) if call_id else resolve_session_id()
        if sid:
            remember_availability_offer(sid, result)

    @staticmethod
    def _reservation_operating_status(requested: datetime) -> JsonDict:
        knowledge = get_restaurant_knowledge()
        status = knowledge.operating_status(requested)
        if not status["available"]:
            return status
        end = requested + timedelta(minutes=RESERVATION_DURATION_MINUTES) - timedelta(
            microseconds=1
        )
        return knowledge.operating_status(end)

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
                if not self._reservation_operating_status(candidate)[
                    "available"
                ]:
                    continue
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
        table_number: int = 0,
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
        chosen_table = int(table_number or 0)
        if chosen_table < 0:
            chosen_table = 0
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
            "table_number": chosen_table,
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
            # create_booking always re-queries live tables below; optional table_number
            # must also match a remembered availability offer (require_offered_table).
            if chosen_table:
                from app.availability_offer import require_offered_table

                require_offered_table(
                    call_id,
                    table_number=chosen_table,
                    date=date,
                    time=time,
                    party_size=party_size,
                )
            location_pref = normalize_preferred_location(preferred_location)
            if location_pref and not chosen_table:
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
                preferred_location="" if chosen_table else location_pref,
                conn=conn,
                require_location_match=bool(location_pref) and not chosen_table,
                table_number=chosen_table,
            )
            table = tables[0] if tables else None
            if not table:
                raise RestaurantServiceError(
                    (
                        f"Table {chosen_table} is no longer available for that slot."
                        if chosen_table
                        else "That slot is no longer available. Offer a new time."
                    ),
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
                SET booking_id = $1,
                    fulfillment_type = 'dine_in'
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
        confirmation_payload = update_booking_confirmation_payload(
            booking_id=booking_id,
            date=date,
            time=time,
            party_size=party_size,
            preferred_location=preferred_location or "",
            seating_preference=seating_preference,
            seating_backup=seating_backup,
            seating_avoid=seating_avoid,
            dietary=dietary,
            occasion=occasion,
            extra_notes=extra_notes if extra_notes is not None else notes,
            customer_name=new_name,
            require_approval_for_paid_items=require_approval_for_paid_items,
        )
        # Party-size edits must cite a fresh check_table_availability for that size.
        if party_size > 0:
            from app.availability_offer import require_fresh_availability_for_party_change
            from app.call_memory import get_reservation_draft as _load_draft

            draft_now = _load_draft(call_id)
            current_party = int(draft_now.get("party_size") or 0)
            if party_size != current_party:
                slot_date = date or str(draft_now.get("date") or "")
                slot_time = time or str(draft_now.get("time") or "")
                require_fresh_availability_for_party_change(
                    call_id,
                    date=slot_date,
                    time=slot_time,
                    party_size=party_size,
                    preferred_location=preferred_location
                    or (
                        seating_preference
                        if isinstance(seating_preference, str)
                        else ""
                    ),
                )
        if confirmed is not True:
            digest = register_pending_confirmation(
                call_id,
                ACTION_UPDATE_CONFIRMED_BOOKING,
                confirmation_payload,
            )
            try:
                await self.persist_call_state(call_id, pending_state_patch(call_id))
            except Exception:
                pass
            return {
                "updated": False,
                "pending": True,
                "readback_required": True,
                "pending_confirmation_hash": digest,
                "proposed": confirmation_payload,
                "message": (
                    "Read every proposed change back to the caller and ask if that is "
                    "correct. Only after an explicit yes call update_confirmed_booking "
                    "with caller_confirmed=true."
                ),
            }
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
        # Fail closed before opening a DB transaction when the gate is not satisfied.
        require_pending_confirmation(
            call_id, ACTION_UPDATE_CONFIRMED_BOOKING, confirmation_payload
        )

        async def operation(conn: Any) -> JsonDict:
            require_pending_confirmation(
                call_id, ACTION_UPDATE_CONFIRMED_BOOKING, confirmation_payload
            )
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
            previous_table_number = table_number
            previous_location = location or ""
            location_changed = bool(
                location_pref
                and normalize_preferred_location(str(location or "")) != location_pref
            )

            if changing_slot or location_changed:
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
            await self._update_order_notes_with_conn(
                conn,
                order_notes=rebuilt_notes,
                booking_id=booking_id,
            )
            await conn.execute(
                "UPDATE orders SET customer_name = $1 WHERE booking_id = $2",
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
            clear_pending_confirmation(call_id, ACTION_UPDATE_CONFIRMED_BOOKING)
            table_reassigned = (
                previous_table_number != table_number
                or (previous_location or "") != (location or "")
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
                "table_reassigned": table_reassigned,
                "previous_table_number": previous_table_number,
                "previous_location": previous_location,
                "seating_preference": draft.get("seating_preference") or "",
            }

        result, replayed = await self._idempotent_write(
            action="update_confirmed_booking",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        if result.get("updated"):
            try:
                await self.persist_call_state(call_id, pending_state_patch(call_id))
            except Exception:
                pass
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
                await self._update_order_notes_with_conn(
                    conn,
                    order_notes=combined,
                    booking_id=row["id"],
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
                    await self._update_order_notes_with_conn(
                        conn,
                        order_notes=combined,
                        order_id=order["id"],
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
            "customer_phone": row["customer_phone"] or "",
            "booked_at": row["booked_at"].isoformat(),
            "date": row["booked_at"].date().isoformat(),
            "time": row["booked_at"].strftime("%H:%M"),
            "party_size": row["party_size"],
            "status": row["status"],
            "table_number": row["table_number"],
            "location": row["location"] or "",
            "notes": row["notes"] or "",
        }

    async def sync_confirmed_draft_from_booking(
        self,
        call_id: str,
        booking_id: int = 0,
    ) -> JsonDict:
        """Reload the live bookings row into call memory — single SoT post-confirm."""
        from app.call_memory import apply_live_booking_to_memory, get_call_memory

        call_id = self._require_call_id(call_id)
        memory = get_call_memory(call_id)
        bid = int(booking_id or memory.get("booking_id") or 0)
        if bid <= 0:
            from app.call_memory import get_reservation_draft

            return get_reservation_draft(call_id)
        live = await self.lookup_booking(booking_id=bid)
        draft = apply_live_booking_to_memory(call_id, live)
        try:
            from app.reservation_draft import flatten_draft

            await self.persist_call_state(
                call_id,
                {
                    **flatten_draft(
                        draft,
                        guest_notes=str(get_call_memory(call_id).get("guest_notes") or ""),
                    ),
                    "table_number": live.get("table_number"),
                    "table_location": live.get("location") or "",
                },
                caller_phone=str(draft.get("customer_phone") or ""),
            )
        except RestaurantServiceError:
            pass
        return draft

    async def reverse_pending_cancellation(self, call_id: str) -> JsonDict:
        from app.call_memory import get_call_memory

        call_id = self._require_call_id(call_id)
        pending = get_pending_confirmation(call_id, ACTION_CANCEL_BOOKING)
        pending_payload = dict((pending or {}).get("payload") or {})
        memory = get_call_memory(call_id)
        try:
            booking_id = int(
                pending_payload.get("booking_id") or memory.get("booking_id") or 0
            )
        except (TypeError, ValueError):
            booking_id = 0
        if booking_id <= 0:
            return {
                "reversed": False,
                "status": "unknown",
                "message": (
                    "I don't have a verified reservation or pending cancellation to change. "
                    "No cancellation action was taken."
                ),
            }
        booking = await self.lookup_booking(booking_id=booking_id)
        if pending:
            state_patch = pending_state_patch(call_id)
            remaining = dict(state_patch.get("pending_confirmations") or {})
            remaining.pop(ACTION_CANCEL_BOOKING, None)
            state_patch["pending_confirmations"] = remaining
            await self.persist_call_state(call_id, state_patch)
            clear_pending_confirmation(call_id, ACTION_CANCEL_BOOKING)
        if booking["status"] == "cancelled":
            message = (
                "That reservation is already cancelled, so I can't say it is unchanged. "
                "No further cancellation action was taken."
            )
        elif pending:
            message = (
                "I stopped the pending cancellation. Your reservation remains "
                f"{booking['status']}."
            )
        else:
            message = f"Your reservation is {booking['status']} and no cancellation is pending."
        return {
            "reversed": bool(pending),
            "booking_id": booking_id,
            "status": booking["status"],
            "message": message,
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
        confirmation_payload = cancel_booking_confirmation_payload(
            booking_id=booking_id,
            customer_name=name,
            customer_phone=phone,
            reason=reason,
        )
        if confirmed is not True:
            digest = register_pending_confirmation(
                call_id,
                ACTION_CANCEL_BOOKING,
                confirmation_payload,
            )
            try:
                await self.persist_call_state(call_id, pending_state_patch(call_id))
            except Exception:
                pass
            return {
                "cancelled": False,
                "pending": True,
                "readback_required": True,
                "pending_confirmation_hash": digest,
                "proposed": confirmation_payload,
                "message": (
                    "Confirm the cancellation with the caller (booking reference and "
                    "name). Only after an explicit yes call cancel_booking with "
                    "caller_confirmed=true."
                ),
            }
        payload = {
            "call_id": call_id,
            "booking_id": booking_id,
            "customer_name": name,
            "customer_phone": phone,
            "reason": reason,
        }
        require_pending_confirmation(
            call_id, ACTION_CANCEL_BOOKING, confirmation_payload
        )

        async def operation(conn: Any) -> JsonDict:
            require_pending_confirmation(
                call_id, ACTION_CANCEL_BOOKING, confirmation_payload
            )
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
                clear_pending_confirmation(call_id, ACTION_CANCEL_BOOKING)
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
            clear_pending_confirmation(call_id, ACTION_CANCEL_BOOKING)
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
        if result.get("cancelled"):
            try:
                await self.persist_call_state(call_id, pending_state_patch(call_id))
            except Exception:
                pass
        return {**result, "idempotent_replay": replayed}

    async def list_menu(self, *, available_only: bool = True) -> JsonDict:
        try:
            knowledge = get_restaurant_knowledge()
            canonical_source_id = knowledge.metadata["source_id"]
            canonical_item_ids = [item["item_id"] for item in knowledge.menu_items]
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, canonical_id, name, aliases, category, price,
                           description, dietary, ingredients, allergens,
                           service_periods, availability_status,
                           knowledge_metadata, source_id, data_version,
                           effective_from, effective_to,
                           COALESCE(price_estimated, FALSE) AS price_estimated,
                           (
                               available = TRUE
                               AND availability_status = 'available'
                               AND (effective_from IS NULL OR effective_from <= CURRENT_DATE)
                               AND (effective_to IS NULL OR effective_to >= CURRENT_DATE)
                           ) AS available
                    FROM menu_items
                    WHERE source_id = $2
                      AND canonical_id = ANY($3::text[])
                      AND (
                          $1::boolean = FALSE
                          OR (
                              available = TRUE
                              AND availability_status = 'available'
                              AND (effective_from IS NULL OR effective_from <= CURRENT_DATE)
                              AND (effective_to IS NULL OR effective_to >= CURRENT_DATE)
                          )
                    )
                    ORDER BY category, name
                    """,
                    available_only,
                    canonical_source_id,
                    canonical_item_ids,
                )
        except Exception as exc:
            raise RestaurantServiceError(
                "Current menu information is temporarily unavailable. I can take a callback message, but I cannot confirm menu availability.",
                code="knowledge_unavailable",
                status=503,
            ) from exc

        now = _restaurant_now()

        def _menu_payload(row: Any) -> JsonDict:
            metadata = _json_value(row["knowledge_metadata"] or {})
            if not isinstance(metadata, dict):
                metadata = {}
            service_status = knowledge.menu_service_status(
                row["service_periods"] or [], now
            )
            available = bool(row["available"]) and bool(service_status["available"])
            return {
                **metadata,
                "id": row["id"],
                "item_id": row["canonical_id"] or "",
                "name": row["name"],
                "aliases": list(row["aliases"] or []),
                "category": row["category"],
                "category_id": f"category.{row['category']}",
                "price": float(row["price"]),
                "price_estimated": bool(row["price_estimated"]),
                "description": row["description"] or "",
                "dietary": list(row["dietary"] or []),
                "dietary_tags": list(row["dietary"] or []),
                "ingredients": list(row["ingredients"] or []),
                "allergens": list(row["allergens"] or []),
                "service_periods": list(row["service_periods"] or []),
                "availability": row["availability_status"],
                "available": available,
                "service_status": service_status["status"],
                "service_message": service_status["customer_message"],
                "source_id": row["source_id"] or "",
                "data_version": row["data_version"] or "",
                "effective_from": str(row["effective_from"] or ""),
                "effective_to": str(row["effective_to"] or ""),
            }
        canonical_rows = [
            row
            for row in rows
            if row["source_id"] == canonical_source_id
            and row["canonical_id"] in canonical_item_ids
        ]
        items = [_menu_payload(row) for row in canonical_rows]
        if available_only:
            items = [item for item in items if item["available"]]
        return {
            "restaurant_name": knowledge.identity["name"],
            "status": "current",
            "items": items,
            "allergen_notice": (
                "Harbor & Hearth uses shared equipment and preparation areas. No item is "
                "guaranteed allergen-free or free from cross-contact. For a severe allergy, "
                "offer the configured kitchen/staff route or an honest callback message."
            ),
        }

    async def find_menu_item(self, item_name: str) -> JsonDict:
        requested = normalize_knowledge_text(item_name)
        if not requested:
            raise RestaurantServiceError("An item_name is required.")
        menu = await self.list_menu(available_only=False)
        items: list[JsonDict] = menu["items"]
        exact = [
            item
            for item in items
            if requested
            in {
                normalize_knowledge_text(item["name"]),
                *(normalize_knowledge_text(alias) for alias in item.get("aliases") or []),
            }
        ]
        if len(exact) == 1:
            return {"match": exact[0], "needs_confirmation": False, "candidates": []}

        requested_tokens = text_tokens(requested)
        contained = []
        for item in items:
            item_tokens = text_tokens(item["name"])
            if requested_tokens and (
                requested_tokens < item_tokens or item_tokens < requested_tokens
            ):
                contained.append(item)
        if len(contained) == 1:
            return {
                "match": None,
                "needs_confirmation": True,
                "candidates": contained,
            }
        if contained:
            contained.sort(key=lambda item: item["name"])
            return {
                "match": None,
                "needs_confirmation": True,
                "candidates": contained[:3],
            }

        ranked = sorted(
            (
                (
                    max(
                        SequenceMatcher(
                            None, requested, normalize_knowledge_text(candidate)
                        ).ratio()
                        for candidate in [item["name"], *(item.get("aliases") or [])]
                    ),
                    item,
                )
                for item in items
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        candidates = [item for score, item in ranked[:3] if score >= 0.72]
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
        modifier_ids: tuple[str, ...] | list[str] = (),
        removals: tuple[str, ...] | list[str] = (),
        substitutions: tuple[str, ...] | list[str] = (),
        order_notes: str = "",
        allergy_notes: str = "",
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
                "candidates": [
                    item
                    for item in (await self.list_menu())["items"]
                    if item.get("item_id") in set(menu_item.get("alternative_item_ids") or [])
                ],
            }
        if menu_item.get("item_id", "").startswith("menu.alcohol.") or "alcohol" in (
            menu_item.get("dietary_tags") or []
        ):
            raise RestaurantServiceError(
                "Alcohol is available as read-only synthetic menu information and cannot be added to an order.",
                code="alcohol_transaction_unsupported",
                status=409,
            )
        customization = get_restaurant_knowledge().resolve_customization(
            menu_item,
            modifier_ids=modifier_ids,
            removals=removals,
            substitutions=substitutions,
        )
        if customization["status"] != "valid":
            return {
                "added": False,
                "needs_confirmation": customization["status"] == "clarification_required",
                "customization_status": customization["status"],
                "message": customization["message"],
                "choices": customization.get("choices", []),
                "candidates": [],
            }
        name = self._validate_name(customer_name) if customer_name else ""
        phone = self._validate_phone(customer_phone) if customer_phone else ""
        notes = notes.strip()[:300]
        order_notes = order_notes.strip()[:500]
        allergy_notes = allergy_notes.strip()[:500]
        payload = {
            "call_id": call_id,
            "menu_item_id": menu_item["id"],
            "quantity": quantity,
            "notes": notes,
            "modifiers": customization["modifiers"],
            "removals": customization["removals"],
            "substitutions": customization["substitutions"],
            "order_notes": order_notes,
            "allergy_notes": allergy_notes,
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
                        notes = CASE WHEN $4 = '' THEN notes ELSE $4 END,
                        allergy_notes = CASE WHEN $5 = '' THEN allergy_notes ELSE $5 END,
                        draft_version = draft_version + 1
                    WHERE id = $6
                    """,
                    resolved_booking or None,
                    name,
                    phone,
                    order_notes,
                    allergy_notes,
                    order_id,
                )
            elif order_notes or allergy_notes:
                await conn.execute(
                    """
                    UPDATE orders
                    SET notes = $1, allergy_notes = $2,
                        draft_version = draft_version + 1
                    WHERE id = $3
                    """,
                    order_notes,
                    allergy_notes,
                    order_id,
                )
            approval_required = await self._paid_item_approval_required(
                conn,
                call_id,
                resolved_booking or int(order.get("booking_id") or 0),
            )
            unit_price = float(menu_item["price"]) + float(customization["price_delta"])
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
                        SET proposed = FALSE, quantity = $1, notes = $2,
                            unit_price = $3, modifiers = $4::jsonb,
                            removals = $5, substitutions = $6::jsonb
                        WHERE id = $7
                        """,
                        quantity,
                        notes,
                        unit_price,
                        json.dumps(customization["modifiers"], sort_keys=True),
                        customization["removals"],
                        json.dumps(customization["substitutions"], sort_keys=True),
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
                    (order_id, menu_item_id, item_name, quantity, unit_price, notes,
                     modifiers, removals, substitutions, proposed)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9::jsonb, $10)
                RETURNING id
                """,
                order_id,
                menu_item["id"],
                menu_item["name"],
                quantity,
                unit_price,
                notes,
                json.dumps(customization["modifiers"], sort_keys=True),
                customization["removals"],
                json.dumps(customization["substitutions"], sort_keys=True),
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
            fulfillment = "dine_in" if booking_id else "pickup"
            created = await conn.fetchrow(
                """
                INSERT INTO orders
                    (session_id, booking_id, customer_name, customer_phone,
                     status, draft_version, fulfillment_type)
                VALUES ($1, $2, $3, $4, 'pending', 1, $5)
                RETURNING id, booking_id, customer_name, customer_phone,
                          draft_version, status, fulfillment_type, FALSE AS existing
                """,
                call_id,
                booking_id or None,
                customer_name,
                customer_phone,
                fulfillment,
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
                   status, total_amount, draft_version, created_at, fulfillment_type,
                   fulfillment_details, notes, allergy_notes
            FROM orders WHERE id = $1
            """,
            order_id,
        )
        if not order:
            raise RestaurantServiceError("Order not found.", code="order_not_found", status=404)
        items = await conn.fetch(
            """
            SELECT oi.id, oi.item_name, oi.quantity, oi.unit_price, oi.subtotal,
                   oi.notes, oi.modifiers, oi.removals, oi.substitutions,
                   mi.canonical_id, mi.dietary,
                   COALESCE(oi.proposed, FALSE) AS proposed
            FROM order_items oi
            LEFT JOIN menu_items mi ON mi.id = oi.menu_item_id
            WHERE oi.order_id = $1 ORDER BY oi.id
            """,
            order_id,
        )
        committed = [item for item in items if not item["proposed"]]
        proposed = [item for item in items if item["proposed"]]
        item_total = sum(float(item["subtotal"]) for item in committed)

        def _item_payload(item: Any) -> JsonDict:
            return {
                "order_item_id": item["id"],
                "item_name": item["item_name"],
                "item_id": item["canonical_id"] or "",
                "quantity": item["quantity"],
                "unit_price": float(item["unit_price"]),
                "subtotal": float(item["subtotal"]),
                "notes": item["notes"] or "",
                "modifiers": _json_value(item["modifiers"] or []),
                "removals": list(item["removals"] or []),
                "substitutions": _json_value(item["substitutions"] or []),
                "dietary_tags": list(item["dietary"] or []),
                "proposed": bool(item["proposed"]),
            }

        stored = order["fulfillment_type"]
        if stored in {"dine_in", "pickup", "delivery"}:
            fulfillment = stored
        else:
            fulfillment = "dine_in" if order["booking_id"] else "pickup"
        delivery_fee = 0.0
        if fulfillment == "delivery":
            delivery_fee = float(_delivery_rule().get("delivery_fee") or 0)
        calculated_total = round(item_total + delivery_fee, 2)
        return {
            "order_id": order["id"],
            "call_id": order["session_id"],
            "booking_id": order["booking_id"],
            "customer_name": order["customer_name"] or "",
            "status": order["status"],
            "draft_version": order["draft_version"],
            "item_total": item_total,
            "fees": ([{"fee_id": "fee.delivery", "name": "delivery fee", "amount": delivery_fee}] if delivery_fee else []),
            "total": calculated_total,
            "items": [_item_payload(item) for item in committed],
            "proposed_items": [_item_payload(item) for item in proposed],
            "fulfillment": fulfillment,
            "fulfillment_type": fulfillment,
            "fulfillment_details": _json_value(order["fulfillment_details"] or {}),
            "order_notes": order["notes"] or "",
            "allergy_notes": order["allergy_notes"] or "",
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

    async def set_order_fulfillment(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        fulfillment_type: str,
        booking_id: int | None = None,
        delivery_address: str = "",
        delivery_instructions: str = "",
    ) -> JsonDict:
        """Replace fulfillment on the existing pending order; never creates a second order."""
        call_id = self._require_call_id(call_id)
        fulfillment = str(fulfillment_type or "").strip().casefold()
        if fulfillment not in {"dine_in", "pickup", "delivery"}:
            raise RestaurantServiceError(
                "fulfillment_type must be dine_in, pickup, or delivery.",
                code="invalid_fulfillment",
                status=400,
            )
        resolved_booking: int | None
        if booking_id is None:
            resolved_booking = None
        else:
            try:
                resolved_booking = int(booking_id)
            except (TypeError, ValueError):
                resolved_booking = 0
            if resolved_booking <= 0:
                resolved_booking = None
        if fulfillment == "dine_in" and not resolved_booking:
            # Fall back to active booking in session when switching to dine-in.
            pool = await get_pool()
            async with pool.acquire() as conn:
                resolved_booking = await self._booking_id_from_session(conn, call_id) or None
            if not resolved_booking:
                raise RestaurantServiceError(
                    "dine_in requires a booking_id (or an active reservation on this call).",
                    code="booking_required",
                    status=409,
                )
        if fulfillment in {"pickup", "delivery"}:
            resolved_booking = None
        fulfillment_details: JsonDict = {}
        if fulfillment == "delivery":
            address = " ".join(delivery_address.split())[:300]
            delivery_rule = _delivery_rule()
            postal_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", address)
            if not address or not postal_match:
                raise RestaurantServiceError(
                    "A complete delivery address with a five-digit postal code is required.",
                    code="delivery_postal_code_required",
                    status=409,
                )
            postal_code = postal_match.group(1)
            eligible_postal_codes = {
                str(value) for value in delivery_rule.get("eligible_postal_codes") or []
            }
            if postal_code not in eligible_postal_codes:
                raise RestaurantServiceError(
                    "That postal code is outside the configured synthetic local delivery zone. The order was not changed.",
                    code="delivery_outside_zone",
                    status=409,
                )
            fulfillment_details = {
                "address": address,
                "instructions": " ".join(delivery_instructions.split())[:300],
                "zone_id": str(delivery_rule.get("delivery_zone_id") or ""),
                "postal_code": postal_code,
                "zone_status": "eligible",
                "provider": "synthetic_local",
                "live_integration": False,
            }
        payload = {
            "call_id": call_id,
            "fulfillment_type": fulfillment,
            "booking_id": resolved_booking or 0,
            "fulfillment_details": fulfillment_details,
        }

        async def operation(conn: Any) -> JsonDict:
            order = await conn.fetchrow(
                """
                SELECT id, status FROM orders
                WHERE session_id = $1 AND status IN ('pending', 'confirmed')
                ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id DESC
                LIMIT 1
                FOR UPDATE
                """,
                call_id,
            )
            if not order:
                raise RestaurantServiceError(
                    "No order exists for this call.",
                    code="order_not_found",
                    status=404,
                )
            if order["status"] == "confirmed":
                raise RestaurantServiceError(
                    "Fulfillment cannot be changed after the order is confirmed.",
                    code="order_already_confirmed",
                    status=409,
                )
            if fulfillment == "dine_in" and resolved_booking:
                booking = await conn.fetchrow(
                    """
                    SELECT id FROM bookings
                    WHERE id = $1 AND status = 'confirmed'
                    """,
                    resolved_booking,
                )
                if not booking:
                    raise RestaurantServiceError(
                        "That booking was not found.",
                        code="booking_not_found",
                        status=404,
                    )
            await conn.execute(
                """
                UPDATE orders
                SET fulfillment_type = $1,
                    booking_id = $2,
                    fulfillment_details = $3::jsonb,
                    draft_version = draft_version + 1
                WHERE id = $4
                """,
                fulfillment,
                resolved_booking,
                json.dumps(fulfillment_details, sort_keys=True),
                order["id"],
            )
            summary = await self._order_summary_with_conn(conn, order["id"])
            return {"updated": True, **summary}

        result, replayed = await self._idempotent_write(
            action="set_order_fulfillment",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def set_order_notes(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        order_notes: str | None = None,
        allergy_notes: str | None = None,
        caller_confirmed: bool = False,
    ) -> JsonDict:
        """Replace explicit order-level notes and advance confirmation integrity."""
        call_id = self._require_call_id(call_id)
        if order_notes is None and allergy_notes is None:
            raise RestaurantServiceError(
                "Provide order_notes or allergy_notes.", code="missing_order_notes"
            )
        cleaned_order_notes = None if order_notes is None else order_notes.strip()[:500]
        cleaned_allergy_notes = None if allergy_notes is None else allergy_notes.strip()[:500]
        payload = {
            "call_id": call_id,
            "order_notes": cleaned_order_notes,
            "allergy_notes": cleaned_allergy_notes,
            "caller_confirmed": bool(caller_confirmed),
        }

        async def operation(conn: Any) -> JsonDict:
            order = await self._lock_order_for_mutation(
                conn,
                call_id=call_id,
                caller_confirmed=caller_confirmed,
            )
            await self._update_order_notes_with_conn(
                conn,
                order_notes=cleaned_order_notes,
                allergy_notes=cleaned_allergy_notes,
                order_id=order["id"],
            )
            return {"updated": True, **await self._order_summary_with_conn(conn, order["id"])}

        result, replayed = await self._idempotent_write(
            action="set_order_notes",
            idempotency_key=idempotency_key,
            call_id=call_id,
            payload=payload,
            operation=operation,
        )
        return {**result, "idempotent_replay": replayed}

    async def update_order_item(
        self,
        *,
        call_id: str,
        idempotency_key: str,
        order_item_id: int,
        quantity: int,
        notes: str | None = None,
        caller_confirmed: bool = False,
    ) -> JsonDict:
        call_id = self._require_call_id(call_id)
        if not 1 <= quantity <= 20:
            raise RestaurantServiceError("Quantity must be between 1 and 20.")
        cleaned_notes = None if notes is None else notes.strip()[:300]
        payload = {
            "call_id": call_id,
            "order_item_id": order_item_id,
            "quantity": quantity,
            "notes": cleaned_notes,
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
                """
                UPDATE order_items
                SET quantity = $1,
                    notes = CASE WHEN $2::text IS NULL THEN notes ELSE $2 END
                WHERE id = $3
                """,
                quantity,
                cleaned_notes,
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
            summary = await self._order_summary_with_conn(conn, order["id"])
            if summary.get("fulfillment") in {"pickup", "delivery"} and (
                not (order["customer_name"] or "").strip()
                or not (order["customer_phone"] or "").strip()
            ):
                raise RestaurantServiceError(
                    "Pickup and delivery orders require a verified name and callback phone before confirmation.",
                    code="fulfillment_contact_required",
                    status=409,
                )
            if summary.get("fulfillment") == "delivery":
                fulfillment_details = summary.get("fulfillment_details", {})
                address = str(fulfillment_details.get("address") or "").strip()
                if not address or fulfillment_details.get("zone_status") != "eligible":
                    raise RestaurantServiceError(
                        "A delivery address in the configured synthetic local zone is required before confirmation.",
                        code="delivery_zone_required",
                        status=409,
                    )
                delivery_minimum = float(_delivery_rule().get("delivery_minimum") or 0)
                if float(summary.get("item_total") or 0) < delivery_minimum:
                    raise RestaurantServiceError(
                        f"Delivery requires a ${delivery_minimum:.2f} food-and-beverage minimum before the delivery fee.",
                        code="delivery_minimum_not_met",
                        status=409,
                    )
            if summary.get("fulfillment") in {"pickup", "delivery"}:
                fulfillment_status = get_restaurant_knowledge().schedule_status(
                    str(summary["fulfillment"]),
                    _restaurant_now(),
                    apply_cutoff=True,
                )
                if not fulfillment_status["available"]:
                    raise RestaurantServiceError(
                        fulfillment_status["customer_message"],
                        code="fulfillment_unavailable",
                        status=409,
                    )
            if not summary["items"]:
                raise RestaurantServiceError(
                    "The draft order is empty.", code="empty_order", status=409
                )
            if any(
                item.get("item_id", "").startswith("menu.alcohol.")
                or "alcohol" in (item.get("dietary_tags") or [])
                for item in summary["items"]
            ):
                raise RestaurantServiceError(
                    "Alcohol is read-only synthetic menu information and cannot be confirmed as an order transaction.",
                    code="alcohol_transaction_unsupported",
                    status=409,
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
                    else (
                        "estimated for local synthetic delivery in 45 to 60 minutes; no live courier is connected"
                        if summary["fulfillment"] == "delivery"
                        else "ready for pickup in about 30 minutes"
                    )
                ),
                "alcohol_verification": "",
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
        """Return one effective canonical topic with explicit unknown states."""
        query = (topic or "").strip()
        try:
            knowledge = get_restaurant_knowledge()
        except KnowledgeFixtureError as exc:
            raise RestaurantServiceError(
                "Current restaurant information is temporarily unavailable. I can take a callback message, but I cannot confirm that policy.",
                code="knowledge_unavailable",
                status=503,
            ) from exc
        identity = knowledge.identity
        metadata = knowledge.metadata
        if not query:
            return {
                "matched": False,
                "status": "missing",
                "answers": [],
                "formatted": "A restaurant information topic is required.",
                "log_unknown": False,
                **metadata,
            }

        topic_match = knowledge.find_topic(query)
        if topic_match.status == "known":
            record = topic_match.records[0]
            result: JsonDict = {
                "matched": True,
                "status": "known",
                "answers": [
                    {
                        "kind": "canonical_topic",
                        "heading": record["topic_id"],
                        "content": record["answer"],
                        "source": record["source_id"],
                    }
                ],
                "formatted": record["answer"],
                "log_unknown": False,
                "restaurant_name": identity["name"],
                "topic_id": record["topic_id"],
                "category_id": record["category_id"],
                "source_id": record["source_id"],
                "schema_version": record["schema_version"],
                "data_version": record["data_version"],
                "effective_from": record["effective_from"],
                "effective_to": record.get("effective_to"),
                "rule": deepcopy(record.get("rule") or {}),
                "escalation_owner": record.get("escalation_owner") or "",
            }
            if record["topic_id"] == "topic.hours":
                result["hours"] = deepcopy(knowledge.raw["hours"])
                resolved_hours = knowledge.resolve_hours_query(query)
                if resolved_hours:
                    result["formatted"] = resolved_hours["customer_message"]
                    result["hours_resolution"] = resolved_hours
                    if resolved_hours["status"] == "unavailable":
                        result.update(
                            {
                                "matched": False,
                                "status": "unavailable",
                                "answers": [],
                            }
                        )
                    else:
                        result["answers"][0]["content"] = resolved_hours[
                            "customer_message"
                        ]
            return result

        if topic_match.status in {"ambiguous", "expired", "future"}:
            return {
                "matched": False,
                "status": "stale" if topic_match.status in {"expired", "future"} else "ambiguous",
                "answers": [],
                "formatted": (
                    "That restaurant information is not currently effective. I won't substitute an older or future policy."
                    if topic_match.status in {"expired", "future"}
                    else "That question could refer to more than one restaurant topic. Please clarify which one you mean."
                ),
                "log_unknown": topic_match.status in {"expired", "future"},
                "restaurant_name": identity["name"],
                "candidate_topic_ids": [record["topic_id"] for record in topic_match.records],
                **metadata,
            }

        # Operator-resolved local FAQs remain an explicit secondary source. They
        # cannot override a canonical topic and carry their own source marker.
        faq_rows: list[JsonDict] = []
        faq_unavailable = False
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT question, answer
                    FROM operator_knowledge
                    WHERE active = TRUE AND restaurant_id = $1
                    ORDER BY updated_at DESC, id DESC
                    """,
                    identity["restaurant_id"],
                )
            faq_rows = [dict(row) for row in rows]
        except Exception:
            faq_unavailable = True

        hits = search_faq_rows(query, faq_rows, limit=1)
        if hits:
            hit = hits[0]
            return {
                "matched": True,
                "status": "known",
                "answers": [
                    {
                        "kind": hit.get("kind"),
                        "heading": hit.get("heading") or "",
                        "content": hit.get("content") or "",
                        "source": hit.get("source") or "",
                    }
                ],
                "formatted": format_knowledge_hits(hits),
                "log_unknown": False,
                "restaurant_name": identity["name"],
                "source_id": "operator_knowledge",
            }
        transfer_available = bool(current_staff_transfer_number())
        return {
            "matched": False,
            "status": "unknown",
            "answers": [],
            "formatted": (
                "That answer is not in the current Harbor & Hearth information. "
                + (
                    "A configured staff transfer can be requested, or I can take a callback message."
                    if transfer_available
                    else "I cannot transfer right now, but I can take a callback message."
                )
            ),
            "log_unknown": True,
            "restaurant_name": identity["name"],
            "transfer_available": transfer_available,
            "operator_source_status": "unavailable" if faq_unavailable else "available",
            **metadata,
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
                SELECT id, restaurant_id, question, answer, source_gap_id, active,
                       created_at, updated_at
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
        restaurant_id = get_restaurant_knowledge().identity["restaurant_id"]
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
                    WHERE restaurant_id = $1 AND LOWER(question) = LOWER($2)
                    FOR UPDATE
                    """,
                    restaurant_id,
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
                        INSERT INTO operator_knowledge
                            (restaurant_id, question, answer, source_gap_id, active)
                        VALUES ($1, $2, $3, $4, TRUE)
                        RETURNING id, question, answer
                        """,
                        restaurant_id,
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
