"""Constrained tool bridge for the development Realtime adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol

from app.native_voice.speech import ToolEvidence


MUTATING_TOOLS = frozenset(
    {
        "create_booking",
        "update_confirmed_booking",
        "cancel_booking",
        "add_guest_note",
        "update_reservation_draft",
        "add_order_item",
        "set_order_fulfillment",
        "set_order_notes",
        "update_order_item",
        "remove_order_item",
        "confirm_order",
    }
)

BOOKING_SCOPED_TOOLS = frozenset(
    {"lookup_booking", "update_confirmed_booking", "cancel_booking", "add_guest_note"}
)

ORDER_SCOPED_TOOLS = frozenset(
    {
        "lookup_order",
        "get_order_summary",
        "add_order_item",
        "set_order_fulfillment",
        "set_order_notes",
        "update_order_item",
        "remove_order_item",
        "confirm_order",
    }
)

KNOWN_TOOLS = frozenset(
    {
        "check_menu_item_availability",
        "get_full_menu",
        "check_table_availability",
        "get_order_summary",
        "add_order_item",
        "set_order_fulfillment",
        "set_order_notes",
        "update_order_item",
        "remove_order_item",
        "confirm_order",
        "get_reservation_draft",
        "update_reservation_draft",
        "create_booking",
        "update_confirmed_booking",
        "lookup_booking",
        "cancel_booking",
        "add_guest_note",
        "lookup_order",
    }
)


def _hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_effect_values(value: Any) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    result: list[str] = []
    for entry in values:
        if isinstance(entry, Mapping):
            option_id = str(entry.get("option_id") or entry.get("id") or entry.get("name") or "").strip()
            selection = str(entry.get("selection") or "").strip()
            result.append(f"{option_id}:{selection}" if selection else option_id)
        else:
            result.append(str(entry).strip())
    return tuple(sorted(result))


def _canonical_phone(value: Any) -> str:
    from app.security import normalize_caller_phone

    raw = str(value or "").strip()
    normalized = normalize_caller_phone(raw)
    if normalized:
        return normalized
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return digits


def _canonical_name(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _is_failure(
    value: Any,
    *,
    allow_negative_availability: bool = False,
    authoritative_read: bool = False,
) -> bool:
    if isinstance(value, Mapping):
        return (
            value.get("ok") is False
            or bool(value.get("error"))
            or bool(
                (value.get("pending") or value.get("readback_required"))
                and not authoritative_read
            )
            or bool((value.get("unavailable") and not allow_negative_availability) or value.get("no_op") or value.get("proposed"))
            or any(key in value and value[key] is False for key in ("added", "updated", "removed", "cancelled", "saved"))
        )
    if isinstance(value, str):
        lowered = value.casefold()
        failure_phrases = (
            "pending confirmation",
            "readback_required",
            "not applied",
            "unchanged",
            "proposed",
            "not added",
            "not saved",
            "no matching",
            "no-op",
            "could not",
            "cannot",
            "failed",
        )
        if not allow_negative_availability:
            failure_phrases += ("unavailable", "not available")
        return bool(re.match(r"^[a-z][a-z0-9_]*:", value.strip())) or any(
            phrase in lowered for phrase in failure_phrases
        )
    return False


_ORDER_READBACK_FIELDS = (
    "order_id",
    "call_id",
    "booking_id",
    "status",
    "draft_version",
    "total",
    "items",
    "proposed_items",
    "fulfillment",
    "fulfillment_type",
    "fulfillment_details",
    "order_notes",
    "allergy_notes",
    "unresolved_fields",
    "state_version",
)
_ORDER_ITEM_FIELDS = (
    "order_item_id",
    "item_id",
    "item_name",
    "quantity",
    "modifiers",
    "removals",
    "substitutions",
    "notes",
)


def _canonical_order_payload(readback: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: readback.get(key) for key in _ORDER_READBACK_FIELDS}
    for key in ("items", "proposed_items"):
        payload[key] = [
            {field: item.get(field) for field in _ORDER_ITEM_FIELDS}
            for item in readback.get(key) or ()
            if isinstance(item, Mapping)
        ]
    return payload


def _order_readback_hash(readback: Mapping[str, Any]) -> str:
    return _hash(_canonical_order_payload(readback))


def _tool_definition(name: str, description: str, properties: Mapping[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": dict(properties),
            "required": required,
            "additionalProperties": False,
        },
    }


class ToolExecutor(Protocol):
    async def invoke(self, name: str, arguments: Mapping[str, Any]) -> Any: ...

    async def readback(self, name: str, arguments: Mapping[str, Any], result: Any) -> Any: ...


class OfflineToolExecutor:
    def __init__(
        self,
        *,
        result: Any = None,
        readback: Any = None,
        order_scope: Mapping[str, Any] | None = None,
        booking_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.result = result
        self.readback_result = readback
        self.order_scope = dict(order_scope or {})
        self.booking_identity = dict(booking_identity or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def invoke(self, name: str, arguments: Mapping[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return self.result if self.result is not None else {"ok": False, "error": "offline_tool_executor_required"}

    async def readback(self, name: str, arguments: Mapping[str, Any], result: Any) -> Any:
        return self.readback_result


@dataclass(frozen=True)
class ToolOutcome:
    name: str
    call_id: str
    arguments: dict[str, Any]
    result: Any
    success: bool
    error: str = ""
    readback: Any = None
    readback_verified: bool = False
    state_version: int = 0
    replayed: bool = False
    facts: dict[str, Any] = field(default_factory=dict)
    confirmation_text: str = ""
    confirmation_hash: str = ""
    operation_id: str = ""
    pending: bool = False

    def as_evidence(self, *, turn_id: str) -> ToolEvidence:
        return ToolEvidence(
            action=self.name,
            call_id=self.call_id,
            turn_id=turn_id,
            state_version=self.state_version,
            success=self.success,
            readback_verified=self.readback_verified,
            facts=self.facts,
            replayed=self.replayed,
            confirmation_text=self.confirmation_text,
            confirmation_hash=self.confirmation_hash,
            pending=self.pending,
        )


class RestaurantToolExecutor:
    """Use the existing rollback tools and service as the authority boundary."""

    def __init__(self, *, service: Any | None = None) -> None:
        from app.services.restaurant import restaurant_service

        self._service = service or restaurant_service
        self._native_service = service
        self.session_id = ""
        self._operation_turn_id = ""
        if service is None:
            from app.tools import db

            self._tools = {
                name: getattr(db, name)
                for name in (
                    "check_table_availability",
                    "create_booking",
                    "update_reservation_draft",
                    "get_reservation_draft",
                    "update_confirmed_booking",
                    "lookup_booking",
                    "cancel_booking",
                    "add_guest_note",
                    "add_order_item",
                    "get_order_summary",
                    "set_order_fulfillment",
                    "set_order_notes",
                    "update_order_item",
                    "remove_order_item",
                    "confirm_order",
                    "lookup_order",
                    "get_full_menu",
                    "check_menu_item_availability",
                )
            }
        else:
            self._tools = {}

    def bind_session(self, session_id: str) -> None:
        if self.session_id and self.session_id != session_id:
            raise ValueError("tool executor is already bound to another session")
        self.session_id = session_id

    def bind_operation_scope(self, turn_id: str) -> None:
        self._operation_turn_id = turn_id

    def clear_operation_scope(self) -> None:
        self._operation_turn_id = ""

    def _native_idempotency_key(self, name: str, arguments: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "session_id": self.session_id,
                    "turn_id": self._operation_turn_id,
                    "operation": name,
                    "arguments": dict(arguments),
                },
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    async def _invoke_native(self, name: str, arguments: Mapping[str, Any]) -> Any:
        args = dict(arguments)
        session_id = str(args.get("session_id") or self.session_id)
        key = self._native_idempotency_key(name, args)
        if name == "check_table_availability":
            return await self._service.check_availability(
                str(args.get("date") or ""),
                str(args.get("time") or ""),
                int(args.get("party_size") or 0),
                preferred_location=str(args.get("preferred_location") or ""),
                call_id=session_id,
            )
        if name == "get_full_menu":
            return await self._service.list_menu(available_only=False)
        if name == "check_menu_item_availability":
            return await self._service.find_menu_item(str(args.get("item_name") or ""))
        if name == "lookup_booking":
            return await self._service.lookup_booking(
                booking_id=int(args.get("booking_id") or 0),
                customer_name=str(args.get("customer_name") or ""),
                customer_phone=str(args.get("customer_phone") or ""),
            )
        if name == "lookup_order":
            return await self._service.lookup_order(
                order_id=int(args.get("order_id") or 0),
                customer_name=str(args.get("customer_name") or ""),
            )
        if name == "get_order_summary":
            result = await self._service.get_order_summary(call_id=session_id)
            if self._native_service is not None and hasattr(self._service, "load_call_state"):
                state = (await self._service.load_call_state(session_id)).get("state") or {}
                result["guest_notes"] = str(state.get("guest_notes") or "")
            return result
        if name == "create_booking":
            return await self._service.create_booking(
                call_id=session_id,
                idempotency_key=key,
                customer_name=str(args.get("name") or ""),
                customer_phone=str(args.get("phone") or ""),
                date=str(args.get("date") or ""),
                time=str(args.get("time") or ""),
                party_size=int(args.get("party_size") or 0),
                notes=str(args.get("notes") or ""),
                confirmed=bool(args.get("caller_confirmed")),
                table_number=int(args.get("table_number") or 0),
            )
        if name == "update_confirmed_booking":
            return await self._service.update_confirmed_booking(
                call_id=session_id,
                idempotency_key=key,
                booking_id=int(args.get("booking_id") or 0),
                confirmed=bool(args.get("caller_confirmed")),
                date=str(args.get("date") or ""),
                time=str(args.get("time") or ""),
                party_size=int(args.get("party_size") or 0),
                seating_preference=args.get("seating_preference"),
                seating_backup=args.get("seating_backup"),
                seating_avoid=args.get("seating_avoid"),
                dietary=args.get("dietary"),
                occasion=args.get("occasion"),
                extra_notes=args.get("extra_notes"),
                customer_name=str(args.get("customer_name") or ""),
                require_approval_for_paid_items=args.get("require_approval_for_paid_items"),
            )
        if name == "cancel_booking":
            return await self._service.cancel_booking(
                call_id=session_id,
                idempotency_key=key,
                booking_id=int(args.get("booking_id") or 0),
                customer_name=str(args.get("customer_name") or ""),
                customer_phone=str(args.get("customer_phone") or ""),
                reason=str(args.get("reason") or ""),
                confirmed=bool(args.get("caller_confirmed")),
            )
        if name == "add_guest_note":
            return await self._service.add_guest_note(
                call_id=session_id,
                idempotency_key=key,
                note=str(args.get("note") or ""),
                booking_id=int(args.get("booking_id") or 0),
            )
        if name == "add_order_item":
            return await self._service.add_order_item(
                call_id=session_id,
                idempotency_key=key,
                item_name=str(args.get("item_name") or ""),
                quantity=int(args.get("quantity") or 1),
                notes=str(args.get("notes") or ""),
                modifier_ids=args.get("modifier_ids") or [],
                removals=args.get("removals") or [],
                substitutions=args.get("substitutions") or [],
                order_notes=str(args.get("order_notes") or ""),
                allergy_notes=str(args.get("allergy_notes") or ""),
                booking_id=int(args.get("booking_id") or 0),
                customer_name=str(args.get("customer_name") or ""),
                customer_phone=str(args.get("customer_phone") or ""),
                caller_confirmed=bool(args.get("caller_confirmed")),
            )
        if name == "set_order_fulfillment":
            return await self._service.set_order_fulfillment(
                call_id=session_id,
                idempotency_key=key,
                fulfillment_type=str(args.get("fulfillment_type") or ""),
                booking_id=args.get("booking_id"),
                delivery_address=str(args.get("delivery_address") or ""),
                delivery_instructions=str(args.get("delivery_instructions") or ""),
            )
        if name == "set_order_notes":
            return await self._service.set_order_notes(
                call_id=session_id,
                idempotency_key=key,
                order_notes=args.get("order_notes"),
                allergy_notes=args.get("allergy_notes"),
                caller_confirmed=bool(args.get("caller_confirmed")),
            )
        if name == "update_order_item":
            return await self._service.update_order_item(
                call_id=session_id,
                idempotency_key=key,
                order_item_id=int(args.get("order_item_id") or 0),
                quantity=int(args.get("quantity") or 0),
                notes=args.get("notes"),
                caller_confirmed=bool(args.get("caller_confirmed")),
            )
        if name == "remove_order_item":
            return await self._service.remove_order_item(
                call_id=session_id,
                idempotency_key=key,
                order_item_id=int(args.get("order_item_id") or 0),
                caller_confirmed=bool(args.get("caller_confirmed")),
            )
        if name == "confirm_order":
            return await self._service.confirm_order(
                call_id=session_id,
                idempotency_key=key,
                expected_draft_version=int(args.get("expected_draft_version") or 0),
                approved=bool(args.get("caller_approved_full_readback")),
            )
        if name == "get_reservation_draft":
            return await self._service.get_reservation_draft(session_id)
        if name == "update_reservation_draft":
            updates = {
                key: args[key]
                for key in (
                    "name", "phone", "date", "time", "party_size", "seating_preference",
                    "seating_backup", "seating_avoid", "dietary", "occasion", "extra_notes",
                    "require_approval_for_paid_items",
                )
                if key in args and args[key] is not None
            }
            return await self._service.update_reservation_draft_native(session_id, updates)
        raise ValueError(f"unsupported_native_tool:{name}")

    async def invoke(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if self._native_service is not None:
            return await self._invoke_native(name, arguments)
        if name == "lookup_order":
            return await self._service.get_order_summary(
                call_id=str(arguments.get("session_id") or "")
            )
        if name == "get_full_menu":
            result = await self._service.list_menu(available_only=False)
            return {
                **result,
                "evidence_source": "restaurant_service.list_menu",
                "evidence_version": max(
                    (str(item.get("data_version") or "") for item in result.get("items") or ()),
                    default="",
                ),
            }
        if name == "check_menu_item_availability":
            result = await self._service.find_menu_item(str(arguments.get("item_name") or ""))
            return {
                **result,
                "evidence_source": "restaurant_service.find_menu_item",
                "evidence_version": max(
                    (str(item.get("data_version") or "") for item in result.get("candidates") or ()),
                    default=str((result.get("match") or {}).get("data_version") or ""),
                ),
            }
        if name == "lookup_booking":
            from app.restaurant_knowledge import get_restaurant_knowledge

            result = dict(
                await self._service.lookup_booking(
                    booking_id=int(arguments.get("booking_id") or 0),
                    customer_name=str(arguments.get("customer_name") or ""),
                    customer_phone=str(arguments.get("customer_phone") or ""),
                )
            )
            result["reference"] = str(result.get("booking_id") or "")
            result["timezone"] = get_restaurant_knowledge().identity["timezone"]
            return result
        tool = self._tools[name]
        args = dict(arguments)
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(args)
        return await tool(**args)

    async def readback(self, name: str, arguments: Mapping[str, Any], result: Any) -> Any:
        """Read authoritative state after every mutation that can change it."""
        if name in {"add_order_item", "set_order_fulfillment", "set_order_notes", "update_order_item", "remove_order_item", "confirm_order"}:
            readback = await self._service.get_order_summary(
                call_id=str(arguments.get("session_id") or "")
            )
            readback = dict(readback)
            readback.setdefault("unresolved_fields", [])
            readback["state_version"] = int(readback.get("draft_version") or 0)
            readback["readback_committed"] = True
            readback["readback_hash"] = _order_readback_hash(readback)
            return readback
        if name == "update_reservation_draft":
            session_id = str(arguments.get("session_id") or "")
            if self._native_service is not None:
                draft = dict(await self._service.get_reservation_draft(session_id))
            else:
                from app.call_memory import get_reservation_draft, hydrate_call_memory

                await hydrate_call_memory(session_id)
                draft = dict(get_reservation_draft(session_id))
            draft["readback_committed"] = True
            return draft
        if name == "add_guest_note":
            booking_id = int(arguments.get("booking_id") or 0)

            if booking_id:
                readback = dict(await self._service.lookup_booking(booking_id=booking_id))
                readback["readback_committed"] = True
                if self._native_service is not None:
                    state = await self._service.load_call_state(str(arguments.get("session_id") or ""))
                    readback["guest_notes"] = str((state.get("state") or {}).get("guest_notes") or "")
                else:
                    from app.call_memory import get_call_memory

                    readback["guest_notes"] = str(get_call_memory(str(arguments.get("session_id") or "")).get("guest_notes") or "")
                return readback
            session_id = str(arguments.get("session_id") or "")
            if self._native_service is not None:
                try:
                    readback = dict(await self._service.get_order_summary(call_id=session_id))
                    if not isinstance(result, Mapping) or "guest_notes" not in result:
                        return {"readback_committed": False}
                    readback["guest_notes"] = str(result.get("guest_notes") or "")
                    readback["readback_committed"] = True
                    readback.setdefault("unresolved_fields", [])
                    readback["state_version"] = int(readback.get("draft_version") or 0)
                    readback["readback_hash"] = _order_readback_hash(readback)
                    return readback
                except Exception:
                    return {"readback_committed": False}
            from app.call_memory import get_call_memory, hydrate_call_memory
            try:
                readback = dict(
                    await self._service.get_order_summary(call_id=session_id)
                )
            except Exception:
                from app.call_memory import get_reservation_draft

                await hydrate_call_memory(session_id)
                readback = dict(get_reservation_draft(session_id))
                readback["guest_notes"] = str(get_call_memory(session_id).get("guest_notes") or "")
                readback["readback_committed"] = True
                readback["note_owner"] = "reservation_draft"
                return readback
            await hydrate_call_memory(session_id)
            readback["guest_notes"] = str(get_call_memory(session_id).get("guest_notes") or "")
            readback.setdefault("unresolved_fields", [])
            readback["state_version"] = int(readback.get("draft_version") or 0)
            readback["readback_committed"] = True
            readback["readback_hash"] = _order_readback_hash(readback)
            return readback
        if name in {"create_booking", "update_confirmed_booking", "cancel_booking"}:
            booking_id = int(arguments.get("booking_id") or 0)
            if not booking_id and isinstance(result, str):
                match = re.search(r"(?:reference|booking)\s+#?\s*(\d+)", result, re.IGNORECASE)
                if match:
                    booking_id = int(match.group(1))
            if booking_id <= 0:
                return None
            try:
                readback = dict(await self._service.lookup_booking(booking_id=booking_id))
                readback["readback_committed"] = True
                return readback
            except Exception:
                return None
        return None


def realtime_tool_definitions() -> list[dict[str, Any]]:
    """Small, strict function surface sent in ``session.update``."""
    return [
        _tool_definition(
            "check_menu_item_availability",
            "Check one exact menu item; never infer a price or availability.",
            {"item_name": {"type": "string"}},
            ["item_name"],
        ),
        _tool_definition(
            "get_full_menu",
            "Read the live menu and prices; do not invent an item or price.",
            {},
            [],
        ),
        _tool_definition(
            "check_table_availability",
            "Check a requested table slot without reserving it.",
            {
                "date": {"type": "string"},
                "time": {"type": "string"},
                "party_size": {"type": "integer", "minimum": 1, "maximum": 24},
                "preferred_location": {"type": "string"},
                "session_id": {"type": "string"},
            },
            ["date", "time", "party_size"],
        ),
        _tool_definition(
            "get_order_summary",
            "Read the authoritative order draft and total.",
            {"session_id": {"type": "string"}},
            ["session_id"],
        ),
        {
            "type": "function",
            "name": "add_order_item",
            "description": "Add an exact available item to the draft; use canonical IDs returned by the restaurant tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "item_name": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
                    "modifier_ids": {"type": "array", "items": {"type": "string"}},
                    "removals": {"type": "array", "items": {"type": "string"}},
                    "substitutions": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                    "order_notes": {"type": "string"},
                    "allergy_notes": {"type": "string"},
                    "booking_id": {"type": "integer"},
                    "customer_name": {"type": "string"},
                    "customer_phone": {"type": "string"},
                    "caller_confirmed": {"type": "boolean"},
                },
                "required": ["session_id", "item_name"],
                "additionalProperties": False,
            },
        },
        _tool_definition(
            "set_order_fulfillment",
            "Set dine-in, pickup, or synthetic local delivery details.",
            {
                "session_id": {"type": "string"},
                "fulfillment_type": {"type": "string", "enum": ["dine_in", "pickup", "delivery"]},
                "booking_id": {"type": "integer"},
                "delivery_address": {"type": "string"},
                "delivery_instructions": {"type": "string"},
            },
            ["session_id", "fulfillment_type"],
        ),
        _tool_definition(
            "set_order_notes",
            "Set or clear authoritative order-level and allergy notes.",
            {
                "session_id": {"type": "string"},
                "order_notes": {"type": ["string", "null"]},
                "allergy_notes": {"type": ["string", "null"]},
                "caller_confirmed": {"type": "boolean"},
            },
            ["session_id"],
        ),
        _tool_definition(
            "update_order_item",
            "Correct one existing order item quantity or notes.",
            {
                "session_id": {"type": "string"},
                "order_item_id": {"type": "integer", "minimum": 1},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 20},
                "notes": {"type": ["string", "null"]},
                "caller_confirmed": {"type": "boolean"},
            },
            ["session_id", "order_item_id", "quantity"],
        ),
        _tool_definition(
            "remove_order_item",
            "Remove one existing order item after caller confirmation.",
            {
                "session_id": {"type": "string"},
                "order_item_id": {"type": "integer", "minimum": 1},
                "caller_confirmed": {"type": "boolean"},
            },
            ["session_id", "order_item_id"],
        ),
        _tool_definition(
            "update_reservation_draft",
            "Save or correct reservation fields before booking.",
            {
                "session_id": {"type": "string"}, "name": {"type": ["string", "null"]},
                "phone": {"type": ["string", "null"]}, "date": {"type": ["string", "null"]},
                "time": {"type": ["string", "null"]}, "party_size": {"type": ["integer", "null"]},
                "seating_preference": {"type": ["string", "null"]},
                "seating_backup": {"type": ["string", "null"]},
                "seating_avoid": {"type": ["string", "null"]}, "dietary": {"type": ["string", "null"]},
                "occasion": {"type": ["string", "null"]}, "extra_notes": {"type": ["string", "null"]},
                "require_approval_for_paid_items": {"type": ["boolean", "null"]},
            },
            ["session_id"],
        ),
        _tool_definition(
            "confirm_order",
            "Commit only after a complete readback, caller approval, and matching draft version.",
            {
                "session_id": {"type": "string"},
                "expected_draft_version": {"type": "integer", "minimum": 1},
                "caller_approved_full_readback": {"type": "boolean"},
            },
            ["session_id", "expected_draft_version", "caller_approved_full_readback"],
        ),
        _tool_definition(
            "get_reservation_draft",
            "Read the reservation draft without treating it as booked.",
            {"session_id": {"type": "string"}},
            ["session_id"],
        ),
        _tool_definition(
            "update_confirmed_booking",
            "Correct an existing booking after authoritative readback and approval.",
            {
                "session_id": {"type": "string"}, "booking_id": {"type": "integer", "minimum": 1},
                "date": {"type": "string"}, "time": {"type": "string"},
                "party_size": {"type": "integer", "minimum": 1},
                "seating_preference": {"type": ["string", "null"]},
                "seating_backup": {"type": ["string", "null"]},
                "seating_avoid": {"type": ["string", "null"]}, "dietary": {"type": ["string", "null"]},
                "occasion": {"type": ["string", "null"]}, "extra_notes": {"type": ["string", "null"]},
                "customer_name": {"type": "string"},
                "require_approval_for_paid_items": {"type": ["boolean", "null"]},
                "caller_confirmed": {"type": "boolean"},
            },
            ["session_id", "booking_id"],
        ),
        _tool_definition(
            "lookup_booking",
            "Read one booking by exact reference or verified customer details.",
            {
                "booking_id": {"type": "integer", "minimum": 1},
                "customer_name": {"type": "string"}, "customer_phone": {"type": "string"},
            },
            [],
        ),
        _tool_definition(
            "add_guest_note",
            "Save a guest instruction on the authoritative reservation or order.",
            {"note": {"type": "string"}, "session_id": {"type": "string"}, "booking_id": {"type": "integer"}},
            ["note", "session_id"],
        ),
        _tool_definition(
            "lookup_order",
            "Read an order only after exact customer verification.",
            {"order_id": {"type": "integer", "minimum": 1}, "customer_name": {"type": "string"}},
            ["order_id", "customer_name"],
        ),
        _tool_definition(
            "create_booking",
            "Create only after every booking field is read back and explicitly approved.",
            {
                "name": {"type": "string"}, "phone": {"type": "string"},
                "date": {"type": "string"}, "time": {"type": "string"},
                "party_size": {"type": "integer"}, "session_id": {"type": "string"},
                "notes": {"type": "string"}, "caller_confirmed": {"type": "boolean"},
                "table_number": {"type": "integer"},
            },
            ["name", "phone", "date", "time", "party_size", "session_id", "caller_confirmed"],
        ),
        _tool_definition(
            "cancel_booking",
            "Cancel only after verified booking details and explicit caller approval.",
            {
                "booking_id": {"type": "integer"}, "customer_name": {"type": "string"},
                "customer_phone": {"type": "string"}, "reason": {"type": "string"},
                "session_id": {"type": "string"}, "caller_confirmed": {"type": "boolean"},
            },
            ["booking_id", "session_id", "caller_confirmed"],
        ),
    ]


class ToolBridge:
    """Validate, deduplicate, execute, and verify constrained model calls."""

    def __init__(
        self,
        executor: ToolExecutor,
        *,
        session_id: str = "",
        scope_resolver: Callable[[str], Awaitable[Mapping[str, Any] | None]] | None = None,
    ) -> None:
        self.executor = executor
        self._calls: dict[str, ToolOutcome] = {}
        self.session_id = session_id
        self._turn_operations: dict[str, dict[str, str]] = {}
        self.scope_resolver = scope_resolver
        self.native_service = getattr(executor, "_native_service", None)
        bind_session = getattr(executor, "bind_session", None)
        if session_id and bind_session is not None:
            bind_session(session_id)

    def bind_session(self, session_id: str) -> None:
        if self.session_id and self.session_id != session_id:
            raise ValueError("tool bridge is already bound to another session")
        self.session_id = session_id
        bind_session = getattr(self.executor, "bind_session", None)
        if bind_session is not None:
            bind_session(session_id)

    def operation_id(self, name: str, arguments: Mapping[str, Any], turn_id: str) -> str:
        payload = json.dumps(
            {
                "session_id": self.session_id,
                "turn_id": turn_id,
                "operation": name,
                "arguments": dict(arguments),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _verified_booking_identity(self) -> tuple[dict[str, Any] | None, str]:
        if not self.session_id:
            return None, "booking_scope_unverified"
        if isinstance(self.executor, OfflineToolExecutor):
            identity = dict(self.executor.booking_identity)
            if identity.get("session_id") not in (None, "", self.session_id):
                return None, "booking_scope_unverified"
            if not identity or not identity.get("booking_id") or not identity.get("customer_name") or not identity.get("customer_phone"):
                return None, "booking_scope_unverified"
            if str(identity.get("status") or "confirmed").casefold() != "confirmed":
                return None, "booking_scope_unverified"
            return {
                "booking_id": int(identity["booking_id"]),
                "customer_name": str(identity.get("customer_name") or ""),
                "customer_phone": str(identity.get("customer_phone") or ""),
            }, ""
        if self.native_service is not None:
            try:
                memory = (await self.native_service.load_call_state(self.session_id)).get("state") or {}
            except Exception:
                return None, "booking_scope_unverified"
        else:
            try:
                from app.call_memory import get_call_memory, hydrate_call_memory

                await hydrate_call_memory(self.session_id)
                memory = get_call_memory(self.session_id)
            except Exception:
                return None, "booking_scope_unverified"
        try:
            booking_id = int(memory.get("booking_id") or 0)
        except (TypeError, ValueError):
            booking_id = 0
        trusted_name = str(memory.get("customer_name") or "").strip()
        trusted_phone = str(memory.get("customer_phone") or "").strip()
        if not booking_id or not trusted_name or not trusted_phone:
            return None, "booking_scope_unverified"
        try:
            service = self.native_service
            if service is None:
                from app.services.restaurant import restaurant_service

                service = restaurant_service
            verified_booking = await service.lookup_booking(booking_id=booking_id)
        except Exception:
            return None, "booking_scope_unverified"
        if (
            _canonical_name(verified_booking.get("customer_name")) != _canonical_name(trusted_name)
            or _canonical_phone(verified_booking.get("customer_phone")) != _canonical_phone(trusted_phone)
            or str(verified_booking.get("status") or "").casefold() != "confirmed"
        ):
            return None, "booking_scope_unverified"
        return {
            "booking_id": booking_id,
            "customer_name": trusted_name,
            "customer_phone": trusted_phone,
        }, ""

    async def _initial_booking_identity(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        if not self.session_id:
            return None, "booking_scope_unverified"
        try:
            booking_id = int(arguments.get("booking_id") or 0)
        except (TypeError, ValueError):
            booking_id = 0
        customer_name = str(arguments.get("customer_name") or "").strip()
        customer_phone = str(arguments.get("customer_phone") or "").strip()
        if not customer_phone or (not booking_id and not customer_name):
            return None, "booking_scope_unverified"
        if isinstance(self.executor, OfflineToolExecutor):
            identity = dict(self.executor.booking_identity)
            if identity.get("session_id") not in (None, "", self.session_id):
                return None, "booking_scope_unverified"
            if not identity:
                return None, "booking_scope_unverified"
            if booking_id and int(identity.get("booking_id") or 0) != booking_id:
                return None, "booking_scope_unverified"
            if customer_name and _canonical_name(identity.get("customer_name")) != _canonical_name(customer_name):
                return None, "booking_scope_unverified"
            if _canonical_phone(identity.get("customer_phone")) != _canonical_phone(customer_phone):
                return None, "booking_scope_unverified"
            return {
                "booking_id": int(identity["booking_id"]),
                "customer_name": str(identity.get("customer_name") or ""),
                "customer_phone": str(identity.get("customer_phone") or ""),
            }, ""
        try:
            service = self.native_service
            if service is None:
                from app.services.restaurant import restaurant_service

                service = restaurant_service
            verified = await service.lookup_booking(
                booking_id=booking_id,
                customer_name="" if booking_id else customer_name,
                customer_phone=customer_phone,
            )
            if customer_name and _canonical_name(verified.get("customer_name")) != _canonical_name(customer_name):
                return None, "booking_scope_unverified"
            if _canonical_phone(verified.get("customer_phone")) != _canonical_phone(customer_phone):
                return None, "booking_scope_unverified"
            await service.persist_call_state(
                self.session_id,
                {
                    "booking_id": verified["booking_id"],
                    "customer_name": verified["customer_name"],
                    "customer_phone": verified["customer_phone"],
                    "booking_date": verified.get("date") or "",
                    "booking_time": verified.get("time") or "",
                    "party_size": verified.get("party_size") or 0,
                    "draft_status": str(verified.get("status") or "").casefold(),
                },
                caller_phone=str(verified.get("customer_phone") or ""),
            )
            if self.native_service is None:
                from app.call_memory import hydrate_call_memory

                await hydrate_call_memory(self.session_id)
                booking_status = str(verified.get("status") or "").casefold()
                if booking_status == "confirmed":
                    from app.call_memory import apply_live_booking_to_memory

                    apply_live_booking_to_memory(self.session_id, verified)
                else:
                    from app.call_memory import update_reservation_draft

                    update_reservation_draft(
                        self.session_id,
                        booking_id=int(verified["booking_id"]),
                        customer_name=str(verified.get("customer_name") or ""),
                        customer_phone=str(verified.get("customer_phone") or ""),
                        date=str(verified.get("date") or ""),
                        time=str(verified.get("time") or ""),
                        party_size=int(verified.get("party_size") or 0),
                        status=booking_status,
                    )
        except Exception:
            return None, "booking_scope_unverified"
        return {
            "booking_id": int(verified["booking_id"]),
            "customer_name": str(verified.get("customer_name") or ""),
            "customer_phone": str(verified.get("customer_phone") or ""),
        }, ""

    async def _scoped_booking_arguments(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        if name not in BOOKING_SCOPED_TOOLS:
            return dict(arguments), ""
        if isinstance(self.executor, OfflineToolExecutor):
            if name == "add_guest_note" and arguments.get("booking_id") in (None, "", 0):
                current = await self.scope_resolver(self.session_id) if self.scope_resolver is not None else self.executor.order_scope
                if isinstance(current, Mapping) and int(current.get("booking_id") or 0):
                    trusted, error = await self._verified_booking_identity()
                    if error:
                        return None, error
                else:
                    scoped = dict(arguments)
                    scoped.update({"session_id": self.session_id, "booking_id": 0})
                    scoped.pop("customer_name", None)
                    scoped.pop("customer_phone", None)
                    return scoped, ""
            elif name == "lookup_booking":
                trusted, error = await self._verified_booking_identity()
                if error:
                    trusted, error = await self._initial_booking_identity(arguments)
                if error:
                    return None, error
            else:
                trusted, error = await self._verified_booking_identity()
                if error:
                    return None, error
            scoped = dict(arguments)
            scoped.update(trusted or {})
            scoped["session_id"] = self.session_id
            if name == "lookup_booking":
                scoped.pop("session_id", None)
            elif name == "add_guest_note":
                scoped.pop("customer_name", None)
                scoped.pop("customer_phone", None)
            return scoped, ""
        if name == "add_guest_note" and self.session_id and arguments.get("booking_id") in (None, "", 0):
            trusted, error = await self._verified_booking_identity()
            if not error:
                scoped = dict(arguments)
                scoped.update(trusted or {})
                scoped["session_id"] = self.session_id
                scoped.pop("customer_name", None)
                scoped.pop("customer_phone", None)
                return scoped, ""
            if self.native_service is not None:
                try:
                    memory = (await self.native_service.load_call_state(self.session_id)).get("state") or {}
                    if int(memory.get("booking_id") or 0):
                        return None, "booking_scope_unverified"
                except Exception:
                    return None, "booking_scope_unverified"
            else:
                from app.call_memory import get_call_memory, hydrate_call_memory

                try:
                    await hydrate_call_memory(self.session_id)
                    memory = get_call_memory(self.session_id)
                    if int(memory.get("booking_id") or 0):
                        return None, "booking_scope_unverified"
                except Exception:
                    return None, "booking_scope_unverified"
            try:
                if self.scope_resolver is not None:
                    current = await self.scope_resolver(self.session_id)
                else:
                    service = self.native_service
                    if service is None:
                        from app.services.restaurant import restaurant_service

                        service = restaurant_service
                    current = await service.get_order_summary(call_id=self.session_id)
            except Exception as exc:
                if getattr(exc, "code", "") != "order_not_found":
                    return None, "booking_scope_unverified"
            else:
                if current is not None and (
                    not isinstance(current, Mapping) or int(current.get("booking_id") or 0)
                ):
                    return None, "booking_scope_unverified"
            scoped = dict(arguments)
            scoped["session_id"] = self.session_id
            scoped["booking_id"] = 0
            scoped.pop("customer_name", None)
            scoped.pop("customer_phone", None)
            return scoped, ""
        if name == "lookup_booking":
            trusted, error = await self._verified_booking_identity()
            if error:
                trusted, error = await self._initial_booking_identity(arguments)
            if error:
                return None, error
            scoped = dict(arguments)
            scoped.update(trusted or {})
            scoped.pop("session_id", None)
            return scoped, ""
        trusted, error = await self._verified_booking_identity()
        if error:
            return None, error
        supplied_booking = arguments.get("booking_id")
        booking_id = int(trusted["booking_id"])
        if supplied_booking not in (None, "", 0):
            try:
                if int(supplied_booking) != booking_id:
                    return None, "booking_scope_mismatch"
            except (TypeError, ValueError):
                return None, "booking_scope_mismatch"
        for key in ("customer_name", "customer_phone"):
            supplied = str(arguments.get(key) or "").strip()
            if not supplied:
                continue
            if key == "customer_phone":
                matches = _canonical_phone(supplied) == _canonical_phone(trusted[key])
            else:
                matches = _canonical_name(supplied) == _canonical_name(trusted[key])
            if not matches:
                return None, "booking_scope_mismatch"
        scoped = dict(arguments)
        scoped.update(trusted)
        scoped["session_id"] = self.session_id
        if name == "lookup_booking":
            scoped.pop("session_id", None)
        elif name == "update_confirmed_booking":
            scoped.pop("customer_phone", None)
        elif name == "add_guest_note":
            scoped.pop("customer_name", None)
            scoped.pop("customer_phone", None)
        return scoped, ""

    async def _scoped_order_arguments(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        if name not in ORDER_SCOPED_TOOLS:
            return dict(arguments), ""
        if not self.session_id:
            return None, "order_scope_unverified"
        scoped = dict(arguments)
        scoped["session_id"] = self.session_id
        try:
            if isinstance(self.executor, OfflineToolExecutor):
                current = await self.scope_resolver(self.session_id) if self.scope_resolver is not None else self.executor.order_scope
            elif self.scope_resolver is not None:
                current = await self.scope_resolver(self.session_id)
            else:
                service = self.native_service
                if service is None:
                    from app.services.restaurant import restaurant_service

                    service = restaurant_service
                current = await service.get_order_summary(call_id=self.session_id)
        except Exception as exc:
            if getattr(exc, "code", "") != "order_not_found":
                return None, "order_scope_unverified"
            current = None
        if name == "lookup_order":
            if not isinstance(current, Mapping):
                return None, "order_scope_unverified"
            try:
                if int(arguments.get("order_id") or 0) != int(current.get("order_id") or 0):
                    return None, "order_scope_unverified"
            except (TypeError, ValueError):
                return None, "order_scope_unverified"
            scoped["order_id"] = current["order_id"]
            if int(current.get("booking_id") or 0):
                trusted, error = await self._verified_booking_identity()
                if error or int(current.get("booking_id") or 0) != int(trusted["booking_id"]):
                    return None, "order_scope_unverified"
                scoped["customer_name"] = trusted["customer_name"]
            else:
                scoped.pop("booking_id", None)
                scoped.pop("customer_name", None)
            return scoped, ""
        if isinstance(current, Mapping):
            current_booking_id = int(current.get("booking_id") or 0)
            if not current_booking_id:
                scoped.pop("booking_id", None)
                if "customer_name" in scoped:
                    scoped["customer_name"] = " ".join(str(scoped["customer_name"]).split())
                if "customer_phone" in scoped:
                    scoped["customer_phone"] = _canonical_phone(scoped["customer_phone"])
                return scoped, ""
            trusted, error = await self._verified_booking_identity()
            if error or current_booking_id != int(trusted["booking_id"]):
                return None, "order_scope_unverified"
            if name == "add_order_item":
                scoped.update(trusted)
            elif name == "set_order_fulfillment":
                scoped["booking_id"] = trusted["booking_id"]
            return scoped, ""
        supplied_booking = arguments.get("booking_id")
        if supplied_booking not in (None, "", 0):
            trusted, error = await self._verified_booking_identity()
            if error:
                return None, "order_scope_unverified"
            try:
                if int(supplied_booking) != int(trusted["booking_id"]):
                    return None, "order_scope_unverified"
            except (TypeError, ValueError):
                return None, "order_scope_unverified"
            if name == "add_order_item":
                scoped.update(trusted)
            elif name == "set_order_fulfillment":
                scoped["booking_id"] = trusted["booking_id"]
        elif name == "add_order_item":
            scoped.pop("booking_id", None)
            if "customer_name" in scoped:
                scoped["customer_name"] = " ".join(str(scoped["customer_name"]).split())
            if "customer_phone" in scoped:
                scoped["customer_phone"] = _canonical_phone(scoped["customer_phone"])
        return scoped, ""

    async def invoke(
        self,
        *,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        turn_id: str,
        state_version: int,
    ) -> ToolOutcome:
        args = dict(arguments)
        supplied_session_id = args.get("session_id")
        if self.session_id and supplied_session_id not in (None, "", self.session_id):
            return ToolOutcome(
                name=name,
                call_id=call_id,
                arguments=args,
                result=None,
                success=False,
                error="session_scope_mismatch",
                state_version=state_version,
            )
        if name in MUTATING_TOOLS and not turn_id:
            return ToolOutcome(
                name=name,
                call_id=call_id,
                arguments=args,
                result=None,
                success=False,
                error="caller_turn_not_finalized",
                state_version=state_version,
            )
        if not call_id or not name:
            outcome = ToolOutcome(name=name, call_id=call_id, arguments=args, result=None, success=False, error="invalid_tool_call", state_version=state_version)
            self._calls[call_id] = outcome
            return outcome
        if name not in KNOWN_TOOLS:
            outcome = ToolOutcome(name=name, call_id=call_id, arguments=args, result=None, success=False, error="unsupported_tool", state_version=state_version)
            self._calls[call_id] = outcome
            return outcome
        raw_operation_fingerprint = f"{self.session_id}:{turn_id}:{name}:{_hash(args)}"
        operation_id = self.operation_id(name, args, turn_id)
        if name in MUTATING_TOOLS and turn_id:
            prior_call_id = self._turn_operations.get(turn_id, {}).get(raw_operation_fingerprint)
            prior = self._calls.get(prior_call_id or "")
            if prior is not None and prior.success and prior.readback_verified:
                return ToolOutcome(
                    name=prior.name,
                    call_id=prior.call_id,
                    arguments=prior.arguments,
                    result=prior.result,
                    success=prior.success,
                    error=prior.error,
                    readback=prior.readback,
                    readback_verified=prior.readback_verified,
                    state_version=prior.state_version,
                    replayed=True,
                    facts=prior.facts,
                    pending=prior.pending,
                )
        scoped_args, scope_error = await self._scoped_booking_arguments(name, args)
        if scope_error:
            outcome = ToolOutcome(
                name=name,
                call_id=call_id,
                arguments=args,
                result=None,
                success=False,
                error=scope_error,
                state_version=state_version,
            )
            self._calls[call_id] = outcome
            return outcome
        args, scope_error = await self._scoped_order_arguments(name, scoped_args or args)
        if scope_error:
            outcome = ToolOutcome(
                name=name,
                call_id=call_id,
                arguments=scoped_args or arguments,
                result=None,
                success=False,
                error=scope_error,
                state_version=state_version,
            )
            self._calls[call_id] = outcome
            return outcome
        args = args or scoped_args or dict(arguments)
        fingerprint = f"{name}:{_hash(args)}"
        operation_fingerprint = f"{self.session_id}:{turn_id}:{name}:{_hash(args)}"
        previous = self._calls.get(call_id)
        if previous is not None:
            if f"{previous.name}:{_hash(previous.arguments)}" != fingerprint:
                return ToolOutcome(
                    name=name,
                    call_id=call_id,
                    arguments=args,
                    result=None,
                    success=False,
                    error="idempotency_conflict",
                    state_version=state_version,
                )
            return ToolOutcome(
                name=previous.name,
                call_id=previous.call_id,
                arguments=previous.arguments,
                result=previous.result,
                success=previous.success,
                error=previous.error,
                readback=previous.readback,
                readback_verified=previous.readback_verified,
                state_version=previous.state_version,
                replayed=True,
                facts=previous.facts,
                pending=previous.pending,
            )

        if name in MUTATING_TOOLS and turn_id:
            prior_call_id = self._turn_operations.get(turn_id, {}).get(operation_fingerprint)
            prior = self._calls.get(prior_call_id or "")
            if prior is not None and prior.success and prior.readback_verified:
                return ToolOutcome(
                    name=prior.name,
                    call_id=prior.call_id,
                    arguments=prior.arguments,
                    result=prior.result,
                    success=prior.success,
                    error=prior.error,
                    readback=prior.readback,
                    readback_verified=prior.readback_verified,
                    state_version=prior.state_version,
                    replayed=True,
                    facts=prior.facts,
                    pending=prior.pending,
                )

        bind_operation_scope = getattr(self.executor, "bind_operation_scope", None)
        clear_operation_scope = getattr(self.executor, "clear_operation_scope", None)
        if bind_operation_scope is not None:
            bind_operation_scope(turn_id)
        try:
            result = await self.executor.invoke(name, args)
        except Exception as exc:  # structured failure; never a success-like response
            outcome = ToolOutcome(name=name, call_id=call_id, arguments=args, result=None, success=False, error=f"tool_exception:{type(exc).__name__}", state_version=state_version)
            self._calls[call_id] = outcome
            return outcome
        finally:
            if clear_operation_scope is not None:
                clear_operation_scope()

        readback = None
        readback_verified = False
        try:
            success = not _is_failure(
                result,
                allow_negative_availability=name in {"check_menu_item_availability", "check_table_availability"},
                authoritative_read=name not in MUTATING_TOOLS,
            )
            error = str(result) if not success else ""
            if success and name in MUTATING_TOOLS:
                try:
                    readback = await self.executor.readback(name, args, result)
                except Exception as exc:
                    success = False
                    error = f"readback_exception:{type(exc).__name__}"
                else:
                    readback_verified = self._verify_readback(name, args, readback, state_version, result)
                    if not readback_verified:
                        success = False
                        error = "database_readback_mismatch"
            facts = self._facts(result, readback, args)
        except Exception as exc:
            success = False
            error = f"bridge_schema_error:{type(exc).__name__}"
            readback_verified = False
            facts = {}
        outcome = ToolOutcome(
            name=name,
            call_id=call_id,
            arguments=args,
            result=result,
            success=success,
            error=error,
            readback=readback,
            readback_verified=readback_verified if name in MUTATING_TOOLS else success,
            state_version=state_version,
            facts=facts,
            operation_id=operation_id,
            pending=(
                isinstance(result, Mapping)
                and bool(result.get("pending") or result.get("readback_required"))
                and bool(result.get("pending_confirmation_hash") or result.get("proposed"))
            ),
        )
        self._calls[call_id] = outcome
        if name in MUTATING_TOOLS and turn_id and outcome.success and outcome.readback_verified:
            self._turn_operations.setdefault(turn_id, {})[operation_fingerprint] = call_id
            self._turn_operations.setdefault(turn_id, {})[raw_operation_fingerprint] = call_id
        return outcome

    @staticmethod
    def _verify_readback(
        name: str,
        arguments: Mapping[str, Any],
        readback: Any,
        state_version: int,
        result: Any = None,
    ) -> bool:
        if readback is None:
            return False
        if isinstance(readback, Mapping):
            if readback.get("ok") is False or readback.get("status") in {"failed", "error"}:
                return False
            if name == "add_guest_note" and readback.get("note_owner") == "reservation_draft":
                note = str(arguments.get("note") or "").casefold()
                stored = " ".join(
                    str(readback.get(key) or "")
                    for key in ("guest_notes", "notes")
                ).casefold()
                return bool(readback.get("readback_committed") and note and note in stored)
            if arguments.get("expected_draft_version") is not None:
                try:
                    if int(readback.get("draft_version")) != int(arguments["expected_draft_version"]):
                        return False
                except (TypeError, ValueError):
                    return False
            if "order_id" in readback:
                if not readback.get("readback_committed"):
                    return False
                if arguments.get("session_id") and readback.get("call_id") != arguments.get("session_id"):
                    return False
                if isinstance(result, Mapping) and result.get("order_id") not in (None, ""):
                    if readback.get("order_id") != result.get("order_id"):
                        return False
                if not all(field in readback for field in _ORDER_READBACK_FIELDS):
                    return False
                if not isinstance(readback.get("items"), list) or not isinstance(readback.get("proposed_items"), list):
                    return False
                if not isinstance(readback.get("unresolved_fields"), list):
                    return False
                if not isinstance(readback.get("readback_hash"), str) or readback["readback_hash"] != _order_readback_hash(readback):
                    return False
                if readback.get("status") not in {"pending", "confirmed"}:
                    return False
                if name == "confirm_order" and readback.get("status") != "confirmed":
                    return False
                for item in [*readback["items"], *readback["proposed_items"]]:
                    if not isinstance(item, Mapping) or not all(field in item for field in _ORDER_ITEM_FIELDS):
                        return False
                    if not item.get("order_item_id") or not item.get("item_id") or not item.get("item_name"):
                        return False
                    if not isinstance(item.get("quantity"), int) or item["quantity"] < 1:
                        return False
                if name == "set_order_fulfillment" and readback.get("fulfillment_type") != arguments.get("fulfillment_type"):
                    return False
                if name == "set_order_fulfillment":
                    details = readback.get("fulfillment_details")
                    if not isinstance(details, Mapping):
                        return False
                    for argument_key, readback_key in (
                        ("delivery_address", "address"),
                        ("delivery_instructions", "instructions"),
                    ):
                        if argument_key in arguments and arguments[argument_key] is not None:
                            if str(details.get(readback_key) or "").strip() != str(arguments[argument_key] or "").strip():
                                return False
                    if "booking_id" in arguments and arguments["booking_id"] not in (None, "", 0):
                        if int(readback.get("booking_id") or 0) != int(arguments["booking_id"]):
                            return False
                if name == "set_order_notes":
                    for field in ("order_notes", "allergy_notes"):
                        if field in arguments and arguments[field] is not None and str(readback.get(field) or "").strip() != str(arguments[field] or "").strip():
                            return False
                if name == "add_order_item" and arguments.get("item_name"):
                    result_item_id = result.get("order_item_id") if isinstance(result, Mapping) else None
                    committed = [
                        item
                        for item in readback["items"]
                        if (
                            result_item_id not in (None, "")
                            and str(item.get("order_item_id")) == str(result_item_id)
                        )
                        or (
                            result_item_id in (None, "")
                            and str(item.get("item_name") or "").casefold()
                            == str(arguments["item_name"]).casefold()
                        )
                    ]
                    if not committed:
                        return False
                    item = committed[0]
                    if str(item.get("item_name") or "").casefold() != str(arguments["item_name"]).casefold():
                        return False
                    if arguments.get("quantity") is not None and int(item.get("quantity") or 0) != int(arguments["quantity"]):
                        return False
                    for argument_key, readback_key in (
                        ("modifier_ids", "modifiers"),
                        ("removals", "removals"),
                        ("substitutions", "substitutions"),
                    ):
                        if argument_key in arguments and _canonical_effect_values(arguments[argument_key]) != _canonical_effect_values(item.get(readback_key)):
                            return False
                    if "notes" in arguments and arguments["notes"] is not None and str(item.get("notes") or "").strip() != str(arguments["notes"] or "").strip():
                        return False
                    for field in ("order_notes", "allergy_notes"):
                        if field in arguments and arguments[field] is not None and str(readback.get(field) or "").strip() != str(arguments[field] or "").strip():
                            return False
                if name == "update_order_item" and arguments.get("order_item_id"):
                    matching = [
                        item
                        for item in readback["items"]
                        if str(item.get("order_item_id")) == str(arguments["order_item_id"])
                    ]
                    if not matching:
                        return False
                    item = matching[0]
                    if arguments.get("quantity") is not None and int(item.get("quantity") or 0) != int(arguments["quantity"]):
                        return False
                    if "notes" in arguments and arguments["notes"] is not None and str(item.get("notes") or "").strip() != str(arguments["notes"] or "").strip():
                        return False
                if name == "remove_order_item" and arguments.get("order_item_id"):
                    ids = {str(item.get("order_item_id")) for item in readback["items"] + readback["proposed_items"]}
                    if str(arguments["order_item_id"]) in ids:
                        return False
                if name == "add_guest_note":
                    note = str(arguments.get("note") or "").casefold()
                    stored = " ".join(str(readback.get(key) or "") for key in ("order_notes", "guest_notes")).casefold()
                    if not note or note not in stored:
                        return False
                return bool(readback.get("order_id") and int(readback.get("draft_version") or 0) > 0)
            if "booking_id" in readback:
                required = ("booking_id", "customer_name", "customer_phone", "status", "date", "time", "readback_committed")
                if not all(field in readback for field in required) or not readback.get("readback_committed"):
                    return False
                expected_booking = arguments.get("booking_id")
                if expected_booking not in (None, "", 0) and int(readback.get("booking_id") or 0) != int(expected_booking):
                    return False
                if isinstance(result, Mapping) and result.get("booking_id") not in (None, "", 0):
                    if int(readback.get("booking_id") or 0) != int(result["booking_id"]):
                        return False
                if name == "cancel_booking" and readback.get("status") != "cancelled":
                    return False
                if name in {"create_booking", "update_confirmed_booking"} and readback.get("status") != "confirmed":
                    return False
                for argument_key, readback_key in (("name", "customer_name"), ("customer_name", "customer_name"), ("phone", "customer_phone"), ("customer_phone", "customer_phone"), ("date", "date"), ("time", "time"), ("party_size", "party_size")):
                    if argument_key not in arguments or arguments[argument_key] in (None, ""):
                        continue
                    expected = arguments[argument_key]
                    actual = readback.get(readback_key)
                    if readback_key == "customer_name":
                        matches = _canonical_name(actual) == _canonical_name(expected)
                    elif readback_key == "customer_phone":
                        matches = _canonical_phone(actual) == _canonical_phone(expected)
                    elif readback_key == "party_size":
                        try:
                            matches = int(actual) == int(expected)
                        except (TypeError, ValueError):
                            matches = False
                    else:
                        matches = str(actual or "").strip().casefold() == str(expected).strip().casefold()
                    if not matches:
                        return False
                if isinstance(result, Mapping):
                    for field in ("booking_id", "customer_name", "customer_phone", "date", "time", "party_size", "location", "notes", "table_number"):
                        if field not in result or field not in readback or result[field] in (None, ""):
                            continue
                        if field == "customer_name":
                            matches = _canonical_name(result[field]) == _canonical_name(readback[field])
                        elif field == "customer_phone":
                            matches = _canonical_phone(result[field]) == _canonical_phone(readback[field])
                        elif field == "party_size":
                            matches = int(result[field]) == int(readback[field])
                        else:
                            matches = str(result[field]).strip().casefold() == str(readback[field]).strip().casefold()
                        if not matches:
                            return False
                if arguments.get("preferred_location"):
                    if _canonical_name(readback.get("location")) != _canonical_name(arguments["preferred_location"]):
                        return False
                stored_notes = str(readback.get("notes") or "").casefold()
                for argument_key in ("notes", "extra_notes"):
                    if arguments.get(argument_key):
                        if str(arguments[argument_key]).strip().casefold() not in stored_notes:
                            return False
                for argument_key, marker in (
                    ("seating_preference", "seating"),
                    ("seating_backup", "backup seating"),
                    ("seating_avoid", "avoid"),
                    ("dietary", "dietary"),
                    ("occasion", "occasion"),
                ):
                    if arguments.get(argument_key):
                        expected_note = f"{marker}: {arguments[argument_key]}".casefold()
                        if expected_note not in stored_notes:
                            return False
                if name == "add_guest_note":
                    note = str(arguments.get("note") or "").casefold()
                    stored = " ".join(str(readback.get(key) or "") for key in ("notes", "guest_notes")).casefold()
                    if not note or note not in stored:
                        return False
                return bool(
                    all(field in readback for field in required)
                    and readback.get("booking_id")
                    and readback.get("customer_name")
                    and readback.get("customer_phone")
                )
            if name == "update_reservation_draft":
                if not readback.get("readback_committed"):
                    return False
                for argument_key, readback_key in (("name", "customer_name"), ("phone", "customer_phone"), ("date", "date"), ("time", "time"), ("party_size", "party_size")):
                    if argument_key in arguments and arguments[argument_key] is not None and readback.get(readback_key) != arguments[argument_key]:
                        return False
                return all(field in readback for field in ("customer_name", "customer_phone", "date", "time", "party_size"))
            return False
        if isinstance(readback, str):
            return False
        return False

    @staticmethod
    def _facts(result: Any, readback: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        subject = {
            key: arguments[key]
            for key in ("item_name", "date", "time", "preferred_location", "session_id", "booking_id", "reference")
            if arguments.get(key) not in (None, "", 0)
        }
        if subject:
            facts["subject"] = subject
        for value in (result, readback):
            if not isinstance(value, Mapping):
                continue
            for key in ("booking_id", "order_id", "reference", "booking_reference"):
                if value.get(key) not in (None, "", 0):
                    facts.setdefault("subject", {})[key] = value[key]
            for key in (
                "items", "proposed_items", "prices", "availability", "booking", "order", "status", "date", "time",
                "customer_name", "customer_phone", "party_size", "location", "notes", "total",
                "fulfillment", "fulfillment_type", "fulfillment_details", "order_notes", "allergy_notes",
                "guest_notes", "unresolved_fields", "state_version", "readback_required",
                "pending_confirmation_hash", "booked_at", "timezone", "reference", "booking_reference", "table_number",
            ):
                if key in value:
                    facts[key] = value[key]
            for key in ("evidence_source", "evidence_version"):
                if value.get(key) not in (None, ""):
                    facts[key] = value[key]
            for item_key in ("items", "proposed_items"):
                if not isinstance(value.get(item_key), list):
                    continue
                canonical_items = [
                    {
                        "id": item.get("item_id"),
                        "name": item.get("item_name") or item.get("name"),
                        "price": item.get("price"),
                        "available": item.get("available"),
                        "modifier_options": item.get("modifier_options") or (),
                        "ingredients": item.get("ingredients") or (),
                        "allergens": item.get("allergens") or (),
                        "dietary_tags": item.get("dietary_tags") or (),
                        "customer_safe_answer": item.get("customer_safe_answer") or "",
                        "quantity": item.get("quantity"),
                        "order_item_id": item.get("order_item_id"),
                        "modifiers": item.get("modifiers") or (),
                        "removals": item.get("removals") or (),
                        "substitutions": item.get("substitutions") or (),
                        "notes": item.get("notes") or "",
                        "unit_price": item.get("unit_price"),
                        "subtotal": item.get("subtotal"),
                    }
                    for item in value[item_key]
                    if isinstance(item, Mapping) and (item.get("item_id") or item.get("item_name") or item.get("name"))
                ]
                if canonical_items:
                    facts["canonical_items" if item_key == "items" else "proposed_items"] = canonical_items
                    if item_key == "items":
                        facts.setdefault("items", []).extend(
                            item["name"] for item in canonical_items if item.get("name")
                        )
                        facts.setdefault("prices", {}).update(
                            {
                                item["id"] or item["name"]: item["price"]
                                for item in canonical_items
                                if item.get("price") is not None
                            }
                        )
                        facts["availability_by_item"] = {
                            item["name"]: ("available" if item.get("available") is True else "unavailable")
                            for item in canonical_items
                            if item.get("name") and item.get("available") is not None
                        }
                        availability = [item.get("available") for item in canonical_items]
                        if len(availability) == 1:
                            facts["availability"] = "available" if availability[0] is True else "unavailable"
            if isinstance(value.get("match"), Mapping):
                match = value["match"]
                facts.setdefault("canonical_items", []).append(
                    {
                        "id": match.get("item_id"),
                        "name": match.get("name"),
                        "price": match.get("price"),
                        "available": match.get("available"),
                        "ingredients": match.get("ingredients") or (),
                        "allergens": match.get("allergens") or (),
                        "dietary_tags": match.get("dietary_tags") or (),
                        "customer_safe_answer": match.get("customer_safe_answer") or "",
                    }
                )
                if match.get("name"):
                    facts.setdefault("items", []).append(match["name"])
                if match.get("price") is not None:
                    facts.setdefault("prices", {})[match.get("item_id") or match.get("name")] = match["price"]
                if match.get("available") is not None:
                    facts["availability"] = "available" if match["available"] is True else "unavailable"
                    if match.get("name"):
                        facts.setdefault("availability_by_item", {})[match["name"]] = facts["availability"]
            if value.get("item"):
                item = value["item"]
                if isinstance(item, Mapping):
                    if item.get("item_id") or item.get("name"):
                        facts.setdefault("canonical_items", []).append({"id": item.get("item_id"), "name": item.get("name")})
                    facts.setdefault("items", []).append(item.get("item_id") or item.get("name"))
                    if item.get("price") is not None:
                        facts.setdefault("prices", {})[item.get("item_id") or item.get("name")] = item["price"]
        if isinstance(result, str):
            item_name = str(arguments.get("item_name") or "")
            if item_name and item_name.casefold() in result.casefold() and not result.casefold().startswith("no matching"):
                facts.setdefault("items", []).append(item_name)
            if re.search(r"\b(?:not currently available|isn't available|is not available|not available|unavailable|sold out|not yet available|out of stock|closed|full|no tables?)\b", result, re.IGNORECASE):
                facts["availability"] = "unavailable"
            elif re.search(r"\b(?:available|open|in stock)\b", result, re.IGNORECASE):
                facts["availability"] = "available"
            prices = re.findall(r"\$\s*(\d+(?:\.\d{1,2})?)", result)
            if prices:
                facts["prices"] = {item_name or "amount": float(prices[-1])}
        return facts
