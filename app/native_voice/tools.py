"""Constrained tool bridge for the development Realtime adapter."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

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


def _is_failure(value: Any) -> bool:
    if isinstance(value, Mapping):
        return (
            value.get("ok") is False
            or bool(value.get("error"))
            or bool(value.get("pending") or value.get("readback_required"))
            or bool(value.get("unavailable") or value.get("no_op") or value.get("proposed"))
            or any(key in value and value[key] is False for key in ("added", "updated", "removed", "cancelled", "saved"))
        )
    if isinstance(value, str):
        lowered = value.casefold()
        return bool(re.match(r"^[a-z][a-z0-9_]*:", value.strip())) or any(
            phrase in lowered
            for phrase in (
                "pending confirmation",
                "readback_required",
                "not applied",
                "unchanged",
                "proposed",
                "not added",
                "not saved",
                "unavailable",
                "not available",
                "no matching",
                "no-op",
                "could not",
                "cannot",
                "failed",
            )
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
        )


class RestaurantToolExecutor:
    """Use the existing rollback tools and service as the authority boundary."""

    def __init__(self) -> None:
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

    async def invoke(self, name: str, arguments: Mapping[str, Any]) -> Any:
        if name == "lookup_order":
            from app.services.restaurant import restaurant_service

            return await restaurant_service.get_order_summary(
                call_id=str(arguments.get("session_id") or "")
            )
        if name == "get_full_menu":
            from app.services.restaurant import restaurant_service

            result = await restaurant_service.list_menu(available_only=False)
            return {
                **result,
                "evidence_source": "restaurant_service.list_menu",
                "evidence_version": max(
                    (str(item.get("data_version") or "") for item in result.get("items") or ()),
                    default="",
                ),
            }
        if name == "check_menu_item_availability":
            from app.services.restaurant import restaurant_service

            result = await restaurant_service.find_menu_item(str(arguments.get("item_name") or ""))
            return {
                **result,
                "evidence_source": "restaurant_service.find_menu_item",
                "evidence_version": max(
                    (str(item.get("data_version") or "") for item in result.get("candidates") or ()),
                    default=str((result.get("match") or {}).get("data_version") or ""),
                ),
            }
        tool = self._tools[name]
        args = dict(arguments)
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(args)
        return await tool(**args)

    async def readback(self, name: str, arguments: Mapping[str, Any], result: Any) -> Any:
        """Read authoritative state after every mutation that can change it."""
        if name in {"add_order_item", "set_order_fulfillment", "set_order_notes", "update_order_item", "remove_order_item", "confirm_order"}:
            from app.services.restaurant import restaurant_service

            readback = await restaurant_service.get_order_summary(
                call_id=str(arguments.get("session_id") or "")
            )
            readback = dict(readback)
            readback.setdefault("unresolved_fields", [])
            readback["state_version"] = int(readback.get("draft_version") or 0)
            readback["readback_committed"] = True
            readback["readback_hash"] = _order_readback_hash(readback)
            return readback
        if name == "update_reservation_draft":
            from app.call_memory import get_reservation_draft, hydrate_call_memory

            session_id = str(arguments.get("session_id") or "")
            await hydrate_call_memory(session_id)
            draft = dict(get_reservation_draft(session_id))
            draft["readback_committed"] = True
            return draft
        if name == "add_guest_note":
            booking_id = int(arguments.get("booking_id") or 0)
            from app.call_memory import get_call_memory

            if booking_id:
                from app.services.restaurant import restaurant_service

                readback = dict(await restaurant_service.lookup_booking(booking_id=booking_id))
                readback["readback_committed"] = True
                readback["guest_notes"] = str(
                    get_call_memory(str(arguments.get("session_id") or "")).get("guest_notes") or ""
                )
                return readback
            from app.services.restaurant import restaurant_service
            from app.call_memory import get_call_memory, hydrate_call_memory

            readback = dict(
                await restaurant_service.get_order_summary(
                    call_id=str(arguments.get("session_id") or "")
                )
            )
            session_id = str(arguments.get("session_id") or "")
            await hydrate_call_memory(session_id)
            readback["guest_notes"] = str(get_call_memory(session_id).get("guest_notes") or "")
            readback.setdefault("unresolved_fields", [])
            readback["state_version"] = int(readback.get("draft_version") or 0)
            readback["readback_committed"] = True
            readback["readback_hash"] = _order_readback_hash(readback)
            return readback
        if name in {"create_booking", "update_confirmed_booking", "cancel_booking"}:
            from app.services.restaurant import restaurant_service

            booking_id = int(arguments.get("booking_id") or 0)
            if not booking_id and isinstance(result, str):
                match = re.search(r"(?:reference|booking)\s+#?\s*(\d+)", result, re.IGNORECASE)
                if match:
                    booking_id = int(match.group(1))
            if booking_id <= 0:
                return None
            try:
                readback = dict(await restaurant_service.lookup_booking(booking_id=booking_id))
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

    def __init__(self, executor: ToolExecutor, *, session_id: str = "") -> None:
        self.executor = executor
        self._calls: dict[str, ToolOutcome] = {}
        self.session_id = session_id
        self._turn_operations: dict[str, dict[str, str]] = {}

    def bind_session(self, session_id: str) -> None:
        if self.session_id and self.session_id != session_id:
            raise ValueError("tool bridge is already bound to another session")
        self.session_id = session_id

    async def _verified_booking_identity(self) -> tuple[dict[str, Any] | None, str]:
        if not self.session_id:
            return None, "booking_scope_unverified"
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
            from app.services.restaurant import restaurant_service

            verified_booking = await restaurant_service.lookup_booking(booking_id=booking_id)
        except Exception:
            return None, "booking_scope_unverified"
        if (
            str(verified_booking.get("customer_name") or "").casefold() != trusted_name.casefold()
            or str(verified_booking.get("customer_phone") or "").strip() != trusted_phone
            or str(verified_booking.get("status") or "").casefold() != "confirmed"
        ):
            return None, "booking_scope_unverified"
        return {
            "booking_id": booking_id,
            "customer_name": trusted_name,
            "customer_phone": trusted_phone,
        }, ""

    async def _scoped_booking_arguments(
        self, name: str, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        if name not in BOOKING_SCOPED_TOOLS:
            return dict(arguments), ""
        if name == "add_guest_note" and self.session_id and arguments.get("booking_id") in (None, "", 0):
            trusted, error = await self._verified_booking_identity()
            if not error:
                scoped = dict(arguments)
                scoped.update(trusted or {})
                scoped["session_id"] = self.session_id
                scoped.pop("customer_name", None)
                scoped.pop("customer_phone", None)
                return scoped, ""
            try:
                from app.call_memory import get_call_memory, hydrate_call_memory

                await hydrate_call_memory(self.session_id)
                memory = get_call_memory(self.session_id)
                if int(memory.get("booking_id") or 0):
                    return None, "booking_scope_unverified"
            except (TypeError, ValueError):
                return None, "booking_scope_unverified"
            scoped = dict(arguments)
            scoped["session_id"] = self.session_id
            scoped["booking_id"] = 0
            scoped.pop("customer_name", None)
            scoped.pop("customer_phone", None)
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
            if supplied and supplied.casefold() != str(trusted[key]).casefold():
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
            from app.services.restaurant import restaurant_service

            current = await restaurant_service.get_order_summary(call_id=self.session_id)
        except Exception:
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
                scoped.pop("customer_name", None)
            return scoped, ""
        if isinstance(current, Mapping):
            current_booking_id = int(current.get("booking_id") or 0)
            if not current_booking_id:
                if name == "add_order_item":
                    scoped.pop("customer_name", None)
                    scoped.pop("customer_phone", None)
                    scoped.pop("booking_id", None)
                elif name == "set_order_fulfillment":
                    scoped.pop("booking_id", None)
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
            scoped.pop("customer_name", None)
            scoped.pop("customer_phone", None)
            scoped.pop("booking_id", None)
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
        operation_fingerprint = f"{fingerprint}:{state_version}"
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
                )

        try:
            result = await self.executor.invoke(name, args)
        except Exception as exc:  # structured failure; never a success-like response
            outcome = ToolOutcome(name=name, call_id=call_id, arguments=args, result=None, success=False, error=f"tool_exception:{type(exc).__name__}", state_version=state_version)
            self._calls[call_id] = outcome
            return outcome

        success = not _is_failure(result)
        error = str(result) if not success else ""
        readback = None
        readback_verified = False
        if success and name in MUTATING_TOOLS:
            try:
                readback = await self.executor.readback(name, args, result)
            except Exception as exc:
                error = f"readback_exception:{type(exc).__name__}"
            else:
                readback_verified = self._verify_readback(name, args, readback, state_version)
                if not readback_verified:
                    error = "database_readback_mismatch"
        facts = self._facts(result, readback, args)
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
        )
        self._calls[call_id] = outcome
        if name in MUTATING_TOOLS and turn_id and outcome.success and outcome.readback_verified:
            self._turn_operations.setdefault(turn_id, {})[operation_fingerprint] = call_id
        return outcome

    @staticmethod
    def _verify_readback(name: str, arguments: Mapping[str, Any], readback: Any, state_version: int) -> bool:
        if readback is None:
            return False
        if isinstance(readback, Mapping):
            if readback.get("ok") is False or readback.get("status") in {"failed", "error"}:
                return False
            if arguments.get("expected_draft_version") is not None:
                try:
                    if int(readback.get("draft_version")) != int(arguments["expected_draft_version"]):
                        return False
                except (TypeError, ValueError):
                    return False
            if "order_id" in readback:
                if not readback.get("readback_committed"):
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
                if name == "add_order_item" and arguments.get("item_name"):
                    if not any(
                        str(item.get("item_name") or "").casefold() == str(arguments["item_name"]).casefold()
                        for item in readback["items"]
                    ):
                        return False
                if name in {"update_order_item", "remove_order_item"} and arguments.get("order_item_id"):
                    ids = {str(item.get("order_item_id")) for item in readback["items"] + readback["proposed_items"]}
                    if name == "update_order_item" and str(arguments["order_item_id"]) not in ids:
                        return False
                    if name == "remove_order_item" and str(arguments["order_item_id"]) in ids:
                        return False
                return bool(readback.get("order_id") and int(readback.get("draft_version") or 0) > 0)
            if "booking_id" in readback:
                required = ("booking_id", "customer_name", "customer_phone", "status", "date", "time", "readback_committed")
                return bool(
                    all(field in readback for field in required)
                    and readback.get("booking_id")
                    and readback.get("customer_name")
                    and readback.get("customer_phone")
                    and readback.get("status") not in {"failed", "error"}
                    and (name == "cancel_booking" or readback.get("status") != "cancelled")
                )
            if name == "update_reservation_draft":
                return bool(
                    readback.get("readback_committed")
                    and all(field in readback for field in ("customer_name", "customer_phone", "date", "time", "party_size"))
                )
            return False
        if isinstance(readback, str):
            return False
        return False

    @staticmethod
    def _facts(result: Any, readback: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        subject = {
            key: arguments[key]
            for key in ("item_name", "date", "time", "preferred_location", "session_id", "booking_id")
            if arguments.get(key) not in (None, "", 0)
        }
        if subject:
            facts["subject"] = subject
        for value in (result, readback):
            if not isinstance(value, Mapping):
                continue
            for key in ("booking_id", "order_id"):
                if value.get(key) not in (None, "", 0):
                    facts.setdefault("subject", {})[key] = value[key]
            for key in ("items", "prices", "availability", "booking", "order", "status"):
                if key in value:
                    facts[key] = value[key]
            for key in ("evidence_source", "evidence_version"):
                if value.get(key) not in (None, ""):
                    facts[key] = value[key]
            if isinstance(value.get("items"), list):
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
                    }
                    for item in value["items"]
                    if isinstance(item, Mapping) and (item.get("item_id") or item.get("item_name") or item.get("name"))
                ]
                if canonical_items:
                    facts["canonical_items"] = canonical_items
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
