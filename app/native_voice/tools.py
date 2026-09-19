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
        return value.get("ok") is False or bool(value.get("error"))
    if isinstance(value, str):
        return bool(re.match(r"^[a-z][a-z0-9_]*:", value.strip()))
    return False


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
        tool = self._tools[name]
        args = dict(arguments)
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(args)
        return await tool(**args)

    async def readback(self, name: str, arguments: Mapping[str, Any], result: Any) -> Any:
        """Read authoritative state after every mutation that can change it."""
        if name in {"add_order_item", "set_order_fulfillment", "set_order_notes", "update_order_item", "remove_order_item", "confirm_order"}:
            from app.services.restaurant import restaurant_service

            return await restaurant_service.get_order_summary(
                call_id=str(arguments.get("session_id") or "")
            )
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
                return await restaurant_service.lookup_booking(booking_id=booking_id)
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

    def __init__(self, executor: ToolExecutor) -> None:
        self.executor = executor
        self._calls: dict[str, ToolOutcome] = {}

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
        fingerprint = f"{name}:{_hash(args)}"
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

        if not call_id or not name:
            outcome = ToolOutcome(name=name, call_id=call_id, arguments=args, result=None, success=False, error="invalid_tool_call", state_version=state_version)
            self._calls[call_id] = outcome
            return outcome
        if name not in KNOWN_TOOLS:
            outcome = ToolOutcome(name=name, call_id=call_id, arguments=args, result=None, success=False, error="unsupported_tool", state_version=state_version)
            self._calls[call_id] = outcome
            return outcome
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
                readback_verified = self._verify_readback(args, readback, state_version)
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
        return outcome

    @staticmethod
    def _verify_readback(arguments: Mapping[str, Any], readback: Any, state_version: int) -> bool:
        if readback is None:
            return False
        if isinstance(readback, Mapping):
            if readback.get("ok") is False or readback.get("status") in {"failed", "error"}:
                return False
            if readback.get("readback_verified") is True:
                return readback.get("state_version", state_version) == state_version
            expected_draft = arguments.get("expected_draft_version")
            if expected_draft is not None and readback.get("draft_version") is not None:
                return int(readback["draft_version"]) == int(expected_draft)
            return bool(readback.get("status") or readback.get("order_id") or readback.get("booking_id"))
        return bool(readback)

    @staticmethod
    def _facts(result: Any, readback: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        for value in (result, readback):
            if not isinstance(value, Mapping):
                continue
            for key in ("items", "prices", "availability", "booking", "order", "status"):
                if key in value:
                    facts[key] = value[key]
            if value.get("item"):
                item = value["item"]
                if isinstance(item, Mapping):
                    facts.setdefault("items", []).append(item.get("item_id") or item.get("name"))
                    if item.get("price") is not None:
                        facts.setdefault("prices", {})[item.get("item_id") or item.get("name")] = item["price"]
        if isinstance(result, str):
            item_name = str(arguments.get("item_name") or "")
            if item_name and item_name.casefold() in result.casefold() and not result.casefold().startswith("no matching"):
                facts.setdefault("items", []).append(item_name)
            if re.search(r"\bavailable\b", result, re.IGNORECASE):
                facts["availability"] = "available"
            elif re.search(r"\b(?:sold out|unavailable|not available)\b", result, re.IGNORECASE):
                facts["availability"] = "unavailable"
            prices = re.findall(r"\$\s*(\d+(?:\.\d{1,2})?)", result)
            if prices:
                facts["prices"] = {item_name or "amount": float(prices[-1])}
        return facts
