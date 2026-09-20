"""Application-owned state and event contracts for native voice.

The model may propose tool arguments, but these dataclasses are the only
representation that can be persisted as order memory.  Conversation history,
audio, and model text are intentionally absent from :class:`OrderState`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    return tuple(str(part) for part in value if str(part))


def _clean_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class CorrectionRecord:
    """Historical correction; the latest value lives in the current state."""

    field: str
    previous_value: Any
    new_value: Any
    source_turn_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorrectionRecord":
        return cls(
            field=str(value.get("field") or ""),
            previous_value=value.get("previous_value"),
            new_value=value.get("new_value"),
            source_turn_id=str(value.get("source_turn_id") or ""),
        )


@dataclass(frozen=True)
class UnresolvedField:
    """An explicit ambiguity or unsupported request awaiting clarification."""

    field: str
    reason: str
    candidates: tuple[str, ...] = ()
    source_turn_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", _tuple(self.candidates))

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "reason": self.reason,
            "candidates": list(self.candidates),
            "source_turn_id": self.source_turn_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UnresolvedField":
        return cls(
            field=str(value.get("field") or ""),
            reason=str(value.get("reason") or ""),
            candidates=_tuple(value.get("candidates")),
            source_turn_id=str(value.get("source_turn_id") or ""),
        )


@dataclass(frozen=True)
class OrderItemState:
    """One canonical menu line in the application-owned order draft."""

    canonical_item_id: str
    item_name: str
    quantity: int
    modifiers: tuple[str, ...] | None = None
    removals: tuple[str, ...] | None = None
    substitutions: tuple[str, ...] | None = None
    source_turn_ids: tuple[str, ...] = ()
    status: str = "draft"
    line_id: str = ""
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.canonical_item_id:
            raise ValueError("canonical_item_id is required for an order item")
        if not self.item_name:
            raise ValueError("item_name is required for an order item")
        if self.quantity < 1:
            raise ValueError("quantity must be positive")
        for name in ("modifiers", "removals", "substitutions", "source_turn_ids"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _tuple(value))

    def merge(self, other: "OrderItemState") -> "OrderItemState":
        """Merge a repeated line while preserving explicit latest fields."""
        if self.canonical_item_id != other.canonical_item_id:
            raise ValueError("cannot merge different canonical items")
        return OrderItemState(
            canonical_item_id=other.canonical_item_id,
            item_name=other.item_name or self.item_name,
            quantity=other.quantity,
            modifiers=self.modifiers if other.modifiers is None else other.modifiers,
            removals=self.removals if other.removals is None else other.removals,
            substitutions=self.substitutions if other.substitutions is None else other.substitutions,
            source_turn_ids=tuple(dict.fromkeys(self.source_turn_ids + other.source_turn_ids)),
            status=other.status,
            line_id=other.line_id or self.line_id,
            notes=self.notes if other.notes is None else other.notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_item_id": self.canonical_item_id,
            "item_name": self.item_name,
            "quantity": self.quantity,
            "modifiers": list(self.modifiers or ()),
            "removals": list(self.removals or ()),
            "substitutions": list(self.substitutions or ()),
            "source_turn_ids": list(self.source_turn_ids),
            "status": self.status,
            "line_id": self.line_id,
            "notes": self.notes or "",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OrderItemState":
        return cls(
            canonical_item_id=str(value.get("canonical_item_id") or ""),
            item_name=str(value.get("item_name") or ""),
            quantity=int(value.get("quantity") or 0),
            modifiers=_tuple(value.get("modifiers")) if "modifiers" in value else None,
            removals=_tuple(value.get("removals")) if "removals" in value else None,
            substitutions=_tuple(value.get("substitutions")) if "substitutions" in value else None,
            source_turn_ids=_tuple(value.get("source_turn_ids")),
            status=str(value.get("status") or "draft"),
            line_id=str(value.get("line_id") or ""),
            notes=str(value.get("notes") or "") if "notes" in value else None,
        )


@dataclass(frozen=True)
class OrderPatch:
    """Facts extracted from one finalized caller turn.

    ``None`` means a field was not mentioned.  An empty string is an explicit
    clear only for fields where the caller said to remove existing text.
    """

    source_turn_id: str
    items: tuple[OrderItemState, ...] = ()
    remove_line_ids: tuple[str, ...] = ()
    order_notes: str | None = None
    allergy_notes: str | None = None
    guest_notes: str | None = None
    fulfillment: str | None = None
    fulfillment_details: dict[str, Any] | None = None
    corrections: tuple[CorrectionRecord, ...] = ()
    unresolved_fields: tuple[UnresolvedField, ...] = ()
    resolved_fields: tuple[str, ...] = ()
    status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "remove_line_ids", _tuple(self.remove_line_ids))
        object.__setattr__(self, "corrections", tuple(self.corrections))
        object.__setattr__(self, "unresolved_fields", tuple(self.unresolved_fields))
        object.__setattr__(self, "resolved_fields", _tuple(self.resolved_fields))
        if self.fulfillment_details is not None:
            object.__setattr__(self, "fulfillment_details", _clean_mapping(self.fulfillment_details))


@dataclass(frozen=True)
class OrderState:
    """Versioned, typed order memory outside model context."""

    schema_version: str = "native-order-state.v1"
    version: int = 0
    items: tuple[OrderItemState, ...] = ()
    order_notes: str = ""
    allergy_notes: str = ""
    guest_notes: str = ""
    fulfillment: str = ""
    fulfillment_details: dict[str, Any] = field(default_factory=dict)
    corrections: tuple[CorrectionRecord, ...] = ()
    unresolved_fields: tuple[UnresolvedField, ...] = ()
    source_turn_ids: tuple[str, ...] = ()
    finalized_turn_ids: tuple[str, ...] = ()
    status: str = "empty"

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("state version cannot be negative")
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "corrections", tuple(self.corrections))
        object.__setattr__(self, "unresolved_fields", tuple(self.unresolved_fields))
        object.__setattr__(self, "source_turn_ids", _tuple(self.source_turn_ids))
        object.__setattr__(self, "finalized_turn_ids", _tuple(self.finalized_turn_ids))
        object.__setattr__(self, "fulfillment_details", _clean_mapping(self.fulfillment_details))

    def mark_turn_finalized(self, turn_id: str) -> "OrderState":
        if not turn_id:
            raise ValueError("turn_id is required")
        if turn_id in self.finalized_turn_ids:
            raise ValueError("turn has already been finalized")
        return OrderState(
            schema_version=self.schema_version,
            version=self.version + 1,
            items=self.items,
            order_notes=self.order_notes,
            allergy_notes=self.allergy_notes,
            guest_notes=self.guest_notes,
            fulfillment=self.fulfillment,
            fulfillment_details=self.fulfillment_details,
            corrections=self.corrections,
            unresolved_fields=self.unresolved_fields,
            source_turn_ids=self.source_turn_ids,
            finalized_turn_ids=self.finalized_turn_ids + (turn_id,),
            status=self.status,
        )

    def apply(self, patch: OrderPatch) -> "OrderState":
        """Apply one finalized turn atomically and increment the state version."""
        if not patch.source_turn_id:
            raise ValueError("source_turn_id is required")
        if patch.source_turn_id in self.source_turn_ids:
            raise ValueError("source turn has already been applied")
        items = list(self.items)
        for incoming in patch.items:
            match_index = next(
                (
                    index
                    for index, existing in enumerate(items)
                    if existing.status != "removed"
                    and (
                        (incoming.line_id and existing.line_id == incoming.line_id)
                        or (
                            not incoming.line_id
                            and not existing.line_id
                            and existing.canonical_item_id == incoming.canonical_item_id
                            and (
                                all(value is None for value in (incoming.modifiers, incoming.removals, incoming.substitutions))
                                or any(value == () for value in (incoming.modifiers, incoming.removals, incoming.substitutions))
                                or (
                                    (incoming.modifiers is None or existing.modifiers == incoming.modifiers)
                                    and (incoming.removals is None or existing.removals == incoming.removals)
                                    and (incoming.substitutions is None or existing.substitutions == incoming.substitutions)
                                )
                            )
                        )
                    )
                ),
                None,
            )
            if match_index is None:
                items.append(
                    OrderItemState(
                        canonical_item_id=incoming.canonical_item_id,
                        item_name=incoming.item_name,
                        quantity=incoming.quantity,
                        modifiers=incoming.modifiers or (),
                        removals=incoming.removals or (),
                        substitutions=incoming.substitutions or (),
                        source_turn_ids=incoming.source_turn_ids,
                        status=incoming.status,
                        line_id=incoming.line_id,
                        notes=incoming.notes or "",
                    )
                )
            else:
                items[match_index] = items[match_index].merge(incoming)
        if patch.remove_line_ids:
            ids = set(patch.remove_line_ids)
            items = [
                OrderItemState(
                    canonical_item_id=item.canonical_item_id,
                    item_name=item.item_name,
                    quantity=item.quantity,
                    modifiers=item.modifiers or (),
                    removals=item.removals or (),
                    substitutions=item.substitutions or (),
                    source_turn_ids=item.source_turn_ids,
                    status="removed" if item.line_id in ids else item.status,
                    line_id=item.line_id,
                    notes=item.notes or "",
                )
                for item in items
            ]

        unresolved = list(self.unresolved_fields)
        if patch.resolved_fields:
            resolved = set(patch.resolved_fields)
            unresolved = [item for item in unresolved if item.field not in resolved]
        for field_value in patch.unresolved_fields:
            unresolved = [item for item in unresolved if item.field != field_value.field]
            unresolved.append(field_value)

        source_turn_ids = tuple(dict.fromkeys(self.source_turn_ids + (patch.source_turn_id,)))
        finalized_turn_ids = tuple(dict.fromkeys(self.finalized_turn_ids + (patch.source_turn_id,)))
        status = patch.status or ("needs_clarification" if unresolved else "draft")
        if not items and not any((patch.order_notes, patch.allergy_notes, patch.guest_notes, patch.fulfillment)):
            status = patch.status or ("needs_clarification" if unresolved else "empty")

        return OrderState(
            schema_version=self.schema_version,
            version=self.version + 1,
            items=tuple(items),
            order_notes=self.order_notes if patch.order_notes is None else patch.order_notes,
            allergy_notes=self.allergy_notes if patch.allergy_notes is None else patch.allergy_notes,
            guest_notes=self.guest_notes if patch.guest_notes is None else patch.guest_notes,
            fulfillment=self.fulfillment if patch.fulfillment is None else patch.fulfillment,
            fulfillment_details=(
                dict(self.fulfillment_details)
                if patch.fulfillment_details is None
                else dict(patch.fulfillment_details)
            ),
            corrections=self.corrections + tuple(patch.corrections),
            unresolved_fields=tuple(unresolved),
            source_turn_ids=source_turn_ids,
            finalized_turn_ids=finalized_turn_ids,
            status=status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "items": [item.to_dict() for item in self.items],
            "order_notes": self.order_notes,
            "allergy_notes": self.allergy_notes,
            "guest_notes": self.guest_notes,
            "fulfillment": self.fulfillment,
            "fulfillment_details": dict(self.fulfillment_details),
            "corrections": [item.to_dict() for item in self.corrections],
            "unresolved_fields": [item.to_dict() for item in self.unresolved_fields],
            "source_turn_ids": list(self.source_turn_ids),
            "finalized_turn_ids": list(self.finalized_turn_ids),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "OrderState":
        data = dict(value or {})
        return cls(
            schema_version=str(data.get("schema_version") or "native-order-state.v1"),
            version=int(data.get("version") or 0),
            items=tuple(OrderItemState.from_dict(item) for item in data.get("items") or []),
            order_notes=str(data.get("order_notes") or ""),
            allergy_notes=str(data.get("allergy_notes") or ""),
            guest_notes=str(data.get("guest_notes") or ""),
            fulfillment=str(data.get("fulfillment") or ""),
            fulfillment_details=_clean_mapping(data.get("fulfillment_details")),
            corrections=tuple(CorrectionRecord.from_dict(item) for item in data.get("corrections") or []),
            unresolved_fields=tuple(UnresolvedField.from_dict(item) for item in data.get("unresolved_fields") or []),
            source_turn_ids=_tuple(data.get("source_turn_ids")),
            finalized_turn_ids=_tuple(data.get("finalized_turn_ids") or data.get("source_turn_ids")),
            status=str(data.get("status") or "empty"),
        )
