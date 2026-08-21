"""Authenticated custom-function API for Retell managed Conversation Flow."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.call_memory import (
    get_call_memory,
    get_reservation_draft as load_reservation_draft,
    hydrate_call_memory,
    set_active_booking,
    update_reservation_draft as save_reservation_draft,
)
from app.config import settings
from app.call_analytics import record_call_event
from app.pending_confirmation import (
    ACTION_CREATE_BOOKING,
    booking_confirmation_payload,
    pending_state_patch,
    register_pending_confirmation,
)
from app.reservation_draft import compose_notes, flatten_draft
from app.security import constant_time_equal
from app.services.restaurant import RestaurantServiceError, restaurant_service


router = APIRouter(prefix="/api/voice-tools", tags=["voice-tools"])
logger = logging.getLogger(__name__)
_tool_secret_header = APIKeyHeader(name="X-Voice-Tool-Secret", auto_error=False)


async def require_voice_tool_secret(
    provided: str | None = Security(_tool_secret_header),
) -> None:
    if not settings.voice_tool_secret:
        raise HTTPException(status_code=503, detail="Voice tool API is not configured")
    if not constant_time_equal(provided, settings.voice_tool_secret):
        raise HTTPException(status_code=401, detail="Invalid or missing voice tool secret")


ToolAuth = Security(require_voice_tool_secret)


class CallRequest(BaseModel):
    call_id: str = Field(min_length=1, max_length=200)


class AvailabilityRequest(BaseModel):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: str = Field(pattern=r"^\d{2}:\d{2}$")
    party_size: int = Field(ge=1, le=12)
    preferred_location: str = Field(default="", max_length=40)
    call_id: str = Field(default="", max_length=200)


class CreateBookingRequest(CallRequest, AvailabilityRequest):
    customer_name: str = Field(min_length=1, max_length=100)
    customer_phone: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=500)
    confirmed: bool
    table_number: int = Field(default=0, ge=0)


class LookupBookingRequest(BaseModel):
    booking_id: int = Field(default=0, ge=0)
    customer_name: str = Field(default="", max_length=100)
    customer_phone: str = Field(default="", max_length=200)


class CancelBookingRequest(CallRequest):
    booking_id: int = Field(gt=0)
    customer_name: str = Field(default="", max_length=100)
    customer_phone: str = Field(default="", max_length=200)
    reason: str = Field(default="", max_length=300)
    confirmed: bool


class AddGuestNoteRequest(CallRequest):
    note: str = Field(min_length=2, max_length=500)
    booking_id: int = Field(default=0, ge=0)


class MenuItemRequest(BaseModel):
    item_name: str = Field(min_length=1, max_length=150)


class AddOrderItemRequest(CallRequest, MenuItemRequest):
    quantity: int = Field(default=1, ge=1, le=20)
    notes: str = Field(default="", max_length=300)
    booking_id: int = Field(default=0, ge=0)
    customer_name: str = Field(default="", max_length=100)
    customer_phone: str = Field(default="", max_length=200)
    confirmed: bool = False


class UpdateOrderItemRequest(CallRequest):
    order_item_id: int = Field(gt=0)
    quantity: int = Field(ge=1, le=20)
    notes: str = Field(default="", max_length=300)
    confirmed: bool = False


class RemoveOrderItemRequest(CallRequest):
    order_item_id: int = Field(gt=0)
    confirmed: bool = False


class ConfirmOrderRequest(CallRequest):
    expected_draft_version: int = Field(ge=1)
    approved: bool


class SetOrderFulfillmentRequest(CallRequest):
    fulfillment_type: str = Field(min_length=1, max_length=20)
    booking_id: int = Field(default=0, ge=0)


class LookupOrderRequest(BaseModel):
    order_id: int = Field(gt=0)
    customer_name: str = Field(min_length=1, max_length=100)


class RestaurantInfoRequest(BaseModel):
    topic: str = Field(default="", max_length=200)


class UpdateReservationDraftRequest(CallRequest):
    customer_name: str | None = Field(default=None, max_length=100)
    customer_phone: str | None = Field(default=None, max_length=200)
    date: str | None = Field(default=None, max_length=10)
    time: str | None = Field(default=None, max_length=5)
    party_size: int | None = Field(default=None, ge=0, le=12)
    seating_preference: str | None = Field(default=None, max_length=80)
    seating_backup: str | None = Field(default=None, max_length=80)
    seating_avoid: str | None = Field(default=None, max_length=80)
    dietary: str | None = Field(default=None, max_length=200)
    occasion: str | None = Field(default=None, max_length=80)
    extra_notes: str | None = Field(default=None, max_length=500)
    require_approval_for_paid_items: bool | None = None


class UpdateConfirmedBookingRequest(CallRequest):
    booking_id: int = Field(default=0, ge=0)
    date: str = Field(default="", max_length=10)
    time: str = Field(default="", max_length=5)
    party_size: int = Field(default=0, ge=0, le=12)
    seating_preference: str | None = Field(default=None, max_length=80)
    seating_backup: str | None = Field(default=None, max_length=80)
    seating_avoid: str | None = Field(default=None, max_length=80)
    dietary: str | None = Field(default=None, max_length=200)
    occasion: str | None = Field(default=None, max_length=80)
    extra_notes: str | None = Field(default=None, max_length=500)
    customer_name: str = Field(default="", max_length=100)
    require_approval_for_paid_items: bool | None = None
    confirmed: bool


class LogUnknownQuestionRequest(CallRequest):
    question: str = Field(min_length=3, max_length=500)
    context_excerpt: str = Field(default="", max_length=500)


def _raise_service_error(error: RestaurantServiceError) -> None:
    raise HTTPException(
        status_code=error.status,
        detail={"code": error.code, "message": error.message},
    ) from error


def _ok(result: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "result": result}


async def _record_tool_safely(
    call_id: str,
    action: str,
    result: dict[str, Any],
) -> None:
    try:
        await record_call_event(
            call_id,
            f"tool_{action}",
            payload={
                "idempotent_replay": bool(result.get("idempotent_replay")),
                "success": True,
            },
        )
    except Exception:
        logger.warning("Could not record tool audit event %s", action, exc_info=True)


def _audit(call_id: str, action: str, result: dict[str, Any]) -> None:
    asyncio.create_task(_record_tool_safely(call_id, action, result))


@router.get("/health", dependencies=[ToolAuth])
async def voice_tool_health() -> dict[str, Any]:
    return {
        "ok": True,
        "writes_enabled": settings.voice_live_writes_enabled,
        "restaurant": settings.restaurant_name,
    }


@router.post("/availability", dependencies=[ToolAuth])
async def check_availability(body: AvailabilityRequest) -> dict[str, Any]:
    try:
        return _ok(
            await restaurant_service.check_availability(
                body.date,
                body.time,
                body.party_size,
                preferred_location=body.preferred_location,
                call_id=body.call_id,
            )
        )
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/bookings/create", dependencies=[ToolAuth])
async def create_booking(
    body: CreateBookingRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.create_booking(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            customer_name=body.customer_name,
            customer_phone=body.customer_phone,
            date=body.date,
            time=body.time,
            party_size=body.party_size,
            notes=body.notes,
            confirmed=body.confirmed,
            preferred_location=body.preferred_location,
            table_number=body.table_number,
        )
        if result.get("created") and result.get("booking_id") and result.get("customer_name"):
            set_active_booking(
                body.call_id,
                booking_id=int(result["booking_id"]),
                customer_name=result["customer_name"],
                customer_phone=result.get("customer_phone", ""),
                party_size=int(result.get("party_size") or 0),
                date=result.get("date") or "",
                time=result.get("time") or "",
                table_number=int(result["table_number"]) if result.get("table_number") else None,
                table_location=result.get("location", ""),
                notes=result.get("notes", ""),
            )
        _audit(body.call_id, "create_booking", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/bookings/lookup", dependencies=[ToolAuth])
async def lookup_booking(body: LookupBookingRequest) -> dict[str, Any]:
    try:
        return _ok(
            await restaurant_service.lookup_booking(
                booking_id=body.booking_id,
                customer_name=body.customer_name,
                customer_phone=body.customer_phone,
            )
        )
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/bookings/cancel", dependencies=[ToolAuth])
async def cancel_booking(
    body: CancelBookingRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.cancel_booking(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            booking_id=body.booking_id,
            customer_name=body.customer_name,
            customer_phone=body.customer_phone,
            reason=body.reason,
            confirmed=body.confirmed,
        )
        _audit(body.call_id, "cancel_booking", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/bookings/notes", dependencies=[ToolAuth])
async def add_guest_note(
    body: AddGuestNoteRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.add_guest_note(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            note=body.note,
            booking_id=body.booking_id,
        )
        _audit(body.call_id, "add_guest_note", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/reservations/draft", dependencies=[ToolAuth])
async def update_reservation_draft(body: UpdateReservationDraftRequest) -> dict[str, Any]:
    await hydrate_call_memory(body.call_id)
    updates = body.model_dump(exclude_unset=True, exclude={"call_id"})
    try:
        draft = save_reservation_draft(body.call_id, updates)
        await restaurant_service.persist_call_state(
            body.call_id,
            flatten_draft(
                draft,
                guest_notes=str(get_call_memory(body.call_id).get("guest_notes") or ""),
            ),
            caller_phone=str(draft.get("customer_phone") or ""),
        )
        return _ok(draft)
    except ValueError as error:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": str(error)}) from error
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/reservations/draft/get", dependencies=[ToolAuth])
async def get_reservation_draft(body: CallRequest) -> dict[str, Any]:
    await hydrate_call_memory(body.call_id)
    draft = load_reservation_draft(body.call_id)
    if int(draft.get("booking_id") or 0) > 0:
        try:
            draft = await restaurant_service.sync_confirmed_draft_from_booking(
                body.call_id, int(draft["booking_id"])
            )
        except RestaurantServiceError as error:
            _raise_service_error(error)
    notes = compose_notes(draft) or str(get_call_memory(body.call_id).get("notes") or "")
    if (
        draft.get("customer_name")
        and draft.get("customer_phone")
        and draft.get("date")
        and draft.get("time")
        and int(draft.get("party_size") or 0) >= 1
        and not int(draft.get("booking_id") or 0)
    ):
        digest = register_pending_confirmation(
            body.call_id,
            ACTION_CREATE_BOOKING,
            booking_confirmation_payload(
                customer_name=str(draft.get("customer_name") or ""),
                customer_phone=str(draft.get("customer_phone") or ""),
                date=str(draft.get("date") or ""),
                time=str(draft.get("time") or ""),
                party_size=int(draft.get("party_size") or 0),
                notes=notes,
            ),
        )
        await restaurant_service.persist_call_state(
            body.call_id, pending_state_patch(body.call_id)
        )
        return _ok({**draft, "pending_confirmation_hash": digest, "readback_required": True})
    memory = get_call_memory(body.call_id)
    return _ok(
        {
            **draft,
            "table_number": memory.get("table_number"),
            "table_location": memory.get("table_location") or "",
        }
    )


@router.post("/bookings/update", dependencies=[ToolAuth])
async def update_confirmed_booking(
    body: UpdateConfirmedBookingRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.update_confirmed_booking(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            booking_id=body.booking_id,
            confirmed=body.confirmed,
            date=body.date,
            time=body.time,
            party_size=body.party_size,
            seating_preference=body.seating_preference,
            seating_backup=body.seating_backup,
            seating_avoid=body.seating_avoid,
            dietary=body.dietary,
            occasion=body.occasion,
            extra_notes=body.extra_notes,
            customer_name=body.customer_name,
            require_approval_for_paid_items=body.require_approval_for_paid_items,
        )
        if result.get("updated") and result.get("customer_name"):
            set_active_booking(
                body.call_id,
                booking_id=int(result["booking_id"]),
                customer_name=result["customer_name"],
                customer_phone=result.get("customer_phone", ""),
                party_size=int(result["party_size"]),
                date=result["date"],
                time=result["time"],
                table_number=int(result["table_number"]) if result.get("table_number") else None,
                table_location=result.get("location", ""),
                notes=result.get("notes", ""),
            )
        _audit(body.call_id, "update_confirmed_booking", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/knowledge/unknown", dependencies=[ToolAuth])
async def log_unknown_question(body: LogUnknownQuestionRequest) -> dict[str, Any]:
    try:
        result = await restaurant_service.log_unknown_question(
            call_id=body.call_id,
            question=body.question,
            context_excerpt=body.context_excerpt,
        )
        _audit(body.call_id, "log_unknown_question", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.get("/menu", dependencies=[ToolAuth])
async def get_menu() -> dict[str, Any]:
    try:
        return _ok(await restaurant_service.list_menu())
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/menu/find", dependencies=[ToolAuth])
async def find_menu_item(body: MenuItemRequest) -> dict[str, Any]:
    try:
        return _ok(await restaurant_service.find_menu_item(body.item_name))
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/orders/items/add", dependencies=[ToolAuth])
async def add_order_item(
    body: AddOrderItemRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.add_order_item(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            item_name=body.item_name,
            quantity=body.quantity,
            notes=body.notes,
            booking_id=body.booking_id,
            customer_name=body.customer_name,
            customer_phone=body.customer_phone,
            caller_confirmed=body.confirmed,
        )
        if result.get("added"):
            _audit(body.call_id, "add_order_item", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/orders/summary", dependencies=[ToolAuth])
async def order_summary(body: CallRequest) -> dict[str, Any]:
    try:
        return _ok(await restaurant_service.get_order_summary(call_id=body.call_id))
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/orders/fulfillment", dependencies=[ToolAuth])
async def set_order_fulfillment(
    body: SetOrderFulfillmentRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.set_order_fulfillment(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            fulfillment_type=body.fulfillment_type,
            booking_id=body.booking_id or None,
        )
        _audit(body.call_id, "set_order_fulfillment", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/orders/items/update", dependencies=[ToolAuth])
async def update_order_item(
    body: UpdateOrderItemRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.update_order_item(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            order_item_id=body.order_item_id,
            quantity=body.quantity,
            notes=body.notes,
            caller_confirmed=body.confirmed,
        )
        _audit(body.call_id, "update_order_item", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/orders/items/remove", dependencies=[ToolAuth])
async def remove_order_item(
    body: RemoveOrderItemRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.remove_order_item(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            order_item_id=body.order_item_id,
            caller_confirmed=body.confirmed,
        )
        _audit(body.call_id, "remove_order_item", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/orders/confirm", dependencies=[ToolAuth])
async def confirm_order(
    body: ConfirmOrderRequest,
    idempotency_key: str = Header(default="", alias="Idempotency-Key"),
) -> dict[str, Any]:
    try:
        result = await restaurant_service.confirm_order(
            call_id=body.call_id,
            idempotency_key=idempotency_key,
            expected_draft_version=body.expected_draft_version,
            approved=body.approved,
        )
        _audit(body.call_id, "confirm_order", result)
        return _ok(result)
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/orders/lookup", dependencies=[ToolAuth])
async def lookup_order(body: LookupOrderRequest) -> dict[str, Any]:
    try:
        return _ok(
            await restaurant_service.lookup_order(
                order_id=body.order_id,
                customer_name=body.customer_name,
            )
        )
    except RestaurantServiceError as error:
        _raise_service_error(error)


@router.post("/restaurant-info", dependencies=[ToolAuth])
async def restaurant_info(body: RestaurantInfoRequest) -> dict[str, Any]:
    try:
        return _ok(await restaurant_service.restaurant_info(body.topic))
    except RestaurantServiceError as error:
        _raise_service_error(error)
