"""LangChain wrappers around the provider-neutral restaurant service.

The managed Retell flow calls ``app.tool_api`` directly. These tools are kept as
the feature-flagged custom-LLM rollback path and therefore share the same
validation, draft-confirmation, and idempotency rules.
"""

from __future__ import annotations

from langchain_core.tools import tool

from app.call_flags import HandoffReason, request_end_call, request_transfer
from app.call_memory import (
    clear_active_booking,
    get_call_memory,
    get_reservation_draft as load_reservation_draft,
    hydrate_call_memory,
    make_idempotency_key,
    resolve_session_id,
    set_active_booking,
    update_call_memory,
    update_reservation_draft as save_reservation_draft,
)
from app.config import settings
from app.pending_confirmation import (
    ACTION_CREATE_BOOKING,
    booking_confirmation_payload,
    pending_state_patch,
    register_pending_confirmation,
)
from app.reservation_draft import (
    compose_notes,
    flatten_draft,
    normalize_preferred_location,
    preferred_location as seating_location,
    speak_draft,
)
from app.services.restaurant import (
    RestaurantServiceError,
    format_availability_speech,
    format_menu_price,
    restaurant_service,
)


def _error_text(error: RestaurantServiceError) -> str:
    return f"{error.code}: {error.message}"


def _format_order(summary: dict) -> str:
    items = "; ".join(
        f"{item['quantity']}x {item['item_name']} (${item['subtotal']:.2f})"
        + (f", notes: {item['notes']}" if item.get("notes") else "")
        for item in summary.get("items", [])
    )
    proposed = "; ".join(
        f"{item['quantity']}x {item['item_name']} (${item['subtotal']:.2f}) waiting for yes"
        for item in summary.get("proposed_items", [])
    )
    nonce = summary.get("summary_nonce") or ""
    text = (
        f"Order #{summary['order_id']} draft version {summary['draft_version']}. "
        f"Items: {items or 'none'}. Total ${summary['total']:.2f}. "
        f"Fulfillment: {summary['fulfillment']}."
    )
    if proposed:
        text += f" Proposed (not added): {proposed}."
    if nonce:
        text += f" summary_nonce={nonce}."
    return text


@tool
async def check_table_availability(
    date: str,
    time: str,
    party_size: int,
    preferred_location: str = "",
    session_id: str = "",
) -> str:
    """Check live table availability without changing the reservation. Date: YYYY-MM-DD; time: HH:MM. Pass preferred_location patio, main, private, or any. Use any when the caller will take any room."""
    session_id = resolve_session_id(session_id)
    raw = (preferred_location or "").strip()
    if raw.casefold() in {"any", "all", "anywhere", "either", "*"}:
        location = ""
    elif raw:
        location = normalize_preferred_location(raw)
    else:
        location = seating_location(load_reservation_draft(session_id))
    try:
        result = await restaurant_service.check_availability(
            date,
            time,
            party_size,
            preferred_location=location,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return format_availability_speech(result)


@tool
async def create_booking(
    name: str,
    phone: str,
    date: str,
    time: str,
    party_size: int,
    session_id: str = "",
    notes: str = "",
    caller_confirmed: bool = False,
) -> str:
    """Create a booking only after the caller confirms name, phone, date, time and party size. Include seating or guest instructions in notes. Never use this to change an existing booking."""
    session_id = resolve_session_id(session_id)
    draft = load_reservation_draft(session_id)
    if not notes.strip():
        notes = compose_notes(draft) or str(get_call_memory(session_id).get("notes") or "")
    try:
        result = await restaurant_service.create_booking(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "create_booking",
                {
                    "name": name,
                    "phone": phone,
                    "date": date,
                    "time": time,
                    "party_size": party_size,
                    "notes": notes,
                },
            ),
            customer_name=name,
            customer_phone=phone,
            date=date,
            time=time,
            party_size=party_size,
            notes=notes,
            confirmed=caller_confirmed,
            preferred_location=seating_location(draft),
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    set_active_booking(
        session_id,
        booking_id=int(result["booking_id"]),
        customer_name=result["customer_name"],
        customer_phone=result.get("customer_phone", ""),
        party_size=int(result["party_size"]),
        date=result["date"],
        time=result["time"],
        table_number=int(result["table_number"]),
        table_location=result.get("location", ""),
        notes=result.get("notes", ""),
    )
    extra = f" Note: {result['notes']}." if result.get("notes") else ""
    order_text = ""
    if result.get("order"):
        order_text = " Pre-order: " + _format_order(result["order"])
    prefix = (
        "Booking already confirmed. Do not create another. "
        if result.get("already_confirmed")
        else "Booking confirmed. "
    )
    return (
        f"{prefix}Reference {result['booking_id']}; table "
        f"{result['table_number']} in {result['location']} for {result['party_size']} "
        f"on {result['date']} at {result['time']} under {result['customer_name']}.{extra}"
        f"{order_text}"
    )


@tool
async def update_reservation_draft(
    session_id: str = "",
    name: str | None = None,
    phone: str | None = None,
    date: str | None = None,
    time: str | None = None,
    party_size: int | None = None,
    seating_preference: str | None = None,
    seating_backup: str | None = None,
    seating_avoid: str | None = None,
    dietary: str | None = None,
    occasion: str | None = None,
    extra_notes: str | None = None,
    require_approval_for_paid_items: bool | None = None,
) -> str:
    """Save or correct reservation details. After a booking exists this also updates the live booking, including the guest name. Never transfer for a name change. Pass only fields the caller just gave. Empty string clears that field."""
    session_id = resolve_session_id(session_id)
    await hydrate_call_memory(session_id)
    updates: dict = {}
    mapping = {
        "customer_name": name,
        "customer_phone": phone,
        "date": date,
        "time": time,
        "party_size": party_size,
        "seating_preference": seating_preference,
        "seating_backup": seating_backup,
        "seating_avoid": seating_avoid,
        "dietary": dietary,
        "occasion": occasion,
        "extra_notes": extra_notes,
        "require_approval_for_paid_items": require_approval_for_paid_items,
    }
    for key, value in mapping.items():
        if value is not None:
            updates[key] = value
    if not updates:
        draft = load_reservation_draft(session_id)
        return "No draft fields changed. " + speak_draft(draft)
    try:
        draft = save_reservation_draft(session_id, updates)
        await restaurant_service.persist_call_state(
            session_id,
            flatten_draft(
                draft,
                guest_notes=str(get_call_memory(session_id).get("guest_notes") or ""),
            ),
            caller_phone=str(draft.get("customer_phone") or ""),
        )
        booking_id = int(draft.get("booking_id") or 0)
        if booking_id > 0:
            result = await restaurant_service.update_confirmed_booking(
                call_id=session_id,
                idempotency_key=make_idempotency_key(
                    "update_confirmed_booking",
                    {"booking_id": booking_id, **updates},
                ),
                booking_id=booking_id,
                confirmed=True,
                date=str(updates.get("date") or ""),
                time=str(updates.get("time") or ""),
                party_size=int(updates["party_size"]) if "party_size" in updates else 0,
                seating_preference=updates.get("seating_preference"),
                seating_backup=updates.get("seating_backup"),
                seating_avoid=updates.get("seating_avoid"),
                dietary=updates.get("dietary"),
                occasion=updates.get("occasion"),
                extra_notes=updates.get("extra_notes"),
                customer_name=str(updates.get("customer_name") or ""),
                require_approval_for_paid_items=updates.get(
                    "require_approval_for_paid_items"
                ),
            )
            if result.get("updated"):
                set_active_booking(
                    session_id,
                    booking_id=int(result["booking_id"]),
                    customer_name=result["customer_name"],
                    customer_phone=result.get("customer_phone", ""),
                    party_size=int(result["party_size"]),
                    date=result["date"],
                    time=result["time"],
                    table_number=int(result["table_number"])
                    if result.get("table_number")
                    else None,
                    table_location=result.get("location", ""),
                    notes=result.get("notes", ""),
                )
                return "Booking updated. " + speak_draft(load_reservation_draft(session_id))
            if result.get("slot_unavailable"):
                return (
                    "Name or notes were not blocked, but the new time is unavailable. "
                    "The existing booking is unchanged. Offer alternatives."
                )
    except ValueError as error:
        return f"invalid_request: {error}"
    except RestaurantServiceError as error:
        return _error_text(error)
    text = "Saved. " + speak_draft(draft)
    date = str(draft.get("date") or "")
    time = str(draft.get("time") or "")
    party_size = int(draft.get("party_size") or 0)
    if date and time and party_size:
        try:
            availability = await restaurant_service.check_availability(
                date,
                time,
                party_size,
                preferred_location=seating_location(draft),
            )
            text += " " + format_availability_speech(availability)
            if not availability.get("available"):
                text += " Draft fields are saved, but this slot is NOT reserved."
        except RestaurantServiceError as error:
            text += " " + _error_text(error)
    return text


@tool
async def get_reservation_draft(session_id: str = "") -> str:
    """Return the authoritative reservation details for this call. Use this instead of guessing from chat when asked what you have so far."""
    session_id = resolve_session_id(session_id)
    await hydrate_call_memory(session_id)
    draft = load_reservation_draft(session_id)
    notes = compose_notes(draft) or str(get_call_memory(session_id).get("notes") or "")
    if (
        draft.get("customer_name")
        and draft.get("customer_phone")
        and draft.get("date")
        and draft.get("time")
        and int(draft.get("party_size") or 0) >= 1
        and not int(draft.get("booking_id") or 0)
    ):
        register_pending_confirmation(
            session_id,
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
        try:
            await restaurant_service.persist_call_state(
                session_id, pending_state_patch(session_id)
            )
        except RestaurantServiceError:
            pass
    return (
        "Host readback facts. Speak this like a person at the stand, not a form. "
        "Do not start with 'I have [name]'. "
        + speak_draft(draft)
    )


@tool
async def update_confirmed_booking(
    session_id: str = "",
    booking_id: int = 0,
    date: str = "",
    time: str = "",
    party_size: int = 0,
    seating_preference: str | None = None,
    seating_backup: str | None = None,
    seating_avoid: str | None = None,
    dietary: str | None = None,
    occasion: str | None = None,
    extra_notes: str | None = None,
    customer_name: str = "",
    require_approval_for_paid_items: bool | None = None,
    caller_confirmed: bool = False,
) -> str:
    """Change time, party size, name, or notes on an existing confirmed booking. Never cancel and recreate. The caller's instruction to change a field is confirmation — set caller_confirmed=true and do it now. Empty string on a note field clears it. 'Forget the fifth person' is party_size 4. If the new slot is taken, offer alternatives and leave the booking unchanged."""
    session_id = resolve_session_id(session_id)
    await hydrate_call_memory(session_id)
    memory = get_call_memory(session_id)
    if booking_id <= 0:
        try:
            booking_id = int(memory.get("booking_id") or 0)
        except (TypeError, ValueError):
            booking_id = 0
    try:
        result = await restaurant_service.update_confirmed_booking(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "update_confirmed_booking",
                {
                    "booking_id": booking_id,
                    "date": date,
                    "time": time,
                    "party_size": party_size,
                    "seating_preference": seating_preference,
                    "dietary": dietary,
                    "extra_notes": extra_notes,
                    "customer_name": customer_name,
                    "require_approval_for_paid_items": require_approval_for_paid_items,
                },
            ),
            booking_id=booking_id,
            confirmed=caller_confirmed,
            date=date,
            time=time,
            party_size=party_size,
            preferred_location=seating_location(
                seating_preference
                if seating_preference is not None
                else load_reservation_draft(session_id)
            ),
            seating_preference=seating_preference,
            seating_backup=seating_backup,
            seating_avoid=seating_avoid,
            dietary=dietary,
            occasion=occasion,
            extra_notes=extra_notes,
            customer_name=customer_name,
            require_approval_for_paid_items=require_approval_for_paid_items,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    if result.get("updated"):
        set_active_booking(
            session_id,
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
        extra = f" Notes: {result['notes']}." if result.get("notes") else ""
        return (
            f"Booking {result['booking_id']} updated. Table {result['table_number']} in "
            f"{result['location']} for {result['party_size']} on {result['date']} at "
            f"{result['time']}.{extra}"
        )
    alternatives = result.get("alternatives") or []
    alt_text = " or ".join(
        f"{row['date']} at {row['display_time']}" for row in alternatives
    )
    return (
        "Requested change was not applied; that slot is unavailable. "
        + (f"Nearest available times: {alt_text}." if alt_text else "No nearby alternative was found.")
        + " The existing booking is unchanged."
    )


@tool
async def lookup_booking(
    booking_id: int = 0,
    customer_name: str = "",
    customer_phone: str = "",
) -> str:
    """Look up a booking by reference, or by exact name plus phone for verification."""
    try:
        result = await restaurant_service.lookup_booking(
            booking_id=booking_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return (
        f"Booking #{result['booking_id']} for {result['customer_name']}: "
        f"party of {result['party_size']} at {result['booked_at']}; "
        f"table {result['table_number']} ({result['location']}); status {result['status']}"
        + (f"; notes {result['notes']}" if result.get("notes") else "")
        + "."
    )


@tool
async def cancel_booking(
    booking_id: int,
    customer_name: str = "",
    customer_phone: str = "",
    reason: str = "",
    session_id: str = "",
    caller_confirmed: bool = False,
) -> str:
    """Cancel after verifying name or phone and receiving explicit confirmation. Cancellation is irreversible. Do not call this if the caller was only asking about the policy or then said not to cancel. To change time, food, or notes, use update_confirmed_booking or the order item tools instead."""
    session_id = resolve_session_id(session_id)
    try:
        result = await restaurant_service.cancel_booking(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "cancel_booking",
                {
                    "booking_id": booking_id,
                    "customer_name": customer_name,
                    "customer_phone": customer_phone,
                    "reason": reason,
                },
            ),
            booking_id=booking_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
            reason=reason,
            confirmed=caller_confirmed,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    memory = get_call_memory(session_id)
    if memory.get("booking_id") == booking_id:
        clear_active_booking(session_id)
    qualifier = " already was" if result.get("already_cancelled") else " has been"
    return f"Booking #{booking_id} for {result['customer_name']}{qualifier} cancelled."


@tool
async def add_guest_note(
    note: str,
    session_id: str = "",
    booking_id: int = 0,
) -> str:
    """Save a guest instruction on the booking notes field, such as window table, high chair, birthday, or kitchen requests. Use this instead of transferring for ordinary special requests."""
    session_id = resolve_session_id(session_id)
    memory = get_call_memory(session_id)
    if booking_id <= 0:
        try:
            booking_id = int(memory.get("booking_id") or 0)
        except (TypeError, ValueError):
            booking_id = 0
    try:
        result = await restaurant_service.add_guest_note(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "add_guest_note",
                {"booking_id": booking_id, "note": note},
            ),
            note=note,
            booking_id=booking_id,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    update_call_memory(session_id, guest_notes=result.get("guest_notes") or note)
    return (
        "Note is on the reservation. Say it like a host, not 'the note is saved': "
        + str(result.get("notes") or note)
    )


@tool
async def get_full_menu() -> str:
    """Return the live menu and prices from the database."""
    try:
        result = await restaurant_service.list_menu()
    except RestaurantServiceError as error:
        return _error_text(error)
    grouped: dict[str, list[str]] = {}
    for item in result["items"]:
        grouped.setdefault(item["category"], []).append(
            f"{item['name']} {format_menu_price(item)}"
        )
    sections = [
        f"{category.title()}: " + ", ".join(items)
        for category, items in grouped.items()
    ]
    return " | ".join(sections) + f" Allergy note: {result['allergen_notice']}"


@tool
async def check_menu_item_availability(item_name: str) -> str:
    """Check a menu item against the live menu. Do not guess fuzzy matches."""
    try:
        result = await restaurant_service.find_menu_item(item_name)
    except RestaurantServiceError as error:
        return _error_text(error)
    if result["match"]:
        item = result["match"]
        return (
            f"{item['name']} is {'available' if item['available'] else 'sold out'} "
            f"at {format_menu_price(item)}."
        )
    if result["candidates"]:
        names = ", ".join(item["name"] for item in result["candidates"])
        return f"No exact match. Ask the caller to choose or clarify: {names}."
    return "No matching menu item was found."


@tool
async def add_order_item(
    session_id: str,
    item_name: str,
    quantity: int = 1,
    notes: str = "",
    booking_id: int = 0,
    customer_name: str = "",
    customer_phone: str = "",
    caller_confirmed: bool = False,
) -> str:
    """Add an exact, caller-approved menu item. For a new draft this is enough. If the order is already confirmed, set caller_confirmed=true after they name the item. Do not call this when the caller is only asking."""
    session_id = resolve_session_id(session_id)
    memory = get_call_memory(session_id)
    booking_id = booking_id or int(memory.get("booking_id") or 0)
    customer_name = customer_name or str(memory.get("customer_name") or "")
    customer_phone = customer_phone or str(memory.get("customer_phone") or "")
    try:
        result = await restaurant_service.add_order_item(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "add_order_item",
                {
                    "item_name": item_name,
                    "quantity": quantity,
                    "notes": notes,
                    "booking_id": booking_id,
                    "caller_confirmed": caller_confirmed,
                },
            ),
            item_name=item_name,
            quantity=quantity,
            notes=notes,
            booking_id=booking_id,
            customer_name=customer_name,
            customer_phone=customer_phone,
            caller_confirmed=caller_confirmed,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    if not result.get("added"):
        if result.get("proposed"):
            return (
                "Item is proposed, not added. Wait for an explicit yes before "
                "attaching a priced item. " + _format_order(result)
            )
        if result.get("unavailable"):
            return f"{result['item']['name']} is currently unavailable."
        candidates = ", ".join(item["name"] for item in result.get("candidates", []))
        return (
            f"No exact item was added. Ask the caller to confirm one of: {candidates}."
            if candidates
            else "No matching menu item was found; ask the caller to clarify."
        )
    return "Item added to draft. " + _format_order(result)


@tool
async def get_order_summary(session_id: str) -> str:
    """Read the current order, including a confirmed reservation pre-order. This does not change anything."""
    try:
        result = await restaurant_service.get_order_summary(
            call_id=resolve_session_id(session_id)
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return _format_order(result) + " Read every item and total, then ask if all details are correct."


@tool
async def update_order_item(
    session_id: str,
    order_item_id: int,
    quantity: int,
    notes: str = "",
    caller_confirmed: bool = False,
) -> str:
    """Correct an item quantity or notes. If the order is already confirmed, set caller_confirmed=true after an explicit yes."""
    session_id = resolve_session_id(session_id)
    try:
        result = await restaurant_service.update_order_item(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "update_order_item",
                {
                    "order_item_id": order_item_id,
                    "quantity": quantity,
                    "notes": notes,
                    "caller_confirmed": caller_confirmed,
                },
            ),
            order_item_id=order_item_id,
            quantity=quantity,
            notes=notes,
            caller_confirmed=caller_confirmed,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return "Draft item updated. " + _format_order(result)


@tool
async def remove_order_item(
    session_id: str,
    order_item_id: int,
    caller_confirmed: bool = False,
) -> str:
    """Remove a caller-selected item. If the order is already confirmed, set caller_confirmed=true after an explicit yes."""
    session_id = resolve_session_id(session_id)
    try:
        result = await restaurant_service.remove_order_item(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "remove_order_item",
                {"order_item_id": order_item_id, "caller_confirmed": caller_confirmed},
            ),
            order_item_id=order_item_id,
            caller_confirmed=caller_confirmed,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return "Draft item removed. " + _format_order(result)


@tool
async def confirm_order(
    session_id: str,
    expected_draft_version: int,
    caller_approved_full_readback: bool = False,
) -> str:
    """Commit a draft only after the caller approves the complete itemized readback."""
    session_id = resolve_session_id(session_id)
    try:
        result = await restaurant_service.confirm_order(
            call_id=session_id,
            idempotency_key=make_idempotency_key(
                "confirm_order",
                {"expected_draft_version": expected_draft_version},
            ),
            expected_draft_version=expected_draft_version,
            approved=caller_approved_full_readback,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return (
        f"Order #{result['order_id']} confirmed. Total ${result['total']:.2f}; "
        f"{result['timing']}."
    )


@tool
async def lookup_order(order_id: int, customer_name: str) -> str:
    """Look up an order only after verifying the exact customer name."""
    try:
        result = await restaurant_service.lookup_order(
            order_id=order_id,
            customer_name=customer_name,
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return _format_order(result) + f" Status: {result['status']}."


@tool
async def request_handoff(session_id: str, reason: HandoffReason, topic: str = "") -> str:
    """Transfer to staff only for an explicit person/manager request, complaint, payment, severe allergy, safety, or outage. Never use this for a name change, water, notes, party size, time, birthday, parking, or menu question."""
    session_id = resolve_session_id(session_id)
    haystack = f"{reason} {topic}".casefold()
    ordinary = (
        "name",
        "water",
        "note",
        "vegetarian",
        "birthday",
        "parking",
        "party",
        "time",
        "table",
        "high chair",
        "window",
        "menu",
        "wine",
        "drink",
    )
    if any(token in haystack for token in ordinary) and reason not in {
        "severe_allergy",
        "safety",
        "payment_or_refund",
        "manager_or_complaint",
    }:
        return (
            "Do not transfer. Update the booking name with update_confirmed_booking "
            "or update_reservation_draft. For water or table requests, say yes and "
            "save add_guest_note. Keep helping. Never say you are connecting them."
        )
    if not settings.staff_transfer_number:
        return (
            "Staff transfer is not available. Never say you are connecting them or "
            "that a team member must handle it. You can change the reservation name, "
            "time, party size, and notes yourself. If they want water on arrival, "
            "say yes and save it as a guest note. Keep hosting."
        )
    request_transfer(session_id, reason)
    return "Staff transfer requested. Tell the caller you are connecting them now."


@tool
async def log_unknown_question(
    question: str,
    session_id: str = "",
    context_excerpt: str = "",
) -> str:
    """Log a restaurant question you cannot answer from search_restaurant_info. Do not transfer. Tell the caller you will get the answer from the team."""
    session_id = resolve_session_id(session_id)
    try:
        result = await restaurant_service.log_unknown_question(
            call_id=session_id,
            question=question,
            context_excerpt=context_excerpt,
            agent_response="We'll get that answer from the team and follow up.",
        )
    except RestaurantServiceError as error:
        return _error_text(error)
    return (
        f"Unknown question logged (id {result['gap_id']}). Keep helping the guest. "
        "Do not say you are connecting them or that a team member must handle it. "
        "For water, say yes we serve still and sparkling at the table and save a note "
        "if they want it chilled. For a name fix, update the booking yourself."
    )


@tool
async def end_call(session_id: str) -> str:
    """End only after final confirmation and a complete spoken goodbye."""
    request_end_call(resolve_session_id(session_id))
    return "Call ending."
