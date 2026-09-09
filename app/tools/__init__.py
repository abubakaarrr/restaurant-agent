"""Grounded read and transactional tools for the rollback LangGraph adapter."""

from app.tools.rag import search_menu, search_restaurant_info
from app.tools.db import (
    check_table_availability,
    create_booking,
    update_reservation_draft,
    get_reservation_draft,
    update_confirmed_booking,
    lookup_booking,
    cancel_booking,
    add_guest_note,
    add_order_item,
    get_order_summary,
    set_order_fulfillment,
    set_order_notes,
    update_order_item,
    remove_order_item,
    confirm_order,
    lookup_order,
    get_full_menu,
    check_menu_item_availability,
    request_handoff,
    log_unknown_question,
    end_call,
)

ALL_TOOLS = [
    search_menu,
    search_restaurant_info,
    check_table_availability,
    create_booking,
    update_reservation_draft,
    get_reservation_draft,
    update_confirmed_booking,
    lookup_booking,
    cancel_booking,
    add_guest_note,
    add_order_item,
    get_order_summary,
    set_order_fulfillment,
    set_order_notes,
    update_order_item,
    remove_order_item,
    confirm_order,
    lookup_order,
    get_full_menu,
    check_menu_item_availability,
    request_handoff,
    log_unknown_question,
    end_call,
]

__all__ = ["ALL_TOOLS"]
