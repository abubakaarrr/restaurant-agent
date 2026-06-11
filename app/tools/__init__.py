"""Restaurant agent tools — RAG search + database operations."""

from app.tools.rag import search_menu, search_restaurant_info
from app.tools.db import (
    check_table_availability,
    create_booking,
    lookup_booking,
    cancel_booking,
    add_order_item,
    confirm_order,
    lookup_order,
    get_full_menu,
    check_menu_item_availability,
)

ALL_TOOLS = [
    search_menu,
    search_restaurant_info,
    check_table_availability,
    create_booking,
    lookup_booking,
    cancel_booking,
    add_order_item,
    confirm_order,
    lookup_order,
    get_full_menu,
    check_menu_item_availability,
]

__all__ = ["ALL_TOOLS"]
