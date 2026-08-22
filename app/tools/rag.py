"""Grounded read tools that avoid embeddings on the live voice path."""

from __future__ import annotations

import re

from langchain_core.tools import tool

from app.services.restaurant import (
    RestaurantServiceError,
    format_menu_price,
    restaurant_service,
)


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2
    }


@tool
async def search_menu(query: str) -> str:
    """Search live menu names, descriptions, categories, dietary tags, and prices."""
    try:
        menu = await restaurant_service.list_menu()
    except RestaurantServiceError as error:
        return f"{error.code}: {error.message}"

    query_tokens = _tokens(query)
    matches: list[dict] = []
    for item in menu["items"]:
        searchable = " ".join(
            [
                item["name"],
                item["category"],
                item["description"],
                " ".join(item["dietary"]),
            ]
        )
        score = len(query_tokens & _tokens(searchable))
        if score:
            matches.append({**item, "_score": score})
    matches.sort(key=lambda item: (-item["_score"], item["name"]))
    if not matches:
        return "No grounded menu result matched that question. Ask the caller to clarify."
    lines = [
        (
            f"{item['name']} ({format_menu_price(item)}, "
            f"{'available' if item['available'] else 'sold out'}): "
            f"{item['description'] or 'No additional description.'} "
            f"Dietary tags: {', '.join(item['dietary']) or 'none listed'}."
        )
        for item in matches[:8]
    ]
    return " ".join(lines) + f" Allergy safety: {menu['allergen_notice']}"


@tool
async def search_restaurant_info(query: str) -> str:
    """Answer hours, address, parking, cancellation, late arrival, patio, birthday cake, and other restaurant policy questions from approved knowledge. If nothing matches, do not transfer; call log_unknown_question."""
    try:
        result = await restaurant_service.restaurant_info(query)
    except RestaurantServiceError as error:
        return f"{error.code}: {error.message}"
    if result.get("formatted"):
        return result["formatted"]
    parts: list[str] = []
    if result.get("restaurant_name"):
        parts.append(f"Restaurant: {result['restaurant_name']}.")
    if result.get("hours_unconfirmed") or result.get("hours_note"):
        parts.append(result.get("hours_note") or result.get("formatted") or "")
    elif result.get("opening_hours"):
        hours = "; ".join(
            f"{day}: {value.get('open', 'closed')}-{value.get('close', 'closed')}"
            for day, value in result["opening_hours"].items()
        )
        parts.append(f"Hours: {hours}.")
    address = " ".join(
        value
        for value in (result.get("street_address", ""), result.get("city", ""))
        if value
    )
    if address:
        parts.append(f"Address: {address}.")
    if result.get("phone_number"):
        parts.append(f"Phone: {result['phone_number']}.")
    if result.get("languages"):
        parts.append(f"Supported languages: {', '.join(result['languages'])}.")
    if parts:
        return " ".join(parts)
    return (
        "No grounded restaurant answer matched that question. "
        "Call log_unknown_question with the caller's words. Do not transfer. "
        "Tell them you will get the answer from the team."
    )
