"""Grounded read tools that avoid embeddings on the live voice path."""

from __future__ import annotations

import re

from langchain_core.tools import tool

from app.call_memory import resolve_session_id
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
async def search_menu(query: str, session_id: str = "") -> str:
    """Search canonical menu ingredients, allergens, dietary tags, availability, and prices."""
    try:
        # Ingredient and allergen questions must remain answerable for sold-out
        # items; availability is reported separately and never inferred.
        menu = await restaurant_service.list_menu(available_only=False)
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
                " ".join(item.get("aliases") or []),
                " ".join(item.get("ingredients") or []),
                " ".join(item.get("allergens") or []),
            ]
        )
        score = len(query_tokens & _tokens(searchable))
        if score:
            matches.append({**item, "_score": score})
    matches.sort(key=lambda item: (-item["_score"], item["name"]))
    if not matches:
        return "No grounded menu result matched that question. Ask the caller to clarify."
    selected = matches[:8]
    def _availability_label(item: dict) -> str:
        if item.get("available"):
            return "available"
        status = str(item.get("availability") or "")
        if status == "sold_out":
            return "sold out"
        if status == "not_yet_available":
            return "not yet available"
        if item.get("effective_status") in {"expired", "future"}:
            return "not currently effective"
        if item.get("service_status") != "available":
            return str(item.get("service_message") or "unavailable").rstrip(".")
        return "not currently effective"

    lines = [
        (
            f"{item['name']} ({format_menu_price(item)}, "
            f"{_availability_label(item)}): "
            f"{item['description'] or 'No additional description.'} "
            f"Ingredients: {', '.join(item.get('ingredients') or []) or 'not listed'}. "
            f"Allergens: {', '.join(item.get('allergens') or []) or 'no recipe allergen listed'}. "
            f"Dietary tags: {', '.join(item['dietary']) or 'none listed'}. "
            f"Cross-contact: {item.get('cross_contact') or menu['allergen_notice']}"
        )
        for item in selected
    ]
    missing: list[str] = []
    lowered = query.casefold()
    if any(word in lowered for word in ("ingredient", "what's in", "what is in")) and any(
        not item.get("ingredients") for item in selected
    ):
        missing.append("ingredients")
    if any(word in lowered for word in ("serving", "how many people", "portion")) and any(
        not item.get("portion") for item in selected
    ):
        missing.append("serving_size")
    suffix = f" Allergy safety: {menu['allergen_notice']}"
    if missing and session_id:
        try:
            gap = await restaurant_service.log_unknown_question(
                call_id=session_id,
                question=query,
                context_excerpt="Missing canonical menu fields: " + ", ".join(missing),
                agent_response="The requested menu detail is not in current canonical data.",
            )
            suffix += (
                f" Missing canonical fields: {', '.join(missing)}. "
                f"Knowledge gap logged (id {gap['gap_id']})."
            )
        except RestaurantServiceError as error:
            suffix += f" Missing canonical fields: {', '.join(missing)}; {error.code}."
    return " ".join(lines) + suffix


@tool
async def search_restaurant_info(query: str, session_id: str = "") -> str:
    """Answer hours, address, parking, cancellation, late arrival, patio, birthday cake, and other restaurant policy questions from approved knowledge. If nothing matches, do not transfer; call log_unknown_question."""
    try:
        result = await restaurant_service.restaurant_info(query)
    except RestaurantServiceError as error:
        return f"{error.code}: {error.message}"
    if result.get("log_unknown"):
        try:
            logged = await restaurant_service.log_unknown_question(
                call_id=resolve_session_id(session_id),
                question=query,
                context_excerpt=f"knowledge_status={result.get('status', 'unknown')}",
                agent_response=result.get("formatted") or "The answer is not in current restaurant information.",
            )
            return (
                (result.get("formatted") or "No grounded restaurant answer matched that question.")
                + f" Knowledge gap logged (id {logged['gap_id']})."
            )
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
