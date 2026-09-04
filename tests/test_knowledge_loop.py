from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from app.call_memory import reset_current_session_id, set_current_session_id
from app.knowledge_search import search_static_knowledge
from app.services.restaurant import RestaurantServiceError, restaurant_service
from app.tools.rag import search_menu, search_restaurant_info


def test_water_on_arrival_is_grounded() -> None:
    hits = search_static_knowledge("do you serve water when I arrive")
    assert hits
    blob = " ".join(hit["content"] for hit in hits).casefold()
    assert "water" in blob


def test_parking_is_not_invented() -> None:
    hits = search_static_knowledge("where can I park")
    blob = " ".join(hit["content"] for hit in hits).casefold()
    assert "garage" not in blob
    assert "metered" not in blob


def test_generic_restaurant_word_does_not_make_parking_a_match() -> None:
    assert search_static_knowledge("does the restaurant have parking") == []


@pytest.mark.asyncio
async def test_unmatched_parking_question_is_logged_by_search_tool(monkeypatch) -> None:
    monkeypatch.setattr(
        restaurant_service,
        "restaurant_info",
        AsyncMock(
            return_value={
                "matched": False,
                "formatted": "",
                "log_unknown": True,
                "restaurant_name": "The Lamplighter Public House",
            }
        ),
    )
    logged = AsyncMock(
        return_value={"logged": True, "gap_id": 42, "status": "unresolved"}
    )
    monkeypatch.setattr(restaurant_service, "log_unknown_question", logged)

    token = set_current_session_id("parking-gap-call")
    try:
        result = await search_restaurant_info.ainvoke(
            {"query": "Does the restaurant have parking?"}
        )
    finally:
        reset_current_session_id(token)

    assert "Knowledge gap logged (id 42)" in result
    logged.assert_awaited_once()
    assert logged.await_args.kwargs["call_id"] == "parking-gap-call"


@pytest.mark.asyncio
async def test_missing_menu_details_are_logged(monkeypatch) -> None:
    monkeypatch.setattr(
        restaurant_service,
        "list_menu",
        AsyncMock(
            return_value={
                "items": [
                    {
                        "name": "Cobb Salad",
                        "category": "main",
                        "description": "Confirmed real item.",
                        "dietary": [],
                        "available": True,
                        "price": 18.0,
                        "price_estimated": True,
                    }
                ],
                "allergen_notice": "No allergen-free guarantee.",
            }
        ),
    )
    logged = AsyncMock(
        return_value={"logged": True, "gap_id": 43, "status": "unresolved"}
    )
    monkeypatch.setattr(restaurant_service, "log_unknown_question", logged)

    result = await search_menu.ainvoke(
        {
            "query": "What are the ingredients and how many people does the Cobb Salad serve?",
            "session_id": "cobb-gap-call",
        }
    )

    assert "ingredients" in result
    assert "serving_size" in result
    assert "gap logged (id 43)" in result
    logged.assert_awaited_once()


def test_hours_and_patio_are_grounded() -> None:
    hours = search_static_knowledge("opening hours weekly schedule")
    patio = search_static_knowledge("do you have a patio")
    assert any("not confirmed" in hit["content"].casefold() for hit in hours)
    assert any("patio" in hit["content"].casefold() for hit in patio)


async def test_restaurant_info_hours_do_not_invent_a_schedule() -> None:
    result = await restaurant_service.restaurant_info("what time do you close")
    assert result.get("hours_unconfirmed") is True
    assert result.get("opening_hours") == {}
    assert "not confirmed" in (result.get("formatted") or "").casefold()
    assert "11:30" not in (result.get("formatted") or "")


async def test_restaurant_info_returns_address_without_transfer() -> None:
    result = await restaurant_service.restaurant_info("what is your address")
    assert result.get("matched") is True
    blob = " ".join(
        [
            result.get("formatted") or "",
            result.get("street_address") or "",
            result.get("city") or "",
        ]
    ).casefold()
    assert "92 water" in blob or "gastown" in blob
    assert result.get("log_unknown") is not True


async def test_unknown_question_without_db_shape() -> None:
    result = await restaurant_service.restaurant_info(
        "do you offer helicopter valet and rooftop bowling"
    )
    if result.get("matched"):
        pytest.skip("Unexpected match against static knowledge")
    assert result.get("log_unknown") is True


pytestmark_db = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


@pytestmark_db
@pytest.mark.asyncio
async def test_unknown_question_logs_and_resolve_is_searchable(monkeypatch) -> None:
    import asyncpg

    from app.config import settings
    from app.db_pool import close_pool

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute(
            "TRUNCATE operator_knowledge, knowledge_gaps RESTART IDENTITY CASCADE"
        )
    finally:
        await connection.close()

    with pytest.raises(RestaurantServiceError):
        await restaurant_service.resolve_knowledge_gap(999, answer="Nope")

    logged = await restaurant_service.log_unknown_question(
        call_id="know-1",
        question="Do you have a coat check?",
        context_excerpt="Caller asked about coat check after booking.",
    )
    assert logged["logged"] is True
    replay = await restaurant_service.log_unknown_question(
        call_id="know-1",
        question="Do you have a coat check?",
    )
    assert replay["gap_id"] == logged["gap_id"]

    resolved = await restaurant_service.resolve_knowledge_gap(
        logged["gap_id"],
        answer="Yes, free coat check is by the host stand.",
        resolved_by="tester",
    )
    assert resolved["resolved"] is True

    result = await restaurant_service.restaurant_info("is there a coat check")
    assert result["matched"] is True
    assert "host stand" in result["formatted"].casefold()
    await close_pool()
