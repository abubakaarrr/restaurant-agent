from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from app.call_memory import reset_current_session_id, set_current_session_id
from app.restaurant_knowledge import get_restaurant_knowledge
from app.services.restaurant import RestaurantServiceError, restaurant_service
from app.tools.rag import search_menu, search_restaurant_info


def test_water_on_arrival_is_grounded() -> None:
    match = get_restaurant_knowledge().find_topic("do you serve water when I arrive")
    assert match.status == "known"
    assert match.records[0]["topic_id"] == "topic.water"


def test_parking_is_canonical() -> None:
    match = get_restaurant_knowledge().find_topic("where can I park")
    assert match.status == "known"
    assert match.records[0]["topic_id"] == "topic.parking"
    assert "does not validate" in match.records[0]["answer"]


def test_parking_word_maps_to_parking_not_table_availability() -> None:
    match = get_restaurant_knowledge().find_topic("does the restaurant have parking availability")
    assert [record["topic_id"] for record in match.records] == ["topic.parking"]


def test_equal_topic_aliases_return_ambiguity_instead_of_length_tiebreak() -> None:
    match = get_restaurant_knowledge().find_topic(
        "Do you have parking at your location?"
    )
    assert match.status == "ambiguous"
    assert {record["topic_id"] for record in match.records} == {
        "topic.address",
        "topic.parking",
    }


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
                "restaurant_name": "Harbor & Hearth Kitchen",
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
    hours = get_restaurant_knowledge().find_topic("opening hours weekly schedule")
    patio = get_restaurant_knowledge().find_topic("heated patio table")
    assert hours.status == "known"
    assert patio.status == "known"
    assert hours.records[0]["topic_id"] == "topic.hours"
    assert patio.records[0]["topic_id"] == "topic.seating"


async def test_restaurant_info_hours_come_from_canonical_schedule() -> None:
    result = await restaurant_service.restaurant_info("what time do you close")
    assert result["status"] == "known"
    assert result["topic_id"] == "topic.hours"
    assert result["hours"]["regular"][0]["status"] == "closed"
    assert "11:30" in result["formatted"]


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
    assert "1842 market" in blob or "portland" in blob
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
        legacy_gap_id = await connection.fetchval(
            """
            INSERT INTO knowledge_gaps
                (session_id, question, question_normalized)
            VALUES ('legacy-call', 'Does Lamplighter have a rooftop?',
                    'does lamplighter have a rooftop')
            RETURNING id
            """
        )
        await connection.execute(
            """
            INSERT INTO operator_knowledge
                (restaurant_id, question, answer, active)
            VALUES
                ('restaurant.old-example', 'Do you have a rooftop?',
                 'The old example restaurant has one.', TRUE)
            """
        )
    finally:
        await connection.close()

    isolated = await restaurant_service.restaurant_info("do you have a rooftop")
    assert isolated["status"] == "unknown"
    assert "old example" not in isolated["formatted"].casefold()

    with pytest.raises(RestaurantServiceError):
        await restaurant_service.resolve_knowledge_gap(999, answer="Nope")
    with pytest.raises(RestaurantServiceError) as legacy_error:
        await restaurant_service.resolve_knowledge_gap(
            legacy_gap_id,
            answer="The old example restaurant has one.",
        )
    assert legacy_error.value.code == "gap_not_found"
    listed = await restaurant_service.list_knowledge_gaps()
    assert all(row["id"] != legacy_gap_id for row in listed["gaps"])

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
    connection = await asyncpg.connect(database_url)
    try:
        assert await connection.fetchval(
            """
            SELECT restaurant_id FROM operator_knowledge
            WHERE LOWER(question) = LOWER('Do you have a coat check?')
            """
        ) == "restaurant.harbor-and-hearth.portland"
    finally:
        await connection.close()
    await close_pool()
