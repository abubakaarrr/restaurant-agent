from __future__ import annotations

from copy import deepcopy
from datetime import date

import pytest

from app.behavior import BehaviorControl, BehaviorMode, TurnObservation, reduce_behavior
from app.agent.graph import tool_limit_response
from app.config import settings
from app.pending_confirmation import order_confirmation_payload, payload_hash
from app.restaurant_knowledge import (
    KnowledgeFixtureError,
    RestaurantKnowledge,
    get_restaurant_knowledge,
)
from app.services.restaurant import RestaurantServiceError, restaurant_service
from app.tools.db import _format_order
from app.tools.rag import search_menu


def test_fixture_is_complete_versioned_synthetic_and_queryable() -> None:
    knowledge = get_restaurant_knowledge()
    meta = knowledge.metadata
    assert meta == {
        "schema_version": "restaurant-knowledge.v1",
        "data_version": "2026.09.07-phase1",
        "fixture_id": "fixture.harbor-and-hearth.phase1",
        "source_id": "source.harbor-and-hearth.synthetic.2026-09-07",
        "effective_from": "2026-09-01",
        "effective_to": None,
        "synthetic": True,
    }
    assert knowledge.identity["name"] == "Harbor & Hearth Kitchen"
    assert knowledge.identity["phone_e164"] == "+15035550148"
    assert knowledge.identity["email"].endswith(".example")

    required_categories = {
        "starters",
        "salads",
        "soups",
        "burgers_sandwiches",
        "mains",
        "vegetarian_vegan",
        "gluten_aware",
        "kids",
        "desserts",
        "non_alcoholic",
        "coffee_tea",
        "beer_wine_cocktails",
        "seasonal",
    }
    assert required_categories <= {
        item["category_id"].removeprefix("category.") for item in knowledge.menu_items
    }
    assert len(knowledge.menu_items) == 29
    for item in knowledge.menu_items:
        assert item["item_id"].startswith("menu.")
        assert item["source_id"] == meta["source_id"]
        assert item["data_version"] == meta["data_version"]
        assert item["effective_from"]
        for field in (
            "ingredients",
            "allergens",
            "dietary_tags",
            "service_periods",
            "modifier_options",
            "removable_ingredients",
            "substitutions",
            "incompatible_choices",
        ):
            assert isinstance(item[field], list), (item["item_id"], field)
        assert item["cross_contact"]
        assert item["availability"] in {
            "available",
            "sold_out",
            "not_yet_available",
        }


def test_menu_covers_allergen_and_dietary_safety_taxonomy() -> None:
    knowledge = get_restaurant_knowledge()
    allergens = {value for item in knowledge.menu_items for value in item["allergens"]}
    assert {
        "dairy",
        "egg",
        "wheat",
        "gluten",
        "soy",
        "sesame",
        "peanut",
        "tree_nut",
        "fish",
        "shellfish",
    } <= allergens
    dietary = {value for item in knowledge.menu_items for value in item["dietary_tags"]}
    assert {"vegetarian", "vegan", "gluten_aware", "kids", "alcohol"} <= dietary


def test_fixture_validation_rejects_dangling_canonical_references() -> None:
    base = get_restaurant_knowledge()
    raw = deepcopy(base.raw)
    raw["menu_items"][0]["modifier_options"] = ["modifier.does-not-exist"]
    with pytest.raises(KnowledgeFixtureError, match="Unknown modifier references"):
        RestaurantKnowledge(raw, path=base.path)


def test_hours_and_required_policy_topics_are_structured() -> None:
    knowledge = get_restaurant_knowledge()
    hours = knowledge.raw["hours"]
    assert {row["day"] for row in hours["regular"]} == {
        "mon", "tue", "wed", "thu", "fri", "sat", "sun"
    }
    assert hours["fulfillment"]["pickup"]["open"] == "11:45"
    assert hours["fulfillment"]["delivery"]["open"] == "12:00"
    assert set(hours["fulfillment"]["delivery"]["eligible_postal_codes"]) == {
        "97201", "97205", "97209", "97210"
    }
    assert hours["fulfillment"]["patio"]["weather_dependent"] is True
    assert hours["fulfillment"]["staff_transfer"]["destination_state"] == "configuration_required"
    assert {row["kind"] for row in hours["exceptions"]} >= {
        "holiday_closure", "temporary_private_event_closure", "holiday_hours"
    }
    periods = {row["id"]: row for row in hours["service_period_catalog"]}
    assert {"service.breakfast", "service.brunch", "service.lunch", "service.dinner"} <= periods.keys()
    assert periods["service.breakfast"]["status"] == "not_offered"

    required_topics = {
        "topic.address", "topic.hours", "topic.parking", "topic.transit",
        "topic.reservations", "topic.seating", "topic.cancellation",
        "topic.late-arrival", "topic.dress-code", "topic.accessibility",
        "topic.wifi", "topic.children", "topic.pets-service-animals",
        "topic.pickup-delivery", "topic.payment", "topic.gratuity-split-checks",
        "topic.refunds", "topic.packaging", "topic.alcohol",
        "topic.events-private-dining", "topic.catering", "topic.gift-cards",
        "topic.lost-found", "topic.weather", "topic.safety-emergency",
    }
    by_id = {topic["topic_id"]: topic for topic in knowledge.topics}
    assert required_topics <= by_id.keys()
    for topic in by_id.values():
        assert topic["category_id"].startswith("category.")
        assert topic["source_id"] == knowledge.metadata["source_id"]
        assert topic["version"] and topic["effective_from"]
        assert topic["answer"] and isinstance(topic["rule"], dict)
        assert topic["escalation_owner"]


@pytest.mark.parametrize(
    ("query", "topic_id"),
    [
        ("dress", "topic.dress-code"),
        ("what is your address", "topic.address"),
        ("is parking available", "topic.parking"),
        ("is a table available", "topic.seating"),
        ("do you have wi fi", "topic.wifi"),
        ("can I get takeaway", "topic.pickup-delivery"),
        ("are you closed on Labor Day", "topic.hours"),
        ("can I bring my dog", "topic.pets-service-animals"),
        ("how late can I arrive", "topic.late-arrival"),
        ("do you accept credit cards", "topic.payment"),
        ("do you deliver", "topic.pickup-delivery"),
        ("do you cater office lunches", "topic.catering"),
    ],
)
def test_topic_matching_uses_word_boundaries(query: str, topic_id: str) -> None:
    match = get_restaurant_knowledge().find_topic(query)
    assert match.status == "known"
    assert [record["topic_id"] for record in match.records] == [topic_id]


def test_unknown_topic_never_borrows_unrelated_fact() -> None:
    match = get_restaurant_knowledge().find_topic("rooftop telescope policy")
    assert match.status == "unknown"
    assert match.records == ()


def test_effective_dates_make_future_and_expired_records_explicit() -> None:
    base = get_restaurant_knowledge()
    raw = deepcopy(base.raw)
    raw["topics"][0]["effective_from"] = "2027-01-01"
    future = RestaurantKnowledge(raw, path=base.path).find_topic(
        raw["topics"][0]["aliases"][0], on_date=date(2026, 9, 7)
    )
    assert future.status == "future"

    raw = deepcopy(base.raw)
    raw["topics"][0]["effective_from"] = "2026-01-01"
    raw["topics"][0]["effective_to"] = "2026-08-31"
    expired = RestaurantKnowledge(raw, path=base.path).find_topic(
        raw["topics"][0]["aliases"][0], on_date=date(2026, 9, 7)
    )
    assert expired.status == "expired"


def test_menu_lookup_exact_alias_spelling_ambiguity_and_unavailable_alternatives() -> None:
    knowledge = get_restaurant_knowledge()
    assert knowledge.find_menu_item("market greens").item["item_id"] == "menu.salad.market-greens"
    assert knowledge.find_menu_item("house salad").item["item_id"] == "menu.salad.market-greens"
    typo = knowledge.find_menu_item("salomn")
    assert typo.status == "ambiguous"
    assert typo.item is None
    assert typo.candidates[0]["item_id"] == "menu.main.cedar-salmon"
    ambiguous = knowledge.find_menu_item("chicken")
    assert ambiguous.status == "ambiguous"
    assert len(ambiguous.candidates) >= 2
    unavailable = knowledge.find_menu_item("salmon dip").item
    assert unavailable["availability"] == "sold_out"
    assert unavailable["alternative_item_ids"] == [
        "menu.main.cedar-salmon", "menu.starter.hearth-bread"
    ]


def test_modifier_semantics_cover_free_paid_removal_unavailable_and_clarification() -> None:
    knowledge = get_restaurant_knowledge()
    burger = knowledge.find_menu_item("Hearth Burger").item
    valid = knowledge.resolve_customization(
        burger,
        modifier_ids=["modifier.extra-cheddar", "modifier.side-fries"],
        removals=["onion jam"],
    )
    assert valid["status"] == "valid"
    assert valid["price_delta"] == 2.0
    assert valid["removals"] == ["onion jam"]

    unavailable = knowledge.resolve_customization(
        burger,
        modifier_ids=["modifier.avocado", "modifier.side-fries"],
    )
    assert unavailable["status"] == "unavailable"

    conflict = knowledge.resolve_customization(
        burger,
        modifier_ids=[
            "modifier.extra-cheddar", "modifier.remove-cheese", "modifier.side-fries"
        ],
    )
    assert conflict["status"] == "clarification_required"

    missing_side = knowledge.resolve_customization(burger)
    assert missing_side["status"] == "clarification_required"

    steak = knowledge.find_menu_item("Steak Frites").item
    temperature = knowledge.resolve_customization(
        steak, modifier_ids=["modifier.steak-temperature"]
    )
    assert temperature["status"] == "clarification_required"
    valid_temperature = knowledge.resolve_customization(
        steak, modifier_ids=["modifier.steak-temperature:medium-rare"]
    )
    assert valid_temperature["status"] == "valid"

    for duplicate in (
        {"modifier_ids": ["modifier.extra-cheddar", "modifier.extra-cheddar", "modifier.side-fries"]},
        {"removals": ["onion jam", "ONION   JAM"], "modifier_ids": ["modifier.side-fries"]},
        {"substitutions": ["modifier.side-salad", "modifier.side-salad"]},
    ):
        rejected = knowledge.resolve_customization(burger, **duplicate)
        assert rejected["status"] == "clarification_required"


def test_order_notes_and_customizations_are_confirmation_integrity_data() -> None:
    summary = {
        "order_id": 17,
        "draft_version": 4,
        "booking_id": 0,
        "fulfillment": "delivery",
        "fulfillment_details": {
            "address": "101 Test Avenue",
            "instructions": "Leave with the front desk",
        },
        "order_notes": "No utensils",
        "allergy_notes": "Severe sesame allergy",
        "fees": [{"fee_id": "fee.delivery", "name": "delivery fee", "amount": 5}],
        "total": 29,
        "items": [
            {
                "item_name": "Hearth Burger",
                "quantity": 1,
                "unit_price": 23,
                "subtotal": 23,
                "notes": "Cut in half",
                "modifiers": [{"option_id": "modifier.extra-cheddar", "name": "extra cheddar", "price_delta": 2}],
                "removals": ["onion jam"],
                "substitutions": [],
            }
        ],
    }
    payload = order_confirmation_payload(summary)
    changed = order_confirmation_payload({**summary, "allergy_notes": "Severe dairy allergy"})
    assert payload["order_notes"] == "No utensils"
    assert payload["allergy_notes"] == "Severe sesame allergy"
    assert payload["items"][0]["modifiers"][0]["price_delta"] == 2
    assert payload_hash(payload) != payload_hash(changed)

    readback = _format_order(summary)
    for expected in (
        "extra cheddar", "remove: onion jam", "Cut in half", "No utensils",
        "Severe sesame allergy", "Delivery address", "front desk", "delivery fee", "$29.00",
    ):
        assert expected.casefold() in readback.casefold()


@pytest.mark.parametrize(
    ("caller_input", "expected_mode", "expected_phrases"),
    [
        (
            "This is the third time this failed.",
            BehaviorMode.DEESCALATING,
            ("acknowledge the concern", "concrete next step"),
        ),
        (
            "Are the fries famous?",
            BehaviorMode.STANDARD,
            ("loyal following", "check whether they're available"),
        ),
        (
            "Are you a real person?",
            BehaviorMode.STANDARD,
            ("virtual host", "restaurant questions"),
        ),
    ],
)
def test_conversation_inputs_execute_behavior_interface(
    caller_input: str,
    expected_mode: BehaviorMode,
    expected_phrases: tuple[str, ...],
) -> None:
    result = reduce_behavior(None, TurnObservation(text=caller_input))
    observable = " ".join(
        value
        for value in (result.directive.direct_reply, result.directive.prompt_instruction)
        if value
    ).casefold()
    assert result.directive.mode is expected_mode
    assert all(phrase in observable for phrase in expected_phrases)


def test_safe_humor_is_suppressed_for_allergy_context() -> None:
    result = reduce_behavior(
        None,
        TurnObservation(text="Are the fries famous and safe for my allergy?"),
    )
    assert result.directive.direct_reply is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_phrases"),
    [
        ("What's in the Market Greens?", ("hazelnut", "tree_nut", "cross-contact")),
        ("Can you guarantee no hazelnut contact?", ("zero cross-contact cannot be guaranteed",)),
    ],
)
async def test_conversation_inputs_execute_grounded_menu_tool(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    expected_phrases: tuple[str, ...],
) -> None:
    item = get_restaurant_knowledge().find_menu_item("Market Greens").item

    async def canonical_menu(*, available_only: bool = True) -> dict:
        return {
            "items": [
                {
                    "name": item["name"],
                    "category": item["category_id"],
                    "description": item["description"],
                    "dietary": item["dietary_tags"],
                    "aliases": item["aliases"],
                    "ingredients": item["ingredients"],
                    "allergens": item["allergens"],
                    "price": item["price"],
                    "price_estimated": False,
                    "available": True,
                    "availability": item["availability"],
                    "cross_contact": item["cross_contact"],
                }
            ],
            "allergen_notice": item["cross_contact"],
        }

    monkeypatch.setattr(restaurant_service, "list_menu", canonical_menu)
    result = (await search_menu.ainvoke({"query": query})).casefold()
    assert all(phrase.casefold() in result for phrase in expected_phrases)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Are you open October 18?", "closes at 4:00 PM"),
        ("Are you open on Thanksgiving?", "closed on Thanksgiving Day"),
        ("Are you open 2026-12-24?", "open 11:30 AM to 8:00 PM"),
        ("Are you open September 8?", "open 11:30 to 22:00"),
    ],
)
async def test_dated_hours_queries_resolve_canonical_exceptions(
    query: str,
    expected: str,
) -> None:
    result = await restaurant_service.restaurant_info(query)
    assert result["topic_id"] == "topic.hours"
    assert expected in result["formatted"]


@pytest.mark.asyncio
async def test_blank_restaurant_topic_is_explicitly_missing() -> None:
    result = await restaurant_service.restaurant_info("  ")
    assert result["matched"] is False
    assert result["status"] == "missing"
    assert result["answers"] == []
    assert "restaurant_name" not in result


@pytest.mark.asyncio
async def test_menu_read_boundary_excludes_stale_restaurant_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge = get_restaurant_knowledge()
    item = knowledge.find_menu_item("Market Greens").item

    def row_for(*, name: str, canonical_id: str, source_id: str) -> dict:
        return {
            "id": 1,
            "canonical_id": canonical_id,
            "name": name,
            "aliases": [],
            "category": "salads",
            "price": 13,
            "description": "Current description",
            "dietary": [],
            "ingredients": [],
            "allergens": [],
            "service_periods": [],
            "availability_status": "available",
            "knowledge_metadata": {},
            "source_id": source_id,
            "data_version": knowledge.metadata["data_version"],
            "effective_from": None,
            "effective_to": None,
            "price_estimated": False,
            "available": True,
        }

    rows = [
        row_for(
            name=item["name"],
            canonical_id=item["item_id"],
            source_id=knowledge.metadata["source_id"],
        ),
        row_for(
            name="Legacy Example Chowder",
            canonical_id="menu.legacy.chowder",
            source_id="source.legacy-example",
        ),
    ]

    class Connection:
        async def fetch(self, query: str, *args: object) -> list[dict]:
            return rows

    class Acquire:
        async def __aenter__(self) -> Connection:
            return Connection()

        async def __aexit__(self, *args: object) -> None:
            return None

    class Pool:
        def acquire(self) -> Acquire:
            return Acquire()

    async def fake_pool() -> Pool:
        return Pool()

    monkeypatch.setattr("app.services.restaurant.get_pool", fake_pool)
    menu = await restaurant_service.list_menu(available_only=False)
    assert [entry["name"] for entry in menu["items"]] == ["Market Greens"]


def test_human_handoff_copy_depends_on_configured_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "staff_transfer_number", "")
    unavailable = reduce_behavior(None, TurnObservation(text="Connect me to a person"))
    assert unavailable.directive.control is BehaviorControl.HANDOFF
    assert "can't transfer" in unavailable.directive.direct_reply.casefold()
    assert "callback" in unavailable.directive.direct_reply.casefold()
    assert "connect you" not in unavailable.directive.direct_reply.casefold()

    monkeypatch.setattr(settings, "staff_transfer_number", "+15035550149")
    configured = reduce_behavior(None, TurnObservation(text="Connect me to a person"))
    assert configured.directive.control is BehaviorControl.HANDOFF
    assert "connect you" in configured.directive.direct_reply.casefold()


@pytest.mark.asyncio
async def test_tool_limit_handoff_copy_depends_on_configured_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "staff_transfer_number", "")
    unavailable = await tool_limit_response({})
    unavailable_text = unavailable["messages"][0].content.casefold()
    assert "can't transfer" in unavailable_text
    assert "callback message" in unavailable_text
    assert "connect you" not in unavailable_text

    monkeypatch.setattr(settings, "staff_transfer_number", "+15035550149")
    configured = await tool_limit_response({})
    assert "connect you" in configured["messages"][0].content.casefold()


@pytest.mark.asyncio
async def test_restaurant_info_unknown_and_provider_unavailable_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable_pool():
        raise RuntimeError("local test database unavailable")

    monkeypatch.setattr("app.services.restaurant.get_pool", unavailable_pool)
    unknown = await restaurant_service.restaurant_info("rooftop telescope policy")
    assert unknown["status"] == "unknown"
    assert unknown["operator_source_status"] == "unavailable"
    assert unknown["log_unknown"] is True
    assert "not in the current" in unknown["formatted"]

    with pytest.raises(RestaurantServiceError) as exc:
        await restaurant_service.list_menu()
    assert exc.value.code == "knowledge_unavailable"
    assert exc.value.status == 503
