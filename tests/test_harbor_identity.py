from __future__ import annotations

from app.restaurant_knowledge import get_restaurant_knowledge
from app.restaurant_settings import DEFAULT_SETTINGS, validate_restaurant_settings_update
from app.services.restaurant import format_menu_price
from db.seed import MENU_ITEMS


def test_default_identity_is_canonical_harbor_and_hearth() -> None:
    knowledge = get_restaurant_knowledge()
    identity = knowledge.identity
    assert DEFAULT_SETTINGS["restaurant_name"] == identity["name"]
    assert DEFAULT_SETTINGS["timezone"] == identity["timezone"]
    assert DEFAULT_SETTINGS["street_address"] == identity["address"]["street"]
    assert DEFAULT_SETTINGS["phone_number"] == identity["phone_e164"]
    assert DEFAULT_SETTINGS["opening_hours"]["fri"] == {
        "open": "11:30",
        "close": "23:00",
    }
    assert DEFAULT_SETTINGS["hours_unconfirmed"] is False


def test_fixture_owned_identity_and_hours_are_not_operator_editable() -> None:
    result = validate_restaurant_settings_update(
        {
            "restaurant_name": "Pilot Bistro",
            "timezone": "America/New_York",
            "opening_hours": {},
        }
    )
    assert result == {}


def test_seed_menu_is_the_canonical_confirmed_fixture() -> None:
    assert len(MENU_ITEMS) >= 25
    assert all(item["source_id"].startswith("source.harbor-and-hearth") for item in MENU_ITEMS)
    assert all(item["price"] > 0 and item["currency"] == "USD" for item in MENU_ITEMS)
    assert all(item["data_version"] == "2026.09.07-phase1" for item in MENU_ITEMS)


def test_format_menu_price_preserves_legacy_estimate_guard() -> None:
    assert "estimated" in format_menu_price({"price": 9, "price_estimated": True})
    assert format_menu_price({"price": 18, "price_estimated": False}) == "$18.00"
