from __future__ import annotations

from app.restaurant_settings import DEFAULT_SETTINGS, validate_restaurant_settings_update
from app.services.restaurant import format_menu_price
from db.seed import MENU_ITEMS, PRICE_CONFIRMED, PRICE_ESTIMATED


def test_default_identity_is_lamplighter_with_unconfirmed_hours() -> None:
    assert DEFAULT_SETTINGS["restaurant_name"] == "The Lamplighter Public House"
    assert DEFAULT_SETTINGS["ai_agent_name"] == "Clough"
    assert DEFAULT_SETTINGS["timezone"] == "America/Vancouver"
    assert DEFAULT_SETTINGS["street_address"] == "92 Water St"
    assert DEFAULT_SETTINGS["phone_number"] == "+16046874424"
    assert DEFAULT_SETTINGS["opening_hours"] == {}
    assert DEFAULT_SETTINGS["hours_unconfirmed"] is True
    assert "not confirmed" in DEFAULT_SETTINGS["hours_note"].casefold()


def test_empty_opening_hours_are_valid() -> None:
    result = validate_restaurant_settings_update({"opening_hours": {}})
    assert result["opening_hours"] == {}


def test_seed_menu_does_not_treat_estimates_as_confirmed() -> None:
    estimated = [row for row in MENU_ITEMS if row[5] is True]
    confirmed = [row for row in MENU_ITEMS if row[5] is False]
    assert estimated
    assert all(PRICE_ESTIMATED in row[3] for row in estimated)
    assert all(PRICE_CONFIRMED in row[3] for row in confirmed)
    confirmed_names = {row[0] for row in confirmed}
    assert confirmed_names == {
        "Caesar Salad",
        "Craft Lager Pitcher",
        "Wings (Wing Wednesday)",
    }
    assert "Tomato Basil Soup" not in {row[0] for row in MENU_ITEMS}


def test_format_menu_price_labels_estimates() -> None:
    assert "estimated" in format_menu_price({"price": 9, "price_estimated": True})
    assert format_menu_price({"price": 18, "price_estimated": False}) == "$18.00"


def test_caesar_salad_is_confirmed_at_eleven_dollars() -> None:
    caesar = next(item for item in MENU_ITEMS if item[0] == "Caesar Salad")
    assert caesar[2] == 11.00
    assert caesar[5] is False
