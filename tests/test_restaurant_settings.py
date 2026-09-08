import json

import pytest

import app.restaurant_settings as restaurant_settings
from app.restaurant_settings import (
    DEFAULT_SETTINGS,
    load_restaurant_settings,
    validate_restaurant_settings_update,
)


def test_valid_operator_settings_are_normalized() -> None:
    result = validate_restaurant_settings_update(
        {
            "restaurant_name": "  Pilot   Bistro ",
            "phone_number": "+14155550123",
            "timezone": "America/New_York",
            "languages": ["English", "Spanish", "English"],
            "opening_hours": {
                "mon": {"open": "11:30", "close": "22:00"}
            },
        }
    )
    assert result["restaurant_name"] == "Pilot Bistro"
    assert result["languages"] == ["English", "Spanish"]
    empty_hours = validate_restaurant_settings_update({"opening_hours": {}})
    assert empty_hours["opening_hours"] == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"phone_number": "415-555-0123"},
        {"timezone": "not/a-timezone"},
        {"languages": []},
        {"languages": ["x" * 41]},
        {"opening_hours": {"monday": {"open": "9", "close": "5"}}},
        {"seating_capacity": 0},
    ],
)
def test_invalid_operator_settings_are_rejected(payload: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_restaurant_settings_update(payload)


def test_stale_restaurant_settings_cannot_override_canonical_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    settings_file = tmp_path / "restaurant-settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "restaurant_name": "The Lamplighter",
                "street_address": "99 Legacy Avenue",
                "opening_hours": {"mon": {"open": "09:00", "close": "23:00"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(restaurant_settings, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(restaurant_settings, "_settings_cache", None)
    loaded = load_restaurant_settings()
    assert loaded["restaurant_id"] == "restaurant.harbor-and-hearth.portland"
    assert loaded["data_version"] == "2026.09.07-phase1"
    assert loaded["restaurant_name"] == DEFAULT_SETTINGS["restaurant_name"]
    assert "mon" not in loaded["opening_hours"]
