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
            "ai_agent_name": "  Avery   Rose ",
            "opening_hours": {"mon": {"open": "11:30", "close": "22:00"}},
        }
    )
    assert result == {"ai_agent_name": "Avery Rose"}


@pytest.mark.parametrize(
    "payload",
    [
        {"ai_agent_name": ""},
        {"ai_agent_name": "x" * 61},
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
                "restaurant_id": DEFAULT_SETTINGS["restaurant_id"],
                "data_version": DEFAULT_SETTINGS["data_version"],
                "restaurant_name": "The Lamplighter",
                "street_address": "99 Legacy Avenue",
                "opening_hours": {"mon": {"open": "09:00", "close": "23:00"}},
                "ai_agent_name": "Avery Rose",
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
    assert loaded["ai_agent_name"] == "Avery Rose"
