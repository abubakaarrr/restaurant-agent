import pytest

from app.restaurant_settings import validate_restaurant_settings_update


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
