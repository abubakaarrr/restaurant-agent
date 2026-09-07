"""Versioned runtime restaurant settings shared by UI, tools, and prompts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.security import is_e164


SETTINGS_FILE = Path(settings.restaurant_settings_file)

HOURS_UNCONFIRMED_NOTE = "Opening hours are unavailable from the canonical local fixture."

DEFAULT_SETTINGS: dict[str, Any] = {
    "restaurant_name": settings.restaurant_name,
    "tagline": "A neighborhood table with a Pacific Northwest hearth",
    "phone_number": "+15035550148",
    "timezone": settings.restaurant_timezone,
    "seating_capacity": 130,
    "street_address": "1842 Market Street",
    "city": "Portland, OR 97205",
    "ai_agent_name": settings.ai_agent_name,
    "languages": ["English"],
    "opening_hours": {
        "tue": {"open": "11:30", "close": "22:00"},
        "wed": {"open": "11:30", "close": "22:00"},
        "thu": {"open": "11:30", "close": "22:00"},
        "fri": {"open": "11:30", "close": "23:00"},
        "sat": {"open": "09:00", "close": "23:00"},
        "sun": {"open": "09:00", "close": "21:00"},
    },
    "hours_unconfirmed": False,
    "hours_note": "Date-specific exceptions in the canonical knowledge fixture override regular hours.",
}

_settings_cache: tuple[float | None, dict[str, Any]] | None = None


def load_restaurant_settings() -> dict[str, Any]:
    global _settings_cache
    if not SETTINGS_FILE.exists():
        return dict(DEFAULT_SETTINGS)
    mtime = SETTINGS_FILE.stat().st_mtime
    if _settings_cache and _settings_cache[0] == mtime:
        return dict(_settings_cache[1])
    saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    merged = {**DEFAULT_SETTINGS, **saved}
    _settings_cache = (mtime, merged)
    return dict(merged)


def save_restaurant_settings(data: dict[str, Any]) -> dict[str, Any]:
    global _settings_cache
    allowed = {
        key: data[key]
        for key in DEFAULT_SETTINGS
        if key in data
    }
    merged = {**load_restaurant_settings(), **allowed}
    SETTINGS_FILE.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _settings_cache = (SETTINGS_FILE.stat().st_mtime, merged)
    return merged


def validate_restaurant_settings_update(data: dict[str, Any]) -> dict[str, Any]:
    """Validate operator-controlled values before they reach prompts/tools."""
    cleaned: dict[str, Any] = {}
    short_text_fields = {
        "restaurant_name": 120,
        "tagline": 200,
        "ai_agent_name": 60,
        "street_address": 200,
        "city": 100,
    }
    for key, maximum in short_text_fields.items():
        if key not in data:
            continue
        value = " ".join(str(data[key]).split())
        if not value and key in {"restaurant_name", "ai_agent_name"}:
            raise ValueError(f"{key} cannot be empty")
        if len(value) > maximum:
            raise ValueError(f"{key} is too long")
        cleaned[key] = value

    if "phone_number" in data:
        phone = str(data["phone_number"]).strip()
        if phone and not is_e164(phone):
            raise ValueError("phone_number must use E.164 format")
        cleaned["phone_number"] = phone

    if "timezone" in data:
        timezone = str(data["timezone"]).strip()
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        cleaned["timezone"] = timezone

    if "seating_capacity" in data:
        capacity = int(data["seating_capacity"])
        if not 1 <= capacity <= 5000:
            raise ValueError("seating_capacity must be between 1 and 5000")
        cleaned["seating_capacity"] = capacity

    if "languages" in data:
        languages = data["languages"]
        if not isinstance(languages, list) or not 1 <= len(languages) <= 5:
            raise ValueError("languages must contain 1-5 entries")
        normalized = []
        for language in languages:
            value = " ".join(str(language).split())
            if not value or len(value) > 40:
                raise ValueError("each language must contain 1-40 characters")
            normalized.append(value)
        cleaned["languages"] = list(dict.fromkeys(normalized))

    if "hours_unconfirmed" in data:
        cleaned["hours_unconfirmed"] = bool(data["hours_unconfirmed"])

    if "hours_note" in data:
        note = " ".join(str(data["hours_note"]).split())
        if len(note) > 500:
            raise ValueError("hours_note is too long")
        cleaned["hours_note"] = note

    if "opening_hours" in data:
        hours = data["opening_hours"]
        if not isinstance(hours, dict):
            raise ValueError("opening_hours must be an object")
        normalized_hours: dict[str, dict[str, str]] = {}
        valid_days = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
        time_pattern = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
        for day, value in hours.items():
            if day not in valid_days or not isinstance(value, dict):
                raise ValueError("opening_hours contains an invalid day")
            opening = str(value.get("open", "")).strip()
            closing = str(value.get("close", "")).strip()
            if not time_pattern.fullmatch(opening) or not time_pattern.fullmatch(closing):
                raise ValueError("opening hours must use 24-hour HH:MM")
            normalized_hours[day] = {"open": opening, "close": closing}
        cleaned["opening_hours"] = normalized_hours

    return cleaned
