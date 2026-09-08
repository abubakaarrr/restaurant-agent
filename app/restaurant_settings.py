"""Versioned runtime restaurant settings shared by UI, tools, and prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.restaurant_knowledge import get_restaurant_knowledge


SETTINGS_FILE = Path(settings.restaurant_settings_file)

HOURS_UNCONFIRMED_NOTE = "Opening hours are unavailable from the canonical local fixture."


def _canonical_defaults() -> dict[str, Any]:
    knowledge = get_restaurant_knowledge()
    identity = knowledge.identity
    address = identity["address"]
    opening_hours = {
        row["day"]: {"open": row["open"], "close": row["close"]}
        for row in knowledge.raw["hours"]["regular"]
        if row.get("status") == "open"
    }
    return {
        "restaurant_id": identity["restaurant_id"],
        "data_version": knowledge.metadata["data_version"],
        "restaurant_name": identity["name"],
        "tagline": identity["tagline"],
        "phone_number": identity["phone_e164"],
        "timezone": identity["timezone"],
        "seating_capacity": sum(
            int(area.get("capacity") or 0) for area in knowledge.raw["dining_areas"]
        ),
        "street_address": address["street"],
        "city": f"{address['city']}, {address['region']} {address['postal_code']}",
        "ai_agent_name": settings.ai_agent_name,
        "languages": list(identity["languages"]),
        "opening_hours": opening_hours,
        "hours_unconfirmed": False,
        "hours_note": (
            "Date-specific exceptions in the canonical knowledge fixture override "
            "regular hours."
        ),
    }


DEFAULT_SETTINGS: dict[str, Any] = _canonical_defaults()
_EDITABLE_SETTINGS = {"ai_agent_name"}

_settings_cache: tuple[float | None, dict[str, Any]] | None = None


def load_restaurant_settings() -> dict[str, Any]:
    global _settings_cache
    canonical = _canonical_defaults()
    if not SETTINGS_FILE.exists():
        return canonical
    mtime = SETTINGS_FILE.stat().st_mtime
    if _settings_cache and _settings_cache[0] == mtime:
        return dict(_settings_cache[1])
    saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or any(
        saved.get(key) != canonical[key]
        for key in ("restaurant_id", "data_version")
    ):
        merged = canonical
        _settings_cache = (mtime, merged)
        return dict(merged)
    merged = {
        **canonical,
        **{key: saved[key] for key in _EDITABLE_SETTINGS if key in saved},
    }
    _settings_cache = (mtime, merged)
    return dict(merged)


def save_restaurant_settings(data: dict[str, Any]) -> dict[str, Any]:
    global _settings_cache
    canonical = _canonical_defaults()
    allowed = {key: data[key] for key in _EDITABLE_SETTINGS if key in data}
    merged = {**load_restaurant_settings(), **allowed}
    persisted = {
        "restaurant_id": canonical["restaurant_id"],
        "data_version": canonical["data_version"],
        **allowed,
    }
    SETTINGS_FILE.write_text(
        json.dumps(persisted, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _settings_cache = (SETTINGS_FILE.stat().st_mtime, merged)
    return merged


def validate_restaurant_settings_update(data: dict[str, Any]) -> dict[str, Any]:
    """Validate operator-controlled values before they reach prompts/tools."""
    cleaned: dict[str, Any] = {}
    if "ai_agent_name" in data:
        agent_name = " ".join(str(data["ai_agent_name"]).split())
        if not agent_name:
            raise ValueError("ai_agent_name cannot be empty")
        if len(agent_name) > 60:
            raise ValueError("ai_agent_name is too long")
        cleaned["ai_agent_name"] = agent_name

    return cleaned
