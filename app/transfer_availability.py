"""Canonical staff-transfer destination availability."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import settings
from app.restaurant_knowledge import KnowledgeFixtureError, get_restaurant_knowledge


def _now(timezone_info: ZoneInfo) -> datetime:
    return datetime.now(timezone_info)


def current_staff_transfer_number(at: datetime | None = None) -> str:
    number = str(settings.staff_transfer_number or "").strip()
    if not number:
        return ""
    try:
        knowledge = get_restaurant_knowledge()
        timezone_info = ZoneInfo(knowledge.identity["timezone"])
        local = at.astimezone(timezone_info) if at and at.tzinfo else at
        local = local.replace(tzinfo=timezone_info) if local else _now(timezone_info)
        schedule = knowledge.raw["hours"]["fulfillment"]["staff_transfer"]
        if local.strftime("%a").casefold() not in schedule["days"]:
            return ""
        current_time = local.strftime("%H:%M")
        if not schedule["open"] <= current_time < schedule["close"]:
            return ""
        operating_status = knowledge.operating_status(local)
    except (KnowledgeFixtureError, KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
        return ""
    if (
        operating_status["kind"] != "regular_hours"
        and not operating_status["available"]
    ):
        return ""
    return number
