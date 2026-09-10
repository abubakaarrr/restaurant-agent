"""Canonical staff-transfer destination availability."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.restaurant_knowledge import KnowledgeFixtureError, get_restaurant_knowledge


def _now(timezone_info: ZoneInfo) -> datetime:
    return datetime.now(timezone_info)


def current_staff_transfer_number(at: datetime | None = None) -> str:
    from app.config import settings

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


def resolve_handoff_destination(reason: str, at: datetime | None = None) -> dict[str, str | bool]:
    owner = {
        "manager_requested": "manager_callback",
        "manager_or_complaint": "manager_callback",
        "payment_or_refund": "payment_support",
        "severe_allergy": "kitchen",
        "safety": "manager_callback",
    }.get(reason, "staff")
    try:
        route = get_restaurant_knowledge().escalation_route(owner)
    except (KnowledgeFixtureError, KeyError, TypeError, ValueError):
        route = None
    if not route:
        return {
            "owner": owner,
            "channel": "unavailable",
            "transfer_number": "",
            "can_transfer": False,
        }
    channel = str(route.get("channel") or "")
    transfer_number = (
        current_staff_transfer_number(at)
        if channel in {"voice_transfer", "conditional"}
        else ""
    )
    return {
        "owner": owner,
        "channel": "voice_transfer" if transfer_number else channel,
        "transfer_number": transfer_number,
        "can_transfer": bool(transfer_number),
    }
