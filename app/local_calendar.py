"""Calendar facts derived from the restaurant clock. One home for 'this Friday'."""

from __future__ import annotations

from datetime import datetime, timedelta


def upcoming_named_dates(now: datetime) -> dict[str, str]:
    """Map this_monday … this_sunday, today, and tomorrow to YYYY-MM-DD.

    'this Friday' is the coming Friday, or today when today is Friday.
    """
    today = now.date() if hasattr(now, "date") else now
    names: dict[str, str] = {
        "today": today.isoformat(),
        "tomorrow": (today + timedelta(days=1)).isoformat(),
    }
    for offset in range(0, 7):
        day = today + timedelta(days=offset)
        weekday = day.strftime("%A").casefold()
        names[f"this_{weekday}"] = day.isoformat()
    return names
