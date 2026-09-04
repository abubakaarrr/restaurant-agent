from app.call_flags import consume_call_control
from app.config import settings
from app.services.restaurant import _combine_notes
from app.tools.db import request_handoff


def test_combine_notes_appends_without_duplicating() -> None:
    assert _combine_notes("", "window table") == "window table"
    assert _combine_notes("window table", "window table") == "window table"
    assert _combine_notes("window table", "high chair") == "window table; high chair"


async def test_handoff_without_staff_number_does_not_transfer(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "staff_transfer_number", "")
    result = await request_handoff.ainvoke(
        {"session_id": "call-notes", "reason": "human_requested"}
    )
    assert "not available" in result.lower()
    assert "connecting" in result.lower()
    assert consume_call_control("call-notes") is None


async def test_handoff_refuses_name_and_water_topics(monkeypatch) -> None:
    monkeypatch.setattr(settings, "staff_transfer_number", "+14155550100")
    result = await request_handoff.ainvoke(
        {
            "session_id": "call-name",
            "reason": "human_requested",
            "topic": "update reservation name to Abubakar",
        }
    )
    assert "do not transfer" in result.lower()
    assert consume_call_control("call-name") is None
    water = await request_handoff.ainvoke(
        {
            "session_id": "call-water",
            "reason": "human_requested",
            "topic": "serve chilled water on arrival",
        }
    )
    assert "water" in water.lower()
    assert consume_call_control("call-water") is None
