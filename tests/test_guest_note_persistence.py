"""Guest free-text notes must survive structured draft re-flattens."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import uuid

import pytest

from app.call_memory import (
    clear_call_memory,
    format_memory_for_prompt,
    update_reservation_draft,
)


pytestmark_db = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1" or not os.getenv("TEST_DATABASE_URL"),
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


@pytestmark_db
@pytest.mark.asyncio
async def test_add_guest_note_survives_reservation_draft_reflatten(monkeypatch) -> None:
    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import restaurant_service
    from app.tools.db import add_guest_note

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)

    suffix = uuid.uuid4().hex[:10]
    call_id = f"guest-note-{suffix}"
    clear_call_memory(call_id)

    update_reservation_draft(
        call_id,
        customer_name="Sam",
        customer_phone="+14155550951",
        date=(datetime.now() + timedelta(days=12)).date().isoformat(),
        time="19:00",
        party_size=4,
        seating_preference="window",
    )

    result = await add_guest_note.ainvoke(
        {
            "session_id": call_id,
            "note": "don't add anything with an extra charge without asking me first",
        }
    )
    assert "extra charge" in result.lower() or "note" in result.lower()

    # Unrelated field patch re-flattens structured notes into top-level notes.
    update_reservation_draft(call_id, dietary="vegetarian")
    prompt = format_memory_for_prompt(call_id)
    assert "don't add anything with an extra charge" in prompt
    assert "vegetarian" in prompt
    assert "window" in prompt
    clear_call_memory(call_id)
    await close_pool()
