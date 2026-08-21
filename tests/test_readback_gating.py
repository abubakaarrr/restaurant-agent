"""Full readback only on terminal pending-confirmation registration."""

from __future__ import annotations

from datetime import datetime, timedelta
import os
import re
import uuid

import pytest

from app.call_memory import clear_call_memory, set_current_session_id


pytestmark_db = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1" or not os.getenv("TEST_DATABASE_URL"),
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)

_FULL_READBACK = re.compile(
    r"readback_required|Read every (?:item|field)|ask if all details are correct",
    re.IGNORECASE,
)


@pytestmark_db
@pytest.mark.asyncio
async def test_three_edits_short_acks_then_one_terminal_readback(monkeypatch) -> None:
    """Three single-field edits → short acks; 'that's everything' → one full readback."""
    from app.config import settings
    from app.db_pool import close_pool
    from app.services.restaurant import restaurant_service
    from app.tools.db import get_order_summary, update_order_item, update_reservation_draft

    database_url = os.environ["TEST_DATABASE_URL"]
    await close_pool()
    monkeypatch.setattr(settings, "database_url", database_url)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)

    suffix = uuid.uuid4().hex[:10]
    call_id = f"readback-{suffix}"
    clear_call_memory(call_id)
    token = set_current_session_id(call_id)

    connection = __import__("asyncpg")
    conn = await connection.connect(database_url)
    try:
        await conn.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary, available)
            VALUES ('Readback Pasta', 'main', 16.00, 'Test', ARRAY['vegetarian'], TRUE)
            ON CONFLICT DO NOTHING
            """
        )
    finally:
        await conn.close()

    try:
        # Three consecutive single-field reservation edits — short acks only.
        replies = []
        for party in (3, 4, 5):
            text = await update_reservation_draft.ainvoke(
                {
                    "session_id": call_id,
                    "party_size": party,
                    "name": "Sam",
                    "phone": "+14155550961",
                    "date": (datetime.now() + timedelta(days=14)).date().isoformat(),
                    "time": "19:00",
                }
                if party == 3
                else {"session_id": call_id, "party_size": party}
            )
            replies.append(text)
            assert "Acknowledge only what changed" in text
            assert not _FULL_READBACK.search(text)

        added = await restaurant_service.add_order_item(
            call_id=call_id,
            idempotency_key=f"{call_id}-add",
            item_name="Readback Pasta",
            quantity=1,
            customer_name="Sam",
            customer_phone="+14155550961",
        )
        assert added["added"] is True
        item_id = added["order_item_id"]

        # Quantity edits stay short.
        for qty in (2, 3, 4):
            text = await update_order_item.ainvoke(
                {
                    "session_id": call_id,
                    "order_item_id": item_id,
                    "quantity": qty,
                }
            )
            replies.append(text)
            assert "Acknowledge only what changed" in text
            assert not _FULL_READBACK.search(text)

        # Terminal "that's everything" path: get_order_summary registers pending.
        summary_text = await get_order_summary.ainvoke({"session_id": call_id})
        assert _FULL_READBACK.search(summary_text)
        assert summary_text.lower().count("readback_required=true") == 1
        assert sum(1 for r in replies if _FULL_READBACK.search(r)) == 0
    finally:
        from app.call_memory import reset_current_session_id

        reset_current_session_id(token)
        clear_call_memory(call_id)
        await close_pool()
