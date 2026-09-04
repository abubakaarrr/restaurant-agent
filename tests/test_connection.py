"""Opt-in PostgreSQL/pgvector integration smoke test."""

import os

import asyncpg
import pytest
from app.config import settings


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 to test a configured PostgreSQL instance",
)
async def test_database_and_pgvector_connection() -> None:
    conn = await asyncpg.connect(settings.database_url)
    try:
        version = await conn.fetchval("SELECT version()")
        assert "PostgreSQL" in version
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )
        assert "bookings" in {row["table_name"] for row in tables}
        distance = await conn.fetchval(
            "SELECT '[1,2,3]'::vector <=> '[1,2,3]'::vector"
        )
        assert float(distance) == pytest.approx(0.0)
    finally:
        await conn.close()
