"""Shared asyncpg connection pool.

Opening a fresh TCP connection + PostgreSQL handshake on every tool call adds
50-200ms of latency. A process-wide pool keeps warm connections ready so each
query reuses an existing connection instead of reconnecting.
"""

from __future__ import annotations

import asyncpg

from app.config import settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide connection pool, creating it on first use."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
    return _pool


async def close_pool() -> None:
    """Close the pool on application shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
