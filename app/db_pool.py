"""Shared asyncpg connection pool.

Opening a fresh TCP connection + PostgreSQL handshake on every tool call adds
50-200ms of latency. A process-wide pool keeps warm connections ready so each
query reuses an existing connection instead of reconnecting.
"""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse, urlunparse

import asyncpg

from app.config import settings

_pool: asyncpg.Pool | None = None

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_REQUIRE_SSL = {"require", "verify-ca", "verify-full"}


def pool_kwargs(database_url: str) -> dict:
    """Connect options that work against local Docker Postgres on Windows.

    `localhost` often resolves to IPv6 (`::1`). Docker's published port is
    IPv4; the WSL relay on `::1` resets during asyncpg's default SSL probe
    (sslmode=prefer), which surfaces as WinError 64 at startup.
    """
    parsed = urlparse(database_url)
    host = (parsed.hostname or "").casefold()
    sslmode = ""
    if parsed.query:
        sslmode = (parse_qs(parsed.query).get("sslmode") or [""])[0].casefold()
    dsn = database_url
    if host == "localhost":
        netloc = parsed.netloc.replace("localhost", "127.0.0.1", 1)
        dsn = urlunparse(parsed._replace(netloc=netloc))
        host = "127.0.0.1"
    kwargs: dict = {"dsn": dsn, "min_size": 2, "max_size": 10, "command_timeout": 30}
    if host in _LOCAL_HOSTS and sslmode not in _REQUIRE_SSL:
        kwargs["ssl"] = False
    return kwargs


async def get_pool() -> asyncpg.Pool:
    """Return the process-wide connection pool, creating it on first use."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(**pool_kwargs(settings.database_url))
    return _pool


async def close_pool() -> None:
    """Close the pool on application shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
