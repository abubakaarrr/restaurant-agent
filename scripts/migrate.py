#!/usr/bin/env python3
"""Apply versioned SQL migrations with checksums and an advisory lock."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path

import asyncpg


ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = ROOT / "db" / "migrations"
SCHEMA_FILE = ROOT / "db" / "schema.sql"
LOCK_ID = 8_110_826


async def apply(*, initialize_schema: bool = False) -> None:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    connection = await asyncpg.connect(database_url)
    try:
        await connection.execute("SELECT pg_advisory_lock($1)", LOCK_ID)
        if initialize_schema:
            await connection.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
            """
        )
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            sql = path.read_text(encoding="utf-8")
            digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            existing = await connection.fetchval(
                "SELECT sha256 FROM schema_migrations WHERE filename = $1",
                path.name,
            )
            if existing:
                if existing != digest:
                    raise RuntimeError(
                        f"Applied migration was modified: {path.name}. "
                        "Create a new migration instead."
                    )
                print(f"unchanged {path.name}")
                continue
            await connection.execute(sql)
            await connection.execute(
                "INSERT INTO schema_migrations (filename, sha256) VALUES ($1, $2)",
                path.name,
                digest,
            )
            print(f"applied   {path.name}")
    finally:
        try:
            await connection.execute("SELECT pg_advisory_unlock($1)", LOCK_ID)
        finally:
            await connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize-schema",
        action="store_true",
        help="Apply db/schema.sql first; use only for an empty database.",
    )
    args = parser.parse_args()
    asyncio.run(apply(initialize_schema=args.initialize_schema))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
