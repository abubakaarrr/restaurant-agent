"""Explicitly authorize the configured database for QA browser voice writes."""
import asyncio
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import settings
from app.db_pool import pool_kwargs
import asyncpg


async def main():
    settings.validate_native_voice_staging()
    marker = os.environ["NATIVE_VOICE_DATABASE_MARKER"].strip()
    options = pool_kwargs(settings.database_url)
    connection = await asyncpg.connect(**{
        key: value for key, value in options.items() if key not in {"min_size", "max_size"}
    })
    try:
        database = await connection.fetchval("SELECT current_database()")
        identifier = '"' + database.replace('"', '""') + '"'
        literal = "'" + marker.replace("'", "''") + "'"
        await connection.execute(
            f"ALTER DATABASE {identifier} SET app.native_voice_staging_marker TO {literal}"
        )
    finally:
        await connection.close()
    # A new connection must inherit the database setting. Verify actual identity too.
    from app.native_voice.database_guard import get_native_voice_pool, close_native_voice_pool
    try:
        await get_native_voice_pool()
        print("QA database authorization verified; existing restaurant data preserved.")
    finally:
        await close_native_voice_pool()


if __name__ == "__main__":
    asyncio.run(main())
