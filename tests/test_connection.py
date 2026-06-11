"""Quick connection test — run after stopping local PostgreSQL 18."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import asyncpg
from app.config import settings


async def main():
    print(f"Connecting to: {settings.database_url}")
    conn = await asyncpg.connect(settings.database_url)

    version = await conn.fetchval("SELECT version()")
    print(f"Connected! PostgreSQL: {version[:40]}")

    tables = await conn.fetch(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
    )
    print(f"Tables found: {[r['table_name'] for r in tables]}")

    vec = await conn.fetchval("SELECT '[1,2,3]'::vector <=> '[1,2,3]'::vector")
    print(f"pgvector working: cosine distance = {vec}")

    await conn.close()
    print("All good — ready to seed!")


asyncio.run(main())
