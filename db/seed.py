"""
Seed script — run once after schema.sql:
    python db/seed.py

By default this inserts only the live tables and menu. Optional legacy/offline
pgvector embeddings require ``--with-embeddings`` and an OpenAI key.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/restaurant_agent")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"

PRICE_ESTIMATED = "Price estimated pending client confirmation."
PRICE_CONFIRMED = "Price confirmed."

# ── Restaurant tables ─────────────────────────────────────────
# Phone bookings support parties up to 10. Each dining room has at least
# one table that seats 10 so main / patio / private can all take a large party.

TABLES = [
    # main dining room
    (1, 2, "main"),
    (2, 2, "main"),
    (3, 4, "main"),
    (4, 4, "main"),
    (5, 6, "main"),
    (6, 8, "main"),
    (7, 10, "main"),
    # private rooms
    (8, 8, "private"),
    (9, 10, "private"),
    # patio / outdoor
    (10, 2, "patio"),
    (11, 4, "patio"),
    (12, 6, "patio"),
    (13, 8, "patio"),
    (14, 10, "patio"),
]


def _menu_description(body: str, *, estimated: bool) -> str:
    flag = PRICE_ESTIMATED if estimated else PRICE_CONFIRMED
    return f"{body} {flag}".strip()


# (name, category, price, description, dietary, price_estimated)
# Dish names are venue-confirmed. Prices are estimated unless price_estimated is False.
MENU_ITEMS = [
    (
        "Rosemary Fries",
        "starter",
        9.00,
        _menu_description("Confirmed real item.", estimated=True),
        ["vegetarian"],
        True,
    ),
    (
        "Fried Cauliflower",
        "starter",
        12.00,
        _menu_description("Confirmed real item.", estimated=True),
        ["vegetarian"],
        True,
    ),
    (
        "Dumplings",
        "starter",
        14.00,
        _menu_description("Confirmed real item. Style unconfirmed.", estimated=True),
        [],
        True,
    ),
    (
        "Chicken Katsu Burger",
        "main",
        19.00,
        _menu_description("Confirmed real item; a reviewer favorite.", estimated=True),
        [],
        True,
    ),
    (
        "Hangover Burger",
        "main",
        20.00,
        _menu_description("Confirmed real item.", estimated=True),
        [],
        True,
    ),
    (
        "Fish and Chips",
        "main",
        21.00,
        _menu_description("Confirmed real item, recommended in reviews.", estimated=True),
        [],
        True,
    ),
    (
        "BLT",
        "main",
        17.00,
        _menu_description("Confirmed real item.", estimated=True),
        [],
        True,
    ),
    (
        "Cobb Salad",
        "main",
        18.00,
        _menu_description("Confirmed real item.", estimated=True),
        [],
        True,
    ),
    (
        "Poutine",
        "main",
        16.00,
        _menu_description("Confirmed real item. Ham hock variant seen.", estimated=True),
        [],
        True,
    ),
    (
        "Sweet Potato Fries",
        "starter",
        8.00,
        _menu_description("Confirmed real item.", estimated=True),
        ["vegetarian"],
        True,
    ),
    (
        "Salted Pretzel Toffee Pudding",
        "dessert",
        11.00,
        _menu_description("Confirmed real item.", estimated=True),
        ["vegetarian"],
        True,
    ),
    (
        "Old Fashioned",
        "drink",
        15.00,
        _menu_description('Confirmed real item; called "best in Gastown" in reviews.', estimated=True),
        [],
        True,
    ),
    (
        "Craft Lager Pitcher",
        "drink",
        18.00,
        _menu_description("Confirmed real item.", estimated=False),
        ["vegetarian"],
        False,
    ),
    (
        "Wings (Wing Wednesday)",
        "special",
        12.50,
        _menu_description(
            "Wednesday only, $12.50 per pound. Not a daily menu item.",
            estimated=False,
        ),
        [],
        False,
    ),
]


# ── Knowledge chunking ────────────────────────────────────────

KNOWLEDGE_FILES = [
    ("menu", "app/knowledge/menu.md"),
    ("slots", "app/knowledge/slots.md"),
    ("info", "app/knowledge/restaurant_info.md"),
]


def chunk_markdown(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks by characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 40]


async def embed(texts: list[str]) -> list[list[float]]:
    oai = AsyncOpenAI(api_key=OPENAI_API_KEY)
    resp = await oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


async def seed(conn: asyncpg.Connection, *, with_embeddings: bool = False) -> None:
    print("Seeding tables...")
    for table_number, capacity, location in TABLES:
        await conn.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES ($1, $2, $3)
            ON CONFLICT (table_number) DO UPDATE SET
                capacity = EXCLUDED.capacity,
                location = EXCLUDED.location
            """,
            table_number,
            capacity,
            location,
        )
    print(f"  -> {len(TABLES)} tables seeded.")

    print("Seeding menu items...")
    live_names = [name.casefold() for name, *_rest in MENU_ITEMS]
    await conn.execute(
        """
        DELETE FROM menu_items
        WHERE NOT (LOWER(name) = ANY($1::text[]))
          AND id NOT IN (
              SELECT DISTINCT menu_item_id FROM order_items
              WHERE menu_item_id IS NOT NULL
          )
        """,
        live_names,
    )
    await conn.execute(
        """
        UPDATE menu_items
        SET available = FALSE
        WHERE NOT (LOWER(name) = ANY($1::text[]))
        """,
        live_names,
    )
    for name, category, price, description, dietary, price_estimated in MENU_ITEMS:
        await conn.execute(
            """
            INSERT INTO menu_items
                (name, category, price, description, dietary, available, price_estimated)
            VALUES ($1, $2, $3, $4, $5, TRUE, $6)
            ON CONFLICT ((LOWER(name))) DO UPDATE SET
                category = EXCLUDED.category,
                price = EXCLUDED.price,
                description = EXCLUDED.description,
                dietary = EXCLUDED.dietary,
                available = TRUE,
                price_estimated = EXCLUDED.price_estimated
            """,
            name,
            category,
            float(price),
            description,
            dietary,
            price_estimated,
        )
    print(f"  -> {len(MENU_ITEMS)} menu items seeded.")

    if not with_embeddings:
        print("Skipping legacy pgvector embeddings (use --with-embeddings to rebuild).")
        print("Seed complete.")
        return

    print("Embedding knowledge files into pgvector...")
    base_path = Path(__file__).parent.parent
    total_chunks = 0

    for source, rel_path in KNOWLEDGE_FILES:
        file_path = base_path / rel_path
        if not file_path.exists():
            print(f"  ! {rel_path} not found, skipping.")
            continue

        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)
        print(f"  Embedding {len(chunks)} chunks from {rel_path}...")
        await conn.execute("DELETE FROM knowledge_chunks WHERE source = $1", source)

        # Embed in batches of 20
        for i in range(0, len(chunks), 20):
            batch = chunks[i:i+20]
            vectors = await embed(batch)
            for chunk_text, vector in zip(batch, vectors):
                await conn.execute(
                    """
                    INSERT INTO knowledge_chunks (source, content, embedding)
                    VALUES ($1, $2, $3::vector)
                    """,
                    source, chunk_text, str(vector)
                )
            total_chunks += len(batch)

    print(f"  -> {total_chunks} chunks embedded.")
    print("Seed complete.")


async def main(*, with_embeddings: bool = False) -> None:
    if with_embeddings and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in .env — cannot embed knowledge files.")
        raise SystemExit(2)

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await seed(conn, with_embeddings=with_embeddings)
    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Rebuild legacy/offline pgvector knowledge chunks.",
    )
    args = parser.parse_args()
    asyncio.run(main(with_embeddings=args.with_embeddings))
