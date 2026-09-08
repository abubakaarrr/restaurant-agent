"""
Seed script — run once after schema.sql:
    python db/seed.py

By default this inserts only the live tables and menu. Optional legacy/offline
pgvector embeddings require ``--with-embeddings`` and an OpenAI key.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.restaurant_knowledge import get_restaurant_knowledge

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/restaurant_agent")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"

PRICE_CONFIRMED = "Price confirmed."
PRICE_ESTIMATED = "Price estimated pending client confirmation."

KNOWLEDGE = get_restaurant_knowledge()

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
    # four reservable high-top tables; the two bar-counter seats are walk-in only
    (15, 4, "bar"),
    (16, 4, "bar"),
    (17, 4, "bar"),
    (18, 4, "bar"),
]


# Backward-compatible public constant used by integration checks.  Each row is
# a fully normalized canonical item rather than the former six-field tuple.
MENU_ITEMS = list(KNOWLEDGE.menu_items)


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
    live_canonical_ids = [item["item_id"] for item in MENU_ITEMS]
    await conn.execute(
        """
        DELETE FROM menu_items
        WHERE (canonical_id IS NULL OR NOT (canonical_id = ANY($1::text[])))
          AND id NOT IN (
              SELECT DISTINCT menu_item_id FROM order_items
              WHERE menu_item_id IS NOT NULL
          )
        """,
        live_canonical_ids,
    )
    await conn.execute(
        """
        UPDATE menu_items
        SET available = FALSE
        WHERE (canonical_id IS NULL OR NOT (canonical_id = ANY($1::text[])))
        """,
        live_canonical_ids,
    )
    for item in MENU_ITEMS:
        metadata = {
            key: value
            for key, value in item.items()
            if key
            not in {
                "name",
                "category_id",
                "price",
                "description",
                "dietary_tags",
                "aliases",
                "ingredients",
                "allergens",
                "service_periods",
                "availability",
                "source_id",
                "data_version",
                "effective_from",
                "effective_to",
            }
        }
        await conn.execute(
            """
            INSERT INTO menu_items
                (name, category, price, description, dietary, available,
                 price_estimated, canonical_id, aliases, ingredients, allergens,
                 service_periods, availability_status, knowledge_metadata,
                 source_id, data_version, effective_from, effective_to)
            VALUES ($1, $2, $3, $4, $5, $6, FALSE, $7, $8, $9, $10,
                    $11, $12, $13::jsonb, $14, $15, $16::date, $17::date)
            ON CONFLICT (canonical_id) WHERE canonical_id IS NOT NULL DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                price = EXCLUDED.price,
                description = EXCLUDED.description,
                dietary = EXCLUDED.dietary,
                available = EXCLUDED.available,
                price_estimated = FALSE,
                canonical_id = EXCLUDED.canonical_id,
                aliases = EXCLUDED.aliases,
                ingredients = EXCLUDED.ingredients,
                allergens = EXCLUDED.allergens,
                service_periods = EXCLUDED.service_periods,
                availability_status = EXCLUDED.availability_status,
                knowledge_metadata = EXCLUDED.knowledge_metadata,
                source_id = EXCLUDED.source_id,
                data_version = EXCLUDED.data_version,
                effective_from = EXCLUDED.effective_from,
                effective_to = EXCLUDED.effective_to
            """,
            item["name"],
            item["category_id"].removeprefix("category."),
            float(item["price"]),
            item["description"],
            item["dietary_tags"],
            item.get("availability") == "available",
            item["item_id"],
            item["aliases"],
            item["ingredients"],
            item["allergens"],
            item["service_periods"],
            item["availability"],
            json.dumps(metadata, sort_keys=True),
            item["source_id"],
            item["data_version"],
            item["effective_from"],
            item.get("effective_to"),
        )
    print(f"  -> {len(MENU_ITEMS)} menu items seeded.")

    print("Seeding canonical restaurant knowledge...")
    meta = KNOWLEDGE.metadata
    records: list[tuple[str, str, str, str, dict]] = [
        (
            KNOWLEDGE.identity["restaurant_id"],
            "identity",
            "category.identity",
            KNOWLEDGE.identity.get("name", ""),
            KNOWLEDGE.identity,
        ),
        ("hours.canonical", "hours", "category.operations", "Operating hours", KNOWLEDGE.raw["hours"]),
        ("style.canonical", "conversation_style", "category.brand", "Brand conversation guidance", KNOWLEDGE.raw["conversation_style"]),
    ]
    records.extend(
        (area["area_id"], "dining_area", "category.seating", area.get("name", ""), area)
        for area in KNOWLEDGE.raw["dining_areas"]
    )
    records.extend(
        (option["option_id"], "modifier", "category.modifiers", option.get("name", ""), option)
        for option in KNOWLEDGE.raw["modifier_options"]
    )
    records.extend(
        (topic["topic_id"], "topic", topic["category_id"], topic.get("answer", ""), topic)
        for topic in KNOWLEDGE.topics
    )
    records.extend(
        (route["route_id"], "escalation_route", "category.escalation", route.get("fallback", ""), route)
        for route in KNOWLEDGE.raw["escalation_routes"]
    )
    canonical_ids = [record[0] for record in records]
    await conn.execute(
        "DELETE FROM restaurant_knowledge_records WHERE source_id = $1 AND NOT (canonical_id = ANY($2::text[]))",
        meta["source_id"],
        canonical_ids,
    )
    for canonical_id, record_type, category_id, display_text, payload in records:
        effective_from = payload.get("effective_from") or meta["effective_from"]
        effective_to = payload.get("effective_to", meta["effective_to"])
        await conn.execute(
            """
            INSERT INTO restaurant_knowledge_records
                (canonical_id, record_type, category_id, source_id, schema_version,
                 data_version, effective_from, effective_to, status, display_text,
                 payload, synthetic)
            VALUES ($1, $2, $3, $4, $5, $6, $7::date, $8::date, $9, $10,
                    $11::jsonb, TRUE)
            ON CONFLICT (canonical_id) DO UPDATE SET
                record_type = EXCLUDED.record_type,
                category_id = EXCLUDED.category_id,
                source_id = EXCLUDED.source_id,
                schema_version = EXCLUDED.schema_version,
                data_version = EXCLUDED.data_version,
                effective_from = EXCLUDED.effective_from,
                effective_to = EXCLUDED.effective_to,
                status = EXCLUDED.status,
                display_text = EXCLUDED.display_text,
                payload = EXCLUDED.payload,
                synthetic = TRUE
            """,
            canonical_id,
            record_type,
            category_id,
            meta["source_id"],
            meta["schema_version"],
            meta["data_version"],
            effective_from,
            effective_to,
            payload.get("status", "current"),
            display_text,
            json.dumps(payload, sort_keys=True),
        )
    print(f"  -> {len(records)} canonical knowledge records seeded.")

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
