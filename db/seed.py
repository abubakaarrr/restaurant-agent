"""
Seed script — run once after schema.sql:
    python db/seed.py

Does two things:
1. Inserts tables and menu items into PostgreSQL
2. Embeds knowledge .md files into pgvector (RAG)
"""

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

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/restaurant_agent")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"

oai = AsyncOpenAI(api_key=OPENAI_API_KEY)


# ── Restaurant tables ─────────────────────────────────────────

TABLES = [
    (1, 2, "main"),
    (2, 2, "main"),
    (3, 4, "main"),
    (4, 4, "main"),
    (5, 4, "main"),
    (6, 6, "main"),
    (7, 6, "main"),
    (8, 8, "private"),
    (9, 8, "private"),
    (10, 2, "patio"),
    (11, 2, "patio"),
    (12, 4, "patio"),
]

MENU_ITEMS = [
    # (name, category, price, description, dietary)
    ("Tomato Basil Soup", "starter", 7.50, "Creamy roasted tomato soup with fresh basil", ["vegetarian", "gluten-free"]),
    ("Garlic Bread", "starter", 5.00, "Toasted sourdough with herb butter and roasted garlic", ["vegetarian"]),
    ("Hummus Platter", "starter", 9.00, "House-made hummus with cucumber, olives, pita", ["vegan", "gluten-free"]),
    ("Spring Rolls", "starter", 8.50, "Crispy vegetable rolls with sweet chili sauce", ["vegan"]),
    ("Chicken Wings", "starter", 12.00, "Grilled wings, choice of BBQ/buffalo/honey-garlic", ["gluten-free"]),
    ("Caprese Salad", "starter", 10.00, "Fresh mozzarella, heirloom tomatoes, aged balsamic", ["vegetarian", "gluten-free"]),
    ("Caesar Salad", "starter", 11.00, "Romaine, parmesan, croutons, house Caesar dressing", ["vegetarian"]),
    ("Grilled Chicken Breast", "main", 18.00, "Free-range chicken, lemon herb sauce, roasted veg", ["gluten-free"]),
    ("Beef Ribeye Steak", "main", 32.00, "280g grain-fed ribeye, chips and house salad", ["gluten-free"]),
    ("Pan-Seared Salmon", "main", 24.00, "Atlantic salmon, dill cream sauce, wild rice", ["gluten-free"]),
    ("Mushroom Risotto", "main", 19.00, "Arborio rice, wild mushrooms, truffle oil, parmesan", ["vegetarian", "gluten-free"]),
    ("Vegetable Pasta Primavera", "main", 16.00, "Penne, roasted seasonal vegetables, tomato herb sauce", ["vegan"]),
    ("Lamb Rack", "main", 36.00, "2 herb-crusted lamb cutlets, red wine jus, minted peas", ["gluten-free"]),
    ("Fish and Chips", "main", 20.00, "Beer-battered barramundi, thick-cut chips, tartare sauce", []),
    ("Chicken Parma", "main", 22.00, "Crumbed chicken, Napoli sauce, ham, mozzarella", []),
    ("New York Cheesecake", "dessert", 10.00, "Baked cheesecake with berry coulis", ["vegetarian"]),
    ("Chocolate Lava Cake", "dessert", 11.00, "Warm dark chocolate cake, molten centre, vanilla ice cream", ["vegetarian"]),
    ("Tiramisu", "dessert", 10.00, "Espresso-soaked sponge, mascarpone, cocoa", ["vegetarian"]),
    ("Fruit Sorbet", "dessert", 8.00, "Rotating daily flavour — vegan and gluten-free", ["vegan", "gluten-free"]),
    ("Ice Cream", "dessert", 8.00, "3 scoops: vanilla, chocolate, or strawberry", ["vegetarian", "gluten-free"]),
    ("Soft Drinks", "drink", 4.00, "Coke, Lemonade, Soda Water, Ginger Beer", ["vegan", "gluten-free"]),
    ("Freshly Squeezed Juice", "drink", 6.00, "Orange, Apple, Watermelon (seasonal)", ["vegan", "gluten-free"]),
    ("Coffee", "drink", 4.50, "Flat White, Cappuccino, Latte — oat milk available", ["vegetarian"]),
    ("Long Black", "drink", 4.00, "Long black coffee (double espresso)", ["vegetarian"]),
    ("Espresso", "drink", 4.00, "Single or double espresso shot", ["vegetarian"]),
    ("Flat White", "drink", 4.50, "Flat white coffee, oat milk available", ["vegetarian"]),
    ("Cappuccino", "drink", 4.50, "Cappuccino, oat milk available", ["vegetarian"]),
    ("Latte", "drink", 4.50, "Latte coffee, oat milk available", ["vegetarian"]),
    ("Sparkling Water", "drink", 3.50, "500ml sparkling water", ["vegan", "gluten-free"]),
    ("Still Water", "drink", 3.50, "500ml still water", ["vegan", "gluten-free"]),
    ("Tea", "drink", 4.00, "English Breakfast, Green, Chamomile, Peppermint", ["vegan", "gluten-free"]),
    ("Mocktail of the Day", "drink", 8.00, "Ask your server for today's creation", ["vegan"]),
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
    resp = await oai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


async def seed(conn: asyncpg.Connection) -> None:
    print("Seeding tables...")
    for table_number, capacity, location in TABLES:
        await conn.execute(
            """
            INSERT INTO tables (table_number, capacity, location)
            VALUES ($1, $2, $3)
            ON CONFLICT (table_number) DO NOTHING
            """,
            table_number, capacity, location
        )
    print(f"  → {len(TABLES)} tables seeded.")

    print("Seeding menu items...")
    for name, category, price, description, dietary in MENU_ITEMS:
        await conn.execute(
            """
            INSERT INTO menu_items (name, category, price, description, dietary)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
            """,
            name, category, float(price), description, dietary
        )
    print(f"  → {len(MENU_ITEMS)} menu items seeded.")

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

    print(f"  → {total_chunks} chunks embedded.")
    print("Seed complete.")


async def main() -> None:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set in .env — cannot embed knowledge files.")
        print("Set your key and re-run, or skip RAG seeding for now.")
        return

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        await seed(conn)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
