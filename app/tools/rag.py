"""RAG tool — semantic search over pgvector knowledge chunks."""

import asyncpg
from openai import AsyncOpenAI
from langchain_core.tools import tool

from app.config import settings

oai = AsyncOpenAI(api_key=settings.openai_api_key)
EMBEDDING_MODEL = "text-embedding-3-small"
SIMILARITY_THRESHOLD = 0.0  # no threshold — always return top results


async def _embed(text: str) -> list[float]:
    resp = await oai.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


async def semantic_search(
    query: str,
    source_filter: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Return top-k most relevant knowledge chunks for a query."""
    vector = await _embed(query)
    conn = await asyncpg.connect(settings.database_url)
    try:
        if source_filter:
            rows = await conn.fetch(
                """
                SELECT content, source,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM knowledge_chunks
                WHERE source = $2
                  AND 1 - (embedding <=> $1::vector) > $4
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                str(vector), source_filter, top_k, SIMILARITY_THRESHOLD,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT content, source,
                       1 - (embedding <=> $1::vector) AS similarity
                FROM knowledge_chunks
                WHERE 1 - (embedding <=> $1::vector) > $3
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                str(vector), top_k, SIMILARITY_THRESHOLD,
            )
        return [dict(r) for r in rows]
    finally:
        await conn.close()


# ── LangChain tool wrappers (bound to the LangGraph agent) ────

@tool
async def search_menu(query: str) -> str:
    """Search the restaurant menu for items, prices, dietary info, and ingredients.
    Use this when the caller asks about food, drinks, dietary options, allergens, or prices.
    """
    chunks = await semantic_search(query, source_filter="menu", top_k=3)
    if not chunks:
        return "No relevant menu information found."
    return "\n\n".join(c["content"] for c in chunks)


@tool
async def search_restaurant_info(query: str) -> str:
    """Search for general restaurant information: location, hours, parking, facilities, FAQs, policies.
    Use this when the caller asks about opening hours, address, parking, accessibility, events, or policies.
    """
    # Search both info and slots sources — hours live in slots, FAQs live in info
    info_chunks = await semantic_search(query, source_filter="info", top_k=3)
    slots_chunks = await semantic_search(query, source_filter="slots", top_k=3)
    all_chunks = info_chunks + slots_chunks
    if not all_chunks:
        return "No relevant information found."
    return "\n\n".join(c["content"] for c in all_chunks)
