"""Token-overlap search over restaurant markdown and operator FAQ rows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
STATIC_SOURCES = (
    ("info", "restaurant_info.md"),
    ("slots", "slots.md"),
)

_STOP = {
    "the",
    "and",
    "for",
    "you",
    "your",
    "are",
    "with",
    "that",
    "this",
    "have",
    "from",
    "what",
    "when",
    "where",
    "how",
    "can",
    "do",
    "does",
    "is",
    "there",
    "any",
    "our",
    "offer",
    "offers",
    "please",
    "just",
    "also",
    "about",
    "call",
    "need",
    "want",
}


def tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (value or "").casefold())
        if len(token) > 2 and token not in _STOP
    }


def normalize_question(value: str) -> str:
    collapsed = " ".join((value or "").casefold().split())
    return collapsed[:300]


def _split_markdown(source: str, text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    heading = source
    body: list[str] = []
    for raw_line in text.splitlines():
        if raw_line.startswith("## "):
            if body:
                chunks.append(
                    {
                        "source": source,
                        "heading": heading,
                        "content": "\n".join(body).strip(),
                    }
                )
            heading = raw_line[3:].strip() or source
            body = [raw_line]
        else:
            body.append(raw_line)
    if body:
        chunks.append(
            {
                "source": source,
                "heading": heading,
                "content": "\n".join(body).strip(),
            }
        )
    return [chunk for chunk in chunks if chunk["content"]]


def load_static_chunks() -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for source, filename in STATIC_SOURCES:
        path = KNOWLEDGE_DIR / filename
        if not path.is_file():
            continue
        chunks.extend(_split_markdown(source, path.read_text(encoding="utf-8")))
    return chunks


def score_text(query: str, *parts: str) -> int:
    query_tokens = tokens(query)
    if not query_tokens:
        return 0
    haystack = tokens(" ".join(parts))
    # Whole-token overlap is deliberately conservative. Substring matching made
    # unrelated words such as "dress" and "address" equivalent.
    return len(query_tokens & haystack)


def search_static_knowledge(query: str, *, limit: int = 4) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for chunk in load_static_chunks():
        points = score_text(query, chunk["heading"], chunk["content"])
        if not points:
            continue
        scored.append({**chunk, "score": points, "kind": "static"})
    scored.sort(key=lambda item: (-item["score"], item["heading"]))
    return scored[:limit]


def search_faq_rows(
    query: str,
    rows: list[dict[str, Any]],
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        points = score_text(query, question, answer)
        if not points:
            continue
        scored.append(
            {
                "kind": "operator_faq",
                "heading": question,
                "content": answer,
                "score": points,
                "source": "operator_knowledge",
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["heading"]))
    return scored[:limit]


def format_knowledge_hits(hits: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for hit in hits:
        heading = hit.get("heading") or ""
        content = " ".join(str(hit.get("content") or "").split())
        if heading:
            lines.append(f"{heading}: {content}")
        else:
            lines.append(content)
    return " ".join(lines)
