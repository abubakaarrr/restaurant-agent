"""Shared caller-turn preparation for local and self-hosted voice flows."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.call_memory import hydrate_call_memory, resolve_session_id
from app.pending_confirmation import begin_caller_turn


logger = logging.getLogger(__name__)

_CANCELLATION_REVERSAL_RE = re.compile(
    r"^\s*(?:"
    r"(?:i\s+was\s+just\s+(?:checking|asking)[,;:]?\s+)?"
    r"(?:please\s+)?(?:do\s+not|don't)\s+cancel"
    r"(?:\s+(?:it|that|the\s+reservation|my\s+reservation)|(?=\s*(?:[,;:.!?]|$)))"
    r"(?:[,;:]?\s+(?:please|i\s+was\s+just\s+(?:checking|asking)))?"
    r"|(?:actually[,;:]?\s+)?no[.,;:]?\s+(?:do\s+not|don't)\s+cancel\s+it\s+yet[.!]?\s+"
    r"i\s+was\s+just\s+checking\s+(?:what\s+)?the\s+cancellation\s+(?:process|policy)\s+is"
    r")[.!?]*\s*",
    re.IGNORECASE,
)


async def process_caller_turn(session_id: str, utterance: str) -> dict[str, Any]:
    """Hydrate confirmation state and apply narrow stateful caller reversals.

    Local streaming and text flows call this once before interpreting or
    executing a caller turn.
    """
    sid = resolve_session_id(session_id)
    await hydrate_call_memory(sid)
    affirmation = begin_caller_turn(sid, utterance)
    reversal_match = _CANCELLATION_REVERSAL_RE.match(utterance or "")
    if not reversal_match:
        return {
            "handled": False,
            "kind": "caller_turn",
            "affirmation": affirmation,
        }

    from app.services.restaurant import restaurant_service

    remaining_intent = bool(
        re.search(r"[a-z0-9]", (utterance or "")[reversal_match.end() :], re.IGNORECASE)
    )
    try:
        result = await restaurant_service.reverse_pending_cancellation(sid)
    except Exception:
        logger.warning(
            "Unable to verify cancellation reversal session=%s", sid, exc_info=True
        )
        return {
            "handled": True,
            "kind": "cancellation_reversal_unavailable",
            "affirmation": affirmation,
            "message": (
                "I couldn't verify the reservation or cancellation state right now. "
                "I have not submitted a cancellation."
            ),
        }
    return {
        "handled": not remaining_intent,
        "kind": (
            "cancellation_reversal_with_remaining_intent"
            if remaining_intent
            else "cancellation_reversal"
        ),
        "affirmation": affirmation,
        **result,
    }
