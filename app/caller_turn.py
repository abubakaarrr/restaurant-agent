"""Shared caller-turn preparation for managed and rollback voice flows."""

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
    r"(?:\s+(?:it|that|the\s+reservation|my\s+reservation))?"
    r"(?:[,;:]?\s+(?:please|i\s+was\s+just\s+(?:checking|asking)))?"
    r"|(?:actually[,;:]?\s+)?no[.,;:]?\s+(?:do\s+not|don't)\s+cancel\s+it\s+yet[.!]?\s+"
    r"i\s+was\s+just\s+checking\s+(?:what\s+)?the\s+cancellation\s+(?:process|policy)\s+is"
    r")[.!?]*\s*$",
    re.IGNORECASE,
)


async def process_caller_turn(session_id: str, utterance: str) -> dict[str, Any]:
    """Hydrate confirmation state and apply narrow stateful caller reversals.

    Every transport must call this once before interpreting or executing a caller
    turn.  Managed flows use the authenticated caller-turn endpoint; rollback
    streaming and text flows call it directly.
    """
    sid = resolve_session_id(session_id)
    await hydrate_call_memory(sid)
    affirmation = begin_caller_turn(sid, utterance)
    if not _CANCELLATION_REVERSAL_RE.fullmatch(utterance or ""):
        return {
            "handled": False,
            "kind": "caller_turn",
            "affirmation": affirmation,
        }

    from app.services.restaurant import restaurant_service

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
        "handled": True,
        "kind": "cancellation_reversal",
        "affirmation": affirmation,
        **result,
    }
