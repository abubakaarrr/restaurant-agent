"""Shared caller-turn preparation for local and self-hosted voice flows."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.call_memory import hydrate_call_memory, resolve_session_id
from app.pending_confirmation import (
    ACTION_CANCEL_BOOKING,
    begin_caller_turn,
    get_pending_confirmation,
)


logger = logging.getLogger(__name__)

_CANCELLATION_REVERSAL_RE = re.compile(
    r"\b(?:do\s+not|don't|not)\s+cancel"
    r"(?:\s+(?:it|that|(?:the|my)\s+(?:reservation|booking))(?:\s+yet)?|(?=\s*(?:[,;:.!?]|$)))",
    re.IGNORECASE,
)
_ORDER_CANCELLATION_RE = re.compile(
    r"\b(?:do\s+not|don't|not)\s+cancel\s+"
    r"(?:(?:the|my)\s+)?(?:pickup|delivery)(?:\s+order)?\b|"
    r"\b(?:do\s+not|don't|not)\s+cancel\s+(?:(?:the|my)\s+)?order\b",
    re.IGNORECASE,
)
_CANCELLATION_ONLY_TAIL_RE = re.compile(
    r"^\s*[,;:.!?]*\s*(?:please|i\s+was\s+just\s+(?:checking|asking)"
    r"(?:\s+(?:what\s+)?the\s+cancellation\s+(?:process|policy)\s+is)?)?"
    r"[.!?]*\s*$",
    re.IGNORECASE,
)
_CANCELLATION_ONLY_PREFIX_RE = re.compile(
    r"^\s*(?:(?:actually|no|wait|please)[,;:.!?\s]*)*"
    r"(?:i\s+was\s+just\s+(?:checking|asking)[,;:.!?\s]*)?$",
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
    text = utterance or ""
    reversal_match = _CANCELLATION_REVERSAL_RE.search(text)
    order_cancellation = _ORDER_CANCELLATION_RE.search(text)
    pending_cancellation = get_pending_confirmation(sid, ACTION_CANCEL_BOOKING)
    if (
        affirmation != "negative"
        or not pending_cancellation
        or (order_cancellation and not reversal_match)
    ):
        return {
            "handled": False,
            "kind": "caller_turn",
            "affirmation": affirmation,
        }

    from app.services.restaurant import restaurant_service

    leading_text = text[: reversal_match.start()] if reversal_match else ""
    remaining_text = text[reversal_match.end() :] if reversal_match else text
    leading_intent = not bool(_CANCELLATION_ONLY_PREFIX_RE.fullmatch(leading_text))
    remaining_intent = leading_intent or not bool(
        _CANCELLATION_ONLY_TAIL_RE.fullmatch(remaining_text)
    )
    try:
        result = await restaurant_service.reverse_pending_cancellation(sid)
    except Exception:
        try:
            await restaurant_service.invalidate_pending_cancellation(sid)
        except Exception:
            logger.warning(
                "Unable to persist cancellation invalidation session=%s",
                sid,
                exc_info=True,
            )
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
