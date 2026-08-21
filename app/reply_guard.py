"""Gate: do not send the previous answer as the reply to a new request."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_ACK = re.compile(
    r"^(thanks|thank you|thx|ok|okay|bye|goodbye|see you|you too|cheers)[\s!.]*$",
    re.IGNORECASE,
)


def is_ack(user_message: str) -> bool:
    return bool(_ACK.match((user_message or "").strip()))


_CLERK_INVENTORY = re.compile(
    r"^I have [A-Z][A-Za-z' -]+,\s*\d"
    r"|^(Patio seating|The \w+ note) is (noted|saved)"
    r"|I've updated your party to"
    r"|Reservation draft updated",
    re.IGNORECASE,
)


def is_clerk_inventory(speech: str) -> bool:
    """True for form-dump replies like 'I have Hamza, 123 654 789, for five people.'"""
    text = " ".join((speech or "").split())
    return bool(_CLERK_INVENTORY.search(text))


def is_repeated_reply(
    user_message: str,
    previous_reply: str,
    new_reply: str,
    *,
    previous_user_message: str = "",
) -> bool:
    """True when this turn's reply is a stale echo of the prior assistant turn.

    Exact duplicates are always flagged when the user said something different —
    including short lines like "Got it, 7:00 PM it is." after an unrelated "yes".
    """
    prev = " ".join((previous_reply or "").split())
    nxt = " ".join((new_reply or "").split())
    if not prev or not nxt:
        return False
    user = " ".join((user_message or "").split())
    prior_user = " ".join((previous_user_message or "").split())
    users_differ = bool(
        prior_user and user and prior_user.casefold() != user.casefold()
    )
    ratio = SequenceMatcher(None, prev.casefold(), nxt.casefold()).ratio()
    if users_differ and (prev.casefold() == nxt.casefold() or ratio >= 0.92):
        logger.error(
            "duplicate_assistant_reply users_differ=1 ratio=%.3f prev=%r new=%r user=%r",
            ratio,
            prev[:160],
            nxt[:160],
            user[:120],
        )
        return True
    # Same user text again (e.g. repeating a question) may legitimately get the same answer.
    if prior_user and not users_differ:
        return False
    if is_ack(user_message):
        return False
    if len(prev) < 12 or len(nxt) < 12:
        return False
    if ratio >= 0.78:
        logger.error(
            "duplicate_assistant_reply users_differ=%s ratio=%.3f prev=%r new=%r user=%r",
            int(users_differ),
            ratio,
            prev[:160],
            nxt[:160],
            user[:120],
        )
        return True
    return False
