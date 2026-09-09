"""Small deterministic contracts for customer-facing voice delivery."""

from __future__ import annotations

import re
from dataclasses import dataclass


TRANSFER_UNAVAILABLE_REPLY = (
    "I can't transfer you right now. I can take a message and callback details."
)
MANAGER_TRANSFER_UNAVAILABLE_REPLY = (
    "I can't transfer you to a manager right now. "
    "I can take a message and callback details."
)
INCOMPLETE_INPUT_REPLY = "Sorry, I didn't catch that. What can I help with?"
FRUSTRATION_REPLY = "You're right—I missed that. What should I fix?"

_MARKDOWN_LINE = re.compile(r"(?m)^\s*(?:#{1,6}\s|>\s|```)")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]+\]\([^)]+\)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_UNORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)([-*+])[ \t]+")
_ORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)(\d+)([.)])[ \t]+")


def _unordered_list_matches(value: str) -> list[re.Match[str]]:
    matches = list(_UNORDERED_LIST_MARKER.finditer(value))
    if len(matches) > 1:
        return matches
    if not matches:
        return []
    match = matches[0]
    before = value[: match.start(2)].rstrip()
    if not before or before.endswith(":"):
        return matches
    return []


def _ordered_list_matches(value: str) -> list[re.Match[str]]:
    matches = list(_ORDERED_LIST_MARKER.finditer(value))
    if not matches:
        return []
    first = matches[0]
    before = value[: first.start(2)].rstrip()
    starts_like_list = not before or before.endswith(":")
    labels = [int(match.group(2)) for match in matches]
    sequential = len(labels) > 1 and all(
        current == previous + 1
        for previous, current in zip(labels, labels[1:])
    )
    return matches if starts_like_list or sequential else []


def sanitize_spoken_text(text: str) -> str:
    """Remove written list markers without deleting grounded numeric values."""
    value = str(text or "")
    unordered = _unordered_list_matches(value)
    if unordered:
        allowed = {match.start() for match in unordered}

        def replace_unordered(match: re.Match[str]) -> str:
            if match.start() not in allowed:
                return match.group(0)
            before = value[: match.start(2)].rstrip()
            if not before or not match.group(1):
                return ""
            return " " if before.endswith(":") else ". "

        value = _UNORDERED_LIST_MARKER.sub(replace_unordered, value)

    ordered = _ordered_list_matches(value)
    if ordered:
        allowed = {match.start() for match in ordered}

        def replace_ordered(match: re.Match[str]) -> str:
            if match.start() not in allowed:
                return match.group(0)
            return f"{match.group(1)}{match.group(2)}, "

        value = _ORDERED_LIST_MARKER.sub(replace_ordered, value)
    return value


def spoken_text_violations(text: str, *, max_words: int = 60) -> tuple[str, ...]:
    """Return observable reasons that a complete reply is unsuitable for speech."""
    value = str(text or "").strip()
    violations: list[str] = []
    if not value:
        violations.append("empty")
    if (
        _MARKDOWN_LINE.search(value)
        or _MARKDOWN_LINK.search(value)
        or _unordered_list_matches(value)
        or _ordered_list_matches(value)
    ):
        violations.append("markdown")
    if _URL.search(value):
        violations.append("url")
    if len(value.split()) > max_words:
        violations.append("too_long")
    return tuple(violations)


@dataclass
class ResponseGenerationGate:
    """Invalidate earlier generation IDs as soon as a newer turn begins."""

    active_response_id: int = -1

    def begin(self, response_id: int) -> None:
        self.active_response_id = int(response_id)

    def allows(self, response_id: int) -> bool:
        return int(response_id) == self.active_response_id
