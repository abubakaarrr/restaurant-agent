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
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_INLINE = re.compile(
    r"(?:\*\*|__|~~)(?=\S)|(?<=\S)(?:\*\*|__|~~)|"
    r"(?<!\*)\*(?=\S)|(?<=\S)\*(?!\*)|`"
)
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_UNORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)([-*+])[ \t]+")
_ORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)(\d+)([.)])[ \t]+")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _unordered_list_matches(value: str) -> list[re.Match[str]]:
    matches = list(_UNORDERED_LIST_MARKER.finditer(value))
    if not matches:
        return []
    match = matches[0]
    before = value[: match.start(2)].rstrip()
    if not before or before.endswith(":") or not match.group(1):
        return matches
    return []


def _ordered_list_matches(value: str) -> list[re.Match[str]]:
    matches = list(_ORDERED_LIST_MARKER.finditer(value))
    if not matches:
        return []
    first = matches[0]
    before = value[: first.start(2)].rstrip()
    first_label = int(first.group(2))
    starts_like_list = (
        not before
        or not first.group(1)
        or (before.endswith(":") and first_label == 1)
    )
    if not starts_like_list:
        return []
    accepted = [first]
    expected = first_label + 1
    for match in matches[1:]:
        label = int(match.group(2))
        if label != expected:
            break
        accepted.append(match)
        expected += 1
    return accepted


def sanitize_spoken_text(text: str) -> str:
    """Remove written formatting without deleting grounded numeric values."""
    value = str(text or "")
    value = _MARKDOWN_LINK.sub(lambda match: match.group(1), value)
    value = _MARKDOWN_LINE.sub("", value)
    value = _MARKDOWN_INLINE.sub("", value)
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
        or _MARKDOWN_INLINE.search(value)
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
class SpokenTextBuffer:
    """Sanitize complete spoken segments without exposing split markup."""

    pending: str = ""

    def feed(self, text: str) -> tuple[str, ...]:
        self.pending += str(text or "")
        end = self._safe_prefix_end()
        if end == 0:
            return ()
        raw = self.pending[:end]
        self.pending = self.pending[end:]
        spoken = sanitize_spoken_text(raw)
        return (spoken,) if spoken else ()

    def flush(self) -> tuple[str, ...]:
        raw = self.pending
        self.pending = ""
        spoken = sanitize_spoken_text(raw)
        return (spoken,) if spoken else ()

    def _safe_prefix_end(self) -> int:
        if self._has_open_markdown_construct():
            return 0
        ordered_markers = {
            match.start(3) for match in _ORDERED_LIST_MARKER.finditer(self.pending)
        }
        matches = [
            match
            for match in _SENTENCE_END.finditer(self.pending)
            if match.start() not in ordered_markers
        ]
        if not matches:
            return 0
        return matches[-1].end()

    def _has_open_markdown_construct(self) -> bool:
        value = self.pending
        bracket = value.rfind("[")
        if bracket > value.rfind("]"):
            return True
        link = value.rfind("](")
        if link >= 0 and link > value.rfind(")"):
            return True
        if value.count("`") % 2:
            return True
        for marker in ("**", "__", "~~"):
            if value.count(marker) % 2:
                return True
        return False


@dataclass
class ResponseGenerationGate:
    """Invalidate earlier generation IDs as soon as a newer turn begins."""

    active_response_id: int = -1

    def begin(self, response_id: int) -> None:
        self.active_response_id = int(response_id)

    def allows(self, response_id: int) -> bool:
        return int(response_id) == self.active_response_id
