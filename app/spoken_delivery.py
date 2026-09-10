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
_MARKDOWN_THEMATIC_BREAK = re.compile(
    r"(?m)^[ \t]*(?:[-*_][ \t]*){3,}[ \t]*(?:\n|$)"
)
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_INLINE = re.compile(
    r"(?:\*\*|__|~~)(?=\S)|(?<=\S)(?:\*\*|__|~~)|"
    r"(?<!\*)\*(?=\S)|(?<=\S)\*(?!\*)|"
    r"(?<![\w_])_(?=\S)|(?<=\S)_(?![\w_])|`"
)
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_UNORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)([-*+])[ \t]+")
_ORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)(\d+)([.)])[ \t]+")
_STREAM_BOUNDARY = re.compile(r"\s+")
_COMPLETE_SEGMENT = re.compile(r"[.!?][\s]*$")
_RANGE_VALUE = re.compile(r"\d+(?::\d+)?(?:\.\d+)?$")
_RANGE_WORDS = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _range_separator(value: str, match: re.Match[str]) -> bool:
    if match.group(2) != "-":
        return False
    left_match = re.search(r"([\w:.]+)\s*$", value[: match.start(2)])
    right_match = re.match(r"\s*([\w:.]+)", value[match.end(2) :])
    if not left_match or not right_match:
        return False
    left = left_match.group(1).casefold()
    right = right_match.group(1).casefold()
    return bool(
        (_RANGE_VALUE.fullmatch(left) and _RANGE_VALUE.fullmatch(right))
        or (left in _RANGE_WORDS and right in _RANGE_WORDS)
        or (
            re.search(
                r"\b\d{1,2}(?::\d{2})?\s*[ap]\.m\.$",
                value[: match.start(2)].rstrip(),
                re.IGNORECASE,
            )
            and _RANGE_VALUE.fullmatch(right)
        )
    )


def _unordered_list_matches(value: str) -> list[re.Match[str]]:
    matches = list(_UNORDERED_LIST_MARKER.finditer(value))
    if not matches:
        return []
    candidates = [match for match in matches if not _range_separator(value, match)]
    if not candidates:
        return []
    match = candidates[0]
    before = value[: match.start(2)].rstrip()
    if len(candidates) >= 2 or not before or before.endswith(":") or not match.group(1):
        return candidates
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
    value = _MARKDOWN_THEMATIC_BREAK.sub("", value)
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
            return " " if before.endswith((":", ".", ",", ";")) else ". "

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
        or _MARKDOWN_THEMATIC_BREAK.search(value)
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
        if _unordered_list_matches(self.pending) or _ordered_list_matches(self.pending):
            return len(self.pending) if _COMPLETE_SEGMENT.search(self.pending) else 0
        boundaries = list(_STREAM_BOUNDARY.finditer(self.pending))
        if len(boundaries) < 2:
            return 0
        end = boundaries[-2].end()
        remainder = self.pending[end:].lstrip()
        if re.match(r"(?:[-*+]\s|\d+[.)]\s|[*_~`]|\[)", remainder):
            return 0
        prefix = self.pending[:end]
        open_positions: list[int] = []
        bracket = prefix.rfind("[")
        if bracket > prefix.rfind("]"):
            open_positions.append(bracket)
        link = prefix.rfind("](")
        if link >= 0 and link > prefix.rfind(")"):
            open_positions.append(link)
        for marker in ("`", "**", "__", "~~"):
            if prefix.count(marker) % 2:
                open_positions.append(prefix.rfind(marker))
        emphasis = re.search(r"(?<![\w_])_(?=\S)[^_]*$", prefix)
        if emphasis:
            open_positions.append(emphasis.start())
        return min(open_positions, default=end)


@dataclass
class ResponseGenerationGate:
    """Invalidate earlier generation IDs as soon as a newer turn begins."""

    active_response_id: int = -1

    def begin(self, response_id: int) -> None:
        self.active_response_id = int(response_id)

    def allows(self, response_id: int) -> bool:
        return int(response_id) == self.active_response_id
