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

_MARKDOWN_LINE = re.compile(r"(?m)^[ \t]*(?:#{1,6}[ \t]+|(?:>[ \t]?)+)")
_MARKDOWN_FENCE = re.compile(
    r"(?m)^[ \t]*(?:`{3,}|~{3,})[^\n]*(?:\n|$)"
)
_MARKDOWN_THEMATIC_BREAK = re.compile(
    r"(?m)^[ \t]*(?:[-*_][ \t]*){3,}[ \t]*(?:\n|$)"
)
_MARKDOWN_SETEXT_UNDERLINE = re.compile(
    r"(?m)^[ \t]*(?:=+|-{1,2})[ \t]*(?:\n|$)"
)
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_INLINE = re.compile(
    r"(?:\*\*|__|~~)(?=\S)|(?<=\S)(?:\*\*|__|~~)|"
    r"(?<!\*)\*(?=\S)|(?<=\S)\*(?!\*)|"
    r"(?<![\w_])_(?=\S)|(?<=\S)_(?![\w_])|`"
)
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_UNORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)([-*+•◦‣])[ \t]+")
_ORDERED_LIST_MARKER = re.compile(r"(?m)(^|[ \t]+)(\d+)([.)])[ \t]+")
_STREAM_BOUNDARY = re.compile(r"\s+")
_COMPLETE_SEGMENT = re.compile(r"[.!?][\s]*$")
_RANGE_VALUE = re.compile(r"\d+(?::\d+)?(?:\.\d+)?$")
_MERIDIEM_TIME_LEFT = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?$", re.IGNORECASE
)
_TIME_RIGHT = re.compile(
    r"^\s*\d{1,2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?(?=\s|$|[,.;])",
    re.IGNORECASE,
)
_CONFIRMATION_CONTEXT = re.compile(
    r"\bconfirmation(?:\s+(?:number|code))?\s*:?\s*$", re.IGNORECASE
)
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
            _MERIDIEM_TIME_LEFT.search(value[: match.start(2)].rstrip())
            and _TIME_RIGHT.match(value[match.end(2) :])
        )
    )


def _unordered_list_candidates(value: str) -> list[re.Match[str]]:
    return [
        match
        for match in _UNORDERED_LIST_MARKER.finditer(value)
        if not _range_separator(value, match)
    ]


def _unordered_list_matches(value: str) -> list[re.Match[str]]:
    candidates = _unordered_list_candidates(value)
    if not candidates:
        return []
    match = candidates[0]
    before = value[: match.start(2)].rstrip()
    if (
        len(candidates) >= 2
        or not before
        or before.endswith(":")
        or not match.group(1)
        or match.group(2) in "•◦‣"
    ):
        return candidates
    return []


def _ordered_list_matches(value: str) -> list[re.Match[str]]:
    matches = list(_ORDERED_LIST_MARKER.finditer(value))
    if not matches:
        return []
    first = matches[0]
    before = value[: first.start(2)].rstrip()
    first_label = int(first.group(2))
    accepted = [first]
    expected = first_label + 1
    for match in matches[1:]:
        label = int(match.group(2))
        if label != expected:
            break
        accepted.append(match)
        expected += 1
    if _CONFIRMATION_CONTEXT.search(before):
        return []
    starts_like_list = (
        not before
        or not first.group(1)
        or before.endswith(":")
        or len(accepted) >= 2
    )
    return accepted if starts_like_list else []


def sanitize_spoken_text(text: str) -> str:
    """Remove written formatting without deleting grounded numeric values."""
    value = str(text or "")
    value = _MARKDOWN_LINK.sub(lambda match: match.group(1), value)
    value = _MARKDOWN_FENCE.sub("", value)
    value = _MARKDOWN_THEMATIC_BREAK.sub("", value)
    value = _MARKDOWN_SETEXT_UNDERLINE.sub("", value)
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
            if len(unordered) == 1 and match.group(2) in "•◦‣":
                return " "
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
        or _MARKDOWN_FENCE.search(value)
        or _MARKDOWN_THEMATIC_BREAK.search(value)
        or _MARKDOWN_SETEXT_UNDERLINE.search(value)
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
        unordered_candidates = _unordered_list_candidates(self.pending)
        unordered_matches = _unordered_list_matches(self.pending)
        ordered_candidates = list(_ORDERED_LIST_MARKER.finditer(self.pending))
        possible_ordered_list = bool(
            ordered_candidates and int(ordered_candidates[0].group(2)) == 1
        )
        ordered_matches = _ordered_list_matches(self.pending)
        if (unordered_candidates and not unordered_matches) or (
            possible_ordered_list and not ordered_matches
        ):
            return 0
        if unordered_matches or ordered_matches:
            return len(self.pending) if _COMPLETE_SEGMENT.search(self.pending) else 0
        boundaries = list(_STREAM_BOUNDARY.finditer(self.pending))
        if len(boundaries) < 2:
            return 0
        end = boundaries[-2].end()
        remainder = self.pending[end:].lstrip()
        if re.match(r"(?:[-*+•◦‣]\s|\d+[.)]\s|[*_~`]|\[)", remainder):
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
