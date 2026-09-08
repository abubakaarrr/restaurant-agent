"""Deterministic caller-behavior policy for voice turns.

The reducer in this module is deliberately narrow.  It responds to explicit
requests and a few mechanical conversation proxies; it does not classify a
caller's identity, health, accent, age, or emotional state.  No caller text is
copied into the generated prompt directive.

Typical use::

    reduction = reduce_behavior(state, TurnObservation(text=transcript))
    state = reduction.state
    directive = reduction.directive

Both the input and output state are immutable, which makes replay, testing,
and per-call persistence straightforward.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from app.config import settings
from app.restaurant_knowledge import (
    KnowledgeFixtureError,
    get_restaurant_knowledge,
    text_tokens,
)
from app.transfer_availability import resolve_handoff_destination


class BehaviorMode(str, Enum):
    """High-level response style selected by the reducer."""

    STANDARD = "standard"
    CONCISE = "concise"
    GUIDED = "guided"
    DEESCALATING = "deescalating"


class PacePreference(str, Enum):
    """A pace explicitly requested by the caller, or inferred from timing."""

    SLOWER = "slower"
    NORMAL = "normal"
    FASTER = "faster"


class AccessibilityPreference(str, Enum):
    """Explicit presentation requests; these are not medical classifications."""

    REPEAT_KEY_DETAILS = "repeat_key_details"
    SPELL_KEY_DETAILS = "spell_key_details"
    PLAIN_LANGUAGE = "plain_language"
    ONE_THING_AT_A_TIME = "one_thing_at_a_time"


class BehaviorControl(str, Enum):
    """Control-plane action to accompany a directive."""

    CONTINUE = "continue"
    HANDOFF = "handoff"
    END_CALL = "end_call"


# Timing proxies intentionally require several adequately sized samples.  A
# single short or noisy utterance must never change behavior.
MIN_TIMED_WORDS = 6
MIN_SPEECH_RATE_SAMPLES = 3
MAX_SPEECH_RATE_SAMPLES = 5
MIN_PLAUSIBLE_WPM = 45.0
MAX_PLAUSIBLE_WPM = 320.0
SLOW_WPM_ENTER = 105.0
SLOW_WPM_EXIT = 125.0
FAST_WPM_ENTER = 190.0
FAST_WPM_EXIT = 170.0

MAX_INTERRUPTION_TIMESTAMPS = 16
BOUNDARY_DECAY_TURNS = 6
BOUNDARY_END_STRIKES = 3
DEESCALATION_HOLD_TURNS = 3
HURRY_HOLD_TURNS = 3


_LANGUAGE_NAMES: dict[str, tuple[str, str]] = {
    "american english": ("en-US", "American English"),
    "british english": ("en-GB", "British English"),
    "bengali": ("bn", "Bengali"),
    "brazilian portuguese": ("pt-BR", "Brazilian Portuguese"),
    "cantonese": ("yue", "Cantonese"),
    "chinese": ("zh", "Chinese"),
    "dutch": ("nl", "Dutch"),
    "english": ("en", "English"),
    "farsi": ("fa", "Farsi"),
    "french": ("fr", "French"),
    "german": ("de", "German"),
    "hindi": ("hi", "Hindi"),
    "italian": ("it", "Italian"),
    "japanese": ("ja", "Japanese"),
    "korean": ("ko", "Korean"),
    "mandarin": ("zh-CN", "Mandarin"),
    "persian": ("fa", "Persian"),
    "polish": ("pl", "Polish"),
    "portuguese": ("pt", "Portuguese"),
    "punjabi": ("pa", "Punjabi"),
    "russian": ("ru", "Russian"),
    "somali": ("so", "Somali"),
    "spanish": ("es", "Spanish"),
    "swahili": ("sw", "Swahili"),
    "turkish": ("tr", "Turkish"),
    "ukrainian": ("uk", "Ukrainian"),
    "urdu": ("ur", "Urdu"),
}
_LOCALE_DISPLAY: dict[str, str] = {}
for _locale_name, (_locale_code, _display_name) in _LANGUAGE_NAMES.items():
    _LOCALE_DISPLAY.setdefault(_locale_code, _display_name)

_LANGUAGE_ALTERNATION = "|".join(
    re.escape(name) for name in sorted(_LANGUAGE_NAMES, key=len, reverse=True)
)
_LANGUAGE_REQUEST_PATTERNS = (
    re.compile(
        rf"\b(?:can|could|would|will)\s+(?:you|we)\s+(?:please\s+)?"
        rf"(?:speak|talk|continue|switch|respond|reply)"
        rf"(?:\s+to\s+me)?\s+(?:back\s+)?(?:in\s+|to\s+)?"
        rf"(?P<language>{_LANGUAGE_ALTERNATION})\b"
    ),
    re.compile(
        rf"\bplease\s+(?:speak|talk|continue|switch|respond|reply)"
        rf"(?:\s+to\s+me)?\s+(?:back\s+)?(?:in\s+|to\s+)?"
        rf"(?P<language>{_LANGUAGE_ALTERNATION})\b"
    ),
    re.compile(
        rf"^(?:speak|talk|continue|switch|respond|reply)"
        rf"(?:\s+to\s+me)?\s+(?:back\s+)?(?:in\s+|to\s+)?"
        rf"(?P<language>{_LANGUAGE_ALTERNATION})\b"
    ),
    re.compile(
        rf"\b(?:i(?:'d| would) like to|i want to|let's)\s+"
        rf"(?:speak|talk|continue|switch)\s+(?:in\s+|to\s+)?"
        rf"(?P<language>{_LANGUAGE_ALTERNATION})\b"
    ),
    re.compile(rf"^(?P<language>{_LANGUAGE_ALTERNATION})[,\s]+please[.!?]*$"),
)


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


@dataclass(frozen=True)
class TimedWord:
    """One ASR word and its per-call or per-turn timestamps in seconds."""

    text: str
    start: float | None
    end: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", "" if self.text is None else str(self.text))
        object.__setattr__(self, "start", _finite_float(self.start))
        object.__setattr__(self, "end", _finite_float(self.end))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TimedWord:
        """Accept common ASR keys without coupling to an ASR vendor."""

        return cls(
            text=value.get("text", value.get("word", "")),
            start=value.get("start", value.get("start_time")),
            end=value.get("end", value.get("end_time")),
        )


@dataclass(frozen=True)
class TurnObservation:
    """Sanitized facts available for one caller turn."""

    text: str = ""
    timed_words: tuple[TimedWord, ...] = field(default_factory=tuple)
    start: float | None = None
    end: float | None = None
    reminder: bool = False
    interrupted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", "" if self.text is None else str(self.text))
        words: list[TimedWord] = []
        for value in self.timed_words or ():
            if isinstance(value, TimedWord):
                words.append(value)
            elif isinstance(value, Mapping):
                words.append(TimedWord.from_mapping(value))
        object.__setattr__(self, "timed_words", tuple(words))
        object.__setattr__(self, "start", _finite_float(self.start))
        object.__setattr__(self, "end", _finite_float(self.end))
        object.__setattr__(self, "reminder", bool(self.reminder))
        object.__setattr__(self, "interrupted", bool(self.interrupted))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TurnObservation:
        """Build an observation from a transport payload."""

        return cls(
            text=value.get("text", value.get("transcript", "")),
            timed_words=tuple(value.get("timed_words", value.get("words", ())) or ()),
            start=value.get("start", value.get("start_time")),
            end=value.get("end", value.get("end_time")),
            reminder=value.get("reminder", value.get("reminder_flag", False)),
            interrupted=value.get("interrupted", value.get("interrupted_flag", False)),
        )


@dataclass(frozen=True)
class BehaviorState:
    """Immutable per-call state consumed and returned by :func:`reduce_behavior`."""

    mode: BehaviorMode = BehaviorMode.STANDARD
    explicit_pace: PacePreference | None = None
    explicit_locale: str | None = None
    accessibility_preferences: tuple[AccessibilityPreference, ...] = field(
        default_factory=tuple
    )

    repair_streak: int = 0
    silence_count: int = 0
    interruption_timestamps: tuple[float, ...] = field(default_factory=tuple)
    interruption_count: int = 0
    boundary_strikes: int = 0
    explicit_handoff_reason: str | None = None

    speech_rate_samples: tuple[float, ...] = field(default_factory=tuple)
    inferred_pace: PacePreference | None = None
    turn_index: int = 0

    # Internal hysteresis/decay state.  These remain public for serialization
    # and deterministic replay.
    stable_turns: int = 0
    boundary_clean_turns: int = 0
    deescalation_hold: int = 0
    hurry_hold: int = 0
    interruption_pressure: int = 0
    last_utterance_fingerprint: str = ""

    # Terminal controls are sticky so a late packet cannot undo a handoff or
    # call termination decision.
    terminal_control: BehaviorControl | None = None
    terminal_reason: str | None = None


@dataclass(frozen=True)
class BehaviorDirective:
    """Safe response style, prompt fragment, and voice-platform controls."""

    mode: BehaviorMode
    prompt_instruction: str
    responsiveness: float
    interruption_sensitivity: float
    reminder_after_seconds: float
    direct_reply: str | None = None
    control: BehaviorControl = BehaviorControl.CONTINUE
    pace: PacePreference | None = None
    locale: str | None = None
    accessibility_preferences: tuple[AccessibilityPreference, ...] = field(
        default_factory=tuple
    )
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reminder_timing_seconds(self) -> float:
        """Readable alias for integrations that call this reminder timing."""

        return self.reminder_after_seconds

    @property
    def reminder_after_ms(self) -> int:
        return int(round(self.reminder_after_seconds * 1000))

    @property
    def retell_controls(self) -> dict[str, float | int]:
        """Return a fresh mapping suitable for a Retell configuration adapter."""

        return {
            "responsiveness": self.responsiveness,
            "interruption_sensitivity": self.interruption_sensitivity,
            "reminder_trigger_ms": self.reminder_after_ms,
        }

    def to_retell_controls(self) -> dict[str, float | int]:
        return self.retell_controls


@dataclass(frozen=True)
class BehaviorReduction:
    """The next immutable state and directive for one reduction."""

    state: BehaviorState
    directive: BehaviorDirective

    @property
    def next_state(self) -> BehaviorState:
        return self.state

    def __iter__(self) -> Iterator[BehaviorState | BehaviorDirective]:
        # Supports the familiar ``state, directive = reduce_behavior(...)``.
        yield self.state
        yield self.directive


_SLOW_PATTERNS = (
    re.compile(r"\b(?:please\s+)?(?:speak|talk|go)\s+(?:a\s+(?:bit|little)\s+)?slower\b"),
    re.compile(r"\bslow\s+down\b"),
    re.compile(r"\bnot\s+so\s+fast\b"),
    re.compile(r"\byou(?:'re|\s+are)\s+(?:speaking|talking|going)\s+too\s+fast\b"),
    re.compile(r"^(?:a\s+(?:bit|little)\s+)?slower(?:[,\s]+please)?[.!?]*$"),
)
_FAST_PATTERNS = (
    re.compile(r"\b(?:please\s+)?(?:speak|talk|go)\s+(?:a\s+(?:bit|little)\s+)?faster\b"),
    re.compile(r"\bspeed\s+up\b"),
    re.compile(r"\byou\s+can\s+(?:speak|talk|go)\s+faster\b"),
    re.compile(r"^(?:a\s+(?:bit|little)\s+)?faster(?:[,\s]+please)?[.!?]*$"),
)
_NORMAL_PACE_PATTERNS = (
    re.compile(r"\b(?:use|at|back\s+to)\s+(?:a\s+)?(?:normal|regular)\s+(?:speed|pace)\b"),
    re.compile(r"\byou\s+can\s+(?:speak|talk)\s+normally\b"),
)
_HURRY_PATTERNS = (
    re.compile(r"\bi(?:'m|\s+am)\s+in\s+a\s+(?:hurry|rush)\b"),
    re.compile(
        r"\b(?:please\s+)?(?:be\s+quick|hurry(?:\s+up)?|make\s+it\s+quick)\b"
    ),
    re.compile(r"\bquickly\s+please\b"),
    re.compile(r"\bi(?:'m|\s+am)\s+short\s+on\s+time\b"),
    re.compile(r"\bi\s+(?:only|just)\s+have\s+\d+\s+(?:seconds?|minutes?)\b"),
)

_REPEAT_PATTERNS = (
    re.compile(r"\b(?:please\s+)?repeat\s+(?:that|it|yourself|the\s+\w+)\b"),
    re.compile(
        r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
        r"repeat(?:\s+(?:that|it|yourself))?(?:[,\s]+please)?[.!?]*$"
    ),
    re.compile(r"\bsay\s+(?:that|it)\s+again\b"),
    re.compile(r"^again(?:[,\s]+please)?[.!?]*$"),
    re.compile(r"\bone\s+more\s+time\b"),
    re.compile(r"\bi\s+(?:didn't|did\s+not)\s+(?:catch|hear)\s+(?:that|you|it)\b"),
    re.compile(r"^come\s+again[.!?]*$"),
    re.compile(r"\bwhat\s+did\s+you\s+say\b"),
)
_CLEAR_REPEAT_PATTERNS = (
    re.compile(r"\b(?:do\s+not|don't|stop)\s+repeat(?:ing)?\b"),
    re.compile(r"\bno\s+need\s+to\s+repeat\b"),
)
_SPELL_PATTERNS = (
    re.compile(
        r"\b(?:please\s+)?spell\s+"
        r"(?:that|it|this|(?:the|your|my)\s+\w+|\w+\s+out)\b"
    ),
    re.compile(r"\bhow\s+do\s+you\s+spell\b"),
    re.compile(r"\bletter\s+by\s+letter\b"),
)
_CLEAR_SPELL_PATTERNS = (
    re.compile(r"\b(?:do\s+not|don't|stop)\s+spell(?:ing)?\b"),
    re.compile(r"\bno\s+need\s+to\s+spell\b"),
)
_PLAIN_LANGUAGE_PATTERNS = (
    re.compile(r"\b(?:use|speak\s+in)\s+(?:simple|plain)\s+(?:words|language)\b"),
    re.compile(r"\bkeep\s+it\s+simple\b"),
)
_CLEAR_PLAIN_LANGUAGE_PATTERNS = (
    re.compile(r"\bno\s+need\s+to\s+simplify\b"),
    re.compile(r"\b(?:use|go\s+back\s+to)\s+normal\s+wording\b"),
)
_ONE_AT_A_TIME_PATTERNS = (
    re.compile(r"\b(?:one\s+(?:thing|question)|a\s+single\s+question)\s+at\s+a\s+time\b"),
    re.compile(r"\bdon't\s+ask\s+(?:me\s+)?multiple\s+questions\b"),
)
_CLEAR_ONE_AT_A_TIME_PATTERNS = (
    re.compile(r"\byou\s+can\s+ask\s+more\s+than\s+one\s+question\b"),
    re.compile(r"\bno\s+need\s+to\s+go\s+one\s+at\s+a\s+time\b"),
)

_CONFUSION_PATTERNS = (
    re.compile(r"\bi\s+(?:do\s+not|don't)\s+understand\b"),
    re.compile(r"\bi(?:'m|\s+am)\s+confused\b"),
    re.compile(r"\bi(?:'m|\s+am)\s+not\s+following\b"),
    re.compile(r"\bwhat\s+do\s+you\s+mean\b"),
    re.compile(r"\bthat\s+(?:does\s+not|doesn't)\s+make\s+sense\b"),
    re.compile(r"\bcan\s+you\s+explain\s+that\b"),
)
_COMPLAINT_PATTERNS = (
    re.compile(r"\byou(?:'re|\s+are)\s+not\s+listening\b"),
    re.compile(r"\bi\s+already\s+told\s+you\b"),
    re.compile(r"\bthis\s+is\s+(?:really\s+)?(?:frustrating|ridiculous|not\s+helpful)\b"),
    re.compile(r"\bthis\s+(?:isn't|is\s+not)\s+helping\b"),
    re.compile(r"\bterrible\s+service\b"),
    re.compile(r"\byou\s+keep\s+(?:asking|repeating|interrupting)\b"),
    re.compile(r"\bstop\s+interrupting\b"),
    re.compile(r"\bthis\s+is\s+the\s+(?:third|fourth|fifth)\s+time\s+this\s+failed\b"),
)
_PERSONAL_IDENTITY_PATTERNS = (
    re.compile(r"\b(?:are|am)\s+(?:you|i)\s+(?:a\s+)?(?:real\s+person|human|ai|robot|bot)\b"),
    re.compile(r"\bwhat\s+(?:are|kind\s+of\s+bot\s+are)\s+you\b"),
)
_SAFE_HUMOR_PATTERNS = (
    re.compile(r"\b(?:are|how)\s+(?:the\s+)?fries\s+(?:famous|popular)\b"),
)

_HANDOFF_MANAGER_PATTERNS = (
    re.compile(
        r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?"
        r"(?:transfer|connect|put)\s+me\s+(?:to|through\s+to)\s+"
        r"(?:a|the|your)\s+(?:manager|supervisor)\b"
    ),
    re.compile(
        r"\bi\s+(?:want|need|would\s+like)\s+to\s+"
        r"(?:speak|talk)\s+(?:to|with)\s+(?:a|the|your)\s+"
        r"(?:manager|supervisor)\b"
    ),
    re.compile(
        r"\b(?:can|could|may)\s+i\s+(?:please\s+)?(?:speak|talk)\s+"
        r"(?:to|with)\s+(?:a|the|your)\s+(?:manager|supervisor)\b"
    ),
    re.compile(
        r"\b(?:let\s+me|please)\s+(?:speak|talk)\s+(?:to|with)\s+"
        r"(?:a|the|your)\s+(?:manager|supervisor)\b"
    ),
    re.compile(
        r"^(?:a|the|your)?\s*(?:manager|supervisor)(?:\s+please)?[.!?]*$"
    ),
)
_HANDOFF_HUMAN_PATTERNS = (
    re.compile(
        r"\b(?:transfer|connect|put)\s+me\s+(?:to|through\s+to)\s+"
        r"(?:a\s+)?(?:human|person|staff\s+member|employee|real\s+person)\b"
    ),
    re.compile(
        r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?"
        r"(?:transfer|connect|put)\s+me\s+(?:to|through\s+to)\s+"
        r"(?:a\s+)?(?:human|person|staff\s+member|employee|real\s+person)\b"
    ),
    re.compile(
        r"\bi\s+(?:want|need|would\s+like)\s+(?:to\s+"
        r"(?:speak|talk)\s+(?:to|with)\s+)?(?:a\s+)?"
        r"(?:human|real\s+person|staff\s+member|employee)\b"
    ),
    re.compile(
        r"\b(?:can|could|may)\s+i\s+(?:please\s+)?(?:speak|talk)\s+"
        r"(?:to|with)\s+(?:a\s+)?"
        r"(?:human|real\s+person|staff\s+member|employee)\b"
    ),
    re.compile(
        r"\b(?:let\s+me|please)\s+(?:speak|talk)\s+(?:to|with)\s+"
        r"(?:a\s+)?(?:human|real\s+person|staff\s+member|employee)\b"
    ),
    re.compile(
        r"^(?:a\s+)?(?:human|real\s+person|staff\s+member)"
        r"(?:\s+please)?[.!?]*$"
    ),
)

_ABUSE_PATTERNS = (
    re.compile(r"\b(?:fuck|screw)\s+you\b"),
    re.compile(r"\bshut\s+up\b"),
    re.compile(
        r"\byou(?:'re|\s+are)\s+(?:a\s+|an\s+)?"
        r"(?:idiot|moron|asshole|bitch|stupid|useless|worthless)\b"
    ),
    re.compile(r"\byou\s+(?:idiot|moron|asshole|bitch)\b"),
    re.compile(r"^(?:idiot|moron|asshole)[.!?]*$"),
    re.compile(r"\b(?:stupid|useless|worthless)\s+(?:bot|agent|thing)\b"),
)
_SEVERE_ABUSE_PATTERNS = (
    re.compile(r"\bi(?:'ll|\s+will)\s+(?:kill|hurt|attack)\s+you\b"),
    re.compile(r"\byou\s+should\s+(?:die|kill\s+yourself)\b"),
)
_OFF_TOPIC_PATTERNS = (
    re.compile(r"\btell\s+me\s+(?:a\s+)?(?:joke|story|poem)\b"),
    re.compile(r"\bwrite\s+(?:me\s+)?(?:a\s+)?(?:poem|essay|song)\b"),
    re.compile(r"\bwhat(?:'s|\s+is)\s+the\s+weather\b"),
    re.compile(r"\bwho\s+won\s+(?:the\s+)?(?:game|match|election)\b"),
    re.compile(r"\b(?:discuss|talk\s+about)\s+politics\b"),
    re.compile(r"\bhelp\s+me\s+with\s+my\s+homework\b"),
)

_SILENCE_MARKERS = {
    "silence",
    "no speech",
    "no audio",
    "caller silent",
}
_UNINTELLIGIBLE_MARKERS = {
    "inaudible",
    "unintelligible",
    "unclear audio",
    "speech not recognized",
    "no transcription",
    "audio unintelligible",
    "unintelligible audio",
}


def _matches(patterns: Sequence[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _normalize_text(text: str) -> str:
    # Bounding work protects the synchronous call path from oversized payloads.
    normalized = unicodedata.normalize("NFKC", text[:4096]).casefold()
    normalized = normalized.replace("’", "'").replace("`", "'")
    return re.sub(r"\s+", " ", normalized).strip()


def _marker_text(text: str) -> str:
    return re.sub(r"[\[\](){}<>.,!?;:_-]+", " ", text).strip()


def _is_silence(text: str) -> bool:
    return not text or _marker_text(text) in _SILENCE_MARKERS


def _is_unintelligible(text: str) -> bool:
    marker = _marker_text(text)
    if marker in _UNINTELLIGIBLE_MARKERS:
        return True
    tokens = marker.split()
    return bool(tokens) and all(token in {"inaudible", "unintelligible"} for token in tokens)


def _token_count(text: str) -> int:
    return sum(
        1
        for token in re.findall(r"[\w']+", text, flags=re.UNICODE)
        if any(character.isalnum() for character in token)
    )


def _utterance_fingerprint(text: str) -> str:
    canonical = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()
    if _token_count(canonical) < 3:
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _detect_pace(text: str) -> PacePreference | None:
    if _matches(_NORMAL_PACE_PATTERNS, text):
        return PacePreference.NORMAL
    if _matches(_SLOW_PATTERNS, text):
        return PacePreference.SLOWER
    if _matches(_FAST_PATTERNS, text):
        return PacePreference.FASTER
    return None


def _detect_locale(text: str) -> str | None:
    for pattern in _LANGUAGE_REQUEST_PATTERNS:
        match = pattern.search(text)
        if match:
            return _LANGUAGE_NAMES[match.group("language")][0]
    return None


def _detect_handoff(text: str) -> str | None:
    if _matches(_HANDOFF_MANAGER_PATTERNS, text):
        return "manager_requested"
    if _matches(_HANDOFF_HUMAN_PATTERNS, text):
        return "human_requested"
    return None


def _detect_boundary(text: str) -> tuple[str | None, bool]:
    if _matches(_SEVERE_ABUSE_PATTERNS, text):
        return "abuse", True
    if _matches(_ABUSE_PATTERNS, text):
        return "abuse", False
    if _matches(_OFF_TOPIC_PATTERNS, text):
        return "off_topic", False
    return None, False


def _speech_rate_sample(observation: TurnObservation, text: str) -> float | None:
    valid_words = [
        word
        for word in observation.timed_words
        if word.start is not None
        and word.end is not None
        and word.end >= word.start
        and _token_count(word.text) > 0
    ]

    if len(valid_words) >= MIN_TIMED_WORDS:
        word_count = len(valid_words)
        word_start = min(word.start for word in valid_words if word.start is not None)
        word_end = max(word.end for word in valid_words if word.end is not None)
        duration = word_end - word_start
        if (
            observation.start is not None
            and observation.end is not None
            and observation.end > observation.start
        ):
            duration = observation.end - observation.start
    else:
        word_count = _token_count(text)
        if (
            word_count < MIN_TIMED_WORDS
            or observation.start is None
            or observation.end is None
            or observation.end <= observation.start
        ):
            return None
        duration = observation.end - observation.start

    if duration < 0.8:
        return None
    words_per_minute = word_count * 60.0 / duration
    if not MIN_PLAUSIBLE_WPM <= words_per_minute <= MAX_PLAUSIBLE_WPM:
        return None
    return round(words_per_minute, 3)


def _update_inferred_pace(
    previous: PacePreference | None,
    samples: tuple[float, ...],
) -> PacePreference | None:
    if len(samples) < MIN_SPEECH_RATE_SAMPLES:
        return previous

    median_wpm = statistics.median(samples)
    if previous is PacePreference.SLOWER:
        if median_wpm >= FAST_WPM_ENTER:
            return PacePreference.FASTER
        return None if median_wpm >= SLOW_WPM_EXIT else PacePreference.SLOWER
    if previous is PacePreference.FASTER:
        if median_wpm <= SLOW_WPM_ENTER:
            return PacePreference.SLOWER
        return None if median_wpm <= FAST_WPM_EXIT else PacePreference.FASTER

    if median_wpm <= SLOW_WPM_ENTER:
        return PacePreference.SLOWER
    if median_wpm >= FAST_WPM_ENTER:
        return PacePreference.FASTER
    return None


def _event_timestamp(observation: TurnObservation, turn_index: int) -> float:
    if observation.end is not None:
        return observation.end
    timed_ends = [word.end for word in observation.timed_words if word.end is not None]
    if timed_ends:
        return max(timed_ends)
    if observation.start is not None:
        return observation.start
    # A logical timestamp keeps replay deterministic when the transport did not
    # provide a clock.
    return float(turn_index)


def _safe_int(value: Any, *, maximum: int = 1_000_000) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(converted, maximum))


def _coerce_enum(value: Any, enum_type: type[Enum]) -> Enum | None:
    if value is None:
        return None
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        return None


def _sanitize_state(state: BehaviorState) -> BehaviorState:
    pace = _coerce_enum(state.explicit_pace, PacePreference)
    inferred = _coerce_enum(state.inferred_pace, PacePreference)
    terminal = _coerce_enum(state.terminal_control, BehaviorControl)
    locale = state.explicit_locale if state.explicit_locale in _LOCALE_DISPLAY else None

    preferences: list[AccessibilityPreference] = []
    for value in state.accessibility_preferences or ():
        preference = _coerce_enum(value, AccessibilityPreference)
        if preference is not None and preference not in preferences:
            preferences.append(preference)

    timestamps = tuple(
        timestamp
        for timestamp in (
            _finite_float(value) for value in (state.interruption_timestamps or ())
        )
        if timestamp is not None
    )[-MAX_INTERRUPTION_TIMESTAMPS:]
    rate_samples = tuple(
        sample
        for sample in (
            _finite_float(value) for value in (state.speech_rate_samples or ())
        )
        if sample is not None and MIN_PLAUSIBLE_WPM <= sample <= MAX_PLAUSIBLE_WPM
    )[-MAX_SPEECH_RATE_SAMPLES:]
    handoff_reason = (
        state.explicit_handoff_reason
        if state.explicit_handoff_reason in {"human_requested", "manager_requested"}
        else None
    )
    terminal_reason = (
        state.terminal_reason
        if state.terminal_reason in {"handoff", "boundary", "silence"}
        else None
    )

    return replace(
        state,
        mode=_coerce_enum(state.mode, BehaviorMode) or BehaviorMode.STANDARD,
        explicit_pace=pace,
        explicit_locale=locale,
        accessibility_preferences=tuple(preferences),
        repair_streak=_safe_int(state.repair_streak, maximum=4),
        silence_count=_safe_int(state.silence_count, maximum=3),
        interruption_timestamps=timestamps,
        interruption_count=_safe_int(state.interruption_count),
        boundary_strikes=_safe_int(state.boundary_strikes, maximum=BOUNDARY_END_STRIKES),
        explicit_handoff_reason=handoff_reason,
        speech_rate_samples=rate_samples,
        inferred_pace=inferred,
        turn_index=_safe_int(state.turn_index),
        stable_turns=_safe_int(state.stable_turns, maximum=2),
        boundary_clean_turns=_safe_int(
            state.boundary_clean_turns, maximum=BOUNDARY_DECAY_TURNS
        ),
        deescalation_hold=_safe_int(
            state.deescalation_hold, maximum=DEESCALATION_HOLD_TURNS
        ),
        hurry_hold=_safe_int(state.hurry_hold, maximum=HURRY_HOLD_TURNS),
        interruption_pressure=_safe_int(state.interruption_pressure, maximum=6),
        last_utterance_fingerprint=(
            state.last_utterance_fingerprint
            if isinstance(state.last_utterance_fingerprint, str)
            else ""
        ),
        terminal_control=terminal,
        terminal_reason=terminal_reason,
    )


def _update_accessibility_preferences(
    previous: tuple[AccessibilityPreference, ...],
    *,
    repeat_requested: bool,
    repeat_cleared: bool,
    spell_requested: bool,
    spell_cleared: bool,
    plain_requested: bool,
    plain_cleared: bool,
    one_at_a_time_requested: bool,
    one_at_a_time_cleared: bool,
) -> tuple[AccessibilityPreference, ...]:
    selected = set(previous)
    changes = (
        (
            AccessibilityPreference.REPEAT_KEY_DETAILS,
            repeat_requested,
            repeat_cleared,
        ),
        (
            AccessibilityPreference.SPELL_KEY_DETAILS,
            spell_requested,
            spell_cleared,
        ),
        (AccessibilityPreference.PLAIN_LANGUAGE, plain_requested, plain_cleared),
        (
            AccessibilityPreference.ONE_THING_AT_A_TIME,
            one_at_a_time_requested,
            one_at_a_time_cleared,
        ),
    )
    for preference, requested, cleared in changes:
        if cleared:
            selected.discard(preference)
        elif requested:
            selected.add(preference)
    return tuple(preference for preference in AccessibilityPreference if preference in selected)


def _choose_mode(
    *,
    pace: PacePreference | None,
    inferred_pace: PacePreference | None,
    accessibility: tuple[AccessibilityPreference, ...],
    repair_streak: int,
    guided_now: bool,
    deescalation_hold: int,
    hurry_hold: int,
    interruption_pressure: int,
) -> BehaviorMode:
    if deescalation_hold > 0:
        return BehaviorMode.DEESCALATING
    if (
        pace is PacePreference.SLOWER
        or accessibility
        or guided_now
        or repair_streak >= 2
        or (pace is None and inferred_pace is PacePreference.SLOWER)
    ):
        return BehaviorMode.GUIDED
    if (
        pace is PacePreference.FASTER
        or hurry_hold > 0
        or interruption_pressure >= 3
        or (pace is None and inferred_pace is PacePreference.FASTER)
    ):
        return BehaviorMode.CONCISE
    return BehaviorMode.STANDARD


def _direct_reply(
    *,
    control: BehaviorControl,
    terminal_reason: str | None,
    handoff_reason: str | None,
    silence_count: int,
    boundary_kind: str | None,
    boundary_strikes: int,
    unintelligible: bool,
    personal_identity: bool,
    safe_humor: bool,
) -> str | None:
    if handoff_reason is not None:
        destination = resolve_handoff_destination(handoff_reason)
        if destination["can_transfer"]:
            return "Of course. I'll connect you with a staff member now."
        if destination["owner"] == "manager_callback":
            return (
                "A manager isn't available by transfer now, but I can take a message "
                "and callback details for the manager."
            )
        return (
            "I can't transfer the call right now, but I can take a message and "
            "callback details for the restaurant team."
        )
    if control is BehaviorControl.END_CALL:
        if terminal_reason == "silence":
            return "I haven't heard you, so I'll end the call for now. Please call back anytime."
        return "I'm ending the call now."
    if personal_identity:
        return (
            f"I'm {settings.ai_agent_name}, Harbor & Hearth's virtual host. "
            "I can help with a reservation, an order, or restaurant questions."
        )
    if safe_humor:
        return (
            "The fries have a loyal following, but I try not to let it go to their heads. "
            "I can check whether they're available right now."
        )

    if silence_count == 1:
        return "Take your time—I'm here when you're ready."
    if silence_count == 2:
        return "Are you still there? I can help with a booking, order, or restaurant question."
    if boundary_kind == "abuse":
        if boundary_strikes == 1:
            return "I want to help, but please keep the conversation respectful."
        return "I can help with the restaurant, but if the abuse continues I'll end the call."
    if boundary_kind == "off_topic":
        if boundary_strikes == 1:
            return (
                "I can help with restaurant bookings, orders, menu questions, and policies. "
                "What do you need?"
            )
        return (
            "I need to keep this call to restaurant requests. "
            "Do you need help with a booking or order?"
        )
    if unintelligible:
        return "Sorry, I didn't catch that. Could you say it one more time?"
    return None


def _prompt_instruction(
    *,
    mode: BehaviorMode,
    pace: PacePreference | None,
    locale: str | None,
    accessibility: tuple[AccessibilityPreference, ...],
    repeat_now: bool,
    spell_now: bool,
    confusion: bool,
    recent_interruptions: bool,
    control: BehaviorControl,
) -> str:
    mode_text = {
        BehaviorMode.STANDARD: "Talk like a host: do the latest request, then one useful question only if needed.",
        BehaviorMode.CONCISE: "Be concise, but do not ignore a change they just asked for.",
        BehaviorMode.GUIDED: (
            "Guide one step at a time, use plain wording, and confirm key details."
        ),
        BehaviorMode.DEESCALATING: (
            "Stay neutral, acknowledge the concern briefly, and offer one concrete next step."
        ),
    }
    instructions = [mode_text[mode]]

    if pace is PacePreference.SLOWER:
        instructions.append("Speak more slowly, using short clauses and clear pauses.")
    elif pace is PacePreference.FASTER:
        instructions.append("Keep delivery brisk without skipping required confirmations.")
    if locale in _LOCALE_DISPLAY:
        instructions.append(f"Respond in {_LOCALE_DISPLAY[locale]} ({locale}).")

    if AccessibilityPreference.REPEAT_KEY_DETAILS in accessibility:
        instructions.append("Repeat key details once when confirming them.")
    if AccessibilityPreference.SPELL_KEY_DETAILS in accessibility:
        instructions.append("Spell names, references, and requested key details clearly.")
    if AccessibilityPreference.PLAIN_LANGUAGE in accessibility:
        instructions.append("Avoid jargon and use plain language.")
    if AccessibilityPreference.ONE_THING_AT_A_TIME in accessibility:
        instructions.append("Never ask more than one question at a time.")

    if repeat_now:
        instructions.append("Repeat or rephrase the immediately preceding answer now.")
    if spell_now:
        instructions.append("Spell the requested item now; clarify only if the item is ambiguous.")
    if confusion:
        instructions.append("Rephrase the last point instead of repeating it verbatim.")
    if recent_interruptions:
        instructions.append("Yield immediately on barge-in and keep the next turn especially short.")
    if control is not BehaviorControl.CONTINUE:
        instructions.append("Use the direct reply and control action; do not generate another answer.")

    instructions.append(
        "Do not infer personal or medical traits from the caller's voice or wording."
    )
    return " ".join(instructions)


def _retell_values(
    mode: BehaviorMode,
    pace: PacePreference | None,
    *,
    recent_interruptions: bool,
    silence_count: int,
    terminal: bool,
) -> tuple[float, float, float]:
    responsiveness, interruption_sensitivity, reminder_after = {
        BehaviorMode.STANDARD: (0.70, 0.65, 8.0),
        BehaviorMode.CONCISE: (0.85, 0.80, 6.0),
        BehaviorMode.GUIDED: (0.50, 0.60, 11.0),
        BehaviorMode.DEESCALATING: (0.45, 0.80, 10.0),
    }[mode]

    if pace is PacePreference.SLOWER:
        responsiveness = min(responsiveness, 0.45)
        reminder_after = max(reminder_after, 12.0)
    elif pace is PacePreference.FASTER:
        responsiveness = max(responsiveness, 0.88)
        reminder_after = min(reminder_after, 6.0)
    if recent_interruptions:
        responsiveness = max(responsiveness, 0.85)
        interruption_sensitivity = max(interruption_sensitivity, 0.90)
    if silence_count == 1:
        reminder_after = min(reminder_after, 6.0)
    elif silence_count >= 2:
        reminder_after = 4.0
    if terminal:
        reminder_after = 0.0

    return (
        round(max(0.0, min(responsiveness, 1.0)), 2),
        round(max(0.0, min(interruption_sensitivity, 1.0)), 2),
        round(max(0.0, reminder_after), 1),
    )


def _deduplicate(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def reduce_behavior(
    state: BehaviorState | None,
    observation: TurnObservation | Mapping[str, Any],
) -> BehaviorReduction:
    """Reduce one caller observation into immutable state and a safe directive.

    The function has no wall-clock reads, I/O, external calls, or mutable global
    state.  If timestamps are absent, interruption events receive the logical
    turn index, preserving deterministic replay.
    """

    if state is None:
        state = BehaviorState()
    if not isinstance(state, BehaviorState):
        raise TypeError("state must be BehaviorState or None")
    state = _sanitize_state(state)

    if isinstance(observation, Mapping):
        observation = TurnObservation.from_mapping(observation)
    if not isinstance(observation, TurnObservation):
        raise TypeError("observation must be TurnObservation or a mapping")

    text = _normalize_text(observation.text)
    silent = _is_silence(text)
    unintelligible = not silent and _is_unintelligible(text)
    meaningful = not silent and not unintelligible
    turn_index = state.turn_index + 1

    pace_request = _detect_pace(text) if meaningful else None
    locale_request = _detect_locale(text) if meaningful else None
    hurry = meaningful and _matches(_HURRY_PATTERNS, text)
    handoff_request = _detect_handoff(text) if meaningful else None
    boundary_kind, severe_boundary = (
        _detect_boundary(text) if meaningful else (None, False)
    )

    repeat_cleared = meaningful and _matches(_CLEAR_REPEAT_PATTERNS, text)
    spell_cleared = meaningful and _matches(_CLEAR_SPELL_PATTERNS, text)
    plain_cleared = meaningful and _matches(_CLEAR_PLAIN_LANGUAGE_PATTERNS, text)
    one_at_a_time_cleared = meaningful and _matches(
        _CLEAR_ONE_AT_A_TIME_PATTERNS, text
    )
    repeat_requested = (
        meaningful
        and not repeat_cleared
        and _matches(_REPEAT_PATTERNS, text)
    )
    spell_requested = (
        meaningful and not spell_cleared and _matches(_SPELL_PATTERNS, text)
    )
    plain_requested = (
        meaningful
        and not plain_cleared
        and _matches(_PLAIN_LANGUAGE_PATTERNS, text)
    )
    one_at_a_time_requested = (
        meaningful
        and not one_at_a_time_cleared
        and _matches(_ONE_AT_A_TIME_PATTERNS, text)
    )
    confusion = meaningful and _matches(_CONFUSION_PATTERNS, text)
    complaint = meaningful and _matches(_COMPLAINT_PATTERNS, text)
    personal_identity = meaningful and _matches(_PERSONAL_IDENTITY_PATTERNS, text)
    caller_tokens = text_tokens(text)
    try:
        forbidden_humor_contexts = [
            text_tokens(context)
            for context in get_restaurant_knowledge().raw["conversation_style"][
                "light_humor"
            ]["forbidden_contexts"]
        ]
        unsafe_humor_context = any(
            context_tokens <= caller_tokens
            for context_tokens in forbidden_humor_contexts
        )
    except (KnowledgeFixtureError, KeyError, TypeError):
        unsafe_humor_context = True
    safe_humor = (
        meaningful
        and not complaint
        and not unsafe_humor_context
        and _matches(_SAFE_HUMOR_PATTERNS, text)
    )

    fingerprint = _utterance_fingerprint(text) if meaningful else ""
    repetition_proxy = bool(
        fingerprint
        and state.last_utterance_fingerprint
        and fingerprint == state.last_utterance_fingerprint
    )

    explicit_pace = pace_request if pace_request is not None else state.explicit_pace
    explicit_locale = (
        locale_request if locale_request is not None else state.explicit_locale
    )
    accessibility = _update_accessibility_preferences(
        state.accessibility_preferences,
        repeat_requested=repeat_requested,
        repeat_cleared=repeat_cleared,
        spell_requested=spell_requested,
        spell_cleared=spell_cleared,
        plain_requested=plain_requested,
        plain_cleared=plain_cleared,
        one_at_a_time_requested=one_at_a_time_requested,
        one_at_a_time_cleared=one_at_a_time_cleared,
    )

    rate_sample = _speech_rate_sample(observation, text) if meaningful else None
    rate_samples = state.speech_rate_samples
    if rate_sample is not None:
        rate_samples = (rate_samples + (rate_sample,))[-MAX_SPEECH_RATE_SAMPLES:]
    inferred_pace = _update_inferred_pace(state.inferred_pace, rate_samples)

    interruption_count = state.interruption_count
    interruption_timestamps = state.interruption_timestamps
    interruption_pressure = state.interruption_pressure
    if observation.interrupted:
        interruption_count += 1
        interruption_timestamps = (
            interruption_timestamps + (_event_timestamp(observation, turn_index),)
        )[-MAX_INTERRUPTION_TIMESTAMPS:]
        interruption_pressure = min(6, interruption_pressure + 2)
    else:
        interruption_pressure = max(0, interruption_pressure - 1)
    recent_interruptions = interruption_pressure >= 3

    silence_count = min(3, state.silence_count + 1) if silent else 0

    repair_signal = bool(
        unintelligible
        or repeat_requested
        or confusion
        or complaint
        or repetition_proxy
    )
    repair_streak = state.repair_streak
    stable_turns = state.stable_turns
    if repair_signal:
        repair_streak = min(4, repair_streak + 1)
        stable_turns = 0
    elif meaningful and boundary_kind is None and handoff_request is None:
        stable_turns += 1
        if stable_turns >= 2:
            repair_streak = max(0, repair_streak - 1)
            stable_turns = 0

    deescalation_hold = state.deescalation_hold
    if complaint or boundary_kind is not None:
        deescalation_hold = DEESCALATION_HOLD_TURNS
    elif meaningful and handoff_request is None:
        deescalation_hold = max(0, deescalation_hold - 1)

    if hurry:
        hurry_hold = HURRY_HOLD_TURNS
    elif meaningful:
        hurry_hold = max(0, state.hurry_hold - 1)
    else:
        hurry_hold = state.hurry_hold

    boundary_strikes = state.boundary_strikes
    boundary_clean_turns = state.boundary_clean_turns
    if boundary_kind is not None:
        boundary_strikes = (
            BOUNDARY_END_STRIKES
            if severe_boundary
            else min(BOUNDARY_END_STRIKES, boundary_strikes + 1)
        )
        boundary_clean_turns = 0
    elif (
        meaningful
        and not complaint
        and handoff_request is None
        and boundary_strikes > 0
    ):
        boundary_clean_turns += 1
        if boundary_clean_turns >= BOUNDARY_DECAY_TURNS:
            boundary_strikes = max(0, boundary_strikes - 1)
            boundary_clean_turns = 0
    elif boundary_strikes == 0:
        boundary_clean_turns = 0

    handoff_reason = handoff_request or (
        state.explicit_handoff_reason
        if state.terminal_control is BehaviorControl.HANDOFF
        else None
    )
    terminal_control = state.terminal_control
    terminal_reason = state.terminal_reason
    destination = (
        resolve_handoff_destination(handoff_reason) if handoff_reason is not None else None
    )
    if terminal_control is BehaviorControl.HANDOFF and not (
        destination and destination["can_transfer"]
    ):
        terminal_control = None
        terminal_reason = None
    if (
        terminal_control is None
        and handoff_request is not None
        and destination
        and destination["can_transfer"]
    ):
        terminal_control = BehaviorControl.HANDOFF
        terminal_reason = "handoff"
    elif (
        terminal_control is None
        and boundary_kind is not None
        and boundary_strikes >= BOUNDARY_END_STRIKES
    ):
        terminal_control = BehaviorControl.END_CALL
        terminal_reason = "boundary"
    elif terminal_control is None and silence_count >= 3:
        terminal_control = BehaviorControl.END_CALL
        terminal_reason = "silence"

    control = terminal_control or BehaviorControl.CONTINUE
    mode = _choose_mode(
        pace=explicit_pace,
        inferred_pace=inferred_pace,
        accessibility=accessibility,
        repair_streak=repair_streak,
        guided_now=bool(confusion or repetition_proxy),
        deescalation_hold=deescalation_hold,
        hurry_hold=hurry_hold,
        interruption_pressure=interruption_pressure,
    )

    next_state = BehaviorState(
        mode=mode,
        explicit_pace=explicit_pace,
        explicit_locale=explicit_locale,
        accessibility_preferences=accessibility,
        repair_streak=repair_streak,
        silence_count=silence_count,
        interruption_timestamps=interruption_timestamps,
        interruption_count=interruption_count,
        boundary_strikes=boundary_strikes,
        explicit_handoff_reason=(
            handoff_reason if terminal_control is BehaviorControl.HANDOFF else None
        ),
        speech_rate_samples=rate_samples,
        inferred_pace=inferred_pace,
        turn_index=turn_index,
        stable_turns=stable_turns,
        boundary_clean_turns=boundary_clean_turns,
        deescalation_hold=deescalation_hold,
        hurry_hold=hurry_hold,
        interruption_pressure=interruption_pressure,
        last_utterance_fingerprint=(
            fingerprint if meaningful else state.last_utterance_fingerprint
        ),
        terminal_control=terminal_control,
        terminal_reason=terminal_reason,
    )

    reasons: list[str] = []
    if pace_request is not None:
        reasons.append("explicit_pace")
    if locale_request is not None:
        reasons.append("explicit_locale")
    if repeat_requested:
        reasons.append("explicit_repetition")
    if spell_requested:
        reasons.append("explicit_spelling")
    if plain_requested or one_at_a_time_requested:
        reasons.append("explicit_presentation")
    if confusion:
        reasons.append("explicit_confusion")
    if complaint:
        reasons.append("explicit_complaint")
    if unintelligible:
        reasons.append("unintelligible_audio")
    if repetition_proxy:
        reasons.append("repetition_proxy")
    if rate_sample is not None and inferred_pace is not None:
        reasons.append("speech_rate_proxy")
    if hurry:
        reasons.append("explicit_hurry")
    if observation.interrupted:
        reasons.append("interrupted")
    if recent_interruptions:
        reasons.append("recent_interruptions")
    if silent:
        reasons.append(f"silence_{silence_count}")
    if boundary_kind is not None:
        reasons.append(f"{boundary_kind}_boundary")
    if handoff_request is not None:
        reasons.append("explicit_handoff")

    effective_pace = (
        None
        if explicit_pace is PacePreference.NORMAL
        else explicit_pace or inferred_pace
    )
    direct_reply = _direct_reply(
        control=control,
        terminal_reason=terminal_reason,
        handoff_reason=handoff_reason,
        silence_count=silence_count,
        boundary_kind=boundary_kind,
        boundary_strikes=boundary_strikes,
        unintelligible=unintelligible,
        personal_identity=personal_identity,
        safe_humor=safe_humor,
    )
    prompt_instruction = _prompt_instruction(
        mode=mode,
        pace=effective_pace,
        locale=explicit_locale,
        accessibility=accessibility,
        repeat_now=repeat_requested,
        spell_now=spell_requested,
        confusion=confusion,
        recent_interruptions=recent_interruptions,
        control=control,
    )
    responsiveness, interruption_sensitivity, reminder_after = _retell_values(
        mode,
        effective_pace,
        recent_interruptions=recent_interruptions,
        silence_count=silence_count,
        terminal=control is not BehaviorControl.CONTINUE,
    )
    directive = BehaviorDirective(
        mode=mode,
        prompt_instruction=prompt_instruction,
        responsiveness=responsiveness,
        interruption_sensitivity=interruption_sensitivity,
        reminder_after_seconds=reminder_after,
        direct_reply=direct_reply,
        control=control,
        pace=effective_pace,
        locale=explicit_locale,
        accessibility_preferences=accessibility,
        reasons=_deduplicate(reasons),
    )
    return BehaviorReduction(state=next_state, directive=directive)


def initial_behavior_state() -> BehaviorState:
    """Return a fresh state for a new call."""

    return BehaviorState()


# Descriptive alias for integrations that prefer a domain-specific verb.
reduce_caller_behavior = reduce_behavior
