"""Security helpers shared by voice transports, tools, and webhooks."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections.abc import Mapping
from typing import Any


_RETELL_SIGNATURE_RE = re.compile(r"^v=(\d+),d=([0-9a-fA-F]{64})$")
_E164_RE = re.compile(r"^\+[1-9]\d{7,14}$")
_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_PHONE_SKIP_WORDS = {
    "plus",
    "and",
    "area",
    "code",
    "number",
    "phone",
    "telephone",
    "callback",
    "my",
    "is",
    "the",
    "country",
    "dash",
    "hyphen",
    "dot",
    "point",
}
_PHONE_TOKEN_RE = re.compile(r"[a-z]+|\d+|\+")


def constant_time_equal(provided: str | None, expected: str) -> bool:
    """Compare two secrets without leaking a useful length/timing signal."""
    if not provided or not expected:
        return False
    provided_bytes = provided.encode("utf-8")
    expected_bytes = expected.encode("utf-8")
    return hmac.compare_digest(
        hashlib.sha256(provided_bytes).digest(),
        hashlib.sha256(expected_bytes).digest(),
    )


def is_e164(value: str) -> bool:
    return bool(_E164_RE.fullmatch(value.strip()))


def _as_e164(digits: str) -> str | None:
    if not digits or digits.startswith("0"):
        return None
    candidate = f"+{digits}"
    return candidate if is_e164(candidate) else None


def _from_national_digits(number: str, default_country_code: str) -> str | None:
    """Map everyday local numbers (including a leading 0) to E.164."""
    code = default_country_code.strip().lstrip("+") or "1"

    # Pakistan mobiles: 03XX-XXXXXXX (11 digits) or 3XXXXXXXXX.
    if len(number) == 11 and number.startswith("03"):
        return _as_e164("92" + number[1:])
    if len(number) == 10 and number.startswith("3") and code == "92":
        return _as_e164("92" + number)

    # UK mobiles: 07XXX XXXXXX.
    if len(number) == 11 and number.startswith("07"):
        return _as_e164("44" + number[1:])

    # NANP: 10 digits, or 1 + 10 digits.
    if len(number) == 10 and number[0] not in {"0", "1"}:
        return _as_e164("1" + number)
    if len(number) == 11 and number.startswith("1") and number[1] not in {"0", "1"}:
        return _as_e164(number)

    # Trunk prefix 0 + national number, e.g. 03098121804.
    if number.startswith("0") and 8 <= len(number) <= 12:
        national = number.lstrip("0")
        if 8 <= len(national) <= 14:
            return _as_e164(code + national)

    if 8 <= len(number) <= 15 and not number.startswith("0"):
        if number.startswith(code):
            return _as_e164(number)
        return _as_e164(number)
    return None


def normalize_caller_phone(value: str, *, default_country_code: str = "1") -> str | None:
    """Turn spoken or typed caller numbers into E.164 for storage.

    Callers usually say a local number such as ``03098121804`` or ``415 555 0123``,
    not ``+14155550123``. Staff/config numbers still use ``is_e164``.
    """
    raw = value.strip()
    if not raw:
        return None
    if is_e164(raw):
        return raw

    tokens = _PHONE_TOKEN_RE.findall(raw.casefold())
    digits: list[str] = []
    had_plus = False
    pending_repeat = 1
    for token in tokens:
        if token == "+":
            had_plus = True
            continue
        if token in {"double", "twice"}:
            pending_repeat = 2
            continue
        if token == "triple":
            pending_repeat = 3
            continue
        if token in _PHONE_SKIP_WORDS:
            pending_repeat = 1
            continue
        if token.isdigit():
            digits.extend(token * pending_repeat)
            pending_repeat = 1
            continue
        digit = _DIGIT_WORDS.get(token)
        if digit is None:
            pending_repeat = 1
            continue
        digits.extend(digit * pending_repeat)
        pending_repeat = 1

    number = "".join(digits)
    if not number:
        return None
    if had_plus:
        return _as_e164(number.lstrip("0"))
    return _from_national_digits(number, default_country_code)


def verify_retell_webhook_signature(
    raw_body: bytes,
    api_key: str,
    signature: str | None,
    *,
    now_ms: int | None = None,
    max_age_ms: int = 5 * 60 * 1000,
) -> bool:
    """Verify Retell's timestamped HMAC and reject replayed requests.

    Retell signs ``raw_body + timestamp`` and sends
    ``X-Retell-Signature: v=<unix-ms>,d=<sha256-hex>``.
    """
    if not api_key or not signature:
        return False
    match = _RETELL_SIGNATURE_RE.fullmatch(signature.strip())
    if not match:
        return False

    timestamp_text, provided_digest = match.groups()
    timestamp = int(timestamp_text)
    current = int(time.time() * 1000) if now_ms is None else now_ms
    if abs(current - timestamp) > max_age_ms:
        return False

    signed_payload = raw_body + timestamp_text.encode("ascii")
    expected_digest = hmac.new(
        api_key.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected_digest, provided_digest.lower())


def canonical_request_hash(payload: Mapping[str, Any]) -> str:
    """Stable request digest used to detect idempotency-key reuse."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def redact_phone(value: str) -> str:
    """Keep enough of a phone number for logs without exposing the full value."""
    value = value.strip()
    if len(value) <= 4:
        return "****"
    return f"{value[:2]}***{value[-2:]}"
