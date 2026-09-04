from __future__ import annotations

import hashlib
import hmac

from app.security import (
    canonical_request_hash,
    constant_time_equal,
    is_e164,
    normalize_caller_phone,
    verify_retell_webhook_signature,
)


def _signature(body: bytes, key: str, timestamp: int) -> str:
    digest = hmac.new(
        key.encode(),
        body + str(timestamp).encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"v={timestamp},d={digest}"


def test_retell_signature_accepts_current_valid_hmac() -> None:
    body = b'{"event":"call_ended"}'
    key = "retell-secret"
    now = 1_800_000_000_000
    assert verify_retell_webhook_signature(
        body,
        key,
        _signature(body, key, now),
        now_ms=now,
    )


def test_retell_signature_rejects_tampering_and_replay() -> None:
    body = b'{"event":"call_ended"}'
    key = "retell-secret"
    now = 1_800_000_000_000
    signature = _signature(body, key, now)
    assert not verify_retell_webhook_signature(
        body + b" ",
        key,
        signature,
        now_ms=now,
    )
    assert not verify_retell_webhook_signature(
        body,
        key,
        signature,
        now_ms=now + 300_001,
    )
    assert not verify_retell_webhook_signature(
        body,
        key,
        "not-a-signature",
        now_ms=now,
    )


def test_constant_time_secret_comparison_fails_closed() -> None:
    assert constant_time_equal("correct", "correct")
    assert not constant_time_equal("wrong", "correct")
    assert not constant_time_equal("", "correct")
    assert not constant_time_equal(None, "correct")
    assert not constant_time_equal("correct", "")


def test_canonical_hash_ignores_mapping_order_not_values() -> None:
    first = canonical_request_hash({"b": 2, "a": 1})
    second = canonical_request_hash({"a": 1, "b": 2})
    changed = canonical_request_hash({"a": 1, "b": 3})
    assert first == second
    assert first != changed


def test_e164_validation() -> None:
    assert is_e164("+14155550123")
    assert is_e164("+442071838750")
    assert not is_e164("415-555-0123")
    assert not is_e164("+012345678")


def test_normalize_caller_phone_accepts_spoken_and_local_forms() -> None:
    expected = "+14155550123"
    assert normalize_caller_phone("415-555-0123") == expected
    assert normalize_caller_phone("(415) 555 0123") == expected
    assert normalize_caller_phone("1 415 555 0123") == expected
    assert normalize_caller_phone("+1 415 555 0123") == expected
    assert (
        normalize_caller_phone("four one five five five five zero one two three")
        == expected
    )
    assert (
        normalize_caller_phone("area code four one five, five five five, oh one two three")
        == expected
    )
    assert normalize_caller_phone("four one five five double five oh one two three") == expected
    assert normalize_caller_phone("03098121804") == "+923098121804"
    assert normalize_caller_phone("0309-8121804") == "+923098121804"
    assert normalize_caller_phone("+92 309 8121804") == "+923098121804"
    assert normalize_caller_phone("555-0123") is None
    assert normalize_caller_phone("") is None
