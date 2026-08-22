"""Authorize Retell custom-LLM WebSocket connections."""

from __future__ import annotations

import time
from threading import Lock

from app.config import settings
from app.security import constant_time_equal

_MINT_TTL_SECONDS = 15 * 60
_lock = Lock()
_minted_calls: dict[str, float] = {}


def remember_retell_call(call_id: str) -> None:
    call_id = (call_id or "").strip()
    if not call_id:
        return
    now = time.monotonic()
    with _lock:
        cutoff = now - _MINT_TTL_SECONDS
        expired = [key for key, created in _minted_calls.items() if created < cutoff]
        for key in expired:
            _minted_calls.pop(key, None)
        _minted_calls[call_id] = now


def is_recent_retell_call(call_id: str) -> bool:
    call_id = (call_id or "").strip()
    if not call_id:
        return False
    with _lock:
        created = _minted_calls.get(call_id)
        if created is None:
            return False
        if time.monotonic() - created > _MINT_TTL_SECONDS:
            _minted_calls.pop(call_id, None)
            return False
        return True


def retell_ws_authorized(call_id: str, provided_token: str | None) -> bool:
    if constant_time_equal(provided_token, settings.retell_ws_token):
        return True
    return is_recent_retell_call(call_id)
