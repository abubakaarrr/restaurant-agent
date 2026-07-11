"""Lightweight module for cross-cutting call-lifecycle flags.

Kept separate from runner.py to avoid circular imports:
  db.py → call_flags  (tools set flags)
  runner.py → call_flags  (runner reads flags)
"""

from __future__ import annotations

_end_call_flags: dict[str, bool] = {}


def request_end_call(session_id: str) -> None:
    """Signal that the call should end after the current turn."""
    _end_call_flags[session_id] = True


def consume_end_call(session_id: str) -> bool:
    """Return True (and clear the flag) if end_call was requested for this session."""
    return _end_call_flags.pop(session_id, False)
