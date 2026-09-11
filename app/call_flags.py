"""Lightweight module for cross-cutting call-lifecycle flags.

Kept separate from runner.py to avoid circular imports:
  db.py → call_flags  (tools set flags)
  runner.py → call_flags  (runner reads flags)
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal


HandoffReason = Literal[
    "human_requested",
    "manager_or_complaint",
    "severe_allergy",
    "unsupported_language",
    "repeated_verification_failure",
    "payment_or_refund",
    "system_outage",
    "safety",
]


@dataclass(frozen=True)
class CallControl:
    action: Literal["end", "transfer"]
    reason: str = ""
    transfer_number: str = ""


_current_control_scope: ContextVar[str] = ContextVar(
    "current_call_control_scope", default=""
)
_control_flags: dict[tuple[str, str], CallControl] = {}


def set_call_control_scope(scope: str):
    return _current_control_scope.set(str(scope or ""))


def reset_call_control_scope(token) -> None:
    _current_control_scope.reset(token)


def _control_key(session_id: str, scope: str | None = None) -> tuple[str, str]:
    active_scope = _current_control_scope.get() if scope is None else str(scope or "")
    return session_id, active_scope


def request_end_call(session_id: str) -> None:
    """Signal that the call should end after the current turn."""
    if session_id:
        _control_flags[_control_key(session_id)] = CallControl(action="end")


def request_transfer(
    session_id: str,
    reason: HandoffReason,
    transfer_number: str,
) -> None:
    """Carry one already-resolved server destination to the transport boundary."""
    destination = str(transfer_number or "").strip()
    if session_id and destination:
        _control_flags[_control_key(session_id)] = CallControl(
            action="transfer",
            reason=reason,
            transfer_number=destination,
        )


def consume_call_control(
    session_id: str, scope: str | None = None
) -> CallControl | None:
    return _control_flags.pop(_control_key(session_id, scope), None)


def clear_call_control(session_id: str, scope: str | None = None) -> None:
    if scope is not None:
        _control_flags.pop(_control_key(session_id, scope), None)
        return
    for key in tuple(_control_flags):
        if key[0] == session_id:
            _control_flags.pop(key, None)


def consume_end_call(session_id: str) -> bool:
    """Return True (and clear the flag) if end_call was requested for this session."""
    control = consume_call_control(session_id)
    return bool(control and control.action == "end")
