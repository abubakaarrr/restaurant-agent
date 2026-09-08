"""Lightweight module for cross-cutting call-lifecycle flags.

Kept separate from runner.py to avoid circular imports:
  db.py → call_flags  (tools set flags)
  runner.py → call_flags  (runner reads flags)
"""

from __future__ import annotations

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


_control_flags: dict[str, CallControl] = {}


def request_end_call(session_id: str) -> None:
    """Signal that the call should end after the current turn."""
    if session_id:
        _control_flags[session_id] = CallControl(action="end")


def request_transfer(
    session_id: str,
    reason: HandoffReason,
    transfer_number: str,
) -> None:
    """Carry one already-resolved server destination to the transport boundary."""
    destination = str(transfer_number or "").strip()
    if session_id and destination:
        _control_flags[session_id] = CallControl(
            action="transfer",
            reason=reason,
            transfer_number=destination,
        )


def consume_call_control(session_id: str) -> CallControl | None:
    return _control_flags.pop(session_id, None)


def clear_call_control(session_id: str) -> None:
    _control_flags.pop(session_id, None)


def consume_end_call(session_id: str) -> bool:
    """Return True (and clear the flag) if end_call was requested for this session."""
    control = consume_call_control(session_id)
    return bool(control and control.action == "end")
