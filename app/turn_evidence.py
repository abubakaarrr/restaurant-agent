"""Same-turn evidence for availability and order-readback gates.

The language model is not trusted to remember a prior tool result. Each
caller turn (action scope) records fresh tool facts. Speech that asserts
availability or confirms money without matching evidence is logged; write
tools can refuse when a required read did not happen in this turn.

No-ops when no turn is active so the HTTP tool API stays usable outside
the custom-LLM /chat loop.
"""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_turn: ContextVar["TurnEvidence | None"] = ContextVar("turn_evidence", default=None)

_POSITIVE_AVAIL = re.compile(
    r"\b(?:patio|table|that (?:time|slot)|a table)\b.{0,50}\b(?:available|is free|are free|is open)\b"
    r"|\b(?:available|is free)\b.{0,50}\b(?:patio|table)\b"
    r"|\bwe (?:can|do) (?:seat|take|book) you\b"
    r"|\byes[,.]? (?:the )?(?:patio|table)\b",
    re.IGNORECASE | re.DOTALL,
)
_NEGATION = re.compile(
    r"\b(?:not|no|n't|isn't|is not|aren't|unavailable|full|can't|cannot|unable)\b",
    re.IGNORECASE,
)
_PATIO = re.compile(r"\bpatio\b", re.IGNORECASE)


@dataclass
class TurnEvidence:
    session_id: str
    scope: str
    availability: list[dict[str, Any]] = field(default_factory=list)
    order_summaries: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last_availability(self) -> dict[str, Any] | None:
        return self.availability[-1] if self.availability else None

    @property
    def last_order_summary(self) -> dict[str, Any] | None:
        return self.order_summaries[-1] if self.order_summaries else None


def begin_turn(session_id: str, scope: str) -> TurnEvidence:
    evidence = TurnEvidence(session_id=session_id or "", scope=scope or "")
    _turn.set(evidence)
    return evidence


def end_turn() -> TurnEvidence | None:
    evidence = _turn.get()
    _turn.set(None)
    return evidence


def current_turn() -> TurnEvidence | None:
    return _turn.get()


def record_availability(result: dict[str, Any]) -> None:
    evidence = current_turn()
    if evidence is None or not isinstance(result, dict):
        return
    evidence.availability.append(
        {
            "nonce": result.get("availability_nonce") or "",
            "available": bool(result.get("available")),
            "preferred_location": str(result.get("preferred_location") or ""),
            "tables": list(result.get("tables") or []),
            "alternatives": list(result.get("alternatives") or []),
            "date": result.get("date"),
            "time": result.get("time"),
        }
    )


def record_order_summary(result: dict[str, Any]) -> None:
    evidence = current_turn()
    if evidence is None or not isinstance(result, dict):
        return
    evidence.order_summaries.append(
        {
            "nonce": result.get("summary_nonce") or "",
            "total": float(result.get("total") or 0),
            "draft_version": int(result.get("draft_version") or 0),
            "order_id": result.get("order_id"),
            "booking_id": result.get("booking_id") or 0,
            "fulfillment": result.get("fulfillment") or "",
        }
    )


def require_order_summary_for_confirm(
    call_id: str,
    expected_draft_version: int,
) -> dict[str, Any] | None:
    """Return the same-turn summary, or raise if a custom-LLM turn is active without one."""
    evidence = current_turn()
    if evidence is None:
        return None
    summary = evidence.last_order_summary
    if not summary:
        from app.services.restaurant import RestaurantServiceError

        raise RestaurantServiceError(
            "Read the order summary out loud before confirming.",
            code="readback_required",
            status=409,
        )
    if int(summary.get("draft_version") or 0) != int(expected_draft_version):
        from app.services.restaurant import RestaurantServiceError

        raise RestaurantServiceError(
            "The order changed after the readback. Get a fresh summary and confirm again.",
            code="draft_version_conflict",
            status=409,
        )
    return summary


def require_order_summary_if_pending_order() -> dict[str, Any] | None:
    """When booking and a pending order share this turn, the summary must already exist."""
    evidence = current_turn()
    if evidence is None:
        return None
    return evidence.last_order_summary


def speech_claims_availability(speech: str) -> bool:
    text = speech or ""
    for match in _POSITIVE_AVAIL.finditer(text):
        window = text[max(0, match.start() - 28) : match.end()]
        if not _NEGATION.search(window):
            return True
    return False


def speech_claims_patio_available(speech: str) -> bool:
    return bool(_PATIO.search(speech or "")) and speech_claims_availability(speech)


def speech_contains_total(speech: str, total: float) -> bool:
    text = (speech or "").replace(",", "")
    cents = f"{total:.2f}"
    if f"${cents}" in text or cents in text:
        return True
    whole = int(total)
    if float(total) == float(whole) and (f"${whole}" in text or f"{whole} dollar" in text.casefold()):
        return True
    return False


def audit_assistant_speech(speech: str) -> list[dict[str, Any]]:
    """Log (do not block) ungrounded availability or missing confirmation totals."""
    evidence = current_turn()
    flags: list[dict[str, Any]] = []
    if evidence is None:
        return flags

    if speech_claims_availability(speech):
        latest = evidence.last_availability
        if not latest:
            flags.append(
                {
                    "code": "ungrounded_availability",
                    "message": "Assistant claimed availability with no same-turn check_table_availability result.",
                }
            )
        elif speech_claims_patio_available(speech):
            patio_tables = [
                row
                for row in (latest.get("tables") or [])
                if str(row.get("location") or "").casefold() == "patio"
            ]
            if not latest.get("available") or (
                str(latest.get("preferred_location") or "").casefold() == "patio"
                and not patio_tables
            ):
                flags.append(
                    {
                        "code": "conflicting_availability",
                        "message": "Assistant claimed patio availability against a negative same-turn tool result.",
                        "nonce": latest.get("nonce"),
                    }
                )

    summary = evidence.last_order_summary
    if summary and float(summary.get("total") or 0) > 0:
        if not speech_contains_total(speech, float(summary["total"])):
            flags.append(
                {
                    "code": "missing_order_total",
                    "message": "Same-turn order summary total did not appear in the assistant reply.",
                    "total": summary["total"],
                    "nonce": summary.get("nonce"),
                }
            )

    for flag in flags:
        evidence.flags.append(flag)
        logger.warning(
            "turn_evidence session=%s scope=%s %s",
            evidence.session_id,
            evidence.scope,
            flag,
        )
    return flags
