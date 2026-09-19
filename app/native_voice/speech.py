"""Blocking speech-evidence gate for native model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_SUCCESS = re.compile(
    r"\b(?:confirmed|booked|reserved|placed|updated|cancelled|canceled|removed|charged)\b",
    re.IGNORECASE,
)
_AVAILABILITY = re.compile(
    r"\b(?:available|unavailable|sold out|open|closed|in stock|out of stock)\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"(?:\$\s*\d+(?:\.\d{1,2})?|\b\d+\.\d{2}\s*(?:dollars?)?)", re.IGNORECASE)
_UNSAFE_FACTUAL = re.compile(
    r"\b(?:the menu|the item|the dish|the order|the booking|the table|the price|the slot|we have|we can seat|it costs?)\b",
    re.IGNORECASE,
)
_SUBJECT_WORDS = re.compile(
    r"\b(?:burger|sandwich|salad|crisp|lemonade|dessert|table|slot|patio|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ToolEvidence:
    action: str
    call_id: str
    turn_id: str
    state_version: int
    success: bool
    readback_verified: bool
    facts: Mapping[str, Any]
    replayed: bool = False

    @property
    def speakable(self) -> bool:
        return self.success and self.readback_verified and not self.replayed


@dataclass(frozen=True)
class SpeechDecision:
    allowed: bool
    text: str
    audio: bytes
    reasons: tuple[str, ...] = ()
    replacement: str = ""


class SpeechGate:
    """Only release model audio whose consequential claims have evidence."""

    replacement = "I can’t confirm that yet. Let me verify it first."

    def evaluate(
        self,
        text: str,
        audio: bytes,
        *,
        evidence: Iterable[ToolEvidence] = (),
        current_state_version: int = 0,
        explicit_claims: Iterable[Mapping[str, Any]] = (),
    ) -> SpeechDecision:
        evidence_list = list(evidence)
        reasons: list[str] = []
        claims = list(explicit_claims)

        success_match = _SUCCESS.search(text or "")
        if success_match:
            matching = [
                item
                for item in evidence_list
                if item.speakable
                and item.state_version == current_state_version
                and self._success_action_supports(item.action, success_match.group(0).casefold())
                and self._subject_matches(text, item.facts)
            ]
            if not matching:
                reasons.append("success_claim_without_matching_readback")

        if _MONEY.search(text or ""):
            spoken_prices: set[float] = set()
            for match in _MONEY.findall(text or ""):
                cleaned = re.sub(r"[^0-9.]", "", match)
                if cleaned:
                    spoken_prices.add(float(cleaned))
            if not any(
                item.speakable
                and item.state_version == current_state_version
                and self._subject_matches(text, item.facts)
                and any(
                    abs(float(value) - spoken_price) < 0.005
                    for value in (item.facts.get("prices") or {}).values()
                    for spoken_price in spoken_prices
                )
                for item in evidence_list
            ):
                reasons.append("price_without_authoritative_evidence")

        if _AVAILABILITY.search(text or ""):
            if not any(
                item.speakable
                and item.state_version == current_state_version
                and item.facts.get("availability")
                and self._subject_matches(text, item.facts)
                for item in evidence_list
            ):
                reasons.append("availability_without_authoritative_evidence")

        for claim in claims:
            if not self._claim_supported(claim, evidence_list, current_state_version):
                reasons.append(f"unsupported_claim:{claim.get('kind', 'unknown')}")

        # A response which presents a restaurant fact without a tool result is
        # blocked even when its wording does not match one of the narrow regexes.
        if _UNSAFE_FACTUAL.search(text or "") and not any(
            item.speakable and item.state_version == current_state_version for item in evidence_list
        ):
            reasons.append("factual_claim_without_authority")

        # No evidence is fine for a short clarification or conversational turn.
        if not reasons:
            return SpeechDecision(allowed=True, text=text, audio=audio)
        return SpeechDecision(
            allowed=False,
            text="",
            audio=b"",
            reasons=tuple(dict.fromkeys(reasons)),
            replacement=self.replacement,
        )

    @staticmethod
    def _success_action_supports(action: str, word: str) -> bool:
        action = action.casefold()
        if word in {"booked", "reserved"}:
            return action in {"create_booking", "update_confirmed_booking"}
        if word in {"cancelled", "canceled"}:
            return action == "cancel_booking"
        if word == "placed":
            return action == "confirm_order"
        if word == "updated":
            return action in {
                "update_confirmed_booking",
                "update_reservation_draft",
                "update_order_item",
                "set_order_notes",
                "set_order_fulfillment",
            }
        if word == "removed":
            return action == "remove_order_item"
        if word == "confirmed":
            return action in {"confirm_order", "create_booking", "update_confirmed_booking"}
        return False

    @staticmethod
    def _claim_supported(
        claim: Mapping[str, Any], evidence: list[ToolEvidence], current_state_version: int
    ) -> bool:
        kind = str(claim.get("kind") or "")
        value = claim.get("value")
        for item in evidence:
            if not item.speakable or item.state_version != current_state_version:
                continue
            facts = item.facts
            if kind == "item" and str(value) in {str(v) for v in facts.get("items", ())}:
                return True
            if kind == "price":
                prices = facts.get("prices") or {}
                if str(value) in {str(v) for v in prices.values()} or value in prices.values():
                    return True
            if kind == "availability" and value == facts.get("availability"):
                return True
            if kind == "success" and SpeechGate._success_action_supports(item.action, str(value).casefold()):
                return True
        return False

    @staticmethod
    def _subject_matches(text: str, facts: Mapping[str, Any]) -> bool:
        normalized = " ".join((text or "").casefold().split())
        items = facts.get("items") or ()
        if isinstance(items, Mapping):
            items = tuple(items)
        item_names = tuple(
            str(item.get("item_name") or item.get("name") or item.get("item_id") or "").casefold()
            if isinstance(item, Mapping)
            else str(item).casefold()
            for item in items
            if item
        )
        if isinstance(subject := facts.get("subject"), Mapping):
            item_names += (str(subject.get("item_name") or "").casefold(),)
        if item_names and any(item_name in normalized for item_name in item_names):
            return True
        subject = facts.get("subject") or {}
        if isinstance(subject, Mapping):
            numbered_subject = re.search(r"\b(?:booking|order)\s*#?\s*(\d+)\b", normalized)
            if numbered_subject:
                expected_id = str(subject.get("booking_id") or subject.get("order_id") or "")
                return bool(expected_id and numbered_subject.group(1) == expected_id)
            spoken_time = re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", normalized)
            expected_time = str(subject.get("time") or "").casefold()
            if spoken_time and expected_time:
                digits = re.sub(r"[^0-9]", "", spoken_time.group(0))
                expected_digits = re.sub(r"[^0-9]", "", expected_time)
                if digits and expected_digits and digits == expected_digits:
                    return True
            for key in ("date", "preferred_location", "booking_id", "order_id", "session_id"):
                value = str(subject.get(key) or "").casefold()
                if value and value in normalized:
                    return True
        return not bool(_SUBJECT_WORDS.search(normalized))
