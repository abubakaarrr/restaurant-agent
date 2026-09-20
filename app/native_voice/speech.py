"""Blocking speech-evidence gate for native model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_SUCCESS = re.compile(
    r"\b(?:confirmed|booked|reserved|placed|added|updated|cancelled|canceled|removed|charged)\b",
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
    r"\b(?:burger|sandwich|salad|crisp|lemonade|dessert|menu|order|booking|reservation|table|slot|patio|\d{1,2}(?::\d{2})?\s*(?:am|pm))\b",
    re.IGNORECASE,
)
_AVAILABLE = re.compile(r"\b(?:available|open|in stock)\b", re.IGNORECASE)
_UNAVAILABLE = re.compile(
    r"\b(?:not currently available|isn't available|is not available|not available|"
    r"unavailable|sold out|not yet available|out of stock|closed|fully booked|full|"
    r"no availability|no tables?)\b",
    re.IGNORECASE,
)
_CONSEQUENTIAL_FOOD_FACT = re.compile(
    r"\b(?:contain(?:s|ed)?|include(?:s|d)?|made\s+with|ingredient(?:s)?|allergen(?:s)?|"
    r"allerg(?:y|ic|ies)|peanuts?|tree\s+nuts?|dairy|gluten|soy|shellfish|"
    r"vegan|vegetarian|cross[- ]contact)\b",
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

        success_matches = list(_SUCCESS.finditer(text or ""))
        for success_match in success_matches:
            if not any(
                item.speakable
                and item.state_version == current_state_version
                and self._success_action_supports(item.action, success_match.group(0).casefold())
                and (fragment := self._claim_fragment(text, success_match.start(), item.facts))
                and (fragment_match := _SUCCESS.search(fragment)) is not None
                and self._success_subject_matches(fragment, item.facts, item.action, fragment_match)
                for item in evidence_list
            ):
                reasons.append("success_claim_without_matching_readback")

        for price_match in _MONEY.finditer(text or ""):
            cleaned = re.sub(r"[^0-9.]", "", price_match.group(0))
            if not cleaned:
                reasons.append("price_without_authoritative_evidence")
                continue
            spoken_price = float(cleaned)
            if not any(
                item.speakable
                and item.state_version == current_state_version
                and (fragment := self._claim_fragment(text, price_match.start(), item.facts))
                and self._subject_matches(fragment, item.facts)
                and any(
                    abs(float(value) - spoken_price) < 0.005
                    for value in (item.facts.get("prices") or {}).values()
                )
                for item in evidence_list
            ):
                reasons.append("price_without_authoritative_evidence")

        for availability_match in _AVAILABILITY.finditer(text or ""):
            if not any(
                item.speakable
                and item.state_version == current_state_version
                and (fragment := self._claim_fragment(text, availability_match.start(), item.facts))
                and (spoken_availability := self._availability_value(fragment))
                and self._availability_matches(
                    spoken_availability,
                    self._fact_availability(fragment, item.facts),
                )
                and self._subject_matches(fragment, item.facts)
                for item in evidence_list
            ):
                reasons.append("availability_without_authoritative_evidence")

        for claim in claims:
            if not self._claim_supported(claim, evidence_list, current_state_version):
                reasons.append(f"unsupported_claim:{claim.get('kind', 'unknown')}")

        for fact_match in _CONSEQUENTIAL_FOOD_FACT.finditer(text or ""):
            if not any(
                item.speakable
                and item.state_version == current_state_version
                and self._food_fact_supported(
                    self._claim_fragment(text, fact_match.start(), item.facts),
                    item.facts,
                )
                for item in evidence_list
            ):
                reasons.append("food_fact_without_authority")

        # A response which presents a restaurant fact without a tool result is
        # blocked even when its wording does not match one of the narrow regexes.
        if _UNSAFE_FACTUAL.search(text or "") and not success_matches and not any(
            item.speakable
            and item.state_version == current_state_version
            and self._subject_matches(text, item.facts)
            for item in evidence_list
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
        if word == "added":
            return action == "add_order_item"
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
            if kind == "item":
                candidates = facts.get("items") or ()
                names = {
                    str(candidate.get("item_name") or candidate.get("name") or candidate.get("item_id") or "")
                    if isinstance(candidate, Mapping)
                    else str(candidate)
                    for candidate in candidates
                }
                if str(value) in names:
                    return True
            if kind == "price":
                prices = facts.get("prices") or {}
                if (str(value) in {str(v) for v in prices.values()} or value in prices.values()) and self._claim_subject_matches(claim, facts):
                    return True
            if kind == "availability" and self._availability_matches(str(value), facts.get("availability")) and self._claim_subject_matches(claim, facts):
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
        prices = facts.get("prices") or {}
        if isinstance(prices, Mapping):
            item_names += tuple(str(name).casefold() for name in prices if name not in {"", "amount"})
        canonical_items = facts.get("canonical_items") or ()
        item_names += tuple(
            str(item.get("name") or "").casefold()
            for item in canonical_items
            if isinstance(item, Mapping) and item.get("name")
        )
        if isinstance(subject := facts.get("subject"), Mapping):
            item_names += (str(subject.get("item_name") or "").casefold(),)
        if item_names:
            for item_name in item_names:
                if not item_name:
                    continue
                pattern = rf"(?<!\w){re.escape(item_name)}(?!\w)"
                match = re.search(pattern, normalized)
                if not match:
                    continue
                if len(item_name.split()) == 1 and (
                    item_name in {"burger", "sandwich", "salad", "dessert", "item", "food"}
                    or not facts.get("canonical_items")
                ):
                    prefix = normalized[:match.start()].rstrip().split()
                    if prefix and prefix[-1] not in {"the", "a", "an", "this", "that"}:
                        continue
                return True
            return False
        subject = facts.get("subject") or {}
        if isinstance(subject, Mapping):
            numbered_subject = re.search(r"\b(?:booking|order)\s*#?\s*(\d+)\b", normalized)
            if numbered_subject:
                expected_id = str(subject.get("booking_id") or subject.get("order_id") or "")
                return bool(expected_id and numbered_subject.group(1) == expected_id)
            spoken_time = re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", normalized)
            expected_time = str(subject.get("time") or "").casefold()
            if spoken_time and expected_time:
                if self._time_minutes(spoken_time.group(0)) == self._time_minutes(expected_time):
                    expected_location = str(subject.get("preferred_location") or "").casefold()
                    mentioned_location = next(
                        (location for location in ("patio", "main", "private") if re.search(rf"\b{location}\b", normalized)),
                        "",
                    )
                    if mentioned_location and expected_location and mentioned_location != expected_location:
                        return False
                    return True
            for key in ("date", "preferred_location", "booking_id", "order_id", "session_id"):
                value = str(subject.get(key) or "").casefold()
                if value and value in normalized:
                    return True
        return not bool(_SUBJECT_WORDS.search(normalized))

    @staticmethod
    def _availability_value(text: str) -> str:
        if _UNAVAILABLE.search(text or ""):
            return "unavailable"
        if _AVAILABLE.search(text or ""):
            return "available"
        return ""

    @staticmethod
    def _claim_fragment(text: str, position: int, facts: Mapping[str, Any] | None = None) -> str:
        left = max(
            (text.rfind(marker, 0, position) for marker in ".!?;"),
            default=-1,
        )
        right_candidates = [text.find(marker, position) for marker in ".!?;" if text.find(marker, position) >= 0]
        right = min(right_candidates, default=len(text))
        fragment = text[left + 1 : right].strip()
        if not facts:
            return fragment
        names = [
            str(item.get("name") or item.get("item_name") or "")
            for item in facts.get("canonical_items") or ()
            if isinstance(item, Mapping) and (item.get("name") or item.get("item_name"))
        ]
        names.extend(str(name) for name in (facts.get("availability_by_item") or {}) if name)
        names.extend(str(name) for name in (facts.get("prices") or {}) if name not in {"", "amount"})
        if not names:
            return fragment
        conjunctions = [match for match in re.finditer(r"\band\b", fragment, re.IGNORECASE)]
        for conjunction in conjunctions:
            before = fragment[: conjunction.start()].strip()
            after = fragment[conjunction.end() :].strip()
            before_has_name = any(re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", before.casefold()) for name in names)
            after_has_name = any(re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", after.casefold()) for name in names)
            if position <= left + 1 + conjunction.start() and before_has_name:
                return before
            if position > left + 1 + conjunction.end() and after_has_name:
                return after
            if position > left + 1 + conjunction.end() and after and not re.match(
                r"^(?:is|are|was|were|has|have|does|did|costs?|runs?)\b", after, re.IGNORECASE
            ):
                return after
        return fragment

    @staticmethod
    def _fact_availability(text: str, facts: Mapping[str, Any]) -> Any:
        by_item = facts.get("availability_by_item")
        if isinstance(by_item, Mapping):
            normalized = " ".join((text or "").casefold().split())
            matches = [
                (str(item), value)
                for item, value in by_item.items()
                if re.search(rf"(?<!\w){re.escape(str(item).casefold())}(?!\w)", normalized)
            ]
            if len(matches) == 1:
                return matches[0][1]
            if len(matches) > 1:
                return None
        return facts.get("availability")

    @staticmethod
    def _time_minutes(value: str) -> int | None:
        match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", value.casefold())
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        if meridiem:
            if hour == 12:
                hour = 0
            if meridiem == "pm":
                hour += 12
        if hour > 23 or minute > 59:
            return None
        return hour * 60 + minute

    @staticmethod
    def _availability_matches(spoken: str, authoritative: Any) -> bool:
        if spoken not in {"available", "unavailable"}:
            return False
        value = str(authoritative or "").casefold()
        if value in {"available", "open", "in stock"}:
            return spoken == "available"
        if value in {"unavailable", "closed", "sold out", "not available", "not yet available", "out of stock", "full", "fully booked", "no availability", "no tables"}:
            return spoken == "unavailable"
        return False

    @staticmethod
    def _claim_subject_matches(claim: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
        subject = claim.get("item") or claim.get("item_name") or claim.get("slot")
        if subject:
            return SpeechGate._subject_matches(str(subject), facts)
        prices = facts.get("prices") or {}
        return not facts.get("items") and not (prices if isinstance(prices, Mapping) else {}) and not (facts.get("subject") or {}).get("item_name")

    @staticmethod
    def _success_subject_matches(
        text: str,
        facts: Mapping[str, Any],
        action: str,
        match: re.Match[str],
    ) -> bool:
        prefix = (text or "")[: match.start()].casefold().rstrip()
        if re.search(r"\b(?:not|never|isn't|wasn't|cannot|can't|no)\s*$", prefix):
            return False
        action = action.casefold()
        if action in {"create_booking", "update_confirmed_booking", "cancel_booking"}:
            required = r"\b(?:booking|reservation|table|slot)\b"
        elif action in {
            "confirm_order",
            "add_order_item",
            "update_order_item",
            "remove_order_item",
            "set_order_notes",
            "set_order_fulfillment",
        }:
            required = r"\b(?:order|item|dish|meal)\b"
        elif action == "update_reservation_draft":
            required = r"\b(?:reservation|booking|draft)\b"
        else:
            required = r"\b(?:booking|reservation|order|item)\b"
        if not re.search(required, (text or ""), re.IGNORECASE):
            return False
        if action in {"add_order_item", "update_order_item", "remove_order_item"}:
            return SpeechGate._subject_matches(text, facts)
        subject = facts.get("subject") or {}
        numbered_subject = re.search(r"\b(?:booking|order)\s*#?\s*(\d+)\b", (text or "").casefold())
        if numbered_subject and isinstance(subject, Mapping):
            expected_id = str(subject.get("booking_id") or subject.get("order_id") or "")
            return bool(expected_id and numbered_subject.group(1) == expected_id)
        return True

    @staticmethod
    def _food_fact_supported(text: str, facts: Mapping[str, Any]) -> bool:
        if not SpeechGate._subject_matches(text, facts):
            return False
        normalized = re.sub(r"[^a-z0-9]+", " ", (text or "").casefold()).split()
        stop_words = {
            "contains", "contain", "contained", "includes", "include", "included", "made", "with",
            "ingredient", "ingredients", "allergen", "allergens", "allergy", "allergic", "is", "are",
            "not", "no", "the", "and", "has", "have", "cross", "contact",
        }
        subject_words = {
            token
            for item in facts.get("canonical_items") or ()
            if isinstance(item, Mapping)
            for token in re.sub(r"[^a-z0-9]+", " ", str(item.get("name") or item.get("item_name") or "").casefold()).split()
        }
        requested = {
            word.rstrip("s")
            for word in normalized
            if len(word) > 2 and word not in stop_words and word not in subject_words
        }
        if not requested:
            return False
        for item in facts.get("canonical_items") or ():
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or item.get("item_name") or "")
            if not name or not re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", (text or "").casefold()):
                continue
            terms: set[str] = set()
            for key in ("ingredients", "allergens", "dietary_tags", "customer_safe_answer"):
                value = item.get(key)
                if isinstance(value, str):
                    terms.update(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())
                elif isinstance(value, (list, tuple)):
                    for entry in value:
                        terms.update(re.sub(r"[^a-z0-9]+", " ", str(entry).casefold()).split())
            if requested & {term.rstrip("s") for term in terms}:
                return True
        return False
