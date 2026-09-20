"""Blocking speech-evidence gate for native model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping


_SUCCESS = re.compile(
    r"\b(?:confirmed|booked|reserved|placed|added|updated|cancelled|canceled|removed|charged|saved|set|completed|submitted|processed)\b",
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
    r"vegan|vegetarian|cross[- ]contact|has|have|calories?|kcal|kilocalories?|"
    r"grams?|milligrams?|mg|sodium|carbs?|protein|fat|sugar|portion)\b",
    re.IGNORECASE,
)
_TIME_TOKEN = re.compile(
    r"\b(?:(?:[01]?\d|2[0-3]):[0-5]\d(?:\s*(?:a\.?m\.?|p\.?m\.?)?)?|\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b)",
    re.IGNORECASE,
)
_DATE_TOKEN = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?|"
    r"\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\s+\d{4})\b",
    re.IGNORECASE,
)
_TIMEZONE_TOKEN = re.compile(
    r"\b(?:UTC|GMT|PST|PDT|MST|MDT|CST|CDT|EST|EDT|[A-Z][a-z]+/[A-Z][a-z_]+)\b"
)
_BOOKING_REFERENCE_TOKEN = re.compile(
    r"\b(?:booking|reservation)\s+(?:reference|ref(?:erence)?|number|no\.?|id)\s*[:#-]?\s*([A-Za-z0-9-]+)\b"
    r"|\b(?:booking|reservation)\s*#\s*([A-Za-z0-9-]+)\b",
    re.IGNORECASE,
)
_WEEKDAY_TOKEN = re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)
_FULFILLMENT_TOKEN = re.compile(r"\b(?:pickup|delivery|dine[ -]?in)\b", re.IGNORECASE)
_EFFECT_MARKER = re.compile(
    r"\b(?:with|without|no|extra|substitute(?:d)?|swap(?:ped)?|instead of)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


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
                and self._success_details_match(fragment, item.facts, item.action, fragment_match)
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
                    for value in (
                        list((item.facts.get("prices") or {}).values())
                        + ([item.facts["total"]] if item.facts.get("total") is not None else [])
                    )
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
        if word in {"completed", "submitted", "processed"}:
            return action in {"confirm_order", "create_booking", "update_confirmed_booking"}
        if word == "saved":
            return action in {
                "update_reservation_draft",
                "add_guest_note",
                "add_order_item",
                "set_order_fulfillment",
                "set_order_notes",
                "update_order_item",
                "remove_order_item",
            }
        if word == "set":
            return action in {
                "create_booking",
                "update_confirmed_booking",
                "update_reservation_draft",
                "set_order_fulfillment",
                "set_order_notes",
            }
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
                plural = "" if item_name.endswith("s") else "s?"
                pattern = rf"(?<!\w){re.escape(item_name)}{plural}(?!\w)"
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
    def _date_matches(spoken: str, expected: str) -> bool:
        expected_date = SpeechGate._parse_date(expected)
        if expected_date is None:
            return False
        value = spoken.strip().replace("/", "-").replace(",", "")
        parsed = None
        includes_year = bool(re.search(r"\b\d{4}\b|\b\d{2}\b$", value))
        for pattern in (
            "%Y-%m-%d", "%m-%d-%Y", "%m-%d-%y", "%B %d %Y", "%b %d %Y",
            "%d %B %Y", "%d %b %Y", "%B %d", "%b %d",
        ):
            try:
                parsed = datetime.strptime(value, pattern).date()
                break
            except ValueError:
                continue
        if parsed is None:
            return False
        return parsed == expected_date if includes_year else parsed.month == expected_date.month and parsed.day == expected_date.day

    @staticmethod
    def _parse_date(value: str) -> date | None:
        normalized = str(value or "").strip()
        try:
            return date.fromisoformat(normalized[:10])
        except ValueError:
            try:
                return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
            except ValueError:
                return None

    @staticmethod
    def _timezone_matches(spoken: str, expected: str) -> bool:
        aliases = {
            "america/los_angeles": {"america/los_angeles", "pst", "pdt", "pacific"},
            "america/new_york": {"america/new_york", "est", "edt", "eastern"},
        }
        normalized_spoken = spoken.casefold()
        normalized_expected = expected.casefold()
        return normalized_spoken == normalized_expected or normalized_spoken in aliases.get(normalized_expected, set())

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
            if action in {"add_order_item", "update_order_item", "remove_order_item"}:
                return SpeechGate._subject_matches(text, facts)
            return False
        if action in {"add_order_item", "update_order_item", "remove_order_item"}:
            return SpeechGate._subject_matches(text, facts)
        subject = facts.get("subject") or {}
        numbered_subject = re.search(r"\b(?:booking|order)\s*#?\s*(\d+)\b", (text or "").casefold())
        if numbered_subject and isinstance(subject, Mapping):
            expected_id = str(subject.get("booking_id") or subject.get("order_id") or "")
            return bool(expected_id and numbered_subject.group(1) == expected_id)
        return True

    @classmethod
    def _success_details_match(
        cls,
        text: str,
        facts: Mapping[str, Any],
        action: str,
        match: re.Match[str],
    ) -> bool:
        action = action.casefold()
        subject = facts.get("subject") if isinstance(facts.get("subject"), Mapping) else {}
        booking = facts.get("booking") if isinstance(facts.get("booking"), Mapping) else {}
        status = str(facts.get("status") or booking.get("status") or "").casefold()
        word = match.group(0).casefold()
        if action in {"create_booking", "update_confirmed_booking", "cancel_booking"}:
            expected_status = "cancelled" if word in {"cancelled", "canceled"} else "confirmed"
            if status != expected_status:
                return False
        if action == "confirm_order" and word in {"placed", "submitted", "processed", "completed", "confirmed"}:
            if status != "confirmed":
                return False
        expected_time = str(facts.get("time") or booking.get("time") or subject.get("time") or "")
        spoken_times = _TIME_TOKEN.findall(text or "")
        if spoken_times and (
            not expected_time
            or any(cls._time_minutes(spoken) != cls._time_minutes(expected_time) for spoken in spoken_times)
        ):
            return False
        expected_date = str(facts.get("date") or booking.get("date") or subject.get("date") or "")
        spoken_dates = _DATE_TOKEN.findall(text or "")
        if spoken_dates and (
            not expected_date
            or any(not cls._date_matches(spoken, expected_date) for spoken in spoken_dates)
        ):
            return False
        weekdays = _WEEKDAY_TOKEN.findall(text or "")
        if weekdays and (not expected_date or not any(cls._weekday_matches(day, expected_date) for day in weekdays)):
            return False
        expected_reference = str(
            facts.get("reference")
            or facts.get("booking_reference")
            or booking.get("reference")
            or subject.get("reference")
            or subject.get("booking_id")
            or ""
        )
        references = [match.group(1) or match.group(2) for match in _BOOKING_REFERENCE_TOKEN.finditer(text or "")]
        if references and (not expected_reference or any(reference != expected_reference for reference in references)):
            return False
        expected_timezone = str(facts.get("timezone") or booking.get("timezone") or "")
        spoken_timezones = _TIMEZONE_TOKEN.findall(text or "")
        if spoken_timezones and (
            not expected_timezone
            or any(not cls._timezone_matches(spoken, expected_timezone) for spoken in spoken_timezones)
        ):
            return False
        fulfillment = _FULFILLMENT_TOKEN.search(text or "")
        if fulfillment:
            expected_fulfillment = str(
                facts.get("fulfillment_type") or facts.get("fulfillment") or ""
            ).replace("-", "").replace(" ", "").casefold()
            spoken_fulfillment = fulfillment.group(0).replace("-", "").replace(" ", "").casefold()
            if not expected_fulfillment or spoken_fulfillment != expected_fulfillment:
                return False
        if action in {
            "add_order_item",
            "update_order_item",
            "remove_order_item",
            "set_order_fulfillment",
            "set_order_notes",
            "confirm_order",
        }:
            items = [item for item in facts.get("items") or () if isinstance(item, Mapping)]
            for item in items:
                name = str(item.get("item_name") or item.get("name") or "").strip()
                if not name:
                    continue
                item_pattern = re.escape(name) + ("" if name.casefold().endswith("s") else "s?")
                if not re.search(rf"(?<!\w){item_pattern}(?!\w)", text or "", re.IGNORECASE):
                    continue
                quantity = re.search(
                    rf"(?:\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:x\s+)?{item_pattern}\b|\b{item_pattern}\s+(?:x\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b)",
                    text or "",
                    re.IGNORECASE,
                )
                if quantity:
                    spoken_quantity = quantity.group(1) or quantity.group(2)
                    spoken_quantity = (
                        _NUMBER_WORDS[spoken_quantity.casefold()]
                        if spoken_quantity.casefold() in _NUMBER_WORDS
                        else int(spoken_quantity)
                    )
                    if item.get("quantity") is None or int(item["quantity"]) != spoken_quantity:
                        return False
                effects = cls._effect_terms(text)
                if effects:
                    authoritative_effects: set[str] = set()
                    for key in ("modifiers", "removals", "substitutions", "notes"):
                        values = item.get(key) or ()
                        values = values if isinstance(values, (list, tuple)) else (values,)
                        for value in values:
                            if isinstance(value, Mapping):
                                value = value.get("name") or value.get("option_id") or value.get("id") or value.get("selection")
                            authoritative_effects.update(str(value or "").casefold().replace("-", " ").split())
                    if not authoritative_effects or not effects <= authoritative_effects:
                        return False
        return True

    @staticmethod
    def _effect_terms(text: str) -> set[str]:
        terms: set[str] = set()
        stop_words = {
            "a", "an", "and", "are", "at", "added", "confirmed", "for", "is", "item", "of",
            "on", "order", "placed", "removed", "saved", "the", "to", "updated", "was", "were", "your",
        }
        for marker in _EFFECT_MARKER.finditer(text or ""):
            tail = re.split(r"[,.;!?]", (text or "")[marker.end():], maxsplit=1)[0]
            words = [word.casefold() for word in re.findall(r"[a-z][a-z-]*", tail)[:3]]
            terms.update(word.replace("-", " ") for word in words if word not in stop_words)
        return terms

    @staticmethod
    def _weekday_matches(spoken: str, expected: str) -> bool:
        normalized = expected.strip().casefold()
        if normalized == spoken.casefold():
            return True
        try:
            parsed = date.fromisoformat(normalized[:10])
        except ValueError:
            try:
                parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
            except ValueError:
                return False
        return parsed.strftime("%A").casefold() == spoken.casefold()

    @staticmethod
    def _food_fact_supported(text: str, facts: Mapping[str, Any]) -> bool:
        if re.search(
            r"\b(?:not|never|no|without|free\s+of|doesn['’]?t|isn['’]?t|aren['’]?t|wasn['’]?t|weren['’]?t)\b",
            (text or "").casefold(),
        ):
            return False
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
            authoritative = {term.rstrip("s") for term in terms}
            if requested <= authoritative:
                return True
        return False
