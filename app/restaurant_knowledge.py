"""Canonical, versioned Harbor & Hearth restaurant knowledge.

The JSON fixture is the source of truth for synthetic restaurant facts.  This
module exposes normalized records to both the seed process and runtime callers
without requiring an embedding service or a network connection.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE_PATH = ROOT / "db" / "fixtures" / "harbor_and_hearth.v1.json"
_WORD_RE = re.compile(r"[a-z0-9]+")
_MONTHS = {
    name: index
    for index, names in enumerate(
        (
            (),
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        )
    )
    for name in names
}


class KnowledgeFixtureError(ValueError):
    """The local canonical fixture is absent or structurally unsafe."""


def normalize_text(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def text_tokens(value: str) -> set[str]:
    return set(_WORD_RE.findall(str(value or "").casefold()))


def _as_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise KnowledgeFixtureError(f"Invalid effective date: {value}") from exc


def _effective_status(record: dict[str, Any], on_date: date) -> str:
    start = _as_date(record.get("effective_from"))
    end = _as_date(record.get("effective_to"))
    if start and on_date < start:
        return "future"
    if end and on_date > end:
        return "expired"
    return str(record.get("status") or "current")


def _merged(defaults: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(defaults)
    value.update(deepcopy(record))
    return value


def _removal_target(option: dict[str, Any], item_id: str) -> str:
    targets = option.get("removes")
    if isinstance(targets, dict):
        return str(targets.get(item_id) or "")
    return str(targets or "")


@dataclass(frozen=True)
class TopicMatch:
    status: str
    records: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class MenuMatch:
    status: str
    item: dict[str, Any] | None = None
    candidates: tuple[dict[str, Any], ...] = ()


class RestaurantKnowledge:
    """Validated, deterministic view over one synthetic dataset version."""

    def __init__(self, raw: dict[str, Any], *, path: Path) -> None:
        self.path = path
        self.raw = deepcopy(raw)
        self._validate()
        source = self.raw["source"]
        common = {
            "schema_version": self.raw["schema_version"],
            "data_version": self.raw["data_version"],
            "source_id": source["source_id"],
            "effective_from": self.raw["effective_from"],
            "effective_to": self.raw.get("effective_to"),
        }
        self.menu_items = tuple(
            _merged({**common, **self.raw["menu_defaults"]}, item)
            for item in self.raw["menu_items"]
        )
        self.topics = tuple(
            _merged({**common, **self.raw["policy_defaults"]}, topic)
            for topic in self.raw["topics"]
        )
        self.modifier_options = {
            option["option_id"]: deepcopy(option)
            for option in self.raw["modifier_options"]
        }

    @classmethod
    def from_path(cls, path: Path | str = DEFAULT_FIXTURE_PATH) -> "RestaurantKnowledge":
        fixture_path = Path(path)
        try:
            raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeFixtureError(
                f"Canonical restaurant knowledge is unavailable: {fixture_path}"
            ) from exc
        if not isinstance(raw, dict):
            raise KnowledgeFixtureError("Canonical restaurant fixture must be an object")
        return cls(raw, path=fixture_path)

    def _validate(self) -> None:
        required = {
            "schema_version",
            "data_version",
            "fixture_id",
            "synthetic",
            "effective_from",
            "source",
            "restaurant",
            "hours",
            "dining_areas",
            "menu_defaults",
            "menu_items",
            "modifier_options",
            "topics",
            "policy_defaults",
            "escalation_routes",
            "conversation_style",
            "conversation_fixtures",
        }
        missing = sorted(required - self.raw.keys())
        if missing:
            raise KnowledgeFixtureError(
                "Canonical restaurant fixture is missing: " + ", ".join(missing)
            )
        if self.raw.get("synthetic") is not True:
            raise KnowledgeFixtureError("Only an explicitly synthetic fixture may load")
        restaurant = self.raw.get("restaurant") or {}
        if restaurant.get("name") != "Harbor & Hearth Kitchen":
            raise KnowledgeFixtureError("Unexpected canonical restaurant identity")
        source_id = str((self.raw.get("source") or {}).get("source_id") or "")
        if not source_id:
            raise KnowledgeFixtureError("source.source_id is required")
        if not str(self.raw.get("schema_version") or ""):
            raise KnowledgeFixtureError("schema_version is required")
        if not str(self.raw.get("data_version") or ""):
            raise KnowledgeFixtureError("data_version is required")
        fixture_start = _as_date(self.raw.get("effective_from"))
        fixture_end = _as_date(self.raw.get("effective_to"))
        if fixture_start is None:
            raise KnowledgeFixtureError("effective_from is required")
        if fixture_end and fixture_end < fixture_start:
            raise KnowledgeFixtureError("effective_to cannot precede effective_from")
        self._require_unique(self.raw["menu_items"], "item_id")
        self._require_unique(self.raw["modifier_options"], "option_id")
        self._require_unique(self.raw["topics"], "topic_id")
        self._require_unique(self.raw["dining_areas"], "area_id")
        self._require_unique(self.raw["escalation_routes"], "route_id")
        self._require_unique(self.raw["conversation_fixtures"], "fixture_id")
        self._require_unique(self.raw["hours"].get("exceptions") or [], "exception_id")
        if len(self.raw["menu_items"]) < 25:
            raise KnowledgeFixtureError("At least 25 canonical menu items are required")

        route_owners = {
            str(route.get("owner") or "") for route in self.raw["escalation_routes"]
        }

        option_ids = {option["option_id"] for option in self.raw["modifier_options"]}
        options_by_id = {
            option["option_id"]: option for option in self.raw["modifier_options"]
        }
        for option in self.raw["modifier_options"]:
            if not option["option_id"].startswith("modifier."):
                raise KnowledgeFixtureError("Modifier identifiers must start with modifier.")
            if option.get("availability") not in {"available", "unavailable"}:
                raise KnowledgeFixtureError(
                    f"Invalid availability for {option['option_id']}"
                )
            try:
                price_delta = float(option.get("price_delta") or 0)
            except (TypeError, ValueError) as exc:
                raise KnowledgeFixtureError(
                    f"Invalid price_delta for {option['option_id']}"
                ) from exc
            if price_delta < 0:
                raise KnowledgeFixtureError(
                    f"price_delta cannot be negative for {option['option_id']}"
                )
            if option.get("requires_clarification") and not option.get("choices"):
                raise KnowledgeFixtureError(
                    f"Clarification choices are required for {option['option_id']}"
                )
            if option.get("kind") == "removal" and not option.get("removes"):
                raise KnowledgeFixtureError(
                    f"Removal target is required for {option['option_id']}"
                )

        item_ids = {item["item_id"] for item in self.raw["menu_items"]}
        list_fields = (
            "aliases",
            "ingredients",
            "allergens",
            "dietary_tags",
            "service_periods",
            "modifier_options",
            "removable_ingredients",
            "substitutions",
            "incompatible_choices",
        )
        for raw_item in self.raw["menu_items"]:
            item = _merged(self.raw["menu_defaults"], raw_item)
            item_id = item["item_id"]
            if not item_id.startswith("menu."):
                raise KnowledgeFixtureError("Menu identifiers must start with menu.")
            if not str(item.get("name") or "") or not str(item.get("description") or ""):
                raise KnowledgeFixtureError(f"Name and description are required for {item_id}")
            if not str(item.get("customer_safe_answer") or ""):
                raise KnowledgeFixtureError(
                    f"customer_safe_answer is required for {item_id}"
                )
            if not str(item.get("category_id") or "").startswith("category."):
                raise KnowledgeFixtureError(f"Invalid category_id for {item_id}")
            try:
                price = float(item.get("price"))
            except (TypeError, ValueError) as exc:
                raise KnowledgeFixtureError(f"Invalid price for {item_id}") from exc
            if price <= 0 or item.get("currency") != "USD":
                raise KnowledgeFixtureError(f"Invalid price or currency for {item_id}")
            if item.get("availability") not in {
                "available",
                "sold_out",
                "not_yet_available",
            }:
                raise KnowledgeFixtureError(f"Invalid availability for {item_id}")
            for field in list_fields:
                if not isinstance(item.get(field), list):
                    raise KnowledgeFixtureError(f"{field} must be a list for {item_id}")
            if not str(item.get("cross_contact") or ""):
                raise KnowledgeFixtureError(f"cross_contact is required for {item_id}")
            item_start = _as_date(item.get("effective_from"))
            item_end = _as_date(item.get("effective_to"))
            if item_start is None or (item_end and item_end < item_start):
                raise KnowledgeFixtureError(f"Invalid effective period for {item_id}")
            referenced_options = set(item["modifier_options"]) | set(item["substitutions"])
            for group in item.get("required_modifier_groups") or []:
                referenced_options.update(group.get("option_ids") or [])
            for conflict in item["incompatible_choices"]:
                referenced_options.update(conflict.get("option_ids") or [])
            unknown_options = sorted(referenced_options - option_ids)
            if unknown_options:
                raise KnowledgeFixtureError(
                    f"Unknown modifier references for {item_id}: {', '.join(unknown_options)}"
                )
            unknown_alternatives = sorted(
                set(item.get("alternative_item_ids") or []) - item_ids
            )
            if unknown_alternatives:
                raise KnowledgeFixtureError(
                    f"Unknown alternatives for {item_id}: {', '.join(unknown_alternatives)}"
                )
            removable = {normalize_text(value) for value in item["removable_ingredients"]}
            for option_id in item["modifier_options"]:
                option = options_by_id[option_id]
                target = _removal_target(option, item_id)
                if option.get("kind") == "removal" and normalize_text(target) not in removable:
                    raise KnowledgeFixtureError(
                        f"Removal target for {option_id} is not removable from {item_id}"
                    )

        for raw_topic in self.raw["topics"]:
            topic = _merged(self.raw["policy_defaults"], raw_topic)
            topic_id = topic["topic_id"]
            if not topic_id.startswith("topic."):
                raise KnowledgeFixtureError("Topic identifiers must start with topic.")
            if not str(topic.get("category_id") or "").startswith("category."):
                raise KnowledgeFixtureError(f"Invalid category_id for {topic_id}")
            if not isinstance(topic.get("aliases"), list) or not topic["aliases"]:
                raise KnowledgeFixtureError(f"Aliases are required for {topic_id}")
            if not str(topic.get("answer") or "") or not isinstance(topic.get("rule"), dict):
                raise KnowledgeFixtureError(f"Answer and structured rule are required for {topic_id}")
            if not str(topic.get("version") or "") or not str(topic.get("escalation_owner") or ""):
                raise KnowledgeFixtureError(f"Version and escalation owner are required for {topic_id}")
            if topic["escalation_owner"] not in route_owners:
                raise KnowledgeFixtureError(
                    f"Unknown escalation owner for {topic_id}: {topic['escalation_owner']}"
                )
            topic_start = _as_date(topic.get("effective_from"))
            topic_end = _as_date(topic.get("effective_to"))
            if topic_start is None or (topic_end and topic_end < topic_start):
                raise KnowledgeFixtureError(f"Invalid effective period for {topic_id}")

    @staticmethod
    def _require_unique(records: Iterable[dict[str, Any]], key: str) -> None:
        values = [str(record.get(key) or "") for record in records]
        if any(not value for value in values) or len(values) != len(set(values)):
            raise KnowledgeFixtureError(f"Every {key} must be present and unique")

    @property
    def identity(self) -> dict[str, Any]:
        return deepcopy(self.raw["restaurant"])

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.raw["schema_version"],
            "data_version": self.raw["data_version"],
            "fixture_id": self.raw["fixture_id"],
            "source_id": self.raw["source"]["source_id"],
            "effective_from": self.raw["effective_from"],
            "effective_to": self.raw.get("effective_to"),
            "synthetic": True,
        }

    def current_menu(self, *, on_date: date | None = None) -> list[dict[str, Any]]:
        today = on_date or self.local_date()
        return [
            deepcopy(item)
            for item in self.menu_items
            if _effective_status(item, today) == "current"
        ]

    def find_menu_item(self, query: str, *, on_date: date | None = None) -> MenuMatch:
        requested = normalize_text(query)
        if not requested:
            return MenuMatch("missing")
        today = on_date or self.local_date()
        current = self.current_menu(on_date=today)
        all_records = list(self.menu_items)

        def names(item: dict[str, Any]) -> set[str]:
            return {
                normalize_text(value)
                for value in [item["name"], *(item.get("aliases") or [])]
                if normalize_text(value)
            }

        exact = [item for item in current if requested in names(item)]
        if len(exact) == 1:
            return MenuMatch("known", deepcopy(exact[0]))
        if len(exact) > 1:
            return MenuMatch("ambiguous", candidates=tuple(deepcopy(exact)))

        stale = [item for item in all_records if requested in names(item)]
        if stale:
            # Exact inactive records are explicit stale/unavailable results,
            # never choices in an ambiguity prompt.
            return MenuMatch(
                _effective_status(stale[0], today),
                item=deepcopy(stale[0]),
            )

        # A partial or spelling match is only ever a clarification candidate.
        # It must never silently select an item or mutate an order.
        requested_tokens = text_tokens(requested)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in current:
            item_names = names(item)
            token_overlap = max(
                (len(requested_tokens & text_tokens(name)) for name in item_names),
                default=0,
            )
            ratio = max(
                (SequenceMatcher(None, requested, name).ratio() for name in item_names),
                default=0.0,
            )
            if token_overlap or ratio >= 0.72:
                ranked.append((token_overlap * 2 + ratio, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
        candidates = tuple(deepcopy(item) for _, item in ranked[:3])
        return MenuMatch("ambiguous" if candidates else "unknown", candidates=candidates)

    def find_topic(self, query: str, *, on_date: date | None = None) -> TopicMatch:
        normalized = normalize_text(query)
        query_tokens = text_tokens(normalized)
        if not query_tokens:
            return TopicMatch("missing")
        today = on_date or self.local_date()
        if "open" in query_tokens and query_tokens & {"table", "tables"}:
            seating = next(
                topic for topic in self.topics if topic["topic_id"] == "topic.seating"
            )
            status = _effective_status(seating, today)
            return TopicMatch(
                "known" if status == "current" else status,
                (deepcopy(seating),),
            )
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        stale: list[dict[str, Any]] = []
        for topic in self.topics:
            best_score = 0
            best_length = 0
            phrases = [topic["topic_id"].removeprefix("topic.").replace("-", " ")]
            phrases.extend(topic.get("aliases") or [])
            for phrase in phrases:
                alias = normalize_text(phrase)
                alias_tokens = text_tokens(alias)
                if not alias_tokens or not alias_tokens <= query_tokens:
                    continue
                if (
                    topic["topic_id"] == "topic.hours"
                    and alias_tokens in ({"open"}, {"close"})
                    and not self._hours_alias_has_context(
                        alias, normalized, query_tokens
                    )
                ):
                    continue
                score = len(alias_tokens) * 10
                if normalized == alias:
                    score += 100
                if alias in normalized:
                    score += 2
                if score > best_score:
                    best_score = score
                    best_length = len(alias)
            if not best_score:
                continue
            status = _effective_status(topic, today)
            if status != "current":
                stale.append(topic)
                continue
            ranked.append((best_score, best_length, topic))
        if not ranked:
            if stale:
                return TopicMatch(
                    _effective_status(stale[0], today), tuple(deepcopy(stale))
                )
            return TopicMatch("unknown")
        ranked.sort(key=lambda value: (-value[0], -value[1], value[2]["topic_id"]))
        top_score, top_length, _ = ranked[0]
        winners = [
            deepcopy(topic)
            for score, length, topic in ranked
            if score == top_score and length == top_length
        ]
        if len(winners) > 1:
            return TopicMatch("ambiguous", tuple(winners))
        return TopicMatch("known", (deepcopy(ranked[0][2]),))

    def resolve_customization(
        self,
        item: dict[str, Any],
        *,
        modifier_ids: Iterable[str] = (),
        removals: Iterable[str] = (),
        substitutions: Iterable[str] = (),
    ) -> dict[str, Any]:
        allowed = set(item.get("modifier_options") or [])
        selected: list[dict[str, Any]] = []
        selected_substitutions: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        option_removals: list[str] = []
        for raw in modifier_ids:
            option_id, separator, choice = str(raw).partition(":")
            if option_id in selected_ids:
                return {
                    "status": "clarification_required",
                    "message": f"Choose {option_id} only once.",
                }
            option = self.modifier_options.get(option_id)
            if not option or option_id not in allowed:
                return {"status": "incompatible", "message": f"{raw} is not available for {item['name']}."}
            if option.get("availability") != "available":
                return {
                    "status": "unavailable",
                    "message": option.get("availability_note") or f"{option['name']} is unavailable.",
                }
            choices = option.get("choices") or []
            if separator and not choices:
                return {
                    "status": "incompatible",
                    "message": f"{option['name']} does not accept a choice value.",
                }
            if option.get("requires_clarification") and (not separator or choice not in choices):
                return {
                    "status": "clarification_required",
                    "message": f"Choose {option['name']}: {', '.join(choices)}.",
                    "choices": list(choices),
                }
            canonical = deepcopy(option)
            if choice:
                canonical["selection"] = choice
            if option.get("kind") == "removal":
                option_removals.append(_removal_target(option, str(item["item_id"])))
            else:
                selected.append(canonical)
            selected_ids.add(option_id)

        explicit_removals = [
            " ".join(str(value).split())
            for value in removals
            if str(value).strip()
        ]
        explicit_removal_ids = [normalize_text(value) for value in explicit_removals]
        if len(explicit_removal_ids) != len(set(explicit_removal_ids)):
            return {
                "status": "clarification_required",
                "message": "Choose each removal only once.",
            }
        normalized_removals = [*explicit_removals, *option_removals]
        normalized_removal_ids = [normalize_text(value) for value in normalized_removals]
        removable = {normalize_text(value): value for value in item.get("removable_ingredients") or []}
        for removal in normalized_removals:
            if normalize_text(removal) not in removable:
                return {
                    "status": "incompatible",
                    "message": f"{removal} cannot be promised as a removal from {item['name']}.",
                }

        substitution_ids = [str(value) for value in substitutions]
        if len(substitution_ids) != len(set(substitution_ids)):
            return {
                "status": "clarification_required",
                "message": "Choose each substitution only once.",
            }
        permitted_substitutions = set(item.get("substitutions") or [])
        if any(value not in permitted_substitutions for value in substitution_ids):
            return {
                "status": "incompatible",
                "message": f"That substitution is not supported for {item['name']}.",
            }
        for value in substitution_ids:
            if value in selected_ids:
                return {
                    "status": "clarification_required",
                    "message": f"Choose {value} as a modifier or substitution, not both.",
                }
            option = self.modifier_options.get(value)
            if not option or option.get("availability") != "available":
                return {"status": "unavailable", "message": f"{value} is unavailable."}
            selected_substitutions.append(deepcopy(option))
            selected_ids.add(value)

        for conflict in item.get("incompatible_choices") or []:
            if set(conflict.get("option_ids") or []) <= selected_ids:
                return {
                    "status": "clarification_required",
                    "message": conflict.get("message") or "Those choices conflict.",
                }

        for group in item.get("required_modifier_groups") or []:
            choices = set(group.get("option_ids") or [])
            count = len(choices & selected_ids)
            if count < int(group.get("min") or 0):
                names = [self.modifier_options[value]["name"] for value in choices]
                return {
                    "status": "clarification_required",
                    "message": f"Choose one {group.get('group_id')}: {', '.join(sorted(names))}.",
                }
            if count > int(group.get("max") or len(choices)):
                return {
                    "status": "clarification_required",
                    "message": f"Choose fewer options for {group.get('group_id')}.",
                }

        return {
            "status": "valid",
            "modifiers": selected,
            "removals": [
                removable[value] for value in dict.fromkeys(normalized_removal_ids)
            ],
            "substitutions": selected_substitutions,
            "price_delta": round(
                sum(
                    float(option.get("price_delta") or 0)
                    for option in [*selected, *selected_substitutions]
                ),
                2,
            ),
        }

    def operating_status(self, at: datetime) -> dict[str, Any]:
        timezone_info = ZoneInfo(self.identity["timezone"])
        local = at.astimezone(timezone_info) if at.tzinfo else at.replace(tzinfo=timezone_info)
        local_date = local.date()

        for exception in self.raw["hours"].get("exceptions") or []:
            start = datetime.fromisoformat(exception["starts_at"]).astimezone(timezone_info)
            end = datetime.fromisoformat(exception["ends_at"]).astimezone(timezone_info)
            if exception["kind"] == "holiday_hours" and local_date == start.date():
                available = start <= local < end
                return {
                    "available": available,
                    "status": "open" if available else "closed",
                    "kind": exception["kind"],
                    "customer_message": (
                        exception["customer_message"]
                        if available
                        else (
                            f"The restaurant is closed at {local.strftime('%I:%M %p').lstrip('0')} "
                            f"on {local.strftime('%B')} {local.day}, {local.year}. "
                            f"{exception['customer_message']}"
                        )
                    ),
                }
            if exception["status"] == "closed" and start <= local < end:
                return {
                    "available": False,
                    "status": "closed",
                    "kind": exception["kind"],
                    "customer_message": exception["customer_message"],
                }

        weekday = local.strftime("%a").casefold()
        regular = next(
            row for row in self.raw["hours"]["regular"] if row["day"] == weekday
        )
        if regular["status"] == "closed":
            return {
                "available": False,
                "status": "closed",
                "kind": "regular_hours",
                "customer_message": f"Harbor & Hearth Kitchen is closed on {local.strftime('%A')}.",
            }
        opening = datetime.combine(
            local_date,
            datetime.strptime(regular["open"], "%H:%M").time(),
            timezone_info,
        )
        closing = datetime.combine(
            local_date,
            datetime.strptime(regular["close"], "%H:%M").time(),
            timezone_info,
        )
        available = opening <= local < closing
        return {
            "available": available,
            "status": "open" if available else "closed",
            "kind": "regular_hours",
            "customer_message": (
                f"Harbor & Hearth Kitchen is open {regular['open']} to {regular['close']} "
                f"on {local.strftime('%A')}."
                if available
                else (
                    f"Harbor & Hearth Kitchen is closed at {local.strftime('%I:%M %p').lstrip('0')} "
                    f"on {local.strftime('%A')}; regular hours are "
                    f"{regular['open']} to {regular['close']}."
                )
            ),
        }

    def schedule_status(
        self,
        schedule_name: str,
        at: datetime,
        *,
        duration_minutes: int = 0,
        apply_cutoff: bool = False,
    ) -> dict[str, Any]:
        timezone_info = ZoneInfo(self.identity["timezone"])
        local = at.astimezone(timezone_info) if at.tzinfo else at.replace(tzinfo=timezone_info)
        operating = self.operating_status(local)
        if not operating["available"]:
            return operating
        schedule = (self.raw["hours"].get("fulfillment") or {}).get(schedule_name)
        if not isinstance(schedule, dict):
            return {
                "available": False,
                "status": "unavailable",
                "kind": "schedule_missing",
                "customer_message": f"Current {schedule_name} hours are unavailable.",
            }
        weekday = local.strftime("%a").casefold()
        close_value = schedule.get("sunday_close") if weekday == "sun" else None
        close_value = str(close_value or schedule.get("close") or "")
        open_value = str(schedule.get("open") or "")
        available = weekday in set(schedule.get("days") or [])
        if available:
            opening = datetime.combine(
                local.date(), datetime.strptime(open_value, "%H:%M").time(), timezone_info
            )
            closing = datetime.combine(
                local.date(), datetime.strptime(close_value, "%H:%M").time(), timezone_info
            )
            if apply_cutoff:
                closing -= timedelta(minutes=int(schedule.get("cutoff_minutes_before_close") or 0))
            available = (
                opening <= local < closing
                and local + timedelta(minutes=duration_minutes) <= closing
            )
        return {
            "available": available,
            "status": "open" if available else "closed",
            "kind": f"{schedule_name}_hours",
            "customer_message": (
                f"{schedule_name.replace('_', ' ').title()} is available now."
                if available
                else (
                    f"{schedule_name.replace('_', ' ').title()} is unavailable at "
                    f"{local.strftime('%I:%M %p').lstrip('0')} on {local.strftime('%A')}; "
                    f"scheduled hours are {open_value} to {close_value}."
                )
            ),
        }

    def menu_service_status(
        self, service_period_ids: Iterable[str], at: datetime
    ) -> dict[str, Any]:
        period_ids = set(service_period_ids)
        operating = self.operating_status(at)
        if not operating["available"] or not period_ids:
            return operating if not operating["available"] else {
                "available": True,
                "status": "available",
                "kind": "all_service_periods",
                "customer_message": "Available during all open service periods.",
            }
        timezone_info = ZoneInfo(self.identity["timezone"])
        local = at.astimezone(timezone_info) if at.tzinfo else at.replace(tzinfo=timezone_info)
        regular = next(
            row
            for row in self.raw["hours"]["regular"]
            if row["day"] == local.strftime("%a").casefold()
        )
        for period in regular.get("service_periods") or []:
            if period.get("id") not in period_ids:
                continue
            opening = datetime.combine(
                local.date(), datetime.strptime(period["open"], "%H:%M").time(), timezone_info
            )
            closing = datetime.combine(
                local.date(), datetime.strptime(period["close"], "%H:%M").time(), timezone_info
            )
            if opening <= local < closing:
                return {
                    "available": True,
                    "status": "available",
                    "kind": str(period["id"]),
                    "customer_message": "The item is available in the current service period.",
                }
        return {
            "available": False,
            "status": "unavailable",
            "kind": "outside_service_period",
            "customer_message": "The item is not available in the current service period.",
        }

    def local_date(self) -> date:
        return datetime.now(ZoneInfo(self.identity["timezone"])).date()

    def _hours_alias_has_context(
        self, alias: str, normalized_query: str, query_tokens: set[str]
    ) -> bool:
        if normalized_query == alias:
            return True
        if alias == "open" and re.search(r"\bopen\s+to\b", normalized_query):
            return False
        if alias == "close" and re.search(r"\bclose\s+to\b", normalized_query):
            return False
        if query_tokens & {
            "table",
            "tables",
            "reservation",
            "reservations",
            "booking",
            "book",
        }:
            return False
        temporal_tokens = {
            "when",
            "time",
            "hours",
            "today",
            "tonight",
            "tomorrow",
            "breakfast",
            "brunch",
            "lunch",
            "dinner",
            "bar",
            "late",
            "until",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
            *_MONTHS,
        }
        for exception in self.raw["hours"].get("exceptions") or []:
            temporal_tokens.update(
                token
                for token in text_tokens(exception.get("exception_id") or "")
                if token not in {"hours", "holiday", "private", "event"}
                and not token.isdigit()
            )
        if re.search(r"\b20\d{2}\s+\d{1,2}\s+\d{1,2}\b", normalized_query):
            return True
        if query_tokens & temporal_tokens:
            return True
        return bool(
            alias == "open"
            and re.fullmatch(
                r"(?:are\s+you|will\s+you\s+be)\s+open(?:\s+(?:now|right\s+now))?",
                normalized_query,
            )
        )

    def resolve_hours_query(
        self, query: str, *, on_date: date | None = None
    ) -> dict[str, Any] | None:
        normalized = normalize_text(query)
        tokens = text_tokens(query)
        requested_date: date | None = None
        local_date = on_date or self.local_date()
        if tokens & {"today", "tonight"}:
            requested_date = local_date
        elif "tomorrow" in tokens:
            requested_date = local_date + timedelta(days=1)
        supplied_year_match = re.search(r"\b(20\d{2})\b", query)
        supplied_year = int(supplied_year_match.group(1)) if supplied_year_match else None
        iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        if iso_match:
            try:
                requested_date = date.fromisoformat(iso_match.group(1))
            except ValueError:
                return None
        if requested_date is None:
            month_match = re.search(
                r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?\b",
                normalized,
            )
            if month_match:
                year = int(month_match.group(3) or local_date.year)
                try:
                    requested_date = date(
                        year,
                        _MONTHS[month_match.group(1)],
                        int(month_match.group(2)),
                    )
                except ValueError:
                    return None

        exceptions = self.raw["hours"].get("exceptions") or []
        requested_year = supplied_year or local_date.year
        mismatched_named_exception = False
        for exception in exceptions:
            start = datetime.fromisoformat(exception["starts_at"])
            end = datetime.fromisoformat(exception["ends_at"])
            exception_date = start.date()
            name_tokens = {
                token
                for token in text_tokens(exception["exception_id"])
                if token not in {"hours", "holiday", "private", "event", str(exception_date.year)}
            }
            named_exception = bool(name_tokens) and name_tokens <= tokens
            if named_exception and requested_year != exception_date.year:
                mismatched_named_exception = True
                continue
            if requested_date == exception_date or named_exception:
                return {
                    "status": exception["status"],
                    "date": exception_date.isoformat(),
                    "starts_at": start.isoformat(),
                    "ends_at": end.isoformat(),
                    "kind": exception["kind"],
                    "customer_message": exception["customer_message"],
                }

        if mismatched_named_exception:
            return {
                "status": "unavailable",
                "date": str(requested_year),
                "kind": "exception_not_published",
                "customer_message": (
                    f"Hours for that {requested_year} holiday are not in the current "
                    "Harbor & Hearth schedule. I won't reuse another year's hours."
                ),
            }

        if requested_date is None:
            return None
        weekday = requested_date.strftime("%a").casefold()
        regular = next(
            row for row in self.raw["hours"]["regular"] if row["day"] == weekday
        )
        display_date = (
            f"{requested_date.strftime('%A, %B')} {requested_date.day}, "
            f"{requested_date.year}"
        )
        if regular["status"] == "closed":
            message = f"Harbor & Hearth Kitchen is closed on {display_date}."
        else:
            message = (
                f"On {display_date}, Harbor & Hearth Kitchen "
                f"is open {regular['open']} to {regular['close']}; kitchen last call is "
                f"{regular['kitchen_last_call']}."
            )
        return {
            "status": regular["status"],
            "date": requested_date.isoformat(),
            "kind": "regular_hours",
            "customer_message": message,
        }

    def escalation_route(self, owner: str) -> dict[str, Any] | None:
        for route in self.raw["escalation_routes"]:
            if route.get("owner") == owner:
                return deepcopy(route)
        return None


@lru_cache(maxsize=1)
def get_restaurant_knowledge() -> RestaurantKnowledge:
    return RestaurantKnowledge.from_path()
