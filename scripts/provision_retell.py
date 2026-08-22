#!/usr/bin/env python3
"""Safely provision one Retell-managed pilot phone number.

The default mode is a local-only dry run. External writes require ``--apply``;
buying a number additionally requires ``--confirm-purchase``.

Retell OpenAPI references used by this module were current on 2026-08-12:
https://docs.retellai.com/api-references/create-phone-number
https://docs.retellai.com/api-references/list-phone-numbers
https://docs.retellai.com/api-references/update-phone-number
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol
from urllib.parse import quote

import httpx
from dotenv import dotenv_values


API_BASE_URL = "https://api.retellai.com"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_SPEC_REVISION = "2026-08-12-9e090f0"
E164_PATTERN = re.compile(r"^\+[1-9]\d{7,14}$")
NANP_NUMBER_PATTERN = re.compile(r"^\+1\d{10}$")
NANP_AREA_CODE_PATTERN = re.compile(r"^[2-9]\d{2}$")
AGENT_VERSION_TAG_PATTERN = re.compile(
    r"^(latest|latest_published|(?!(?:latest|latest_published|v\d+)$)"
    r"[a-z][a-z0-9_-]{0,19})$"
)
RETELL_MANAGED_NUMBER_TYPES = {"retell-twilio", "retell-telnyx"}
SUPPORTED_COUNTRIES = {"US", "CA"}
SUPPORTED_PROVIDERS = {"twilio", "telnyx"}


class ConfigurationError(ValueError):
    """The local environment does not describe a safe provisioning request."""


class ProvisioningError(RuntimeError):
    """The requested provisioning action could not be completed safely."""


class PurchaseConfirmationRequired(ProvisioningError):
    """A purchase was needed but the explicit confirmation flag was absent."""


class RetellAPIError(ProvisioningError):
    """Retell returned an error or an invalid response."""


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def validate_e164(value: str, field_name: str = "phone number") -> str:
    """Validate a strict E.164-shaped number and return it unchanged."""
    cleaned = value.strip()
    if not E164_PATTERN.fullmatch(cleaned):
        raise ConfigurationError(
            f"{field_name} must be E.164: '+' followed by 8-15 digits "
            "(for example, +14155551234), with no spaces or punctuation."
        )
    return cleaned


def validate_pilot_number(value: str, field_name: str = "RETELL_PHONE_NUMBER") -> str:
    """Validate a US/Canada (+1 NANP) pilot number."""
    cleaned = validate_e164(value, field_name)
    if not NANP_NUMBER_PATTERN.fullmatch(cleaned):
        raise ConfigurationError(
            f"{field_name} must be a US/Canada NANP number in the form +1 "
            "followed by exactly 10 digits."
        )
    return cleaned


def validate_country(value: str) -> str:
    country = value.strip().upper()
    if country not in SUPPORTED_COUNTRIES:
        raise ConfigurationError(
            "RETELL_COUNTRY_CODE must be US or CA; Retell-managed pilot "
            "numbers are limited to those countries."
        )
    return country


def validate_area_code(value: str | None, country: str) -> int | None:
    """Validate the documented US-only, three-digit area-code selector."""
    cleaned = _optional_text(value)
    if cleaned is None:
        return None
    if country != "US":
        raise ConfigurationError(
            "RETELL_AREA_CODE can only be used with RETELL_COUNTRY_CODE=US; "
            "Retell's Create Phone Number API documents area_code as US-only."
        )
    if not NANP_AREA_CODE_PATTERN.fullmatch(cleaned):
        raise ConfigurationError(
            "RETELL_AREA_CODE must be a three-digit NANP area code whose first "
            "digit is 2-9 (for example, 415)."
        )
    return int(cleaned)


def validate_agent_version(value: str) -> int | str:
    """Validate Retell's documented AgentVersionReference union."""
    cleaned = value.strip()
    if cleaned.isdigit():
        return int(cleaned)
    if not AGENT_VERSION_TAG_PATTERN.fullmatch(cleaned):
        raise ConfigurationError(
            "RETELL_AGENT_VERSION must be a non-negative integer, "
            "'latest', 'latest_published', or a valid lowercase environment "
            "tag (up to 20 characters; not v followed only by digits)."
        )
    return cleaned


def _positive_float(value: str, field_name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{field_name} must be a number.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigurationError(
            f"{field_name} must be a finite number greater than zero."
        )
    return parsed


@dataclass(frozen=True)
class ProvisioningConfig:
    """Validated configuration sourced from environment variables."""

    agent_id: str
    agent_version: int | str = "latest_published"
    country_code: str = "US"
    area_code: int | None = None
    phone_number: str | None = None
    fallback_number: str | None = None
    number_provider: str = "twilio"
    nickname: str = "restaurant-agent-pilot"
    timeout_seconds: float = 20.0
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        require_api_key: bool = False,
    ) -> "ProvisioningConfig":
        if env is None:
            file_values = {
                key: value
                for key, value in dotenv_values(
                    os.path.join(PROJECT_ROOT, ".env")
                ).items()
                if value is not None
            }
            source = {**file_values, **os.environ}
        else:
            source = env

        agent_id = (source.get("RETELL_AGENT_ID") or "").strip()
        if not agent_id:
            raise ConfigurationError(
                "RETELL_AGENT_ID is required. Create and publish the managed "
                "Conversation Flow agent first, then export its agent ID."
            )

        country = validate_country(source.get("RETELL_COUNTRY_CODE", "US"))
        area_code = validate_area_code(source.get("RETELL_AREA_CODE"), country)

        raw_phone = _optional_text(source.get("RETELL_PHONE_NUMBER"))
        phone_number = (
            validate_pilot_number(raw_phone, "RETELL_PHONE_NUMBER")
            if raw_phone
            else None
        )
        if phone_number is not None and area_code is not None:
            raise ConfigurationError(
                "Set only one number selector: RETELL_PHONE_NUMBER for an exact "
                "number to reuse/request, or RETELL_AREA_CODE for a new US "
                "number. Do not set both."
            )

        raw_fallback = _optional_text(source.get("RETELL_FALLBACK_NUMBER"))
        fallback_number = (
            validate_e164(raw_fallback, "RETELL_FALLBACK_NUMBER")
            if raw_fallback
            else None
        )
        if phone_number and fallback_number == phone_number:
            raise ConfigurationError(
                "RETELL_FALLBACK_NUMBER cannot be the same as RETELL_PHONE_NUMBER."
            )

        provider = source.get("RETELL_NUMBER_PROVIDER", "twilio").strip().lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ConfigurationError(
                "RETELL_NUMBER_PROVIDER must be 'twilio' or 'telnyx'."
            )
        if country == "CA" and provider != "twilio":
            raise ConfigurationError(
                "Canadian Retell-managed numbers require "
                "RETELL_NUMBER_PROVIDER=twilio; Retell documents Telnyx "
                "managed numbers as US-only."
            )

        nickname = source.get(
            "RETELL_PHONE_NUMBER_NICKNAME", "restaurant-agent-pilot"
        ).strip()
        if not nickname:
            raise ConfigurationError(
                "RETELL_PHONE_NUMBER_NICKNAME cannot be empty."
            )

        api_key = _optional_text(source.get("RETELL_API_KEY"))
        if require_api_key and not api_key:
            raise ConfigurationError(
                "RETELL_API_KEY is required with --apply. Use a scoped key with "
                "Deploy > Phone edit access; do not place it in version control."
            )

        timeout_seconds = _positive_float(
            source.get("RETELL_HTTP_TIMEOUT_SECONDS", "20"),
            "RETELL_HTTP_TIMEOUT_SECONDS",
        )

        return cls(
            api_key=api_key,
            agent_id=agent_id,
            agent_version=validate_agent_version(
                source.get("RETELL_AGENT_VERSION", "latest_published")
            ),
            country_code=country,
            area_code=area_code,
            phone_number=phone_number,
            fallback_number=fallback_number,
            number_provider=provider,
            nickname=nickname,
            timeout_seconds=timeout_seconds,
        )


def build_inbound_agents(config: ProvisioningConfig) -> list[dict[str, Any]]:
    """Build the post-2026 weighted inbound-agent binding."""
    return [
        {
            "agent_id": config.agent_id,
            "agent_version": config.agent_version,
            "weight": 1,
        }
    ]


def _validate_documented_create_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if an undocumented or ambiguous purchase field appears.

    This intentionally isolates the purchase payload contract. It only permits
    fields present in the 2026-08-12 Create Phone Number OpenAPI schema and
    rejects simultaneous exact-number and area-code selectors.
    """
    allowed_fields = {
        "inbound_agents",
        "area_code",
        "nickname",
        "number_provider",
        "country_code",
        "phone_number",
        "fallback_number",
    }
    unexpected = set(payload) - allowed_fields
    if unexpected:
        raise ConfigurationError(
            "Refusing undocumented Create Phone Number fields: "
            + ", ".join(sorted(unexpected))
        )

    if "area_code" in payload and "phone_number" in payload:
        raise ConfigurationError(
            "Create Phone Number payload cannot combine area_code and phone_number."
        )

    has_exact_number = "phone_number" in payload
    has_country = "country_code" in payload
    if has_exact_number == has_country:
        raise ConfigurationError(
            "Purchase payload must select exactly one of phone_number or "
            "country_code."
        )

    agents = payload.get("inbound_agents")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
        or agents[0].get("weight") != 1
        or not isinstance(agents[0].get("agent_id"), str)
        or not agents[0]["agent_id"].strip()
    ):
        raise ConfigurationError(
            "Pilot purchase payload must bind exactly one inbound agent at weight 1."
        )

    agent_version = agents[0].get("agent_version")
    if isinstance(agent_version, bool):
        raise ConfigurationError("agent_version cannot be a boolean.")
    if isinstance(agent_version, int):
        if agent_version < 0:
            raise ConfigurationError("agent_version cannot be negative.")
    elif isinstance(agent_version, str):
        validate_agent_version(agent_version)
    else:
        raise ConfigurationError(
            "agent_version must be a documented integer or string reference."
        )

    nickname = payload.get("nickname")
    if not isinstance(nickname, str) or not nickname.strip():
        raise ConfigurationError("Purchase payload nickname must be non-empty.")

    provider = payload.get("number_provider")
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(
            "Purchase payload number_provider must be twilio or telnyx."
        )

    if "phone_number" in payload:
        phone_number = validate_pilot_number(
            str(payload["phone_number"]), "phone_number"
        )
    else:
        phone_number = None
        country = validate_country(str(payload["country_code"]))
        if country == "CA" and provider != "twilio":
            raise ConfigurationError(
                "Canadian purchase payloads require the Twilio provider."
            )

    if "area_code" in payload:
        area_code = payload["area_code"]
        if isinstance(area_code, bool) or not isinstance(area_code, int):
            raise ConfigurationError("Purchase payload area_code must be an integer.")
        validate_area_code(str(area_code), str(payload.get("country_code", "")))

    if "fallback_number" in payload:
        fallback = validate_e164(
            str(payload["fallback_number"]), "fallback_number"
        )
        if phone_number and fallback == phone_number:
            raise ConfigurationError(
                "Purchase payload fallback_number cannot equal phone_number."
            )


def _validate_documented_update_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed around the small documented PATCH field subset we use."""
    allowed_fields = {"inbound_agents", "nickname", "fallback_number"}
    unexpected = set(payload) - allowed_fields
    if unexpected:
        raise ConfigurationError(
            "Refusing undocumented Update Phone Number fields: "
            + ", ".join(sorted(unexpected))
        )

    # Reuse the stricter create validator with a policy-only selector. The
    # selector is removed before the actual PATCH request.
    validation_copy = dict(payload)
    validation_copy["country_code"] = "US"
    validation_copy["number_provider"] = "twilio"
    _validate_documented_create_payload(validation_copy)


def build_create_phone_payload(config: ProvisioningConfig) -> dict[str, Any]:
    """Build the documented purchase payload after strict local validation."""
    payload: dict[str, Any] = {
        "inbound_agents": build_inbound_agents(config),
        "nickname": config.nickname,
        "number_provider": config.number_provider,
    }

    # Avoid relying on undocumented selector combinations.
    if config.phone_number:
        payload["phone_number"] = config.phone_number
    else:
        payload["country_code"] = config.country_code
        if config.area_code is not None:
            payload["area_code"] = config.area_code

    if config.fallback_number:
        payload["fallback_number"] = config.fallback_number

    _validate_documented_create_payload(payload)
    return payload


def build_update_phone_payload(config: ProvisioningConfig) -> dict[str, Any]:
    """Build the documented idempotent binding update payload."""
    payload: dict[str, Any] = {
        "inbound_agents": build_inbound_agents(config),
        "nickname": config.nickname,
    }
    # Omission preserves an existing fallback; a supplied value configures it.
    if config.fallback_number:
        payload["fallback_number"] = config.fallback_number
    _validate_documented_update_payload(payload)
    return payload


def _normalized_agents(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        try:
            weight = float(item.get("weight"))
        except (TypeError, ValueError):
            return None
        normalized.append(
            {
                "agent_id": item.get("agent_id"),
                "agent_version": item.get("agent_version"),
                "weight": weight,
            }
        )
    return normalized


def compute_update_changes(
    existing: Mapping[str, Any], desired: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return only meaningful differences between existing and desired state."""
    changes: dict[str, dict[str, Any]] = {}
    for key, desired_value in desired.items():
        existing_value = existing.get(key)
        if key == "inbound_agents":
            existing_value = _normalized_agents(existing_value)
            desired_value = _normalized_agents(desired_value)
        if existing_value != desired_value:
            changes[key] = {"from": existing_value, "to": desired_value}
    return changes


def _redact(text: str, *secrets: str | None) -> str:
    redacted = text
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _redact_structure(value: Any, *secrets: str | None) -> Any:
    """Recursively remove secrets before serializing user-visible results."""
    if isinstance(value, str):
        return _redact(value, *secrets)
    if isinstance(value, list):
        return [_redact_structure(item, *secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_structure(item, *secrets) for item in value)
    if isinstance(value, Mapping):
        return {
            _redact(str(key), *secrets): _redact_structure(item, *secrets)
            for key, item in value.items()
        }
    return value


class PhoneNumberClient(Protocol):
    def list_phone_numbers(self) -> list[dict[str, Any]]: ...

    def create_phone_number(self, payload: Mapping[str, Any]) -> dict[str, Any]: ...

    def update_phone_number(
        self, phone_number: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]: ...


class RetellPhoneNumberClient:
    """Small HTTP client limited to documented phone-number endpoints."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=API_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "restaurant-agent-retell-pilot/1.0",
            },
            timeout=timeout_seconds,
            transport=transport,
        )

    def __enter__(self) -> "RetellPhoneNumberClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _error_detail(self, response: httpx.Response) -> str:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, Mapping):
                detail = str(body.get("message") or body.get("detail") or "")
        except (ValueError, TypeError):
            detail = ""
        return _redact(detail[:500], self._api_key)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            response = self._client.request(
                method, path, params=params, json=payload
            )
        except httpx.TimeoutException as exc:
            raise RetellAPIError(
                f"Retell {method} {path} timed out. Check connectivity and "
                "increase RETELL_HTTP_TIMEOUT_SECONDS if needed."
            ) from exc
        except httpx.RequestError as exc:
            raise RetellAPIError(
                f"Could not reach Retell for {method} {path}. Check DNS, TLS, "
                "proxy, and firewall settings."
            ) from exc

        if response.is_error:
            detail = self._error_detail(response)
            action = {
                400: "Recheck the locally validated fields against the linked API spec.",
                401: "Verify RETELL_API_KEY and the selected Retell workspace.",
                402: "Add a payment method or credits in the Retell workspace.",
                403: "Grant the API key Deploy > Phone access.",
                422: "Verify that the agent, version, and phone number exist in this workspace.",
                429: "Wait and retry after the Retell rate limit resets.",
            }.get(response.status_code, "Retry later or contact Retell support.")
            suffix = f" Retell message: {detail}" if detail else ""
            raise RetellAPIError(
                f"Retell API returned HTTP {response.status_code} for "
                f"{method} {path}.{suffix} {action}"
            )

        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise RetellAPIError(
                f"Retell returned non-JSON content for {method} {path}."
            ) from exc

    def list_phone_numbers(self) -> list[dict[str, Any]]:
        """Read all pages from the current v2 list endpoint."""
        items: list[dict[str, Any]] = []
        pagination_key: str | None = None
        seen_keys: set[str] = set()

        while True:
            params: dict[str, Any] = {
                "limit": 1000,
                "sort_order": "descending",
            }
            if pagination_key:
                params["pagination_key"] = pagination_key

            page = self._request("GET", "/v2/list-phone-numbers", params=params)
            if not isinstance(page, Mapping) or not isinstance(
                page.get("items"), list
            ):
                raise RetellAPIError(
                    "Retell list response did not contain the documented items array."
                )
            for item in page["items"]:
                if not isinstance(item, Mapping):
                    raise RetellAPIError(
                        "Retell list response contained a non-object phone number."
                    )
                items.append(dict(item))

            if not page.get("has_more"):
                return items

            next_key = page.get("pagination_key")
            if not isinstance(next_key, str) or not next_key:
                raise RetellAPIError(
                    "Retell set has_more without a pagination_key."
                )
            if next_key in seen_keys:
                raise RetellAPIError(
                    "Retell repeated a pagination_key; stopped to avoid a loop."
                )
            seen_keys.add(next_key)
            pagination_key = next_key

    def create_phone_number(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        _validate_documented_create_payload(payload)
        result = self._request("POST", "/create-phone-number", payload=payload)
        if not isinstance(result, Mapping) or not result.get("phone_number"):
            raise RetellAPIError(
                "Retell purchase response lacked the documented phone_number."
            )
        return dict(result)

    def update_phone_number(
        self, phone_number: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        validated = validate_pilot_number(phone_number, "phone_number")
        _validate_documented_update_payload(payload)
        encoded = quote(validated, safe="")
        result = self._request(
            "PATCH",
            f"/update-phone-number/{encoded}",
            payload=payload,
        )
        if (
            not isinstance(result, Mapping)
            or result.get("phone_number") != validated
        ):
            raise RetellAPIError(
                "Retell update response did not contain the requested phone_number."
            )
        return dict(result)


def _safe_number_summary(number: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phone_number": number.get("phone_number"),
        "phone_number_type": number.get("phone_number_type"),
        "nickname": number.get("nickname"),
        "inbound_agents": number.get("inbound_agents"),
        "fallback_number": number.get("fallback_number"),
    }


def _dry_run_plan(
    config: ProvisioningConfig, *, confirm_purchase: bool
) -> dict[str, Any]:
    purchase_payload = build_create_phone_payload(config)
    update_payload = build_update_phone_payload(config)
    if config.phone_number:
        selection = {
            "strategy": "exact_phone_number",
            "phone_number": config.phone_number,
            "on_apply": (
                "list all pages; reuse and update the exact match; if absent, "
                "purchase only when --confirm-purchase is also present"
            ),
        }
    else:
        selection = {
            "strategy": "new_number",
            "country_code": config.country_code,
            "area_code": config.area_code,
            "on_apply": (
                "list existing numbers, then purchase because no exact "
                "RETELL_PHONE_NUMBER was configured"
            ),
        }

    return {
        "mode": "dry-run",
        "api_base_url": API_BASE_URL,
        "docs_spec_revision": DOCS_SPEC_REVISION,
        "network_calls_made": 0,
        "external_writes_made": 0,
        "purchase_confirmation_seen": confirm_purchase,
        "selection": selection,
        "create_payload_if_purchase_authorized": purchase_payload,
        "update_payload_if_reused": update_payload,
        "next_step": (
            "Review this plan. Use --apply to permit binding updates. A number "
            "purchase additionally requires --confirm-purchase."
        ),
    }


def execute(
    config: ProvisioningConfig,
    *,
    apply: bool = False,
    confirm_purchase: bool = False,
    client: PhoneNumberClient | None = None,
) -> dict[str, Any]:
    """Validate, plan, and optionally apply the phone-number configuration."""
    if not apply:
        return _dry_run_plan(config, confirm_purchase=confirm_purchase)

    if client is None:
        raise ProvisioningError(
            "An authenticated Retell client is required in apply mode."
        )

    numbers = client.list_phone_numbers()
    existing: dict[str, Any] | None = None
    if config.phone_number:
        matches = [
            item
            for item in numbers
            if item.get("phone_number") == config.phone_number
        ]
        if len(matches) > 1:
            raise ProvisioningError(
                "Retell returned duplicate records for RETELL_PHONE_NUMBER; "
                "refusing to choose one."
            )
        existing = matches[0] if matches else None

    if existing is not None:
        number_type = existing.get("phone_number_type")
        if number_type not in RETELL_MANAGED_NUMBER_TYPES:
            raise ProvisioningError(
                "RETELL_PHONE_NUMBER exists but is not Retell-managed "
                f"(phone_number_type={number_type!r}); refusing to repurpose it."
            )

        desired = build_update_phone_payload(config)
        changes = compute_update_changes(existing, desired)
        if not changes:
            return {
                "mode": "apply",
                "action": "unchanged",
                "message": "Existing Retell-managed number already matches.",
                "number": _safe_number_summary(existing),
                "external_writes_made": 0,
            }

        updated = client.update_phone_number(config.phone_number, desired)
        return {
            "mode": "apply",
            "action": "updated",
            "changes": changes,
            "number": _safe_number_summary(updated),
            "external_writes_made": 1,
        }

    if not confirm_purchase:
        target = (
            f"Configured number {config.phone_number} was not found."
            if config.phone_number
            else "No RETELL_PHONE_NUMBER was configured for reuse."
        )
        raise PurchaseConfirmationRequired(
            f"{target} Purchasing a new recurring-charge number requires both "
            "--apply and --confirm-purchase. No purchase was made."
        )

    created = client.create_phone_number(build_create_phone_payload(config))
    created_number = validate_pilot_number(
        str(created.get("phone_number")), "Retell response phone_number"
    )
    expected_number_type = f"retell-{config.number_provider}"
    if created.get("phone_number_type") != expected_number_type:
        raise RetellAPIError(
            "Retell purchase response did not identify the requested managed "
            f"provider type ({expected_number_type}). Inspect the Retell "
            "dashboard immediately."
        )
    if (
        config.phone_number is not None
        and created_number != config.phone_number
    ):
        raise RetellAPIError(
            "Retell returned a different phone number than the exact number "
            "requested. Inspect the Retell dashboard immediately."
        )

    return {
        "mode": "apply",
        "action": "purchased",
        "number": _safe_number_summary(created),
        "external_writes_made": 1,
        "follow_up": (
            "Store the returned E.164 number as RETELL_PHONE_NUMBER so future "
            "runs reuse it idempotently."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or safely apply Retell pilot number provisioning from "
            "environment variables. Default: dry-run with no network calls."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Permit authenticated list/update calls. Does not authorize purchase.",
    )
    parser.add_argument(
        "--confirm-purchase",
        action="store_true",
        help=(
            "Explicitly acknowledge recurring number charges. Effective only "
            "together with --apply."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config: ProvisioningConfig | None = None
    try:
        config = ProvisioningConfig.from_env(require_api_key=args.apply)
        if not args.apply:
            result = execute(
                config,
                apply=False,
                confirm_purchase=args.confirm_purchase,
            )
        else:
            assert config.api_key is not None
            with RetellPhoneNumberClient(
                config.api_key,
                timeout_seconds=config.timeout_seconds,
            ) as client:
                result = execute(
                    config,
                    apply=True,
                    confirm_purchase=args.confirm_purchase,
                    client=client,
                )
        safe_result = _redact_structure(result, config.api_key)
        print(json.dumps(safe_result, indent=2, sort_keys=True))
        return 0
    except (ConfigurationError, ProvisioningError) as exc:
        api_key = config.api_key if config else os.environ.get("RETELL_API_KEY")
        print(f"ERROR: {_redact(str(exc), api_key)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
