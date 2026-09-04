"""Unit tests for safe Retell pilot provisioning (no real network calls)."""

from __future__ import annotations

import json
from typing import Any, Mapping

import httpx
import pytest

from scripts.provision_retell import (
    ConfigurationError,
    ProvisioningConfig,
    ProvisioningError,
    PurchaseConfirmationRequired,
    RetellAPIError,
    RetellPhoneNumberClient,
    build_create_phone_payload,
    build_inbound_agents,
    execute,
    main,
    validate_agent_version,
    validate_e164,
)


BASE_ENV = {
    "RETELL_AGENT_ID": "agent_test_123",
    "RETELL_AGENT_VERSION": "3",
    "RETELL_COUNTRY_CODE": "US",
    "RETELL_PHONE_NUMBER_NICKNAME": "restaurant-agent-pilot",
}


def make_config(**overrides: str) -> ProvisioningConfig:
    env = {**BASE_ENV, **overrides}
    return ProvisioningConfig.from_env(env)


class NoNetworkClient:
    """Fails the test if dry-run attempts any client operation."""

    def list_phone_numbers(self) -> list[dict[str, Any]]:
        raise AssertionError("dry-run attempted a network read")

    def create_phone_number(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        raise AssertionError("dry-run attempted a purchase")

    def update_phone_number(
        self, phone_number: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        raise AssertionError("dry-run attempted an update")


class FakeClient:
    def __init__(self, numbers: list[dict[str, Any]]) -> None:
        self.numbers = numbers
        self.list_calls = 0
        self.create_calls: list[dict[str, Any]] = []
        self.update_calls: list[tuple[str, dict[str, Any]]] = []

    def list_phone_numbers(self) -> list[dict[str, Any]]:
        self.list_calls += 1
        return self.numbers

    def create_phone_number(
        self, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        copied = dict(payload)
        self.create_calls.append(copied)
        return {
            **copied,
            "phone_number": copied.get("phone_number", "+14155550123"),
            "phone_number_type": "retell-twilio",
        }

    def update_phone_number(
        self, phone_number: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        copied = dict(payload)
        self.update_calls.append((phone_number, copied))
        return {
            **copied,
            "phone_number": phone_number,
            "phone_number_type": "retell-twilio",
        }


@pytest.mark.parametrize(
    "value",
    [
        "+14155551234",
        "+442079460123",
    ],
)
def test_validate_e164_accepts_strict_numbers(value: str) -> None:
    assert validate_e164(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "14155551234",
        "+1 (415) 555-1234",
        "+0123456789",
        "+123",
        "+1234567890123456",
    ],
)
def test_validate_e164_rejects_invalid_numbers(value: str) -> None:
    with pytest.raises(ConfigurationError, match="E.164"):
        validate_e164(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("3", 3),
        ("latest", "latest"),
        ("latest_published", "latest_published"),
        ("prod", "prod"),
        ("pilot-1", "pilot-1"),
    ],
)
def test_validate_agent_version_matches_current_union(
    value: str, expected: int | str
) -> None:
    assert validate_agent_version(value) == expected


@pytest.mark.parametrize("value", ["-1", "Prod", "v3", "3.0", "has space"])
def test_validate_agent_version_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ConfigurationError, match="RETELL_AGENT_VERSION"):
        validate_agent_version(value)


def test_country_area_and_provider_constraints() -> None:
    with pytest.raises(ConfigurationError, match="US-only"):
        make_config(RETELL_COUNTRY_CODE="CA", RETELL_AREA_CODE="416")

    with pytest.raises(ConfigurationError, match="Telnyx"):
        make_config(
            RETELL_COUNTRY_CODE="CA",
            RETELL_NUMBER_PROVIDER="telnyx",
        )

    with pytest.raises(ConfigurationError, match="first digit"):
        make_config(RETELL_AREA_CODE="015")

    with pytest.raises(ConfigurationError, match="finite"):
        make_config(RETELL_HTTP_TIMEOUT_SECONDS="nan")


def test_exact_number_and_area_code_are_mutually_exclusive() -> None:
    with pytest.raises(ConfigurationError, match="only one number selector"):
        make_config(
            RETELL_PHONE_NUMBER="+14155550123",
            RETELL_AREA_CODE="415",
        )


def test_target_and_fallback_cannot_be_equal() -> None:
    with pytest.raises(ConfigurationError, match="cannot be the same"):
        make_config(
            RETELL_PHONE_NUMBER="+14155550123",
            RETELL_FALLBACK_NUMBER="+14155550123",
        )


def test_weighted_binding_and_purchase_payload_use_current_fields() -> None:
    config = make_config(
        RETELL_AREA_CODE="415",
        RETELL_FALLBACK_NUMBER="+14155551234",
    )

    assert build_inbound_agents(config) == [
        {
            "agent_id": "agent_test_123",
            "agent_version": 3,
            "weight": 1,
        }
    ]
    assert build_create_phone_payload(config) == {
        "inbound_agents": build_inbound_agents(config),
        "nickname": "restaurant-agent-pilot",
        "number_provider": "twilio",
        "country_code": "US",
        "area_code": 415,
        "fallback_number": "+14155551234",
    }


def test_exact_number_purchase_payload_avoids_selector_combinations() -> None:
    config = make_config(RETELL_PHONE_NUMBER="+14155550123")
    payload = build_create_phone_payload(config)

    assert payload["phone_number"] == "+14155550123"
    assert "country_code" not in payload
    assert "area_code" not in payload


def test_dry_run_never_uses_client_or_exposes_api_key() -> None:
    config = ProvisioningConfig.from_env(
        {
            **BASE_ENV,
            "RETELL_AREA_CODE": "415",
            "RETELL_API_KEY": "secret_key_must_not_appear",
        }
    )

    result = execute(
        config,
        apply=False,
        confirm_purchase=True,
        client=NoNetworkClient(),
    )

    rendered = json.dumps(result)
    assert result["mode"] == "dry-run"
    assert result["network_calls_made"] == 0
    assert result["external_writes_made"] == 0
    assert "secret_key_must_not_appear" not in rendered


def test_cli_recursively_redacts_api_key_from_dry_run_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api_key = "secret_key_even_if_reused_in_a_label"
    monkeypatch.setenv("RETELL_AGENT_ID", "agent_test_123")
    monkeypatch.setenv("RETELL_AGENT_VERSION", "3")
    monkeypatch.setenv("RETELL_COUNTRY_CODE", "US")
    monkeypatch.setenv("RETELL_NUMBER_PROVIDER", "twilio")
    monkeypatch.setenv("RETELL_API_KEY", api_key)
    monkeypatch.setenv(
        "RETELL_PHONE_NUMBER_NICKNAME",
        f"pilot-{api_key}",
    )
    monkeypatch.delenv("RETELL_PHONE_NUMBER", raising=False)
    monkeypatch.delenv("RETELL_AREA_CODE", raising=False)
    monkeypatch.delenv("RETELL_FALLBACK_NUMBER", raising=False)

    assert main([]) == 0

    output = capsys.readouterr()
    assert api_key not in output.out
    assert api_key not in output.err
    assert "[REDACTED]" in output.out


def test_apply_without_purchase_confirmation_cannot_purchase() -> None:
    config = make_config(RETELL_AREA_CODE="415")
    client = FakeClient([])

    with pytest.raises(PurchaseConfirmationRequired, match="No purchase was made"):
        execute(
            config,
            apply=True,
            confirm_purchase=False,
            client=client,
        )

    assert client.list_calls == 1
    assert client.create_calls == []
    assert client.update_calls == []


def test_apply_with_both_flags_can_purchase_once() -> None:
    config = make_config(RETELL_AREA_CODE="415")
    client = FakeClient([])

    result = execute(
        config,
        apply=True,
        confirm_purchase=True,
        client=client,
    )

    assert result["action"] == "purchased"
    assert result["external_writes_made"] == 1
    assert client.list_calls == 1
    assert len(client.create_calls) == 1
    assert client.update_calls == []


def test_apply_reuses_exact_number_without_writing_when_already_matching() -> None:
    config = make_config(
        RETELL_PHONE_NUMBER="+14155550123",
        RETELL_FALLBACK_NUMBER="+14155551234",
    )
    existing = {
        "phone_number": "+14155550123",
        "phone_number_type": "retell-twilio",
        "nickname": "restaurant-agent-pilot",
        "inbound_agents": [
            {
                "agent_id": "agent_test_123",
                "agent_version": 3,
                "weight": 1.0,
            }
        ],
        "fallback_number": "+14155551234",
    }
    client = FakeClient([existing])

    result = execute(config, apply=True, client=client)

    assert result["action"] == "unchanged"
    assert result["external_writes_made"] == 0
    assert client.list_calls == 1
    assert client.create_calls == []
    assert client.update_calls == []


def test_http_client_uses_current_paginated_v2_list_without_real_network() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer test_api_key"
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "items": [{"phone_number": "+14155550111"}],
                    "has_more": True,
                    "pagination_key": "page-2",
                },
            )
        return httpx.Response(
            200,
            json={
                "items": [{"phone_number": "+14155550222"}],
                "has_more": False,
            },
        )

    with RetellPhoneNumberClient(
        "test_api_key",
        transport=httpx.MockTransport(handler),
    ) as client:
        numbers = client.list_phone_numbers()

    assert [item["phone_number"] for item in numbers] == [
        "+14155550111",
        "+14155550222",
    ]
    assert [request.url.path for request in requests] == [
        "/v2/list-phone-numbers",
        "/v2/list-phone-numbers",
    ]
    assert requests[0].url.params["limit"] == "1000"
    assert requests[1].url.params["pagination_key"] == "page-2"


def test_http_client_encodes_e164_update_path_without_real_network() -> None:
    seen_url = ""
    config = make_config(RETELL_PHONE_NUMBER="+14155550123")
    payload = {
        "inbound_agents": build_inbound_agents(config),
        "nickname": "restaurant-agent-pilot",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_url
        seen_url = str(request.url)
        assert request.method == "PATCH"
        assert json.loads(request.content) == payload
        return httpx.Response(
            200,
            json={
                **payload,
                "phone_number": "+14155550123",
                "phone_number_type": "retell-twilio",
            },
        )

    with RetellPhoneNumberClient(
        "test_api_key",
        transport=httpx.MockTransport(handler),
    ) as client:
        client.update_phone_number("+14155550123", payload)

    assert "/update-phone-number/%2B14155550123" in seen_url


def test_http_error_redacts_api_key_without_real_network() -> None:
    api_key = "key_that_must_never_be_printed"

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"message": f"invalid credential {api_key}"},
        )

    with RetellPhoneNumberClient(
        api_key,
        transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RetellAPIError) as exc_info:
            client.list_phone_numbers()

    assert api_key not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


def test_apply_updates_only_configured_exact_retell_number() -> None:
    config = make_config(
        RETELL_PHONE_NUMBER="+14155550123",
        RETELL_FALLBACK_NUMBER="+14155551234",
    )
    existing = {
        "phone_number": "+14155550123",
        "phone_number_type": "retell-twilio",
        "nickname": "old-name",
        "inbound_agents": [],
        "fallback_number": None,
    }
    client = FakeClient([existing])

    result = execute(config, apply=True, client=client)

    assert result["action"] == "updated"
    assert result["external_writes_made"] == 1
    assert client.create_calls == []
    assert client.update_calls == [
        (
            "+14155550123",
            {
                "inbound_agents": build_inbound_agents(config),
                "nickname": "restaurant-agent-pilot",
                "fallback_number": "+14155551234",
            },
        )
    ]


def test_apply_refuses_to_repurpose_custom_number() -> None:
    config = make_config(RETELL_PHONE_NUMBER="+14155550123")
    client = FakeClient(
        [
            {
                "phone_number": "+14155550123",
                "phone_number_type": "custom",
            }
        ]
    )

    with pytest.raises(ProvisioningError, match="not Retell-managed"):
        execute(config, apply=True, client=client)

    assert client.create_calls == []
    assert client.update_calls == []
