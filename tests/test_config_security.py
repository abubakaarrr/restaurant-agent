from __future__ import annotations

import pytest

from app.config import Settings


def _production_settings(**overrides) -> Settings:
    values = {
        "app_env": "production",
        "dashboard_api_key": "d" * 48,
        "session_secret": "s" * 48,
        "voice_tool_secret": "t" * 48,
        "retell_api_key": "key_123456789",
        "retell_agent_id": "agent_123456789",
        "retell_ws_token": "w" * 48,
        "staff_transfer_number": "",
        "login_username": "admin",
        "login_password": "a-strong-unique-password",
        "allowed_origins": "https://agent.example.com",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_secure_production_configuration_passes() -> None:
    _production_settings().validate_runtime_security()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("dashboard_api_key", "short", "DASHBOARD_API_KEY"),
        ("session_secret", "short", "SESSION_SECRET"),
        ("voice_tool_secret", "", "VOICE_TOOL_SECRET"),
        ("allowed_origins", "*", "ALLOWED_ORIGINS"),
        ("allowed_origins", "http://agent.example.com", "HTTPS"),
        ("staff_transfer_number", "555-1234", "E.164"),
    ],
)
def test_unsafe_production_configuration_is_rejected(
    field: str,
    value: str,
    expected: str,
) -> None:
    configured = _production_settings(**{field: value})
    with pytest.raises(RuntimeError, match=expected):
        configured.validate_runtime_security()


def test_widget_requires_public_key_and_domains() -> None:
    configured = _production_settings(
        widget_enabled=True,
        retell_public_key="",
        widget_allowed_domains="",
    )
    with pytest.raises(RuntimeError, match="RETELL_PUBLIC_KEY"):
        configured.validate_runtime_security()


def test_admin_username_and_missing_staff_transfer_are_allowed() -> None:
    configured = _production_settings(
        login_username="admin",
        login_password="admin",
        staff_transfer_number="",
    )
    configured.validate_runtime_security()
