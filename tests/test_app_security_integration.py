from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
import app.restaurant_settings as restaurant_settings


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DB_INTEGRATION") != "1",
    reason="Set RUN_DB_INTEGRATION=1 with TEST_DATABASE_URL",
)


def _retell_signature(body: bytes, key: str) -> str:
    timestamp = str(int(time.time() * 1000))
    digest = hmac.new(
        key.encode(),
        body + timestamp.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"v={timestamp},d={digest}"


def test_dashboard_csrf_voice_auth_and_signed_webhook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(settings, "database_url", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setattr(settings, "dashboard_api_key", "dashboard-test-key")
    monkeypatch.setattr(settings, "voice_tool_secret", "voice-tool-test-key")
    monkeypatch.setattr(settings, "login_username", "operator")
    monkeypatch.setattr(settings, "login_password", "strong-password")
    monkeypatch.setattr(settings, "retell_api_key", "retell-webhook-test-key")
    monkeypatch.setattr(settings, "enable_legacy_vapi", False)
    monkeypatch.setattr(settings, "widget_enabled", True)
    monkeypatch.setattr(settings, "widget_mode", "hybrid")
    monkeypatch.setattr(settings, "retell_public_key", "public_widget_key")
    monkeypatch.setattr(settings, "retell_agent_id", "voice_agent_test")
    monkeypatch.setattr(settings, "retell_chat_agent_id", "chat_agent_test")
    monkeypatch.setattr(settings, "recaptcha_site_key", "recaptcha_site_test")
    monkeypatch.setattr(
        restaurant_settings,
        "SETTINGS_FILE",
        tmp_path / "restaurant-settings.json",
    )

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        widget = client.get("/widget-demo")
        assert widget.status_code == 200
        assert "retell-widget-v2.js" in widget.text
        assert "public_widget_key" in widget.text
        assert "retell-webhook-test-key" not in widget.text

        assert client.get("/api/history").status_code == 401
        assert (
            client.get(
                "/api/history",
                headers={"X-API-Key": "dashboard-test-key"},
            ).status_code
            == 200
        )

        login = client.post(
            "/login",
            data={"username": "operator", "password": "strong-password"},
            follow_redirects=False,
        )
        assert login.status_code == 302
        page = client.get("/")
        match = re.search(r"csrfToken:\s+\"([^\"]+)\"", page.text)
        assert match
        csrf = match.group(1)
        assert client.put("/api/settings", json={}).status_code == 403
        assert (
            client.put(
                "/api/settings",
                headers={"X-CSRF-Token": csrf},
                json={"ai_agent_name": "Avery Rose"},
            ).status_code
            == 200
        )

        assert client.get("/api/voice-tools/health").status_code == 401
        assert (
            client.get(
                "/api/voice-tools/health",
                headers={"X-Voice-Tool-Secret": "voice-tool-test-key"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/vapi/llm/chat/completions",
                headers={"x-vapi-secret": "anything"},
                json={},
            ).status_code
            == 404
        )

        call_id = f"security-test-{uuid.uuid4()}"
        payload = {
            "event": "call_ended",
            "call": {"call_id": call_id, "call_status": "ended"},
        }
        raw = json.dumps(payload, separators=(",", ":")).encode()
        signature = _retell_signature(raw, settings.retell_api_key)
        first = client.post(
            "/api/retell/webhook",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Retell-Signature": signature,
            },
        )
        second = client.post(
            "/api/retell/webhook",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Retell-Signature": signature,
            },
        )
        assert first.status_code == 204
        assert second.status_code == 204
        assert (
            client.post(
                "/api/retell/webhook",
                content=raw + b" ",
                headers={"X-Retell-Signature": signature},
            ).status_code
            == 401
        )
