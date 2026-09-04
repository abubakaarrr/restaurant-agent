from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.config import settings
from app.tool_api import router
import app.tool_api as tool_api


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "voice_tool_secret", "test-tool-secret")
    monkeypatch.setattr(tool_api, "_audit", lambda *args, **kwargs: None)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client


def test_tool_api_requires_dedicated_secret(client: TestClient) -> None:
    assert client.get("/api/voice-tools/health").status_code == 401
    assert (
        client.get(
            "/api/voice-tools/health",
            headers={"X-Voice-Tool-Secret": "wrong"},
        ).status_code
        == 401
    )
    response = client.get(
        "/api/voice-tools/health",
        headers={"X-Voice-Tool-Secret": "test-tool-secret"},
    )
    assert response.status_code == 200


def test_availability_returns_typed_envelope(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_check(date: str, time: str, party_size: int, **kwargs):
        return {
            "available": True,
            "date": date,
            "time": time,
            "party_size": party_size,
            "tables": [],
            "alternatives": [],
        }

    monkeypatch.setattr(tool_api.restaurant_service, "check_availability", fake_check)
    response = client.post(
        "/api/voice-tools/availability",
        headers={"X-Voice-Tool-Secret": "test-tool-secret"},
        json={"date": "2026-09-01", "time": "19:00", "party_size": 2},
    )
    assert response.status_code == 200
    assert response.json()["result"]["available"] is True


def test_write_forwards_idempotency_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    async def fake_create_booking(**kwargs):
        captured.update(kwargs)
        return {"booking_id": 9, "created": True}

    monkeypatch.setattr(
        tool_api.restaurant_service,
        "create_booking",
        fake_create_booking,
    )
    response = client.post(
        "/api/voice-tools/bookings/create",
        headers={
            "X-Voice-Tool-Secret": "test-tool-secret",
            "Idempotency-Key": "retell-call-1-booking-1",
        },
        json={
            "call_id": "call-1",
            "customer_name": "Taylor",
            "customer_phone": "+14155550123",
            "date": "2026-09-01",
            "time": "19:00",
            "party_size": 2,
            "notes": "",
            "confirmed": True,
        },
    )
    assert response.status_code == 200
    assert captured["idempotency_key"] == "retell-call-1-booking-1"
    assert captured["confirmed"] is True
