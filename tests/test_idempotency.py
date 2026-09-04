from __future__ import annotations

import json

import pytest

import app.services.restaurant as restaurant_module
from app.config import settings
from app.services.restaurant import (
    RestaurantService,
    RestaurantServiceError,
    WritesDisabledError,
)


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeConnection:
    def __init__(self) -> None:
        self.ledger: dict[tuple[str, str], dict] = {}

    def transaction(self):
        return _AsyncContext(self)

    async def fetchrow(self, query: str, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("INSERT INTO voice_action_idempotency"):
            action, key, call_id, request_hash = args
            ledger_key = (action, key)
            if ledger_key in self.ledger:
                return None
            self.ledger[ledger_key] = {
                "request_hash": request_hash,
                "status": "processing",
                "response": None,
                "call_id": call_id,
            }
            return {"id": 1}
        if "FROM voice_action_idempotency" in normalized:
            action, key = args
            return self.ledger.get((action, key))
        raise AssertionError(f"Unexpected fetchrow query: {normalized}")

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split())
        if normalized.startswith("UPDATE voice_action_idempotency"):
            action, key, response = args
            row = self.ledger[(action, key)]
            row["status"] = "completed"
            row["response"] = json.loads(response)
            return "UPDATE 1"
        raise AssertionError(f"Unexpected execute query: {normalized}")


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self):
        return _AsyncContext(self.connection)


@pytest.mark.asyncio
async def test_idempotent_write_executes_once_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection()
    pool = FakePool(connection)

    async def fake_get_pool():
        return pool

    monkeypatch.setattr(restaurant_module, "get_pool", fake_get_pool)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    service = RestaurantService()
    executions = 0

    async def operation(_conn):
        nonlocal executions
        executions += 1
        return {"booking_id": 42}

    first, first_replay = await service._idempotent_write(
        action="create_booking",
        idempotency_key="request-123",
        call_id="call-1",
        payload={"date": "2026-09-01"},
        operation=operation,
    )
    second, second_replay = await service._idempotent_write(
        action="create_booking",
        idempotency_key="request-123",
        call_id="call-1",
        payload={"date": "2026-09-01"},
        operation=operation,
    )

    assert first == second == {"booking_id": 42}
    assert first_replay is False
    assert second_replay is True
    assert executions == 1


@pytest.mark.asyncio
async def test_idempotency_key_cannot_be_reused_for_different_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection()

    async def fake_get_pool():
        return FakePool(connection)

    monkeypatch.setattr(restaurant_module, "get_pool", fake_get_pool)
    monkeypatch.setattr(settings, "voice_live_writes_enabled", True)
    service = RestaurantService()

    async def operation(_conn):
        return {"ok": True}

    await service._idempotent_write(
        action="cancel_booking",
        idempotency_key="request-456",
        call_id="call-1",
        payload={"booking_id": 1},
        operation=operation,
    )
    with pytest.raises(RestaurantServiceError, match="different inputs"):
        await service._idempotent_write(
            action="cancel_booking",
            idempotency_key="request-456",
            call_id="call-1",
            payload={"booking_id": 2},
            operation=operation,
        )


@pytest.mark.asyncio
async def test_live_writes_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "voice_live_writes_enabled", False)
    service = RestaurantService()

    async def operation(_conn):
        raise AssertionError("operation must not run")

    with pytest.raises(WritesDisabledError):
        await service._idempotent_write(
            action="create_booking",
            idempotency_key="request-789",
            call_id="call-1",
            payload={},
            operation=operation,
        )
