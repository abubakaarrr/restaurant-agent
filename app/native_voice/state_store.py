"""Persistence boundary for application-owned native voice state."""

from __future__ import annotations

import json
from typing import Protocol

from app.native_voice.contracts import OrderState


class StateVersionConflict(RuntimeError):
    """A stale adapter attempted to overwrite newer application state."""


class OrderStateStore(Protocol):
    async def load(self, session_id: str) -> OrderState: ...

    async def save(self, session_id: str, state: OrderState, *, expected_version: int) -> None: ...


class InMemoryOrderStateStore:
    """Deterministic store used by offline acceptance tests and local probes."""

    def __init__(self) -> None:
        self._states: dict[str, OrderState] = {}

    async def load(self, session_id: str) -> OrderState:
        return self._states.get(session_id, OrderState())

    async def save(self, session_id: str, state: OrderState, *, expected_version: int) -> None:
        current = self._states.get(session_id, OrderState())
        if current.version != expected_version:
            raise StateVersionConflict(
                f"state version conflict for {session_id}: expected {expected_version}, current {current.version}"
            )
        self._states[session_id] = OrderState.from_dict(state.to_dict())


class CallSessionOrderStateStore:
    """Persist only typed state in the existing ``call_sessions.state`` JSONB.

    This is intentionally not wired into production startup.  It uses the
    existing application persistence boundary and never stores transcript or
    audio content.
    """

    async def load(self, session_id: str) -> OrderState:
        from app.native_voice.database_guard import get_native_voice_pool

        pool = await get_native_voice_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state->'native_voice_order' AS native_state FROM call_sessions WHERE session_id = $1",
                session_id,
            )
        value = row["native_state"] if row else None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        return OrderState.from_dict(value if isinstance(value, dict) else None)

    async def save(self, session_id: str, state: OrderState, *, expected_version: int) -> None:
        from app.native_voice.database_guard import get_native_voice_pool

        pool = await get_native_voice_pool()
        payload = json.dumps(state.to_dict(), sort_keys=True)
        async with pool.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE call_sessions
                    SET state = COALESCE(call_sessions.state, '{}'::jsonb)
                               || jsonb_build_object('native_voice_order', $2::jsonb),
                        updated_at = NOW()
                    WHERE session_id = $1
                      AND COALESCE(NULLIF(call_sessions.state->'native_voice_order'->>'version', '')::int, 0) = $3
                    RETURNING session_id
                    """,
                    session_id,
                    payload,
                    expected_version,
                )
                if row is None and expected_version == 0:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO call_sessions (session_id, state)
                        VALUES ($1, jsonb_build_object('native_voice_order', $2::jsonb))
                        ON CONFLICT (session_id) DO NOTHING
                        RETURNING session_id
                        """,
                        session_id,
                        payload,
                    )
                if row is None:
                    raise StateVersionConflict(
                        f"state version conflict for {session_id}: expected {expected_version}"
                    )
