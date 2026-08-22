"""Durable behavior-state snapshots for custom-LLM reconnect safety."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields
from enum import Enum
from typing import Any

from app.behavior import BehaviorState, initial_behavior_state
from app.db_pool import get_pool


logger = logging.getLogger(__name__)
_STATE_FIELDS = {item.name for item in fields(BehaviorState)}


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def serialize_behavior_state(state: BehaviorState) -> dict[str, Any]:
    return _plain(asdict(state))


def deserialize_behavior_state(value: Any) -> BehaviorState:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return initial_behavior_state()
    if not isinstance(value, dict):
        return initial_behavior_state()
    accepted = {key: value[key] for key in _STATE_FIELDS if key in value}
    try:
        return BehaviorState(**accepted)
    except (TypeError, ValueError):
        return initial_behavior_state()


async def load_behavior_state(call_id: str) -> BehaviorState:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            value = await conn.fetchval(
                "SELECT behavior_state FROM call_sessions WHERE session_id = $1",
                call_id,
            )
        return deserialize_behavior_state(value)
    except Exception:
        logger.warning("Could not load behavior state for %s", call_id, exc_info=True)
        return initial_behavior_state()


async def save_behavior_state(call_id: str, state: BehaviorState) -> None:
    payload = json.dumps(serialize_behavior_state(state))
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO call_sessions
                    (session_id, provider, behavior_state, updated_at)
                VALUES ($1, 'retell', $2::jsonb, NOW())
                ON CONFLICT (session_id) DO UPDATE
                SET behavior_state = EXCLUDED.behavior_state,
                    updated_at = NOW()
                """,
                call_id,
                payload,
            )
    except Exception:
        logger.warning("Could not persist behavior state for %s", call_id, exc_info=True)
