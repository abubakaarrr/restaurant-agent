"""Minimal, retention-aware call telemetry persistence."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings
from app.db_pool import get_pool


logger = logging.getLogger(__name__)


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str, ensure_ascii=False)


def _event_id(payload: dict[str, Any], raw_body: bytes) -> str:
    explicit = payload.get("event_id") or payload.get("id")
    if explicit:
        return str(explicit)
    return hashlib.sha256(raw_body).hexdigest()


def _call_id(call: dict[str, Any]) -> str:
    return str(call.get("call_id") or call.get("id") or "")


def _safe_call_payload(call: dict[str, Any]) -> dict[str, Any]:
    """Allowlist operational fields and omit customer PII by default."""
    safe_keys = (
        "call_id",
        "call_type",
        "agent_id",
        "agent_version",
        "start_timestamp",
        "end_timestamp",
        "duration_ms",
        "disconnection_reason",
        "call_status",
        "latency",
        "latency_metrics",
        "transfer_destination",
        "transfer_successful",
    )
    safe = {key: call[key] for key in safe_keys if key in call}
    analysis = call.get("call_analysis") or call.get("analysis")
    if isinstance(analysis, dict):
        safe["analysis"] = {
            key: analysis[key]
            for key in (
                "user_sentiment",
                "call_successful",
                "call_summary",
                "custom_analysis_data",
            )
            if key in analysis
        }
    if settings.store_call_transcripts:
        for key in ("transcript", "transcript_object", "transcript_with_tool_calls"):
            if key in call:
                safe[key] = call[key]
    return safe


async def record_call_event(
    call_id: str,
    event_type: str,
    *,
    response_id: int | None = None,
    duration_ms: int | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if not call_id:
        return
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO call_events
                (call_id, provider, event_type, response_id, duration_ms, payload)
            VALUES ($1, 'retell', $2, $3, $4, $5::jsonb)
            """,
            call_id,
            event_type,
            response_id,
            duration_ms,
            _json(payload or {}),
        )


async def ingest_retell_webhook(payload: dict[str, Any], raw_body: bytes) -> bool:
    """Persist one lifecycle webhook. Returns False for a duplicate delivery."""
    event_type = str(payload.get("event") or payload.get("event_type") or "unknown")
    call = payload.get("call") if isinstance(payload.get("call"), dict) else {}
    call_id = _call_id(call)
    event_id = _event_id(payload, raw_body)
    safe_payload = _safe_call_payload(call)
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            inserted = await conn.fetchrow(
                """
                INSERT INTO provider_webhook_events
                    (provider, event_id, event_type, call_id, payload)
                VALUES ('retell', $1, $2, $3, $4::jsonb)
                ON CONFLICT (provider, event_id) DO NOTHING
                RETURNING event_id
                """,
                event_id,
                event_type,
                call_id,
                _json(safe_payload),
            )
            if not inserted:
                return False

            if call_id:
                ended = event_type in {
                    "call_ended",
                    "call_analyzed",
                    "transfer_ended",
                }
                await conn.execute(
                    """
                    INSERT INTO call_sessions
                        (session_id, provider, metadata, ended_at, updated_at)
                    VALUES ($1, 'retell', $2::jsonb, $3, NOW())
                    ON CONFLICT (session_id) DO UPDATE
                    SET provider = 'retell',
                        metadata = call_sessions.metadata || EXCLUDED.metadata,
                        ended_at = COALESCE(EXCLUDED.ended_at, call_sessions.ended_at),
                        updated_at = NOW()
                    """,
                    call_id,
                    _json(safe_payload),
                    datetime.now(timezone.utc).replace(tzinfo=None) if ended else None,
                )
                await conn.execute(
                    """
                    INSERT INTO call_events
                        (call_id, provider, event_type, duration_ms, payload)
                    VALUES ($1, 'retell', $2, $3, $4::jsonb)
                    """,
                    call_id,
                    event_type,
                    call.get("duration_ms"),
                    _json(safe_payload),
                )
    return True


async def purge_expired_call_data() -> None:
    """Delete retained call telemetry beyond the configured policy."""
    days = max(1, settings.call_data_retention_days)
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM call_events WHERE created_at < $1", cutoff)
        await conn.execute(
            "DELETE FROM provider_webhook_events WHERE received_at < $1",
            cutoff,
        )
        await conn.execute(
            """
            UPDATE call_sessions
            SET metadata = '{}', behavior_state = '{}', state = '{}'
            WHERE ended_at IS NOT NULL AND ended_at < $1
            """,
            cutoff,
        )
    logger.info("Applied call-data retention policy (%s days)", days)
