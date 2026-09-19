"""Realtime WebSocket transport and privacy-safe event recording."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Mapping, Protocol


class RealtimeTransport(Protocol):
    async def send(self, event: Mapping[str, Any]) -> None: ...

    async def receive(self) -> Mapping[str, Any]: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class ProtocolEvent:
    sequence: int
    event_type: str
    recorded_at: float
    payload: dict[str, Any] = field(default_factory=dict)


class EventRecorder:
    """Record protocol shape and hashes, never raw audio or credentials."""

    def __init__(self) -> None:
        self.events: list[ProtocolEvent] = []

    def record(self, event: Mapping[str, Any], *, event_type: str | None = None) -> ProtocolEvent:
        safe = self._safe_payload(event)
        item = ProtocolEvent(
            sequence=len(self.events) + 1,
            event_type=event_type or str(event.get("type") or "unknown"),
            recorded_at=monotonic(),
            payload=safe,
        )
        self.events.append(item)
        return item

    def record_audio(self, event_type: str, audio: bytes, **extra: Any) -> ProtocolEvent:
        return self.record(
            {
                "type": event_type,
                "audio_bytes": len(audio),
                "audio_sha256": hashlib.sha256(audio).hexdigest(),
                **extra,
            }
        )

    @staticmethod
    def _safe_payload(event: Mapping[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in event.items():
            if key in {"authorization", "api_key", "client_secret", "audio", "delta"}:
                if key in {"audio", "delta"} and isinstance(value, str):
                    try:
                        raw = base64.b64decode(value)
                    except Exception:
                        raw = value.encode("utf-8")
                    safe[f"{key}_bytes"] = len(raw)
                    safe[f"{key}_sha256"] = hashlib.sha256(raw).hexdigest()
                continue
            if isinstance(value, Mapping):
                safe[key] = EventRecorder._safe_payload(value)
            elif isinstance(value, list):
                safe[key] = [EventRecorder._safe_payload(item) if isinstance(item, Mapping) else item for item in value]
            else:
                safe[key] = value
        return safe


class WebSocketRealtimeTransport:
    """Server-to-server GA Realtime WebSocket transport."""

    def __init__(self, socket: Any) -> None:
        self._socket = socket

    @classmethod
    async def connect(cls, *, api_key: str, model: str) -> "WebSocketRealtimeTransport":
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the development Realtime adapter")
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt; the development adapter requires websockets") from exc
        socket = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={model}",
            additional_headers={"Authorization": f"Bearer {api_key}"},
            max_size=None,
        )
        return cls(socket)

    async def send(self, event: Mapping[str, Any]) -> None:
        await self._socket.send(json.dumps(dict(event), separators=(",", ":")))

    async def receive(self) -> Mapping[str, Any]:
        value = await self._socket.recv()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    async def close(self) -> None:
        await self._socket.close()


class MemoryRealtimeTransport:
    """Scriptable transport for deterministic protocol tests."""

    def __init__(self, incoming: list[Mapping[str, Any]] | None = None) -> None:
        self.incoming = list(incoming or [])
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, event: Mapping[str, Any]) -> None:
        self.sent.append(dict(event))

    async def receive(self) -> Mapping[str, Any]:
        if not self.incoming:
            raise RuntimeError("MemoryRealtimeTransport has no scripted event")
        return dict(self.incoming.pop(0))

    async def close(self) -> None:
        self.closed = True
