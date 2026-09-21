"""Completed caller-turn assembly for the Realtime transcript stream."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True)
class CompletedCallerTurn:
    turn_id: str
    version: int
    transcript: str
    finalized_at: float


class TurnAssembler:
    """Keep transcript deltas provisional until one completion event arrives."""

    def __init__(self) -> None:
        self._deltas: dict[str, list[str]] = {}
        self._completed: dict[str, CompletedCallerTurn] = {}
        self._next_version = 1
        self._active_turn_id = ""

    @property
    def active_turn_id(self) -> str:
        return self._active_turn_id

    def start(self, turn_id: str) -> None:
        if not turn_id:
            raise ValueError("turn_id is required")
        self._active_turn_id = turn_id
        self._deltas.setdefault(turn_id, [])

    def add_delta(self, delta: str, *, turn_id: str = "") -> None:
        resolved = turn_id or self._active_turn_id
        if not resolved or resolved in self._completed:
            return
        self._deltas.setdefault(resolved, []).append(delta or "")

    def finalize(self, transcript: str | None = None, *, turn_id: str = "") -> CompletedCallerTurn:
        resolved = turn_id or self._active_turn_id
        if not resolved:
            raise ValueError("no active caller turn")
        existing = self._completed.get(resolved)
        if existing is not None:
            return existing
        assembled = transcript if transcript is not None else "".join(self._deltas.get(resolved, []))
        completed = CompletedCallerTurn(
            turn_id=resolved,
            version=self._next_version,
            transcript=assembled,
            finalized_at=monotonic(),
        )
        self._next_version += 1
        self._completed[resolved] = completed
        self._deltas.pop(resolved, None)
        return completed

    def completed(self, turn_id: str = "") -> CompletedCallerTurn | None:
        return self._completed.get(turn_id or self._active_turn_id)

    def reset(self) -> None:
        """Invalidate provisional text after an interruption."""
        if self._active_turn_id:
            self._deltas.pop(self._active_turn_id, None)
            self._completed.pop(self._active_turn_id, None)
        self._active_turn_id = ""
