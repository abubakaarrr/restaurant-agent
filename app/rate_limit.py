"""Small in-process limiter for the single-node pilot.

The public Retell hosted widget has its own key/domain/reCAPTCHA controls. This
limiter protects local demo endpoints; move it to Redis before horizontal
scaling.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class SlidingWindowLimiter:
    def __init__(self, *, limit: int, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, request: Request, *, scope: str) -> None:
        client = request.client.host if request.client else "unknown"
        key = f"{scope}:{client}"
        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests; try again shortly",
                    headers={"Retry-After": str(retry_after)},
                )
            events.append(now)


chat_limiter = SlidingWindowLimiter(limit=30)
web_call_limiter = SlidingWindowLimiter(limit=5)
login_limiter = SlidingWindowLimiter(limit=5, window_seconds=300)
