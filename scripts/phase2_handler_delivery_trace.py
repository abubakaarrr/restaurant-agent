#!/usr/bin/env python3
"""Execute the Retell handler against a deterministic interruption transport."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import WebSocketDisconnect

import app.retell_handler as handler
from app.behavior import BehaviorState
from app.call_flags import clear_call_control, request_transfer


CALL_ID = "phase2-handler-trace"


class TraceWebSocket:
    def __init__(self, old_id: int, new_id: int, trace: list[dict[str, Any]]) -> None:
        self.old_id = old_id
        self.new_id = new_id
        self.trace = trace
        self.receive_count = 0
        self.old_turn_started = asyncio.Event()
        self.new_response_complete = asyncio.Event()

    async def receive_text(self) -> str:
        self.receive_count += 1
        if self.receive_count == 1:
            self.trace.append({"event": "customer_turn", "response_id": self.old_id})
            return json.dumps(
                {
                    "interaction_type": "response_required",
                    "response_id": self.old_id,
                    "transcript": [{"role": "user", "content": "Old request"}],
                }
            )
        if self.receive_count == 2:
            await asyncio.wait_for(self.old_turn_started.wait(), timeout=1)
            self.trace.append({"event": "customer_turn", "response_id": self.new_id})
            return json.dumps(
                {
                    "interaction_type": "response_required",
                    "response_id": self.new_id,
                    "transcript": [
                        {"role": "user", "content": "Old request"},
                        {"role": "user", "content": "New request"},
                    ],
                }
            )
        await asyncio.wait_for(self.new_response_complete.wait(), timeout=1)
        raise WebSocketDisconnect()

    async def send_text(self, payload: str) -> None:
        message = json.loads(payload)
        if message.get("response_type") != "response":
            return
        self.trace.append(
            {
                "event": "assistant_delivery",
                "response_id": message["response_id"],
                "content": message.get("content", ""),
                "content_complete": bool(message.get("content_complete")),
                "end_call": bool(message.get("end_call")),
                "transfer_number": message.get("transfer_number") or None,
            }
        )
        if (
            message["response_id"] == self.new_id
            and message.get("content_complete")
        ):
            self.new_response_complete.set()


async def execute(old_id: int, new_id: int) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    websocket = TraceWebSocket(old_id, new_id, trace)

    async def fake_load(_call_id: str) -> BehaviorState:
        return BehaviorState()

    async def fake_save(_call_id: str, _state: BehaviorState) -> None:
        return None

    async def fake_caller_turn(_call_id: str, user_text: str) -> dict[str, Any]:
        if user_text == "Old request":
            websocket.old_turn_started.set()
            request_transfer(CALL_ID, "human_requested", "+14155550123")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {"handled": True, "message": "STALE DIRECT REPLY"}
        return {"handled": False, "kind": "caller_turn", "affirmation": None}

    async def fake_stream(*_args, **_kwargs):
        yield "New response."

    handler.load_behavior_state = fake_load
    handler.save_behavior_state = fake_save
    handler.process_caller_turn = fake_caller_turn
    handler.stream_agent_tokens = fake_stream
    handler._record_background = lambda *_args, **_kwargs: None

    try:
        await handler.handle_retell_connection(websocket, CALL_ID)
    finally:
        clear_call_control(CALL_ID)

    new_turn_index = next(
        index
        for index, event in enumerate(trace)
        if event == {"event": "customer_turn", "response_id": new_id}
    )
    delivered_after_new_turn = [
        event
        for event in trace[new_turn_index + 1 :]
        if event["event"] == "assistant_delivery"
    ]
    stale = [
        event for event in delivered_after_new_turn if event["response_id"] != new_id
    ]
    leaked_controls = [
        event
        for event in delivered_after_new_turn
        if event["response_id"] == new_id
        and (event["end_call"] or event["transfer_number"])
    ]
    return {
        "trace": trace,
        "stale_response_incidents": len(stale),
        "cancelled_control_leaks": len(leaked_controls),
    }


def main() -> int:
    old_id = int(sys.argv[1])
    new_id = int(sys.argv[2])
    print(json.dumps(asyncio.run(execute(old_id, new_id)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
