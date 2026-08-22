"""Bounded sequential tool executor for the custom-LLM rollback adapter.

Reads run before writes in the same model batch so get_order_summary can
satisfy the confirm_order gate, and create_booking can land before add_order_item.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from app.tools import ALL_TOOLS

_TOOLS = {tool.name: tool for tool in ALL_TOOLS}
_READ_TOOLS = {
    "search_menu",
    "search_restaurant_info",
    "check_table_availability",
    "get_reservation_draft",
    "lookup_booking",
    "get_order_summary",
    "lookup_order",
    "get_full_menu",
    "check_menu_item_availability",
}


def _call_name(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("name") or "")
    return str(getattr(call, "name", "") or "")


def _call_args(call: Any) -> dict:
    if isinstance(call, dict):
        return dict(call.get("args") or {})
    return dict(getattr(call, "args", None) or {})


def _call_id(call: Any) -> str:
    if isinstance(call, dict):
        return str(call.get("id") or "")
    return str(getattr(call, "id", "") or "")


async def tool_node(state: dict, config: RunnableConfig) -> dict:
    message = state["messages"][-1]
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    tool_calls.sort(key=lambda call: 0 if _call_name(call) in _READ_TOOLS else 1)
    outputs: list[ToolMessage] = []
    for call in tool_calls:
        name = _call_name(call)
        tool = _TOOLS.get(name)
        call_id = _call_id(call)
        if tool is None:
            outputs.append(
                ToolMessage(
                    content=f"unknown_tool: {name}",
                    tool_call_id=call_id,
                    name=name,
                )
            )
            continue
        try:
            result = await tool.ainvoke(_call_args(call), config=config)
            if isinstance(result, str):
                content = result
            else:
                content = json.dumps(result, default=str)
        except Exception as error:
            content = f"error: {error}"
        outputs.append(
            ToolMessage(content=content, tool_call_id=call_id, name=name)
        )
    return {
        "messages": outputs,
        "tool_iterations": int(state.get("tool_iterations", 0)) + 1,
    }
