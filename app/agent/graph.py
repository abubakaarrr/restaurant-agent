"""
LangGraph state graph for the restaurant AI receptionist.

Flow:
  START → agent → (has tool calls?) → tool_node → agent → ... → END
                        ↓ (no tool calls)
                       END

The rollback agent decides which tools to call based on the caller's message.
Available tools:
  - search_menu              → RAG on menu.md
  - search_restaurant_info   → RAG on restaurant_info.md + slots.md
  - check_table_availability → live PostgreSQL query
  - create_booking           → atomic PostgreSQL insert with row lock
  - check_menu_item_availability → live stock check
"""

from __future__ import annotations
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.agent.state import RestaurantAgentState
from app.agent.nodes import generate_response, tool_node
from app.config import settings

# ── Build graph ───────────────────────────────────────────────

MAX_TOOL_ITERATIONS = 3


def route_after_agent(state: RestaurantAgentState) -> Literal["tools", "limit", "__end__"]:
    message = state["messages"][-1]
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        return END
    if state.get("tool_iterations", 0) >= MAX_TOOL_ITERATIONS:
        return "limit"
    return "tools"


async def tool_limit_response(state: RestaurantAgentState) -> dict:
    if settings.staff_transfer_number:
        recovery = "I can connect you with the restaurant staff."
    else:
        recovery = "I can't transfer right now, but I can take a callback message for the restaurant staff."
    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry, I couldn't complete that safely. "
                    + recovery
                )
            )
        ]
    }

builder = StateGraph(RestaurantAgentState)

builder.add_node("agent", generate_response)
builder.add_node("tools", tool_node)
builder.add_node("limit", tool_limit_response)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"tools": "tools", "limit": "limit", END: END},
)
builder.add_edge("tools", "agent")  # after tool result → back to agent for final response
builder.add_edge("limit", END)

restaurant_agent = builder.compile()
restaurant_agent.name = "restaurant_receptionist"
