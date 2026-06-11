"""
LangGraph state graph for the restaurant AI receptionist.

Flow:
  START → agent → (has tool calls?) → tool_node → agent → ... → END
                        ↓ (no tool calls)
                       END

The agent (Claude) decides which tools to call based on the caller's message.
Available tools:
  - search_menu              → RAG on menu.md
  - search_restaurant_info   → RAG on restaurant_info.md + slots.md
  - check_table_availability → live PostgreSQL query
  - create_booking           → atomic PostgreSQL insert with row lock
  - check_menu_item_availability → live stock check
"""

from __future__ import annotations
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.agent.configuration import AgentConfiguration
from app.agent.state import RestaurantAgentState
from app.agent.nodes import generate_response, tool_node

# ── Build graph ───────────────────────────────────────────────

builder = StateGraph(RestaurantAgentState, config_schema=AgentConfiguration)

builder.add_node("agent", generate_response)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition,          # LangGraph built-in: routes to "tools" if tool_calls present
    {"tools": "tools", END: END},
)
builder.add_edge("tools", "agent")  # after tool result → back to agent for final response

restaurant_agent = builder.compile()
restaurant_agent.name = "restaurant_receptionist"
