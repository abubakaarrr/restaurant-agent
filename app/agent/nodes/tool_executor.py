"""Tool executor node — runs whichever tool Claude decided to call."""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from app.tools import ALL_TOOLS

# LangGraph's built-in ToolNode handles calling each tool and
# wrapping the result in a ToolMessage automatically.
tool_node = ToolNode(ALL_TOOLS)
