"""State definition for the restaurant agent graph."""

from __future__ import annotations
from langgraph.graph import MessagesState


class RestaurantAgentState(MessagesState):
    """
    Extends MessagesState with restaurant-specific session context.
    MessagesState already provides: messages: list[BaseMessage]
    """
    session_id: str = ""
    caller_phone: str = ""
    turn_count: int = 0
