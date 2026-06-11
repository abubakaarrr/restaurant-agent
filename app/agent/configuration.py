"""Agent configuration — swappable at runtime via LangGraph config."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional

from langchain_core.runnables import RunnableConfig, ensure_config
from app.config import settings


@dataclass(kw_only=True)
class AgentConfiguration:
    """Runtime configuration passed into the LangGraph graph."""

    restaurant_name: str = field(default_factory=lambda: settings.restaurant_name)
    model: str = field(default="gpt-4o")
    max_response_tokens: int = field(default=300)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> AgentConfiguration:
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
