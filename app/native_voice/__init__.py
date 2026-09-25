"""Development-only OpenAI Realtime integration.

Nothing in the production application imports this package.  The package is
deliberately explicit at its boundary: callers construct an adapter, supply a
transport, and own the lifecycle of a synthetic development session.
"""

from app.native_voice.contracts import (
    CorrectionRecord,
    OrderItemState,
    OrderPatch,
    OrderState,
    UnresolvedField,
)

__all__ = [
    "CorrectionRecord",
    "OrderItemState",
    "OrderPatch",
    "OrderState",
    "UnresolvedField",
]
