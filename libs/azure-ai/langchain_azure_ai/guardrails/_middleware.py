"""Backward-compatibility shim.

The implementation has moved to
:mod:`langchain_azure_ai.agents.middleware._middleware`.
"""

from langchain_azure_ai.agents.middleware._middleware import (  # noqa: F401
    _resolve_state_schema,
    apply_middleware,
)

__all__ = [
    "apply_middleware",
    "_resolve_state_schema",
]
