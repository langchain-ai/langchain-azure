"""Backward-compatibility shim.

The implementation has moved to
:mod:`langchain_azure_ai.agents.middleware._middleware`.
"""

from langchain_azure_ai.agents.middleware._middleware import (  # noqa: F401
    _resolve_state_schema,
)

__all__ = [
    "_resolve_state_schema",
]
