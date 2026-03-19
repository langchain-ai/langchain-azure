"""Guardrails for Azure AI LangChain/LangGraph integrations.

This module provides middleware and utilities for adding safety guardrails
to any LangGraph ``StateGraph``.

Classes:
    AzureContentSafetyMiddleware: AgentMiddleware that screens messages using
        Azure AI Content Safety.

Exceptions:
    ContentSafetyViolationError: Raised when content safety violations are
        detected with ``action='block'``.

Functions:
    apply_middleware: Add AgentMiddleware before/after hooks to any LangGraph
        StateGraph (not specific to the Agent Service).
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.guardrails._content_safety import (
        AzureContentSafetyMiddleware,
        ContentSafetyViolationError,
    )
    from langchain_azure_ai.guardrails._middleware import apply_middleware

__all__ = [
    "apply_middleware",
    "AzureContentSafetyMiddleware",
    "ContentSafetyViolationError",
]

_module_lookup = {
    "apply_middleware": "langchain_azure_ai.guardrails._middleware",
    "AzureContentSafetyMiddleware": "langchain_azure_ai.guardrails._content_safety",
    "ContentSafetyViolationError": "langchain_azure_ai.guardrails._content_safety",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
