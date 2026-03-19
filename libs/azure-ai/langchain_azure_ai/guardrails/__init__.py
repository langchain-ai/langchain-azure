"""Guardrails for Azure AI LangChain/LangGraph integrations.

.. deprecated::
    This module is preserved for backward compatibility.  The canonical
    location is :mod:`langchain_azure_ai.agents.middleware`.

    Update your imports::

        # Old
        from langchain_azure_ai.guardrails import AzureContentSafetyMiddleware
        # New
        from langchain_azure_ai.agents.middleware import AzureContentSafetyMiddleware
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.middleware._content_safety import (
        AzureContentSafetyImageMiddleware,
        AzureContentSafetyMiddleware,
        AzurePromptShieldMiddleware,
        AzureProtectedMaterialMiddleware,
        ContentSafetyViolationError,
    )

__all__ = [
    "AzureContentSafetyMiddleware",
    "AzureContentSafetyImageMiddleware",
    "AzureProtectedMaterialMiddleware",
    "AzurePromptShieldMiddleware",
    "ContentSafetyViolationError",
]

_module_lookup = {
    "AzureContentSafetyMiddleware": "langchain_azure_ai.agents.middleware._content_safety",
    "AzureContentSafetyImageMiddleware": "langchain_azure_ai.agents.middleware._content_safety",
    "AzureProtectedMaterialMiddleware": "langchain_azure_ai.agents.middleware._content_safety",
    "AzurePromptShieldMiddleware": "langchain_azure_ai.agents.middleware._content_safety",
    "ContentSafetyViolationError": "langchain_azure_ai.agents.middleware._content_safety",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



