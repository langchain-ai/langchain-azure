"""Document loaders provided by Azure AI Foundry."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.document_loaders.content_understanding import (
        AzureAIContentUnderstandingLoader,
        OutputMode,
    )

_MODULE_MAP = {
    "AzureAIContentUnderstandingLoader": (
        "langchain_azure_ai.document_loaders.content_understanding"
    ),
    "OutputMode": "langchain_azure_ai.document_loaders.content_understanding",
}


def __getattr__(name: str) -> Any:
    """Lazy-load document loader classes and type aliases."""
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AzureAIContentUnderstandingLoader",
    "OutputMode",
]
