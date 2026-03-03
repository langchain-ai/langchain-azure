"""LangGraph stores backed by Azure AI services.

This package provides ``AzureAIMemoryStore``, a persisted LangGraph
``BaseStore`` implementation that uses Azure AI Memory Stores via the
Azure AI Projects SDK V2 (``azure-ai-projects>=2.0.0b4``).

Example:
    ```python
    from langchain_azure_ai.stores import AzureAIMemoryStore
    from azure.identity import DefaultAzureCredential

    store = AzureAIMemoryStore(
        memory_store_name="my-memory-store",
        endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
        credential=DefaultAzureCredential(),
    )

    store.put(("users", "alice"), "prefs", {"theme": "dark"})
    item = store.get(("users", "alice"), "prefs")
    ```
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.stores.azure_ai_memory import AzureAIMemoryStore

__all__ = ["AzureAIMemoryStore"]

_module_lookup = {
    "AzureAIMemoryStore": "langchain_azure_ai.stores.azure_ai_memory",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
