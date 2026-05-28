"""LangGraph stores backed by Azure AI services.

This package provides ``AzureAIMemoryStore``, a persisted LangGraph
``BaseStore`` implementation that uses Azure AI Memory Stores via the
Azure AI Projects SDK V2 (``azure-ai-projects>=2.0.0b4``).

Example:
    ```python
    from langchain_azure_ai.stores.memory import AzureAIMemoryStore
    from azure.identity import DefaultAzureCredential

    store = AzureAIMemoryStore(
        memory_store_name="my-memory-store",
        project_endpoint="https://my-resource.services.ai.azure.com/api/projects/my-project",
        credential=DefaultAzureCredential(),
    )

    store.put("users", "alice", {"content": "User prefers a dark theme"})
    item = store.get("users", "alice")
    ```
"""

from langchain_azure_ai.stores.memory.azure_ai_memory import (
    CONTENT_KEY,
    NAMESPACE_AUTHENTICATED_USER,
    AzureAIMemoryStore,
)

__all__ = ["AzureAIMemoryStore", "NAMESPACE_AUTHENTICATED_USER", "CONTENT_KEY"]
