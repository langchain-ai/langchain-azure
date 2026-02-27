"""**Memory** module for Azure AI Foundry Memory integration with LangChain.

Azure AI Foundry Memory provides a managed, long-term memory layer that extracts and
consolidates durable facts across chat sessions, partitioned by scope (user/tenant).

**Main helpers:**

```output
FoundryMemoryChatMessageHistory, FoundryMemoryRetriever
```
"""  # noqa: E501

from langchain_azure_ai.memory.foundry_memory import (
    FoundryMemoryChatMessageHistory,
    FoundryMemoryRetriever,
)

__all__ = [
    "FoundryMemoryChatMessageHistory",
    "FoundryMemoryRetriever",
]
