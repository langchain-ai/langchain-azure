"""**Chat message history** stores a history of the message interactions in a chat.

**Class hierarchy:**

```output
BaseChatMessageHistory --> <name>ChatMessageHistory  # Examples: CosmosDBChatMessageHistory
```

**Main helpers:**

```output
AIMessage, HumanMessage, BaseMessage
```
"""  # noqa: E501

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.chat_message_histories.azure_ai_memory import (
        AzureAIMemoryChatMessageHistory,
    )
    from langchain_azure_ai.chat_message_histories.cosmos_db import (
        CosmosDBChatMessageHistory,
    )

__all__ = [
    "AzureAIMemoryChatMessageHistory",
    "AzureAIMemoryRetriever",
    "CosmosDBChatMessageHistory",
]

_module_lookup = {
    "AzureAIMemoryChatMessageHistory": (
        "langchain_azure_ai.chat_message_histories.azure_ai_memory"
    ),
    "AzureAIMemoryRetriever": "langchain_azure_ai.retrievers.azure_ai_memory_retriever",
    "CosmosDBChatMessageHistory": (
        "langchain_azure_ai.chat_message_histories.cosmos_db"
    ),
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
