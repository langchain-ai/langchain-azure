"""**Chat message history** stores a history of the message interactions in a chat.

**Class hierarchy:**

.. code-block::

    BaseChatMessageHistory --> <name>ChatMessageHistory  # Examples: CosmosDBChatMessageHistory

**Main helpers:**

.. code-block::

    AIMessage, HumanMessage, BaseMessage

"""  # noqa: E501

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.chat_message_histories.cosmos_db import (
        CosmosDBChatMessageHistory,
    )

__all__ = [
    "CosmosDBChatMessageHistory",
]

_module_lookup = {
    "CosmosDBChatMessageHistory": "langchain_azure_ai.chat_message_histories.cosmos_db",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
