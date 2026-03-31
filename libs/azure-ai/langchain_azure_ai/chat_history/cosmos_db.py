"""Azure CosmosDB Memory History — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Import directly from there instead::

    from langchain_azure_cosmosdb import CosmosDBChatMessageHistory
"""

import warnings

from langchain_azure_cosmosdb.langchain._chat_history import (  # noqa: F401
    CosmosDBChatMessageHistory,
)

warnings.warn(
    "Importing CosmosDBChatMessageHistory from "
    "'langchain_azure_ai.chat_history.cosmos_db' is deprecated. "
    "Use 'from langchain_azure_cosmosdb import "
    "CosmosDBChatMessageHistory' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CosmosDBChatMessageHistory"]
