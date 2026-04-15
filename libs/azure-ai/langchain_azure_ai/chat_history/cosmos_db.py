"""Azure CosmosDB Memory History — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Install and import directly from there instead::

    pip install langchain-azure-cosmosdb
    from langchain_azure_cosmosdb import CosmosDBChatMessageHistory
"""

import warnings
from typing import Any

__all__ = ["CosmosDBChatMessageHistory"]  # noqa: F822


def __getattr__(name: str) -> Any:
    if name == "CosmosDBChatMessageHistory":
        warnings.warn(
            "Importing CosmosDBChatMessageHistory from "
            "'langchain_azure_ai.chat_history.cosmos_db' is deprecated. "
            "Use 'from langchain_azure_cosmosdb import "
            "CosmosDBChatMessageHistory' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from langchain_azure_cosmosdb import (
                CosmosDBChatMessageHistory,
            )

            return CosmosDBChatMessageHistory
        except ImportError:
            raise ImportError(
                "langchain-azure-cosmosdb is required for "
                "CosmosDBChatMessageHistory. "
                "Install it with: pip install langchain-azure-cosmosdb"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
