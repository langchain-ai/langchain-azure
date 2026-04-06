"""Vector Store for CosmosDB NoSql — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Install and import directly from there instead::

    pip install langchain-azure-cosmosdb
    from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch
"""

import warnings
from typing import Any

_DEPRECATED_NAMES = {
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBNoSqlVectorStoreRetriever",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        warnings.warn(
            f"Importing {name} from "
            "'langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql' is deprecated. "
            f"Use 'from langchain_azure_cosmosdb import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import langchain_azure_cosmosdb

            _map: dict[str, Any] = {
                "AzureCosmosDBNoSqlVectorSearch": (
                    langchain_azure_cosmosdb.AzureCosmosDBNoSqlVectorSearch
                ),
                "AzureCosmosDBNoSqlVectorStoreRetriever": (
                    langchain_azure_cosmosdb.AzureCosmosDBNoSqlVectorStoreRetriever
                ),
            }
            return _map[name]
        except ImportError:
            raise ImportError(
                f"langchain-azure-cosmosdb is required for {name}. "
                "Install it with: pip install langchain-azure-cosmosdb"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
