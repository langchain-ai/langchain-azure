"""Vector Store for Azure DocumentDB (with MongoDB compatibility) — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Install and import directly from there instead::

    pip install langchain-azure-cosmosdb
    from langchain_azure_cosmosdb import AzureDocumentDBVectorSearch
"""

import warnings
from typing import Any

_DEPRECATED_NAMES = {
    "AzureDocumentDBVectorSearch",
    "AzureCosmosDBMongoVCoreVectorSearch",
    "CosmosDBSimilarityType",
    "CosmosDBVectorSearchCompression",
    "CosmosDBVectorSearchType",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        warnings.warn(
            f"Importing {name} from "
            "'langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore' is "
            "deprecated. "
            f"Use 'from langchain_azure_cosmosdb import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import langchain_azure_cosmosdb

            _map: dict[str, Any] = {
                "AzureDocumentDBVectorSearch": (
                    langchain_azure_cosmosdb.AzureDocumentDBVectorSearch
                ),
                "AzureCosmosDBMongoVCoreVectorSearch": (
                    langchain_azure_cosmosdb.AzureCosmosDBMongoVCoreVectorSearch
                ),
                "CosmosDBSimilarityType": (
                    langchain_azure_cosmosdb.CosmosDBSimilarityType
                ),
                "CosmosDBVectorSearchCompression": (
                    langchain_azure_cosmosdb.CosmosDBVectorSearchCompression
                ),
                "CosmosDBVectorSearchType": (
                    langchain_azure_cosmosdb.CosmosDBVectorSearchType
                ),
            }
            return _map[name]
        except ImportError:
            raise ImportError(
                f"langchain-azure-cosmosdb is required for {name}. "
                "Install it with: pip install langchain-azure-cosmosdb"
            )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
