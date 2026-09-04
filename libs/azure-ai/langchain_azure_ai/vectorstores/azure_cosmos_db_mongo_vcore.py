"""Legacy compatibility imports for the renamed Azure DocumentDB integration.

This module has moved to ``langchain_azure_documentdb``.
Install and import directly from there instead::

    pip install langchain-azure-documentdb
    from langchain_azure_documentdb import AzureDocumentDBVectorSearch
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

_DOCUMENTDB_NAMES = {
    "AzureDocumentDBVectorSearch": "AzureDocumentDBVectorSearch",
    "AzureCosmosDBMongoVCoreVectorSearch": "AzureDocumentDBVectorSearch",
    "CosmosDBSimilarityType": "AzureDocumentDBSimilarityType",
    "CosmosDBVectorSearchCompression": "AzureDocumentDBVectorSearchCompression",
    "CosmosDBVectorSearchType": "AzureDocumentDBVectorSearchType",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        documentdb_name = _DOCUMENTDB_NAMES[name]
        warnings.warn(
            f"Importing {name} from "
            "'langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore' is "
            "deprecated. "
            "Use 'from langchain_azure_documentdb import "
            f"{documentdb_name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import langchain_azure_documentdb

            return getattr(langchain_azure_documentdb, documentdb_name)
        except ImportError as exc:
            raise ImportError(
                f"langchain-azure-documentdb is required for {name}. "
                "Install it with: pip install langchain-azure-documentdb"
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
