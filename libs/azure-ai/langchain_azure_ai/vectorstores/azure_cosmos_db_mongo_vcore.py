"""Backward-compatible alias for the renamed ``azure_documentdb`` module.

This module previously hosted ``AzureCosmosDBMongoVCoreVectorSearch``. The
implementation has moved to ``azure_documentdb``; this module re-exports the
public symbols so existing deep imports keep working.
"""

from langchain_azure_ai.vectorstores.azure_documentdb import (
    AzureCosmosDBMongoVCoreVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchCompression,
    CosmosDBVectorSearchType,
)

__all__ = [
    "AzureCosmosDBMongoVCoreVectorSearch",
    "CosmosDBSimilarityType",
    "CosmosDBVectorSearchCompression",
    "CosmosDBVectorSearchType",
]
