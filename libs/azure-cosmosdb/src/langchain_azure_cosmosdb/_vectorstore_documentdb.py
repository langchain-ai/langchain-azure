"""Compatibility exports for the Azure DocumentDB vector store.

Use ``langchain_azure_documentdb`` for new applications.
"""

from langchain_azure_documentdb import (
    AzureDocumentDBSimilarityType,
    AzureDocumentDBVectorSearch,
    AzureDocumentDBVectorSearchCompression,
    AzureDocumentDBVectorSearchType,
)

CosmosDBSimilarityType = AzureDocumentDBSimilarityType
CosmosDBVectorSearchCompression = AzureDocumentDBVectorSearchCompression
CosmosDBVectorSearchType = AzureDocumentDBVectorSearchType
AzureCosmosDBMongoVCoreVectorSearch = AzureDocumentDBVectorSearch

__all__ = [
    "AzureCosmosDBMongoVCoreVectorSearch",
    "AzureDocumentDBSimilarityType",
    "AzureDocumentDBVectorSearch",
    "AzureDocumentDBVectorSearchCompression",
    "AzureDocumentDBVectorSearchType",
    "CosmosDBSimilarityType",
    "CosmosDBVectorSearchCompression",
    "CosmosDBVectorSearchType",
]