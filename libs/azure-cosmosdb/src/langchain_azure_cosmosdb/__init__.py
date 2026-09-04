"""Azure CosmosDB integrations for LangChain and LangGraph."""

from typing import Any

from langchain_azure_cosmosdb._cache import AzureCosmosDBNoSqlSemanticCache
from langchain_azure_cosmosdb._chat_history import CosmosDBChatMessageHistory
from langchain_azure_cosmosdb._langgraph_cache import CosmosDBCacheSync
from langchain_azure_cosmosdb._langgraph_checkpoint_store import CosmosDBSaverSync
from langchain_azure_cosmosdb._langgraph_store import CosmosDBStore
from langchain_azure_cosmosdb._query_constructor import AzureCosmosDbNoSQLTranslator
from langchain_azure_cosmosdb._request_charge import (
    CosmosDBRequestCharge,
    CosmosDBRequestChargeCallback,
)
from langchain_azure_cosmosdb._vectorstore import (
    AzureCosmosDBNoSqlVectorSearch,
    AzureCosmosDBNoSqlVectorStoreRetriever,
)
from langchain_azure_cosmosdb.aio import (
    AsyncAzureCosmosDBNoSqlSemanticCache,
    AsyncAzureCosmosDBNoSqlVectorSearch,
    AsyncAzureCosmosDBNoSqlVectorStoreRetriever,
    AsyncCosmosDBChatMessageHistory,
    AsyncCosmosDBStore,
    CosmosDBCache,
    CosmosDBSaver,
)

_DOCUMENTDB_EXPORTS = {
    "AzureCosmosDBMongoVCoreVectorSearch": "AzureDocumentDBVectorSearch",
    "AzureDocumentDBSimilarityType": "AzureDocumentDBSimilarityType",
    "AzureDocumentDBVectorSearch": "AzureDocumentDBVectorSearch",
    "AzureDocumentDBVectorSearchCompression": (
        "AzureDocumentDBVectorSearchCompression"
    ),
    "AzureDocumentDBVectorSearchType": "AzureDocumentDBVectorSearchType",
    "CosmosDBSimilarityType": "AzureDocumentDBSimilarityType",
    "CosmosDBVectorSearchCompression": "AzureDocumentDBVectorSearchCompression",
    "CosmosDBVectorSearchType": "AzureDocumentDBVectorSearchType",
}

__all__ = [
    "AsyncAzureCosmosDBNoSqlSemanticCache",
    "AsyncAzureCosmosDBNoSqlVectorSearch",
    "AsyncAzureCosmosDBNoSqlVectorStoreRetriever",
    "AsyncCosmosDBChatMessageHistory",
    "AsyncCosmosDBStore",
    "AzureCosmosDBMongoVCoreVectorSearch",
    "AzureCosmosDBNoSqlSemanticCache",
    "AzureDocumentDBSimilarityType",
    "AzureDocumentDBVectorSearch",
    "AzureDocumentDBVectorSearchCompression",
    "AzureDocumentDBVectorSearchType",
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBNoSqlVectorStoreRetriever",
    "AzureCosmosDbNoSQLTranslator",
    "CosmosDBSimilarityType",
    "CosmosDBRequestCharge",
    "CosmosDBRequestChargeCallback",
    "CosmosDBVectorSearchCompression",
    "CosmosDBVectorSearchType",
    "CosmosDBCache",
    "CosmosDBCacheSync",
    "CosmosDBChatMessageHistory",
    "CosmosDBSaver",
    "CosmosDBSaverSync",
    "CosmosDBStore",
]


def __getattr__(name: str) -> Any:
    if name in _DOCUMENTDB_EXPORTS:
        try:
            import langchain_azure_documentdb

            return getattr(langchain_azure_documentdb, _DOCUMENTDB_EXPORTS[name])
        except ImportError as exc:
            raise ImportError(
                f"langchain-azure-documentdb is required for {name}. "
                "Install it with: pip install langchain-azure-documentdb"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
