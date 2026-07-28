from unittest.mock import MagicMock

import pytest
from langchain_azure_cosmosdb import (
    AzureCosmosDBMongoVCoreVectorSearch,
    AzureDocumentDBVectorSearch,
    __all__,
)

EXPECTED_ALL = [
    "AsyncAzureCosmosDBNoSqlSemanticCache",
    "AsyncAzureCosmosDBNoSqlVectorSearch",
    "AsyncAzureCosmosDBNoSqlVectorStoreRetriever",
    "AsyncCosmosDBChatMessageHistory",
    "AsyncCosmosDBStore",
    "AzureCosmosDBNoSqlSemanticCache",
    "AzureDocumentDBVectorSearch",
    "AzureCosmosDBMongoVCoreVectorSearch",
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBNoSqlVectorStoreRetriever",
    "AzureCosmosDbNoSQLTranslator",
    "CosmosDBSimilarityType",
    "CosmosDBVectorSearchCompression",
    "CosmosDBVectorSearchType",
    "CosmosDBCache",
    "CosmosDBCacheSync",
    "CosmosDBChatMessageHistory",
    "CosmosDBSaver",
    "CosmosDBSaverSync",
    "CosmosDBStore",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)


def test_legacy_documentdb_alias_warns() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        vectorstore = AzureCosmosDBMongoVCoreVectorSearch(
            collection=MagicMock(),
            embedding=MagicMock(),
        )

    assert isinstance(vectorstore, AzureDocumentDBVectorSearch)
