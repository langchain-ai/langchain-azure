from langchain_azure_documentdb import (
    AzureDocumentDBSimilarityType,
    AzureDocumentDBVectorSearch,
    AzureDocumentDBVectorSearchCompression,
    AzureDocumentDBVectorSearchType,
)


def test_documentdb_vector_search_import() -> None:
    assert AzureDocumentDBVectorSearch
    assert AzureDocumentDBSimilarityType
    assert AzureDocumentDBVectorSearchCompression
    assert AzureDocumentDBVectorSearchType