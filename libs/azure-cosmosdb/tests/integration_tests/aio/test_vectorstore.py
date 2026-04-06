# type: ignore
"""Integration tests for AsyncAzureCosmosDBNoSqlVectorSearch."""

import asyncio
import logging
import os
import uuid
from typing import Any, List

import pytest
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

logging.basicConfig(level=logging.DEBUG)

azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-3-large")

HOST = os.environ.get("COSMOSDB_ENDPOINT", "")
KEY = os.environ.get("COSMOSDB_KEY", "")

pytestmark = pytest.mark.skipif(
    not HOST or not KEY,
    reason="COSMOSDB_ENDPOINT/COSMOSDB_KEY not set",
)


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _get_documents() -> List[Document]:
    return [
        Document(
            page_content="Border Collies are intelligent, "
            "energetic herders skilled in outdoor activities.",
            metadata={
                "a": 1,
                "origin": "Border Collies were developed in "
                "the border region between Scotland and England.",
            },
        ),
        Document(
            page_content="Golden Retrievers are friendly, "
            "loyal companions with excellent retrieving skills.",
            metadata={
                "a": 2,
                "origin": "Golden Retrievers originated "
                "in Scotland in the mid-19th century.",
            },
        ),
        Document(
            page_content="Labrador Retrievers are playful, "
            "eager learners and skilled retrievers.",
            metadata={
                "a": 1,
                "origin": "Labrador Retrievers were first developed "
                "in Newfoundland (now part of Canada).",
            },
        ),
        Document(
            page_content="Australian Shepherds are agile, energetic "
            "herders excelling in outdoor tasks.",
            metadata={
                "a": 2,
                "b": 1,
                "origin": "Despite the name, Australian Shepherds were "
                "developed in the United States in the 19th century.",
            },
        ),
        Document(
            page_content="German Shepherds are brave, "
            "loyal protectors excelling in versatile tasks.",
            metadata={
                "a": 1,
                "b": 2,
                "origin": "German Shepherds were developed in Germany in the "
                "late 19th century for herding and guarding sheep.",
            },
        ),
        Document(
            page_content="Standard Poodles are intelligent, "
            "energetic learners excelling in agility.",
            metadata={
                "a": 2,
                "b": 3,
                "origin": "Standard Poodles originated in Germany "
                "as water retrievers.",
            },
        ),
    ]


def _get_vector_indexing_policy(embedding_type: str) -> dict:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": embedding_type}],
        "fullTextIndexes": [{"path": "/text"}],
    }


def _get_vector_embedding_policy(
    distance_function: str, data_type: str, dimensions: int
) -> dict:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": data_type,
                "dimensions": dimensions,
                "distanceFunction": distance_function,
            }
        ]
    }


def _get_full_text_policy() -> dict:
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": [{"path": "/text", "language": "en-US"}],
    }


@pytest.fixture()
def async_cosmos_client() -> Any:
    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

    return AsyncCosmosClient(HOST, KEY)


@pytest.fixture()
def partition_key() -> Any:
    from azure.cosmos import PartitionKey

    return PartitionKey(path="/id")


@pytest.fixture()
def azure_openai_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        api_key=SecretStr(openai_api_key),
        model=model_name,
        dimensions=1536,
    )


async def safe_delete_database(client: Any, db_name: str) -> None:
    try:
        await client.delete_database(db_name)
    except Exception:
        pass


async def test_async_from_documents_cosine_distance(
    async_cosmos_client: Any,
    partition_key: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    from langchain_azure_cosmosdb import AsyncAzureCosmosDBNoSqlVectorSearch

    db_name = _unique_name("async_vs_cosine_db")
    container_name = _unique_name("async_vs_cosine_ctr")
    documents = _get_documents()

    try:
        store = await AsyncAzureCosmosDBNoSqlVectorSearch.create(
            cosmos_client=async_cosmos_client,
            embedding=azure_openai_embeddings,
            database_name=db_name,
            container_name=container_name,
            vector_embedding_policy=_get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            indexing_policy=_get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=_get_full_text_policy(),
            full_text_search_enabled=True,
        )

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        await store.aadd_texts(texts=texts, metadatas=metadatas)
        await asyncio.sleep(2)

        output = await store.asimilarity_search(
            "Which dog breed is considered a herder?", k=5
        )
        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content
    finally:
        await safe_delete_database(async_cosmos_client, db_name)


async def test_async_add_texts_and_delete(
    async_cosmos_client: Any,
    partition_key: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    from langchain_azure_cosmosdb import AsyncAzureCosmosDBNoSqlVectorSearch

    db_name = _unique_name("async_vs_del_db")
    container_name = _unique_name("async_vs_del_ctr")

    try:
        store = await AsyncAzureCosmosDBNoSqlVectorSearch.create(
            cosmos_client=async_cosmos_client,
            embedding=azure_openai_embeddings,
            database_name=db_name,
            container_name=container_name,
            vector_embedding_policy=_get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            indexing_policy=_get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=_get_full_text_policy(),
            full_text_search_enabled=True,
        )

        ids = await store.aadd_texts(
            texts=["Hello world", "Goodbye world"],
            metadatas=[{"source": "test"}, {"source": "test"}],
        )
        assert len(ids) == 2
        await asyncio.sleep(2)

        output = await store.asimilarity_search("Hello", k=2)
        assert len(output) == 2

        await store.adelete_document_by_id(ids[0])
        await asyncio.sleep(2)

        output = await store.asimilarity_search("Hello", k=2)
        assert len(output) == 1
    finally:
        await safe_delete_database(async_cosmos_client, db_name)


async def test_async_similarity_search_with_score(
    async_cosmos_client: Any,
    partition_key: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    from langchain_azure_cosmosdb import AsyncAzureCosmosDBNoSqlVectorSearch

    db_name = _unique_name("async_vs_score_db")
    container_name = _unique_name("async_vs_score_ctr")
    documents = _get_documents()

    try:
        store = await AsyncAzureCosmosDBNoSqlVectorSearch.create(
            cosmos_client=async_cosmos_client,
            embedding=azure_openai_embeddings,
            database_name=db_name,
            container_name=container_name,
            vector_embedding_policy=_get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            indexing_policy=_get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=_get_full_text_policy(),
            full_text_search_enabled=True,
        )

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        await store.aadd_texts(texts=texts, metadatas=metadatas)
        await asyncio.sleep(2)

        results = await store.asimilarity_search_with_score(
            "Which dog breed is considered a herder?", k=3
        )
        assert results
        assert len(results) == 3
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
        assert "Border Collies" in results[0][0].page_content
    finally:
        await safe_delete_database(async_cosmos_client, db_name)


async def test_async_hybrid_search(
    async_cosmos_client: Any,
    partition_key: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    from langchain_azure_cosmosdb import AsyncAzureCosmosDBNoSqlVectorSearch

    db_name = _unique_name("async_vs_hybrid_db")
    container_name = _unique_name("async_vs_hybrid_ctr")
    documents = _get_documents()

    try:
        store = await AsyncAzureCosmosDBNoSqlVectorSearch.create(
            cosmos_client=async_cosmos_client,
            embedding=azure_openai_embeddings,
            database_name=db_name,
            container_name=container_name,
            vector_embedding_policy=_get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            indexing_policy=_get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=_get_full_text_policy(),
            full_text_search_enabled=True,
        )

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        await store.aadd_texts(texts=texts, metadatas=metadatas)
        await asyncio.sleep(2)

        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = await store.asimilarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            search_type="hybrid",
            full_text_rank_filter=full_text_rank_filter,
        )
        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content
    finally:
        await safe_delete_database(async_cosmos_client, db_name)
