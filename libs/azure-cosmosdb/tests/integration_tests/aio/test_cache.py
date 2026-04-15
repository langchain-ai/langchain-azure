# type: ignore
"""Integration tests for AsyncAzureCosmosDBNoSqlSemanticCache."""

import os
import uuid
from typing import Any, Dict

import pytest
from langchain_core.outputs import Generation
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

HOST = os.environ.get("COSMOSDB_ENDPOINT", "")
KEY = os.environ.get("COSMOSDB_KEY", "")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
model_name = os.environ.get("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-3-large")

pytestmark = pytest.mark.skipif(
    not HOST or not KEY,
    reason="COSMOSDB_ENDPOINT/COSMOSDB_KEY not set",
)


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture()
def async_cosmos_client() -> Any:
    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

    return AsyncCosmosClient(HOST, KEY)


@pytest.fixture()
def azure_openai_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        api_key=SecretStr(openai_api_key),
        model=model_name,
        dimensions=400,
    )


def _indexing_policy(index_type: str) -> dict:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": index_type}],
    }


def _vector_embedding_policy(distance_function: str) -> dict:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": distance_function,
                "dimensions": 400,
            }
        ]
    }


def _get_container_properties() -> Dict[str, Any]:
    from azure.cosmos import PartitionKey

    return {"partition_key": PartitionKey(path="/id")}


async def _safe_delete_database(client: Any, db_name: str) -> None:
    try:
        await client.delete_database(db_name)
    except Exception:
        pass


async def test_async_cache_cosine_quantizedflat(
    async_cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    from langchain_azure_cosmosdb import AsyncAzureCosmosDBNoSqlSemanticCache

    db_name = _unique_name("async_cache_cos_db")
    container_name = _unique_name("async_cache_cos_ctr")

    try:
        cache = await AsyncAzureCosmosDBNoSqlSemanticCache.create(
            cosmos_client=async_cosmos_client,
            embedding=azure_openai_embeddings,
            database_name=db_name,
            container_name=container_name,
            vector_embedding_policy=_vector_embedding_policy("cosine"),
            indexing_policy=_indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )

        llm_string = "test-async-cache-llm"
        await cache.aupdate("foo", llm_string, [Generation(text="fizz")])
        result = await cache.alookup("foo", llm_string)
        assert result == [Generation(text="fizz")]

        await cache.aclear(llm_string=llm_string)
    finally:
        await _safe_delete_database(async_cosmos_client, db_name)


async def test_async_cache_euclidean_quantizedflat(
    async_cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    from langchain_azure_cosmosdb import AsyncAzureCosmosDBNoSqlSemanticCache

    db_name = _unique_name("async_cache_euc_db")
    container_name = _unique_name("async_cache_euc_ctr")

    try:
        cache = await AsyncAzureCosmosDBNoSqlSemanticCache.create(
            cosmos_client=async_cosmos_client,
            embedding=azure_openai_embeddings,
            database_name=db_name,
            container_name=container_name,
            vector_embedding_policy=_vector_embedding_policy("euclidean"),
            indexing_policy=_indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties={},
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )

        llm_string = "test-async-cache-euc-llm"
        await cache.aupdate("foo", llm_string, [Generation(text="fizz")])
        result = await cache.alookup("foo", llm_string)
        assert result == [Generation(text="fizz")]

        await cache.aclear(llm_string=llm_string)
    finally:
        await _safe_delete_database(async_cosmos_client, db_name)


async def test_async_cache_custom_partition_key() -> None:
    from azure.cosmos import PartitionKey
    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
    from langchain_azure_cosmosdb.aio import (
        AsyncAzureCosmosDBNoSqlSemanticCache,
    )

    db_name = f"async_cache_custom_pk_{uuid.uuid4().hex[:8]}"
    container_name = "test_async_cache_custom_pk"

    async_cosmos_client = AsyncCosmosClient(HOST, KEY)

    try:
        azure_openai_embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=SecretStr(openai_api_key),
            model=model_name,
            dimensions=400,
        )

        cache = await AsyncAzureCosmosDBNoSqlSemanticCache.create(
            cosmos_client=async_cosmos_client,
            embedding=azure_openai_embeddings,
            database_name=db_name,
            container_name=container_name,
            vector_embedding_policy=_vector_embedding_policy("cosine"),
            indexing_policy=_indexing_policy("quantizedFlat"),
            cosmos_container_properties={
                "partition_key": PartitionKey(path="/metadata/prompt")
            },
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "text",
                "embedding_field": "embedding",
            },
        )

        llm_string = "test-async-custom-pk"
        await cache.aupdate("foo", llm_string, [Generation(text="fizz")])
        result = await cache.alookup("foo", llm_string)
        assert result == [Generation(text="fizz")]

        await cache.aclear(llm_string=llm_string)

        result2 = await cache.alookup("foo", llm_string)
        assert result2 is None
    finally:
        await _safe_delete_database(async_cosmos_client, db_name)
