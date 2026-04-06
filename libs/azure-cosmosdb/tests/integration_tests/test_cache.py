"""Test Azure CosmosDB NoSql cache functionality."""
# mypy: disable-error-code=union-attr

import os
from typing import Any, Dict

import pytest
from langchain_azure_cosmosdb import AzureCosmosDBNoSqlSemanticCache
from langchain_core.globals import get_llm_cache, set_llm_cache
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


@pytest.fixture()
def cosmos_client() -> Any:
    from azure.cosmos import CosmosClient

    return CosmosClient(HOST, KEY)


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
        dimensions=400,
    )


# cosine, euclidean, innerproduct
def indexing_policy(index_type: str) -> dict:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": index_type}],
    }


def vector_embedding_policy(distance_function: str) -> dict:
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


cosmos_database_properties_test: Dict[str, Any] = {}


def test_azure_cosmos_db_nosql_semantic_cache_cosine_quantizedflat(
    cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("cosine"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm_string = "test-cache-llm-string"
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_cosine_flat(
    cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("cosine"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm_string = "test-cache-llm-string"
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_dotproduct_quantizedflat(
    cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("dotproduct"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm_string = "test-cache-llm-string"
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_dotproduct_flat(
    cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("dotproduct"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm_string = "test-cache-llm-string"
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_euclidean_quantizedflat(
    cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("euclidean"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm_string = "test-cache-llm-string"
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_euclidean_flat(
    cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("euclidean"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_container_properties(),
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm_string = "test-cache-llm-string"
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def _get_custom_pk_container_properties() -> Dict[str, Any]:
    from azure.cosmos import PartitionKey

    return {"partition_key": PartitionKey(path="/metadata/prompt")}


def test_azure_cosmos_db_nosql_semantic_cache_custom_partition_key(
    cosmos_client: Any,
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("cosine"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=_get_custom_pk_container_properties(),
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
            database_name="cache_custom_pk_test",
        )
    )

    llm_string = "test-cache-custom-pk"
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache — exercises the non-/id pk query path
    get_llm_cache().clear(llm_string=llm_string)

    # verify cleared
    cache_output = get_llm_cache().lookup("foo", llm_string)
    assert cache_output is None

    # clean up database
    cosmos_client.delete_database("cache_custom_pk_test")
