"""Test` Azure CosmosDB NoSql cache functionality."""

import os
from typing import Any, Dict

import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_azure_ai.vectorstores.cache import AzureCosmosDBNoSqlSemanticCache

HOST = "COSMOS_DB_URI"
KEY = "COSMOS_DB_KEY"
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")


@pytest.fixture()
def cosmos_client() -> Any:
    from azure.cosmos import CosmosClient

    return CosmosClient(HOST, KEY)


@pytest.fixture()
def partition_key() -> Any:
    from azure.cosmos import PartitionKey

    return PartitionKey(path="/id")


@pytest.fixture()
def azure_openai_embeddings() -> OpenAIEmbeddings:
    openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        model=model_name,
        chunk_size=1,
    )
    return openai_embeddings


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
                "dimensions": 1536,
            }
        ]
    }


cosmos_container_properties_test = {"partition_key": partition_key}
cosmos_database_properties_test: Dict[str, Any] = {}


def test_azure_cosmos_db_nosql_semantic_cache_cosine_quantizedflat(
    cosmos_client: Any,
    azure_openai_embeddings: OpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("cosine"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_cosine_flat(
    cosmos_client: Any,
    azure_openai_embeddings: OpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("cosine"),
            indexing_policy=indexing_policy("flat"),
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_dotproduct_quantizedflat(
    cosmos_client: Any,
    azure_openai_embeddings: OpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("dotProduct"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_dotproduct_flat(
    cosmos_client: Any,
    azure_openai_embeddings: OpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("dotProduct"),
            indexing_policy=indexing_policy("flat"),
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_euclidean_quantizedflat(
    cosmos_client: Any,
    azure_openai_embeddings: OpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("euclidean"),
            indexing_policy=indexing_policy("quantizedFlat"),
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


def test_azure_cosmos_db_nosql_semantic_cache_euclidean_flat(
    cosmos_client: Any,
    azure_openai_embeddings: OpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=azure_openai_embeddings,
            vector_embedding_policy=vector_embedding_policy("euclidean"),
            indexing_policy=indexing_policy("flat"),
            cosmos_container_properties=cosmos_container_properties_test,
            cosmos_database_properties=cosmos_database_properties_test,
            vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)
