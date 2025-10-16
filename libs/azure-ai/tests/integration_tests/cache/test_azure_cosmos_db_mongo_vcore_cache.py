"""Test Azure CosmosDB cache functionality.

Required to run this test:
    - a recent 'pymongo' Python package available
    - an Azure CosmosDB Mongo vCore instance
    - one environment variable set:
        export MONGODB_VCORE_URI="connection string for azure cosmos db mongo vCore"
"""
# mypy: disable-error-code=union-attr

import os
import uuid
from typing import Any

import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation
from langchain_openai import AzureOpenAIEmbeddings

from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
    CosmosDBSimilarityType,
    CosmosDBVectorSearchType,
)
from langchain_azure_ai.vectorstores.cache import AzureCosmosDBMongoVCoreSemanticCache

INDEX_NAME = "langchain-test-index"
NAMESPACE = "langchain_test_db.langchain_test_collection"
CONNECTION_STRING: str = os.environ.get("MONGODB_VCORE_URI", "")
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")
num_lists = 3
dimensions = 1536
similarity_algorithm = CosmosDBSimilarityType.COS
kind = CosmosDBVectorSearchType.VECTOR_IVF
m = 16
ef_construction = 64
ef_search = 40
max_degree = 32
l_build = 50
l_search = 40
score_threshold = 0.1
application_name = "langchainpy"


def _has_env_vars() -> bool:
    return all(["MONGODB_VCORE_URI" in os.environ])


def random_string() -> str:
    return str(uuid.uuid4())


@pytest.fixture()
def azure_openai_embeddings() -> Any:
    openai_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        model=model_name, chunk_size=1
    )

    return openai_embeddings


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_inner_product(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_inner_product(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=kind,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_hnsw(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_inner_product_hnsw(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_hnsw(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_inner_product_hnsw(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=CosmosDBVectorSearchType.VECTOR_HNSW,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_diskann(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_inner_product_diskann(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_diskann(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=similarity_algorithm,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)


@pytest.mark.requires("pymongo")
@pytest.mark.skipif(
    not _has_env_vars(), reason="Missing Azure CosmosDB Mongo vCore env. vars"
)
def test_azure_cosmos_db_semantic_cache_multi_inner_product_diskann(
    azure_openai_embeddings: AzureOpenAIEmbeddings,
) -> None:
    set_llm_cache(
        AzureCosmosDBMongoVCoreSemanticCache(
            cosmosdb_connection_string=CONNECTION_STRING,
            embedding=azure_openai_embeddings,
            database_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            num_lists=num_lists,
            similarity=CosmosDBSimilarityType.IP,
            kind=CosmosDBVectorSearchType.VECTOR_DISKANN,
            dimensions=dimensions,
            m=m,
            ef_construction=ef_construction,
            max_degree=max_degree,
            l_build=l_build,
            l_search=l_search,
            ef_search=ef_search,
            score_threshold=score_threshold,
            application_name=application_name,
        )
    )

    llm = AzureAIChatCompletionsModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted([(k, v) for k, v in params.items()]))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by AzureAIEmbeddingsModel
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)
