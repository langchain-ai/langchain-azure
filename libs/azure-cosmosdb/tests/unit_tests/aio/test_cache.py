"""Unit tests for AsyncAzureCosmosDBNoSqlSemanticCache."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_azure_cosmosdb._cache import _hash
from langchain_azure_cosmosdb.aio._cache import (
    AsyncAzureCosmosDBNoSqlSemanticCache,
)
from langchain_core.outputs import Generation

# ---- helpers ---------------------------------------------------------------

VECTOR_SEARCH_FIELDS = {"text_field": "text", "embedding_field": "embedding"}

VEC_POLICY = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 3,
        }
    ]
}

IDX_POLICY = {"vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}]}

CONTAINER_PROPS = {"partition_key": "/id"}
DB_PROPS = {}


def _make_cache() -> AsyncAzureCosmosDBNoSqlSemanticCache:
    mock_embedding = MagicMock()
    mock_embedding.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_embedding.embed_query.return_value = [0.1, 0.2, 0.3]

    return AsyncAzureCosmosDBNoSqlSemanticCache(
        embedding=mock_embedding,
        vector_embedding_policy=VEC_POLICY,
        indexing_policy=IDX_POLICY,
        cosmos_container_properties=CONTAINER_PROPS,
        cosmos_database_properties=DB_PROPS,
        vector_search_fields=VECTOR_SEARCH_FIELDS,
    )


# ---- _cache_name -----------------------------------------------------------


def test_cache_name_consistent() -> None:
    cache = _make_cache()
    name1 = cache._cache_name("gpt-4-some-config")
    name2 = cache._cache_name("gpt-4-some-config")
    assert name1 == name2
    assert name1 == f"cache:{_hash('gpt-4-some-config')}"


def test_cache_name_different_inputs() -> None:
    cache = _make_cache()
    assert cache._cache_name("model-a") != cache._cache_name("model-b")


# ---- sync methods raise NotImplementedError --------------------------------


def test_lookup_raises() -> None:
    cache = _make_cache()
    with pytest.raises(NotImplementedError):
        cache.lookup("prompt", "llm_string")


def test_update_raises() -> None:
    cache = _make_cache()
    with pytest.raises(NotImplementedError):
        cache.update("prompt", "llm_string", [Generation(text="x")])


def test_clear_raises() -> None:
    cache = _make_cache()
    with pytest.raises(NotImplementedError):
        cache.clear()


# ---- aupdate (mock-based) --------------------------------------------------


async def test_aupdate_calls_add_texts() -> None:
    cache = _make_cache()

    mock_vs = AsyncMock()
    mock_vs.aadd_texts = AsyncMock(return_value=["id1"])

    cache._cache_dict["cache:" + _hash("llm1")] = mock_vs

    gens = [Generation(text="hello")]
    await cache.aupdate("my prompt", "llm1", gens)

    mock_vs.aadd_texts.assert_awaited_once()
    call_kwargs = mock_vs.aadd_texts.call_args
    assert call_kwargs[1]["texts"] == ["my prompt"]
    metadata = call_kwargs[1]["metadatas"][0]
    assert metadata["prompt"] == "my prompt"
    assert metadata["llm_string"] == "llm1"


async def test_aupdate_rejects_non_generation() -> None:
    cache = _make_cache()
    with pytest.raises(ValueError, match="only supports caching of normal LLM"):
        await cache.aupdate("prompt", "llm", ["not a generation"])  # type: ignore[list-item]
