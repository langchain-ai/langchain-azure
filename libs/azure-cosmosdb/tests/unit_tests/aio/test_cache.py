"""Unit tests for AsyncAzureCosmosDBNoSqlSemanticCache."""

from typing import Any
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


# ---------------------------------------------------------------------------
# Score threshold filtering
# ---------------------------------------------------------------------------


async def test_alookup_filters_by_score_threshold() -> None:
    """Results below score_threshold are cache misses."""
    from unittest.mock import AsyncMock, MagicMock

    from langchain_azure_cosmosdb.aio._cache import (
        AsyncAzureCosmosDBNoSqlSemanticCache,
    )

    cache = AsyncAzureCosmosDBNoSqlSemanticCache.__new__(
        AsyncAzureCosmosDBNoSqlSemanticCache
    )
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": "cosine"}]
    }
    cache.score_threshold = 0.8

    mock_vs = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["gen"]'}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.3)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))

    assert await cache.alookup("prompt", "llm_string") is None


async def test_alookup_returns_result_above_threshold() -> None:
    """Results above score_threshold are returned."""
    from unittest.mock import AsyncMock, MagicMock

    from langchain_azure_cosmosdb.aio._cache import (
        AsyncAzureCosmosDBNoSqlSemanticCache,
    )
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = AsyncAzureCosmosDBNoSqlSemanticCache.__new__(
        AsyncAzureCosmosDBNoSqlSemanticCache
    )
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": "cosine"}]
    }
    cache.score_threshold = 0.5

    mock_vs = AsyncMock()
    mock_doc = MagicMock()
    gen = Generation(text="cached response")
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.9)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))

    result = await cache.alookup("prompt", "llm_string")
    assert result is not None
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Distance function-specific threshold tests
# ---------------------------------------------------------------------------


def _make_async_threshold_cache(dist_fn: str, threshold: float) -> Any:
    from langchain_azure_cosmosdb.aio._cache import (
        AsyncAzureCosmosDBNoSqlSemanticCache,
    )

    cache = AsyncAzureCosmosDBNoSqlSemanticCache.__new__(
        AsyncAzureCosmosDBNoSqlSemanticCache
    )
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": dist_fn}]
    }
    cache.score_threshold = threshold
    return cache


async def test_cosine_high_score_is_cache_hit() -> None:
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = _make_async_threshold_cache("cosine", 0.5)
    mock_vs = AsyncMock()
    gen = Generation(text="cached")
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.9)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))
    assert await cache.alookup("p", "l") is not None


async def test_cosine_low_score_is_cache_miss() -> None:
    cache = _make_async_threshold_cache("cosine", 0.5)
    mock_vs = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["x"]'}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.3)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))
    assert await cache.alookup("p", "l") is None


async def test_dotproduct_high_score_is_cache_hit() -> None:
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = _make_async_threshold_cache("dotproduct", 0.5)
    mock_vs = AsyncMock()
    gen = Generation(text="cached")
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.8)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))
    assert await cache.alookup("p", "l") is not None


async def test_dotproduct_low_score_is_cache_miss() -> None:
    cache = _make_async_threshold_cache("dotproduct", 0.5)
    mock_vs = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["x"]'}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.2)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))
    assert await cache.alookup("p", "l") is None


async def test_euclidean_low_distance_is_cache_hit() -> None:
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = _make_async_threshold_cache("euclidean", 0.5)
    mock_vs = AsyncMock()
    gen = Generation(text="cached")
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.1)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))
    assert await cache.alookup("p", "l") is not None


async def test_euclidean_high_distance_is_cache_miss() -> None:
    cache = _make_async_threshold_cache("euclidean", 0.5)
    mock_vs = AsyncMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["x"]'}
    mock_vs.asimilarity_search_with_score.return_value = [(mock_doc, 0.9)]
    setattr(cache, "_aget_llm_cache", AsyncMock(return_value=mock_vs))
    assert await cache.alookup("p", "l") is None
