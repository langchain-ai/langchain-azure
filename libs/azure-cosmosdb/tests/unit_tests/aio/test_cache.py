"""Unit tests for AsyncAzureCosmosDBNoSqlSemanticCache."""

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.outputs import Generation

from langchain_azure_cosmosdb._cache import _hash
from langchain_azure_cosmosdb.aio._cache import (
    AsyncAzureCosmosDBNoSqlSemanticCache,
)

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

    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

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


# ---- entries are scoped to their llm_string --------------------------------

_LLM_STRING_FILTER = "c.metadata.llm_string = "
_LLM_A = "model_name=\"gpt-4o\", stop='\\n'"
_LLM_B = "model_name=\"gpt-4o-mini\", stop='\\n'"


class _SharedContainer:
    """In-memory stand-in for the one container that every llm_string shares."""

    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def matching(self, llm_string: Optional[str]) -> List[Dict[str, Any]]:
        return [
            item
            for item in self.items
            if llm_string is None or item["metadata"]["llm_string"] == llm_string
        ]

    async def _query(self, llm_string: Optional[str]) -> AsyncIterator[Dict[str, Any]]:
        for item in self.matching(llm_string):
            yield {"id": item["id"]}

    def query_items(
        self,
        query: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        values = {p["name"]: p["value"] for p in parameters or []}
        llm_string = values["@llm_string"] if "@llm_string" in query else None
        return self._query(llm_string)

    async def delete_item(self, item: str, partition_key: Any) -> None:
        self.items = [i for i in self.items if i["id"] != item]


class _SharedContainerVectorStore:
    """Vector store over a shared container; it filters only when asked to."""

    def __init__(self, container: _SharedContainer) -> None:
        self._container = container

    async def aadd_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        for text, metadata in zip(texts, metadatas):
            self._container.items.append(
                {"id": uuid.uuid4().hex, "text": text, "metadata": metadata}
            )

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, where: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        llm_string = None
        if where is not None:
            assert where.startswith(_LLM_STRING_FILTER)
            llm_string = json.loads(where[len(_LLM_STRING_FILTER) :])
        return [
            (Document(page_content=item["text"], metadata=item["metadata"]), 1.0)
            for item in self._container.matching(llm_string)
            if item["text"] == query
        ][:k]


def _make_shared_container_cache() -> Tuple[Any, _SharedContainer]:
    cache = _make_async_threshold_cache("cosine", 0.5)
    cache._pk_parts = ["id"]
    cache._pk_sql = "id"
    container = _SharedContainer()
    cache._cache_dict = {
        cache._cache_name(llm_string): _SharedContainerVectorStore(container)
        for llm_string in (_LLM_A, _LLM_B)
    }
    return cache, container


async def test_alookup_ignores_entries_for_other_llm_strings() -> None:
    """A cached answer for one model must not be returned for another."""
    cache, _ = _make_shared_container_cache()
    await cache.aupdate("What is 2+2?", _LLM_A, [Generation(text="from A")])

    assert await cache.alookup("What is 2+2?", _LLM_B) is None
    result = await cache.alookup("What is 2+2?", _LLM_A)
    assert result is not None
    assert [g.text for g in result] == ["from A"]


async def test_aclear_with_llm_string_keeps_other_llm_strings() -> None:
    """``aclear(llm_string=...)`` must only delete that llm_string's entries."""
    cache, container = _make_shared_container_cache()
    await cache.aupdate("prompt", _LLM_A, [Generation(text="from A")])
    await cache.aupdate("prompt", _LLM_B, [Generation(text="from B")])

    await cache.aclear(llm_string=_LLM_A)

    assert [item["metadata"]["llm_string"] for item in container.items] == [_LLM_B]
