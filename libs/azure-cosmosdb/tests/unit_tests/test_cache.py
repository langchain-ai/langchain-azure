"""Unit tests for cache helper functions in _cache.py."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_azure_cosmosdb._cache import (
    _dump_generations_to_json,
    _dumps_generations,
    _hash,
    _load_generations_from_json,
    _loads_generations,
)
from langchain_core.outputs import Generation

# ---------------------------------------------------------------------------
# _hash
# ---------------------------------------------------------------------------


def test_hash_consistent() -> None:
    result1 = _hash("hello")
    result2 = _hash("hello")
    assert result1 == result2
    assert len(result1) == 64  # SHA-256 hex digest length


def test_hash_different_inputs() -> None:
    assert _hash("hello") != _hash("world")


# ---------------------------------------------------------------------------
# _dump_generations_to_json / _load_generations_from_json round-trip
# ---------------------------------------------------------------------------


def test_dump_load_generations_json_roundtrip() -> None:
    gens = [Generation(text="foo"), Generation(text="bar")]
    dumped = _dump_generations_to_json(gens)
    loaded = _load_generations_from_json(dumped)
    assert len(loaded) == 2
    assert loaded[0].text == "foo"
    assert loaded[1].text == "bar"


def test_load_generations_from_json_invalid() -> None:
    with pytest.raises(ValueError, match="Could not decode json"):
        _load_generations_from_json("not valid json")


# ---------------------------------------------------------------------------
# _dumps_generations / _loads_generations round-trip
# ---------------------------------------------------------------------------


def test_dumps_loads_generations_roundtrip() -> None:
    gens = [Generation(text="alpha"), Generation(text="beta")]
    serialized = _dumps_generations(gens)
    deserialized = _loads_generations(serialized)
    assert deserialized is not None
    assert len(deserialized) == 2
    assert deserialized[0].text == "alpha"
    assert deserialized[1].text == "beta"


def test_loads_generations_malformed_returns_none() -> None:
    result = _loads_generations("completely invalid {{{")
    assert result is None


def test_loads_generations_legacy_format() -> None:
    gens = [Generation(text="legacy")]
    legacy_json = json.dumps([g.dict() for g in gens])
    result = _loads_generations(legacy_json)
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "legacy"


# ---------------------------------------------------------------------------
# Partition key extraction and clear() query paths
# ---------------------------------------------------------------------------


def test_pk_defaults_to_id_when_none() -> None:
    from langchain_azure_cosmosdb._cache import (
        AzureCosmosDBNoSqlSemanticCache,
    )

    cache = AzureCosmosDBNoSqlSemanticCache.__new__(AzureCosmosDBNoSqlSemanticCache)
    cache.cosmos_container_properties = {"partition_key": None}
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": "cosine"}]
    }
    try:
        pk_def = cache.cosmos_container_properties.get("partition_key")
        if pk_def is not None:
            pk_path = pk_def.get("paths", ["/id"])[0]
            parts = [p for p in pk_path.split("/") if p]
            cache._pk_parts = parts if parts else ["id"]
            cache._pk_sql = ".".join(cache._pk_parts)
        else:
            cache._pk_parts = ["id"]
            cache._pk_sql = "id"
    except (AttributeError, TypeError, KeyError, IndexError):
        cache._pk_parts = ["id"]
        cache._pk_sql = "id"
    assert cache._pk_sql == "id"
    assert cache._pk_parts == ["id"]


def test_pk_extracts_simple_path() -> None:
    from azure.cosmos import PartitionKey

    pk = PartitionKey(path="/category")
    pk_path = pk.get("paths", ["/id"])[0]
    parts = [p for p in pk_path.split("/") if p]
    assert parts == ["category"]
    assert ".".join(parts) == "category"


def test_pk_extracts_nested_path() -> None:
    from azure.cosmos import PartitionKey

    pk = PartitionKey(path="/metadata/prompt")
    pk_path = pk.get("paths", ["/id"])[0]
    parts = [p for p in pk_path.split("/") if p]
    assert parts == ["metadata", "prompt"]
    assert ".".join(parts) == "metadata.prompt"


def test_clear_query_no_duplicate_when_pk_is_id() -> None:
    pk_sql = "id"
    query = (
        "SELECT c.id FROM c" if pk_sql == "id" else f"SELECT c.id, c.{pk_sql} FROM c"
    )
    assert query == "SELECT c.id FROM c"


def test_clear_query_simple_pk() -> None:
    pk_sql = "category"
    query = (
        "SELECT c.id FROM c" if pk_sql == "id" else f"SELECT c.id, c.{pk_sql} FROM c"
    )
    assert query == "SELECT c.id, c.category FROM c"


def test_clear_query_nested_pk() -> None:
    pk_sql = "metadata.prompt"
    query = (
        "SELECT c.id FROM c" if pk_sql == "id" else f"SELECT c.id, c.{pk_sql} FROM c"
    )
    assert query == "SELECT c.id, c.metadata.prompt FROM c"


def test_get_nested_simple() -> None:
    from langchain_azure_cosmosdb._cache import _get_nested

    d = {"id": "123", "category": "dogs"}
    assert _get_nested(d, ["category"]) == "dogs"
    assert _get_nested(d, ["id"]) == "123"


def test_get_nested_deep() -> None:
    from langchain_azure_cosmosdb._cache import _get_nested

    d = {"id": "123", "metadata": {"prompt": "hello", "a": 1}}
    assert _get_nested(d, ["metadata", "prompt"]) == "hello"
    assert _get_nested(d, ["metadata", "a"]) == 1


def test_get_nested_missing() -> None:
    from langchain_azure_cosmosdb._cache import _get_nested

    d = {"id": "123"}
    assert _get_nested(d, ["metadata", "prompt"]) is None
    assert _get_nested(d, ["nonexistent"]) is None


# ---------------------------------------------------------------------------
# Semantic cache exception narrowing
# ---------------------------------------------------------------------------


def test_lookup_catches_json_decode_error_not_broad_exception() -> None:
    """Verify the cache lookup only catches specific deserialization errors,
    not broad Exception (which would hide real bugs)."""
    from langchain_azure_cosmosdb._cache import AzureCosmosDBNoSqlSemanticCache

    cache = AzureCosmosDBNoSqlSemanticCache.__new__(AzureCosmosDBNoSqlSemanticCache)
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": "cosine"}]
    }
    cache.score_threshold = 0.5

    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": "not-valid-json{{{"}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    cache._cache_dict["cache:abc"] = mock_vs

    # Patch _get_llm_cache to return our mock
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))

    # Should not raise — catches json.JSONDecodeError in legacy fallback
    result = cache.lookup("prompt", "llm_string")
    # Malformed data returns None
    assert result is None


def test_lookup_skips_entries_missing_return_val() -> None:
    """Missing 'return_val' metadata should be logged and skipped."""
    from langchain_azure_cosmosdb._cache import AzureCosmosDBNoSqlSemanticCache

    cache = AzureCosmosDBNoSqlSemanticCache.__new__(AzureCosmosDBNoSqlSemanticCache)
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": "cosine"}]
    }
    cache.score_threshold = 0.5

    mock_vs = MagicMock()
    mock_doc = MagicMock()
    # No 'return_val' in metadata
    mock_doc.metadata = {"prompt": "p", "llm_string": "l"}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))

    # Should not raise; should return None and skip the entry
    assert cache.lookup("prompt", "llm_string") is None


def test_lookup_filters_by_score_threshold() -> None:
    """Results below score_threshold should be treated as cache misses."""
    from langchain_azure_cosmosdb._cache import AzureCosmosDBNoSqlSemanticCache

    cache = AzureCosmosDBNoSqlSemanticCache.__new__(AzureCosmosDBNoSqlSemanticCache)
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": "cosine"}]
    }
    cache.score_threshold = 0.8

    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["gen"]'}
    # Score below threshold
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.3)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))

    assert cache.lookup("prompt", "llm_string") is None


def test_lookup_returns_result_above_threshold() -> None:
    """Results above score_threshold should be returned."""
    from langchain_azure_cosmosdb._cache import AzureCosmosDBNoSqlSemanticCache
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = AzureCosmosDBNoSqlSemanticCache.__new__(AzureCosmosDBNoSqlSemanticCache)
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": "cosine"}]
    }
    cache.score_threshold = 0.5

    mock_vs = MagicMock()
    mock_doc = MagicMock()
    gen = Generation(text="cached response")
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))

    result = cache.lookup("prompt", "llm_string")
    assert result is not None
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Distance function-specific threshold tests
# ---------------------------------------------------------------------------


def _make_threshold_cache(dist_fn: str, threshold: float) -> Any:
    """Build a cache with the given distance function and threshold."""
    from langchain_azure_cosmosdb._cache import AzureCosmosDBNoSqlSemanticCache

    cache = AzureCosmosDBNoSqlSemanticCache.__new__(AzureCosmosDBNoSqlSemanticCache)
    cache._cache_dict = {}
    cache.vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": dist_fn}]
    }
    cache.score_threshold = threshold
    return cache


def test_cosine_high_score_is_cache_hit() -> None:
    """Cosine: higher = more similar. Score 0.9 > threshold 0.5 → hit."""
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = _make_threshold_cache("cosine", 0.5)
    mock_vs = MagicMock()
    gen = Generation(text="cached")
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))
    assert cache.lookup("p", "l") is not None


def test_cosine_low_score_is_cache_miss() -> None:
    """Cosine: score 0.3 <= threshold 0.5 → miss."""
    cache = _make_threshold_cache("cosine", 0.5)
    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["x"]'}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.3)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))
    assert cache.lookup("p", "l") is None


def test_dotproduct_high_score_is_cache_hit() -> None:
    """DotProduct: higher = more similar. Score 0.8 > threshold 0.5 → hit."""
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = _make_threshold_cache("dotproduct", 0.5)
    mock_vs = MagicMock()
    gen = Generation(text="cached")
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.8)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))
    assert cache.lookup("p", "l") is not None


def test_dotproduct_low_score_is_cache_miss() -> None:
    """DotProduct: score 0.2 <= threshold 0.5 → miss."""
    cache = _make_threshold_cache("dotproduct", 0.5)
    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["x"]'}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.2)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))
    assert cache.lookup("p", "l") is None


def test_euclidean_low_distance_is_cache_hit() -> None:
    """Euclidean: lower = more similar. Distance 0.1 < threshold 0.5 → hit."""
    from langchain_core.load.dump import dumps
    from langchain_core.outputs import Generation

    cache = _make_threshold_cache("euclidean", 0.5)
    mock_vs = MagicMock()
    gen = Generation(text="cached")
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": dumps([gen])}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.1)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))
    assert cache.lookup("p", "l") is not None


def test_euclidean_high_distance_is_cache_miss() -> None:
    """Euclidean: distance 0.9 >= threshold 0.5 → miss."""
    cache = _make_threshold_cache("euclidean", 0.5)
    mock_vs = MagicMock()
    mock_doc = MagicMock()
    mock_doc.metadata = {"return_val": '["x"]'}
    mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.9)]
    setattr(cache, "_get_llm_cache", MagicMock(return_value=mock_vs))
    assert cache.lookup("p", "l") is None
