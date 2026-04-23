"""Unit tests for AsyncAzureCosmosDBNoSqlVectorSearch."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_azure_cosmosdb.aio._vectorstore import (
    AsyncAzureCosmosDBNoSqlVectorSearch,
)
from langchain_core.embeddings import Embeddings

# ---- helpers ---------------------------------------------------------------


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


DEFAULT_VS_FIELDS: Dict[str, Any] = {
    "text_field": "text",
    "embedding_field": "embedding",
}

DEFAULT_VEC_POLICY: Dict[str, Any] = {
    "vectorEmbeddings": [
        {
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 3,
        }
    ]
}

DEFAULT_IDX_POLICY: Dict[str, Any] = {
    "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}]
}

DEFAULT_CONTAINER_PROPS: Dict[str, Any] = {"partition_key": "/id"}
DEFAULT_DB_PROPS: Dict[str, Any] = {}


def _make_store(
    text_field: str = "text",
    embedding_field: str = "embedding",
    metadata_key: str = "metadata",
    table_alias: str = "c",
    search_type: str = "vector",
    full_text_search_enabled: bool = False,
) -> AsyncAzureCosmosDBNoSqlVectorSearch:
    """Build a store instance directly, bypassing the async ``create`` factory."""
    mock_client = MagicMock()
    mock_database = MagicMock()
    mock_container = AsyncMock()

    vs_fields = {"text_field": text_field, "embedding_field": embedding_field}

    return AsyncAzureCosmosDBNoSqlVectorSearch(
        cosmos_client=mock_client,
        embedding=FakeEmbeddings(),
        database=mock_database,
        container=mock_container,
        vector_embedding_policy=DEFAULT_VEC_POLICY,
        indexing_policy=DEFAULT_IDX_POLICY,
        cosmos_container_properties=DEFAULT_CONTAINER_PROPS,
        cosmos_database_properties=DEFAULT_DB_PROPS,
        vector_search_fields=vs_fields,
        database_name="testdb",
        container_name="testcontainer",
        search_type=search_type,
        metadata_key=metadata_key,
        table_alias=table_alias,
        full_text_search_enabled=full_text_search_enabled,
    )


# ---- _construct_query: vector search type ----------------------------------


def test_construct_query_vector() -> None:
    store = _make_store()
    query, params = store._construct_query(
        k=4,
        search_type="vector",
        embeddings=[0.1, 0.2, 0.3],
    )
    assert "SELECT TOP @limit" in query
    assert "VectorDistance" in query
    assert "ORDER BY VectorDistance" in query
    assert "RRF" not in query


# ---- _construct_query: hybrid search produces RRF -------------------------


def test_construct_query_hybrid() -> None:
    store = _make_store()
    ftr = [{"search_field": "text", "search_text": "hello world"}]
    query, params = store._construct_query(
        k=4,
        search_type="hybrid",
        embeddings=[0.1, 0.2, 0.3],
        full_text_rank_filter=ftr,
    )
    assert "RRF" in query
    assert "VectorDistance" in query
    assert "FullTextScore" in query


# ---- _construct_query: weights validation ----------------------------------


def test_construct_query_hybrid_wrong_weights_raises() -> None:
    store = _make_store()
    ftr = [{"search_field": "text", "search_text": "hello world"}]
    # 1 FullTextScore + 1 VectorDistance = 2 components, but 3 weights
    with pytest.raises(ValueError, match="weights must have 2 elements"):
        store._construct_query(
            k=4,
            search_type="hybrid",
            embeddings=[0.1, 0.2, 0.3],
            full_text_rank_filter=ftr,
            weights=[0.3, 0.3, 0.4],
        )


# ---- _generate_projection_fields ------------------------------------------


def test_projection_vector_default() -> None:
    store = _make_store()
    projection = store._generate_projection_fields(None, "vector")
    assert "as text" in projection
    assert "as metadata" in projection
    assert "SimilarityScore" in projection


def test_projection_hybrid_includes_similarity() -> None:
    store = _make_store()
    projection = store._generate_projection_fields(None, "hybrid")
    assert "SimilarityScore" in projection
    assert "VectorDistance" in projection


def test_projection_full_text_search_no_similarity() -> None:
    store = _make_store()
    projection = store._generate_projection_fields(None, "full_text_search")
    assert "SimilarityScore" not in projection


def test_projection_with_embedding() -> None:
    store = _make_store(embedding_field="content_vector")
    projection = store._generate_projection_fields(None, "vector", with_embedding=True)
    assert "as content_vector" in projection


# ---- _build_parameters ----------------------------------------------------


def test_build_parameters_vector() -> None:
    store = _make_store()
    params = store._build_parameters(
        k=5,
        search_type="vector",
        embeddings=[0.1, 0.2],
        projection_mapping=None,
    )
    names = {p["name"] for p in params}
    assert "@limit" in names
    assert "@textKey" in names
    assert "@metadataKey" in names
    assert "@embeddingKey" in names
    assert "@embeddings" in names

    limit_param = next(p for p in params if p["name"] == "@limit")
    assert limit_param["value"] == 5


def test_build_parameters_with_weights() -> None:
    store = _make_store()
    params = store._build_parameters(
        k=5,
        search_type="hybrid",
        embeddings=[0.1, 0.2],
        weights=[0.5, 0.5],
    )
    names = {p["name"] for p in params}
    assert "@weights" in names


# ---- embeddings property ---------------------------------------------------


def test_embeddings_property() -> None:
    store = _make_store()
    assert isinstance(store.embeddings, FakeEmbeddings)


# ---- aadd_texts (mock-based) ----------------------------------------------


async def test_aadd_texts_calls_create_item() -> None:
    store = _make_store()
    container: AsyncMock = store._container  # type: ignore[assignment]
    container.create_item.side_effect = lambda item: {"id": item["id"]}

    ids = await store.aadd_texts(
        texts=["hello", "world"],
        metadatas=[{"k": "v1"}, {"k": "v2"}],
        ids=["id1", "id2"],
    )

    assert ids == ["id1", "id2"]
    assert container.create_item.call_count == 2

    first_call_arg = container.create_item.call_args_list[0][0][0]
    assert first_call_arg["id"] == "id1"
    assert first_call_arg["text"] == "hello"
    assert first_call_arg["metadata"] == {"k": "v1"}
    assert "embedding" in first_call_arg


# ---- adelete (mock-based) --------------------------------------------------


async def test_adelete_calls_delete_item() -> None:
    store = _make_store()
    container: AsyncMock = store._container  # type: ignore[assignment]

    result = await store.adelete(ids=["id1", "id2"])

    assert result is True
    assert container.delete_item.call_count == 2
    container.delete_item.assert_any_call("id1", partition_key="id1")
    container.delete_item.assert_any_call("id2", partition_key="id2")


async def test_adelete_none_raises() -> None:
    store = _make_store()
    with pytest.raises(ValueError, match="No document ids provided"):
        await store.adelete(ids=None)


# ---- sync stubs raise NotImplementedError ----------------------------------


def test_add_texts_raises() -> None:
    store = _make_store()
    with pytest.raises(NotImplementedError):
        store.add_texts(texts=["hello"])


def test_similarity_search_raises() -> None:
    store = _make_store()
    with pytest.raises(NotImplementedError):
        store.similarity_search(query="hello")


def test_from_texts_raises() -> None:
    with pytest.raises(NotImplementedError):
        AsyncAzureCosmosDBNoSqlVectorSearch.from_texts(
            texts=["hello"], embedding=FakeEmbeddings()
        )
