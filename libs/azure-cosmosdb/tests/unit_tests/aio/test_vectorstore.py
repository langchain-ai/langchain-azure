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

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def aembed_query(self, text: str) -> List[float]:
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
    container.execute_item_batch.side_effect = [
        [{"resourceBody": {"id": "id1"}}],
        [{"resourceBody": {"id": "id2"}}],
    ]

    ids = await store.aadd_texts(
        texts=["hello", "world"],
        metadatas=[{"k": "v1"}, {"k": "v2"}],
        ids=["id1", "id2"],
    )

    assert ids == ["id1", "id2"]
    assert container.execute_item_batch.call_count == 2

    # Each call has one item (partition key /id means each item is a separate batch)
    first_call = container.execute_item_batch.call_args_list[0]
    batch_ops = first_call[0][0]  # positional arg: batch_operations
    assert len(batch_ops) == 1
    item = batch_ops[0][1][0]  # ("create", (item,), {})
    assert item["id"] == "id1"
    assert item["text"] == "hello"
    assert item["metadata"] == {"k": "v1"}
    assert "embedding" in item


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


# ---- factory method & client ownership tests --------------------------------


async def test_from_endpoint_and_aad_sets_owns_client() -> None:
    """Factory method sets _owns_client and stores the client."""
    from unittest.mock import patch

    mock_client = AsyncMock()
    mock_db = AsyncMock()
    mock_container = AsyncMock()
    mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
    mock_db.create_container_if_not_exists = AsyncMock(return_value=mock_container)
    mock_container.execute_item_batch = AsyncMock(
        return_value=[{"resourceBody": {"id": "1"}}]
    )

    with patch(
        "azure.cosmos.aio.CosmosClient",
        return_value=mock_client,
    ):
        store = await AsyncAzureCosmosDBNoSqlVectorSearch.from_endpoint_and_aad(
            endpoint="https://fake.documents.azure.com:443/",
            credential=MagicMock(),
            texts=["hello"],
            embedding=FakeEmbeddings(),
            vector_embedding_policy=DEFAULT_VEC_POLICY,
            indexing_policy=DEFAULT_IDX_POLICY,
            cosmos_container_properties=DEFAULT_CONTAINER_PROPS,
            cosmos_database_properties={"id": "testdb"},
            vector_search_fields=DEFAULT_VS_FIELDS,
        )
        assert store._owns_client is True


async def test_from_endpoint_and_key_sets_owns_client() -> None:
    """Factory method with key sets _owns_client."""
    from unittest.mock import patch

    mock_client = AsyncMock()
    mock_db = AsyncMock()
    mock_container = AsyncMock()
    mock_client.create_database_if_not_exists = AsyncMock(return_value=mock_db)
    mock_db.create_container_if_not_exists = AsyncMock(return_value=mock_container)
    mock_container.execute_item_batch = AsyncMock(
        return_value=[{"resourceBody": {"id": "1"}}]
    )

    with patch(
        "azure.cosmos.aio.CosmosClient",
        return_value=mock_client,
    ):
        store = await AsyncAzureCosmosDBNoSqlVectorSearch.from_endpoint_and_key(
            endpoint="https://fake.documents.azure.com:443/",
            key="fakekey",
            texts=["hello"],
            embedding=FakeEmbeddings(),
            vector_embedding_policy=DEFAULT_VEC_POLICY,
            indexing_policy=DEFAULT_IDX_POLICY,
            cosmos_container_properties=DEFAULT_CONTAINER_PROPS,
            cosmos_database_properties={"id": "testdb"},
            vector_search_fields=DEFAULT_VS_FIELDS,
        )
        assert store._owns_client is True


async def test_close_closes_client_when_owned() -> None:
    """close() calls client.close() when _owns_client is True."""
    store = _make_store()
    store._owns_client = True
    store._cosmos_client = AsyncMock()

    await store.close()
    store._cosmos_client.close.assert_awaited_once()


async def test_close_skips_when_not_owned() -> None:
    """close() does nothing when _owns_client is False."""
    store = _make_store()
    store._cosmos_client = AsyncMock()

    await store.close()
    store._cosmos_client.close.assert_not_awaited()


async def test_context_manager_calls_close() -> None:
    """__aexit__ calls close()."""
    store = _make_store()
    store._owns_client = True
    store._cosmos_client = AsyncMock()

    async with store:
        pass

    store._cosmos_client.close.assert_awaited_once()


async def test_factory_cleans_up_on_error() -> None:
    """Client is closed if _afrom_kwargs raises."""
    from unittest.mock import patch

    mock_client = AsyncMock()
    # Make create_database_if_not_exists raise to simulate failure
    mock_client.create_database_if_not_exists = AsyncMock(
        side_effect=RuntimeError("simulated failure")
    )

    with patch(
        "azure.cosmos.aio.CosmosClient",
        return_value=mock_client,
    ):
        with pytest.raises(RuntimeError, match="simulated failure"):
            await AsyncAzureCosmosDBNoSqlVectorSearch.from_endpoint_and_key(
                endpoint="https://fake.documents.azure.com:443/",
                key="fakekey",
                texts=["hello"],
                embedding=FakeEmbeddings(),
                vector_embedding_policy=DEFAULT_VEC_POLICY,
                indexing_policy=DEFAULT_IDX_POLICY,
                cosmos_container_properties=DEFAULT_CONTAINER_PROPS,
                cosmos_database_properties={"id": "testdb"},
                vector_search_fields=DEFAULT_VS_FIELDS,
            )
        mock_client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# SQL injection prevention
# ---------------------------------------------------------------------------


def test_async_construct_query_rejects_injection_in_projection() -> None:
    store = _make_store()
    with pytest.raises(ValueError, match="not a valid CosmosDB NoSQL identifier"):
        store._construct_query(
            k=4,
            search_type="vector",
            embeddings=[0.1, 0.2, 0.3],
            projection_mapping={"id OR 1=1--": "alias"},
        )


def test_async_construct_query_rejects_injection_in_search_field() -> None:
    store = _make_store()
    with pytest.raises(ValueError, match="not a valid CosmosDB NoSQL identifier"):
        store._construct_query(
            k=4,
            search_type="full_text_ranking",
            full_text_rank_filter=[
                {"search_field": "field); DROP", "search_text": "hi"}
            ],
        )


# ---------------------------------------------------------------------------
# Async embedding calls
# ---------------------------------------------------------------------------


async def test_aadd_texts_uses_aembed_not_sync() -> None:
    """aadd_texts should call aembed_documents, not embed_documents."""
    tracking = {"sync": False, "async": False}

    class TrackingEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            tracking["sync"] = True
            return [[0.1] for _ in texts]

        def embed_query(self, text: str) -> List[float]:
            return [0.1]

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            tracking["async"] = True
            return [[0.1] for _ in texts]

        async def aembed_query(self, text: str) -> List[float]:
            return [0.1]

    store = _make_store()
    store._embedding = TrackingEmbeddings()
    store._container.execute_item_batch.return_value = [{"resourceBody": {"id": "1"}}]

    await store.aadd_texts(texts=["hello"], ids=["1"])

    assert tracking["async"] is True
    assert tracking["sync"] is False


# ---------------------------------------------------------------------------
# threshold=0.0 handling
# ---------------------------------------------------------------------------


async def test_async_threshold_zero_is_respected() -> None:
    store = _make_store()

    async def fake_query_items(**kwargs: Any) -> Any:
        for item in [
            {"id": "d1", "text": "hi", "metadata": {}, "SimilarityScore": 0.001}
        ]:
            yield item

    setattr(store._container, "query_items", fake_query_items)

    results = await store._aexecute_query(
        query="SELECT ...",
        search_type="vector_score_threshold",
        parameters=[],
        with_embedding=False,
        projection_mapping=None,
        threshold=0.0,
    )
    assert len(results) == 1


async def test_async_threshold_none_defaults_to_zero() -> None:
    store = _make_store()

    async def fake_query_items(**kwargs: Any) -> Any:
        for item in [
            {"id": "d1", "text": "hi", "metadata": {}, "SimilarityScore": 0.001}
        ]:
            yield item

    setattr(store._container, "query_items", fake_query_items)

    results = await store._aexecute_query(
        query="SELECT ...",
        search_type="vector_score_threshold",
        parameters=[],
        with_embedding=False,
        projection_mapping=None,
        threshold=None,
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Async batch insertion
# ---------------------------------------------------------------------------


async def test_async_batch_shared_pk() -> None:
    store = _make_store()
    store._container.execute_item_batch.return_value = [
        {"resourceBody": {"id": "1"}},
        {"resourceBody": {"id": "2"}},
    ]
    result = await store._abatch_insert(
        [{"id": "1", "cat": "A"}, {"id": "2", "cat": "A"}], ["/cat"]
    )
    assert result == ["1", "2"]
    store._container.execute_item_batch.assert_called_once()


async def test_async_batch_different_pks() -> None:
    store = _make_store()
    store._container.execute_item_batch.side_effect = [
        [{"resourceBody": {"id": "1"}}],
        [{"resourceBody": {"id": "2"}}],
    ]
    result = await store._abatch_insert(
        [{"id": "1", "cat": "A"}, {"id": "2", "cat": "B"}], ["/cat"]
    )
    assert result == ["1", "2"]
    assert store._container.execute_item_batch.call_count == 2


async def test_async_batch_empty() -> None:
    store = _make_store()
    result = await store._abatch_insert([], ["/id"])
    assert result == []
    store._container.execute_item_batch.assert_not_called()


async def test_async_batch_over_100() -> None:
    store = _make_store()
    items = [{"id": str(i), "cat": "same"} for i in range(150)]
    store._container.execute_item_batch.side_effect = [
        [{"resourceBody": {"id": str(i)}} for i in range(100)],
        [{"resourceBody": {"id": str(i)}} for i in range(100, 150)],
    ]
    result = await store._abatch_insert(items, ["/cat"])
    assert len(result) == 150
    assert store._container.execute_item_batch.call_count == 2


async def test_aadd_texts_empty_raises() -> None:
    store = _make_store()
    with pytest.raises(ValueError, match="Texts can not be null or empty"):
        await store.aadd_texts(texts=[])
