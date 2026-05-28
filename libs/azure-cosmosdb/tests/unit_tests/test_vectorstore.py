"""Unit tests for AzureCosmosDBNoSqlVectorSearch field validation and projection."""

from typing import Any, Iterator, List

import pytest
from langchain_azure_cosmosdb._vectorstore import (
    AzureCosmosDBNoSqlVectorSearch,
    _validate_sql_identifier,
)
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# _validate_sql_identifier – valid identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "page_content",
        "embedding",
        "my_metadata",
        "text",
        "_private",
        "field123",
        "A",
    ],
)
def test_validate_sql_identifier_valid(name: str) -> None:
    """Valid identifiers should not raise."""
    _validate_sql_identifier(name, "test_field")  # no exception expected


# ---------------------------------------------------------------------------
# _validate_sql_identifier – invalid identifier patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "my-field",  # hyphen
        "my field",  # space
        "123abc",  # starts with digit
        "field.name",  # dot
        "field@name",  # @
        "",  # empty string
    ],
)
def test_validate_sql_identifier_invalid_pattern(name: str) -> None:
    """Identifiers with invalid characters or patterns should raise ValueError."""
    with pytest.raises(ValueError, match="not a valid CosmosDB NoSQL identifier"):
        _validate_sql_identifier(name, "test_field")


# ---------------------------------------------------------------------------
# _validate_sql_identifier – reserved keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "SELECT",
        "select",
        "From",
        "WHERE",
        "NULL",
        "ORDER",
        "VALUE",
        "top",
    ],
)
def test_validate_sql_identifier_reserved_keyword(name: str) -> None:
    """Reserved keywords should raise ValueError regardless of case."""
    with pytest.raises(ValueError, match="reserved CosmosDB NoSQL keyword"):
        _validate_sql_identifier(name, "test_field")


# ---------------------------------------------------------------------------
# _generate_projection_fields – non-default metadata_key / embedding_field
# ---------------------------------------------------------------------------


def _make_store_stub(
    text_field: str = "text",
    embedding_field: str = "embedding",
    metadata_key: str = "metadata",
    table_alias: str = "c",
) -> Any:
    """Return a minimal object that mimics the attributes used by
    _generate_projection_fields without constructing a full store."""

    class _Stub:
        _vector_search_fields = {
            "text_field": text_field,
            "embedding_field": embedding_field,
        }
        _metadata_key = metadata_key
        _table_alias = table_alias

        # Bind the method directly so we can call it without a real instance
        _generate_projection_fields = (
            AzureCosmosDBNoSqlVectorSearch._generate_projection_fields
        )

    return _Stub()


def test_projection_defaults_vector() -> None:
    """Default field names should appear verbatim in the projection alias."""
    stub = _make_store_stub()
    projection = stub._generate_projection_fields(None, "vector")
    assert "as text," in projection or "as text " in projection
    assert "as metadata" in projection
    assert "as VectorScore" in projection


def test_projection_custom_metadata_key() -> None:
    """Custom metadata_key should be used as the SQL alias for the metadata field."""
    stub = _make_store_stub(metadata_key="my_meta")
    projection = stub._generate_projection_fields(None, "vector")
    assert "as my_meta" in projection
    assert "as metadata" not in projection


def test_projection_custom_embedding_field_with_embedding() -> None:
    """Custom embedding_field is used as the SQL alias when with_embedding=True."""
    stub = _make_store_stub(embedding_field="content_vector")
    projection = stub._generate_projection_fields(None, "vector", with_embedding=True)
    assert "as content_vector" in projection
    assert "as embedding" not in projection


def test_projection_custom_text_field() -> None:
    """Custom text_field should be used as the SQL alias for the text field."""
    stub = _make_store_stub(text_field="page_content")
    projection = stub._generate_projection_fields(None, "vector")
    assert "as page_content" in projection


def test_projection_hybrid_custom_fields() -> None:
    """Non-default metadata and embedding fields produce correct SQL aliases."""
    stub = _make_store_stub(
        text_field="page_content",
        embedding_field="content_vector",
        metadata_key="doc_meta",
    )
    projection = stub._generate_projection_fields(None, "hybrid", with_embedding=True)
    assert "as page_content" in projection
    assert "as doc_meta" in projection
    assert "as content_vector" in projection
    # Hardcoded names must NOT appear when custom names differ
    assert "as metadata" not in projection
    assert "as embedding" not in projection


# ---------------------------------------------------------------------------
# SQL injection prevention — _construct_query
# ---------------------------------------------------------------------------


def _make_full_store(
    text_field: str = "text",
    embedding_field: str = "embedding",
    metadata_key: str = "metadata",
    table_alias: str = "c",
    search_type: str = "vector",
) -> AzureCosmosDBNoSqlVectorSearch:
    """Build a store with mocked Cosmos client/database/container."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_database = MagicMock()
    mock_container = MagicMock()
    mock_client.create_database_if_not_exists.return_value = mock_database
    mock_database.create_container_if_not_exists.return_value = mock_container

    return AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=mock_client,
        embedding=_FakeEmbeddings(),
        vector_embedding_policy={
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 3,
                }
            ]
        },
        indexing_policy={
            "vectorIndexes": [{"path": "/embedding", "type": "quantizedFlat"}]
        },
        cosmos_container_properties={"partition_key": MagicMock(path="/id")},
        cosmos_database_properties={},
        vector_search_fields={
            "text_field": text_field,
            "embedding_field": embedding_field,
        },
        database_name="testdb",
        container_name="testcontainer",
        search_type=search_type,
        metadata_key=metadata_key,
        table_alias=table_alias,
        create_container=False,
    )


class _FakeEmbeddings(Embeddings):
    """Minimal embeddings for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]


def test_construct_query_rejects_injection_in_projection_key() -> None:
    store = _make_full_store()
    with pytest.raises(ValueError, match="not a valid CosmosDB NoSQL identifier"):
        store._construct_query(
            k=4,
            search_type="vector",
            embeddings=[0.1, 0.2, 0.3],
            projection_mapping={"id; DROP TABLE c--": "alias"},
        )


def test_construct_query_rejects_injection_in_projection_alias() -> None:
    store = _make_full_store()
    with pytest.raises(ValueError, match="not a valid CosmosDB NoSQL identifier"):
        store._construct_query(
            k=4,
            search_type="vector",
            embeddings=[0.1, 0.2, 0.3],
            projection_mapping={"name": "alias) UNION SELECT *--"},
        )


def test_construct_query_rejects_reserved_keyword_in_projection() -> None:
    store = _make_full_store()
    with pytest.raises(ValueError, match="reserved CosmosDB NoSQL keyword"):
        store._construct_query(
            k=4,
            search_type="vector",
            embeddings=[0.1, 0.2, 0.3],
            projection_mapping={"SELECT": "alias"},
        )


def test_construct_query_rejects_injection_in_search_field() -> None:
    store = _make_full_store()
    with pytest.raises(ValueError, match="not a valid CosmosDB NoSQL identifier"):
        store._construct_query(
            k=4,
            search_type="full_text_ranking",
            full_text_rank_filter=[
                {"search_field": "text; DROP", "search_text": "hello"}
            ],
        )


def test_construct_query_accepts_valid_projection() -> None:
    store = _make_full_store()
    query, _ = store._construct_query(
        k=4,
        search_type="vector",
        embeddings=[0.1, 0.2, 0.3],
        projection_mapping={"name": "doc_name", "text": "content"},
    )
    assert "doc_name" in query
    assert "content" in query


# ---------------------------------------------------------------------------
# Generator input handling
# ---------------------------------------------------------------------------


def test_add_texts_with_generator() -> None:
    store = _make_full_store()
    store._container.execute_item_batch.side_effect = [
        [{"resourceBody": {"id": "0"}}],
        [{"resourceBody": {"id": "1"}}],
        [{"resourceBody": {"id": "2"}}],
    ]

    def text_gen() -> Iterator[str]:
        yield "hello"
        yield "world"
        yield "foo"

    ids = store.add_texts(texts=text_gen())
    assert len(ids) == 3


def test_add_texts_empty_raises() -> None:
    store = _make_full_store()
    with pytest.raises(ValueError, match="Texts can not be null or empty"):
        store.add_texts(texts=[])


def test_add_texts_empty_generator_raises() -> None:
    store = _make_full_store()
    with pytest.raises(ValueError, match="Texts can not be null or empty"):
        store.add_texts(texts=(x for x in []))


# ---------------------------------------------------------------------------
# threshold=0.0 handling
# ---------------------------------------------------------------------------


def test_execute_query_threshold_zero_keeps_results() -> None:
    store = _make_full_store()
    store._container.query_items.return_value = [
        {"id": "d1", "text": "hi", "metadata": {}, "VectorScore": 0.001}
    ]
    results = store._execute_query(
        query="SELECT TOP 1 ...",
        search_type="vector_score_threshold",
        parameters=[],
        with_embedding=False,
        projection_mapping=None,
        threshold=0.0,
    )
    assert len(results) == 1


def test_execute_query_threshold_none_defaults_to_zero() -> None:
    store = _make_full_store()
    store._container.query_items.return_value = [
        {"id": "d1", "text": "hi", "metadata": {}, "VectorScore": 0.001}
    ]
    results = store._execute_query(
        query="SELECT TOP 1 ...",
        search_type="vector_score_threshold",
        parameters=[],
        with_embedding=False,
        projection_mapping=None,
        threshold=None,
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Batch insertion
# ---------------------------------------------------------------------------


def test_batch_insert_shared_partition_key() -> None:
    store = _make_full_store()
    store._container.execute_item_batch.return_value = [
        {"resourceBody": {"id": "1"}},
        {"resourceBody": {"id": "2"}},
    ]
    items = [{"id": "1", "cat": "A"}, {"id": "2", "cat": "A"}]
    store._batch_insert(items, ["/cat"])
    store._container.execute_item_batch.assert_called_once()
    assert len(store._container.execute_item_batch.call_args[0][0]) == 2


def test_batch_insert_different_partition_keys() -> None:
    store = _make_full_store()
    store._container.execute_item_batch.side_effect = [
        [{"resourceBody": {"id": "1"}}],
        [{"resourceBody": {"id": "2"}}],
    ]
    items = [{"id": "1", "cat": "A"}, {"id": "2", "cat": "B"}]
    store._batch_insert(items, ["/cat"])
    assert store._container.execute_item_batch.call_count == 2


def test_batch_insert_over_100_splits() -> None:
    store = _make_full_store()
    items = [{"id": str(i), "cat": "same"} for i in range(150)]
    store._container.execute_item_batch.side_effect = [
        [{"resourceBody": {"id": str(i)}} for i in range(100)],
        [{"resourceBody": {"id": str(i)}} for i in range(100, 150)],
    ]
    store._batch_insert(items, ["/cat"])
    assert store._container.execute_item_batch.call_count == 2


def test_batch_insert_empty() -> None:
    store = _make_full_store()
    store._batch_insert([], ["/id"])
    store._container.execute_item_batch.assert_not_called()


def test_batch_insert_nested_pk_path() -> None:
    store = _make_full_store()
    store._container.execute_item_batch.return_value = [{"resourceBody": {"id": "1"}}]
    items = [{"id": "1", "metadata": {"category": "docs"}}]
    store._batch_insert(items, ["/metadata/category"])
    pk_arg = store._container.execute_item_batch.call_args[1]["partition_key"]
    assert pk_arg == "docs"


# ---------------------------------------------------------------------------
# Sync context manager
# ---------------------------------------------------------------------------


def test_sync_vectorstore_close() -> None:
    store = _make_full_store()
    store._owns_client = True
    store.close()
    store._cosmos_client.close.assert_called_once()


def test_sync_vectorstore_close_skips_when_not_owned() -> None:
    store = _make_full_store()
    store.close()
    store._cosmos_client.close.assert_not_called()


def test_sync_vectorstore_context_manager() -> None:
    store = _make_full_store()
    store._owns_client = True
    with store as s:
        assert s is store
    store._cosmos_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------


def test_retriever_get_relevant_documents() -> None:
    """Retriever._get_relevant_documents delegates to vectorstore."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    store = _make_full_store(search_type="vector")
    retriever = store.as_retriever(search_type="vector", k=3)

    with patch.object(
        store,
        "similarity_search",
        return_value=[Document(page_content="doc1")],
    ) as mock_search:
        docs = retriever.invoke("test query")
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs[1]["k"] == 3
        assert call_kwargs[1]["search_type"] == "vector"
        assert len(docs) == 1
        assert docs[0].page_content == "doc1"


async def test_retriever_aget_relevant_documents() -> None:
    """Retriever._aget_relevant_documents works for ainvoke."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    store = _make_full_store(search_type="vector")
    retriever = store.as_retriever(search_type="vector", k=3)

    with patch.object(
        store,
        "similarity_search",
        return_value=[Document(page_content="doc1")],
    ) as mock_search:
        docs = await retriever.ainvoke("test query")
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs[1]["k"] == 3
        assert call_kwargs[1]["search_type"] == "vector"
        assert len(docs) == 1
        assert docs[0].page_content == "doc1"


def test_retriever_invalid_search_type_raises() -> None:
    """Retriever rejects invalid search types."""
    store = _make_full_store()
    with pytest.raises(ValueError, match="not allowed"):
        store.as_retriever(search_type="invalid_type")


def test_retriever_all_search_types_accepted() -> None:
    """Retriever accepts all valid search types."""
    store = _make_full_store()
    for st in [
        "vector",
        "vector_score_threshold",
        "full_text_search",
        "full_text_ranking",
        "hybrid",
        "hybrid_score_threshold",
    ]:
        retriever = store.as_retriever(search_type=st)
        assert retriever.search_type == st


# ---------------------------------------------------------------------------
# MMR tests
# ---------------------------------------------------------------------------


def test_mmr_by_vector_forces_with_embedding() -> None:
    """MMR must fetch fetch_k candidates with embeddings, then rerank to k."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    store = _make_full_store(search_type="vector")

    fake_docs = [
        (Document(page_content="d1", metadata={"embedding": [0.1, 0.2, 0.3]}), 0.9),
        (Document(page_content="d2", metadata={"embedding": [0.4, 0.5, 0.6]}), 0.8),
        (Document(page_content="d3", metadata={"embedding": [0.7, 0.8, 0.9]}), 0.7),
        (Document(page_content="d4", metadata={"embedding": [0.2, 0.3, 0.4]}), 0.6),
        (Document(page_content="d5", metadata={"embedding": [0.5, 0.6, 0.7]}), 0.5),
    ]
    with patch.object(
        store, "vector_search_with_score", return_value=fake_docs
    ) as mock_search:
        results = store.max_marginal_relevance_search_by_vector(
            embedding=[0.1, 0.2, 0.3], k=2, fetch_k=5
        )
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["with_embedding"] is True
        # Should fetch fetch_k candidates
        assert call_kwargs["k"] == 5
        # Should return at most k results
        assert len(results) <= 2


def test_mmr_search_delegates_to_mmr_by_vector() -> None:
    """max_marginal_relevance_search embeds query then calls by_vector."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    store = _make_full_store(search_type="vector")
    expected_docs = [Document(page_content="result")]

    with patch.object(
        store, "max_marginal_relevance_search_by_vector", return_value=expected_docs
    ) as mock_mmr:
        docs = store.max_marginal_relevance_search("test query", k=2, fetch_k=5)
        mock_mmr.assert_called_once()
        call_kwargs = mock_mmr.call_args[1]
        assert call_kwargs["k"] == 2
        assert call_kwargs["fetch_k"] == 5
        assert docs == expected_docs


def test_similarity_search_by_vector_returns_docs() -> None:
    """similarity_search_by_vector returns documents without scores."""
    from unittest.mock import patch

    from langchain_core.documents import Document

    store = _make_full_store(search_type="vector")
    fake_results = [
        (Document(page_content="doc1"), 0.9),
        (Document(page_content="doc2"), 0.7),
    ]
    with patch.object(
        store, "vector_search_with_score", return_value=fake_results
    ) as mock_search:
        docs = store.similarity_search_by_vector([0.1, 0.2, 0.3], k=2)
        mock_search.assert_called_once()
        assert len(docs) == 2
        assert docs[0].page_content == "doc1"
        assert docs[1].page_content == "doc2"


# ---------------------------------------------------------------------------
# Vectorstore threshold tests per distance function
# ---------------------------------------------------------------------------


def _make_store_with_dist_fn(dist_fn: str) -> AzureCosmosDBNoSqlVectorSearch:
    return _make_full_store(search_type="vector_score_threshold")


def _patch_policy(store: Any, dist_fn: str) -> None:
    store._vector_embedding_policy = {
        "vectorEmbeddings": [{"distanceFunction": dist_fn}]
    }


def test_cosine_threshold_keeps_high_score() -> None:
    """Cosine: score 0.9 > threshold 0.5 → kept."""
    store = _make_store_with_dist_fn("cosine")
    _patch_policy(store, "cosine")
    store._container.query_items.return_value = [
        {"id": "1", "text": "doc", "metadata": {}, "VectorScore": 0.9}
    ]
    results = store.similarity_search(
        "q", k=1, search_type="vector_score_threshold", threshold=0.5
    )
    assert len(results) == 1


def test_cosine_threshold_skips_low_score() -> None:
    """Cosine: score 0.3 <= threshold 0.5 → skipped."""
    store = _make_store_with_dist_fn("cosine")
    _patch_policy(store, "cosine")
    store._container.query_items.return_value = [
        {"id": "1", "text": "doc", "metadata": {}, "VectorScore": 0.3}
    ]
    results = store.similarity_search(
        "q", k=1, search_type="vector_score_threshold", threshold=0.5
    )
    assert len(results) == 0


def test_dotproduct_threshold_keeps_high_score() -> None:
    """DotProduct: score 0.8 > threshold 0.5 → kept."""
    store = _make_store_with_dist_fn("dotproduct")
    _patch_policy(store, "dotproduct")
    store._container.query_items.return_value = [
        {"id": "1", "text": "doc", "metadata": {}, "VectorScore": 0.8}
    ]
    results = store.similarity_search(
        "q", k=1, search_type="vector_score_threshold", threshold=0.5
    )
    assert len(results) == 1


def test_dotproduct_threshold_skips_low_score() -> None:
    """DotProduct: score 0.2 <= threshold 0.5 → skipped."""
    store = _make_store_with_dist_fn("dotproduct")
    _patch_policy(store, "dotproduct")
    store._container.query_items.return_value = [
        {"id": "1", "text": "doc", "metadata": {}, "VectorScore": 0.2}
    ]
    results = store.similarity_search(
        "q", k=1, search_type="vector_score_threshold", threshold=0.5
    )
    assert len(results) == 0


def test_euclidean_threshold_keeps_low_distance() -> None:
    """Euclidean: distance 0.1 < threshold 0.5 → kept."""
    store = _make_store_with_dist_fn("euclidean")
    _patch_policy(store, "euclidean")
    store._container.query_items.return_value = [
        {"id": "1", "text": "doc", "metadata": {}, "VectorScore": 0.1}
    ]
    results = store.similarity_search(
        "q", k=1, search_type="vector_score_threshold", threshold=0.5
    )
    assert len(results) == 1


def test_euclidean_threshold_skips_high_distance() -> None:
    """Euclidean: distance 0.9 >= threshold 0.5 → skipped."""
    store = _make_store_with_dist_fn("euclidean")
    _patch_policy(store, "euclidean")
    store._container.query_items.return_value = [
        {"id": "1", "text": "doc", "metadata": {}, "VectorScore": 0.9}
    ]
    results = store.similarity_search(
        "q", k=1, search_type="vector_score_threshold", threshold=0.5
    )
    assert len(results) == 0
