"""Unit tests for _utils.py helper functions."""

import numpy as np
import pytest
from langchain_azure_cosmosdb._utils import (
    DistanceStrategy,
    cosine_similarity,
    filter_complex_metadata,
    maximal_marginal_relevance,
)
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_empty_x() -> None:
    result = cosine_similarity([], [[1.0, 2.0]])
    assert len(result) == 0


def test_cosine_similarity_empty_y() -> None:
    result = cosine_similarity([[1.0, 2.0]], [])
    assert len(result) == 0


def test_cosine_similarity_matching_dimensions() -> None:
    X = [[1.0, 0.0], [0.0, 1.0]]
    Y = [[1.0, 0.0], [0.0, 1.0]]
    result = cosine_similarity(X, Y)
    assert result.shape == (2, 2)
    np.testing.assert_almost_equal(result[0][0], 1.0)
    np.testing.assert_almost_equal(result[1][1], 1.0)
    np.testing.assert_almost_equal(result[0][1], 0.0)


def test_cosine_similarity_mismatched_dimensions() -> None:
    X = [[1.0, 0.0, 0.0]]
    Y = [[1.0, 0.0]]
    with pytest.raises(
        ValueError, match="Number of columns in X and Y must be the same"
    ):
        cosine_similarity(X, Y)


# ---------------------------------------------------------------------------
# maximal_marginal_relevance
# ---------------------------------------------------------------------------


def test_mmr_empty_embedding_list() -> None:
    query = np.array([1.0, 0.0])
    result = maximal_marginal_relevance(query, [], k=4)
    assert result == []


def test_mmr_k_zero() -> None:
    query = np.array([1.0, 0.0])
    embeddings = [np.array([1.0, 0.0])]
    result = maximal_marginal_relevance(query, embeddings, k=0)
    assert result == []


def test_mmr_returns_correct_indices() -> None:
    query = np.array([1.0, 0.0])
    embeddings = [
        np.array([1.0, 0.0]),
        np.array([0.9, 0.1]),
        np.array([0.0, 1.0]),
    ]
    result = maximal_marginal_relevance(query, embeddings, k=2, lambda_mult=0.5)
    assert len(result) == 2
    # Most similar should be first
    assert result[0] == 0


# ---------------------------------------------------------------------------
# filter_complex_metadata
# ---------------------------------------------------------------------------


def test_filter_complex_metadata_removes_disallowed_types() -> None:
    docs = [
        Document(
            page_content="test",
            metadata={"name": "Alice", "tags": ["a", "b"], "nested": {"k": "v"}},
        )
    ]
    filtered = filter_complex_metadata(docs)
    assert "name" in filtered[0].metadata
    assert "tags" not in filtered[0].metadata
    assert "nested" not in filtered[0].metadata


def test_filter_complex_metadata_keeps_allowed_types() -> None:
    docs = [
        Document(
            page_content="test",
            metadata={"name": "Bob", "age": 42, "score": 3.14, "active": True},
        )
    ]
    filtered = filter_complex_metadata(docs)
    assert filtered[0].metadata == {
        "name": "Bob",
        "age": 42,
        "score": 3.14,
        "active": True,
    }


def test_filter_complex_metadata_does_not_mutate_originals() -> None:
    """Original documents must not be modified."""
    original_metadata = {"name": "Alice", "tags": ["a", "b"], "count": 5}
    docs = [Document(page_content="test", metadata=original_metadata)]

    filtered = filter_complex_metadata(docs)

    # Original document's metadata is unchanged
    assert "tags" in docs[0].metadata
    assert docs[0].metadata == {"name": "Alice", "tags": ["a", "b"], "count": 5}
    # Filtered result has tags removed
    assert "tags" not in filtered[0].metadata
    # They are different objects
    assert filtered[0] is not docs[0]


def test_filter_complex_metadata_empty_metadata() -> None:
    docs = [Document(page_content="test", metadata={})]
    filtered = filter_complex_metadata(docs)
    assert filtered[0].metadata == {}


def test_filter_complex_metadata_all_filtered_out() -> None:
    docs = [
        Document(
            page_content="test",
            metadata={"list_val": [1, 2], "dict_val": {"a": 1}},
        )
    ]
    filtered = filter_complex_metadata(docs)
    assert filtered[0].metadata == {}


# ---------------------------------------------------------------------------
# DistanceStrategy enum
# ---------------------------------------------------------------------------


def test_distance_strategy_values() -> None:
    assert DistanceStrategy.COSINE == "COSINE"
    assert DistanceStrategy.EUCLIDEAN_DISTANCE == "EUCLIDEAN_DISTANCE"
    assert DistanceStrategy.MAX_INNER_PRODUCT == "MAX_INNER_PRODUCT"
    assert DistanceStrategy.DOT_PRODUCT == "DOT_PRODUCT"
    assert DistanceStrategy.JACCARD == "JACCARD"


def test_filter_complex_metadata_preserves_document_id() -> None:
    """Document.id must be preserved through filtering."""
    docs = [
        Document(id="doc-1", page_content="a", metadata={"k": "v", "bad": [1, 2]}),
        Document(id="doc-2", page_content="b", metadata={"n": 3}),
    ]
    filtered = filter_complex_metadata(docs)
    assert [d.id for d in filtered] == ["doc-1", "doc-2"]
    # Filtering still happened.
    assert filtered[0].metadata == {"k": "v"}
    assert filtered[1].metadata == {"n": 3}
