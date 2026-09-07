"""Unit tests for AzureDocumentDBVectorSearch."""

from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import Embeddings

from langchain_azure_documentdb import (
    AzureDocumentDBVectorSearch,
    AzureDocumentDBVectorSearchType,
)

EMBEDDING_KEY = "vectorContent"
TEXT_KEY = "textContent"


class FakeEmbeddings(Embeddings):
    """Fake embeddings for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    def embed_query(self, text: str) -> List[float]:
        return [float(1.0)] * 9 + [float(0.0)]


def _make_vectorstore() -> Tuple[AzureDocumentDBVectorSearch, MagicMock]:
    """Create a vectorstore instance with a mocked collection.

    Returns both the vectorstore and the MagicMock so tests can set
    ``mock_collection.aggregate.return_value`` directly (mypy-safe).
    """
    mock_collection: MagicMock = MagicMock()
    embeddings = FakeEmbeddings()
    vectorstore = AzureDocumentDBVectorSearch(
        collection=mock_collection,
        embedding=embeddings,
        text_key=TEXT_KEY,
        embedding_key=EMBEDDING_KEY,
    )
    return vectorstore, mock_collection


def _make_search_result(
    text: str,
    embedding: List[float],
    index: int,
    extra_metadata: Optional[Dict] = None,
) -> Dict:
    """Build a fake aggregation result document."""
    metadata: Dict = {} if extra_metadata is None else dict(extra_metadata)
    return {
        "similarityScore": 1.0 - index * 0.1,
        "document": {
            "_id": f"id_{index}",
            TEXT_KEY: text,
            EMBEDDING_KEY: embedding,
            "metadata": metadata,
        },
    }


def test_add_texts_accepts_single_pass_iterable() -> None:
    vectorstore, mock_collection = _make_vectorstore()
    texts = (value for value in ["one", "two", "three"])

    vectorstore.add_texts(texts)

    documents = mock_collection.insert_many.call_args.args[0]
    assert [document[TEXT_KEY] for document in documents] == ["one", "two", "three"]
    assert [document["metadata"] for document in documents] == [{}, {}, {}]


@pytest.mark.parametrize(
    "kind",
    [
        AzureDocumentDBVectorSearchType.VECTOR_IVF,
        AzureDocumentDBVectorSearchType.VECTOR_HNSW,
        AzureDocumentDBVectorSearchType.VECTOR_DISKANN,
    ],
)
def test_similarity_search_forwards_oversampling(
    kind: AzureDocumentDBVectorSearchType,
) -> None:
    vectorstore, mock_collection = _make_vectorstore()
    mock_collection.aggregate.return_value = []

    vectorstore.similarity_search("query", kind=kind, oversampling=7.5)

    pipeline = mock_collection.aggregate.call_args.args[0]
    assert pipeline[0]["$search"]["cosmosSearch"]["oversampling"] == 7.5


class TestMMRWithoutEmbedding:
    """Tests that max_marginal_relevance_search works even when with_embedding=False."""

    def test_mmr_default_no_embedding_in_metadata(self) -> None:
        """Regression test: MMR search must not raise KeyError when with_embedding
        is False (the default). Previously accessing doc.metadata[embedding_key]
        in the MMR step would raise KeyError because embeddings were not stored."""
        vectorstore, mock_collection = _make_vectorstore()
        fake_embedding = [1.0] * 9 + [0.0]
        results = [
            _make_search_result("foo", [1.0] * 9 + [float(i)], i) for i in range(3)
        ]

        mock_collection.aggregate.return_value = iter(results)

        # with_embedding=False is the default; must not raise KeyError
        docs = vectorstore.max_marginal_relevance_search_by_vector(
            embedding=fake_embedding,
            k=2,
            fetch_k=3,
            kind=AzureDocumentDBVectorSearchType.VECTOR_IVF,
            with_embedding=False,
        )

        assert len(docs) == 2
        # Embeddings must NOT be present in metadata when with_embedding=False
        for doc in docs:
            assert EMBEDDING_KEY not in doc.metadata

    def test_mmr_with_embedding_true_keeps_embedding(self) -> None:
        """When with_embedding=True, embeddings should remain in doc metadata."""
        vectorstore, mock_collection = _make_vectorstore()
        fake_embedding = [1.0] * 9 + [0.0]
        results = [
            _make_search_result("foo", [1.0] * 9 + [float(i)], i) for i in range(3)
        ]

        mock_collection.aggregate.return_value = iter(results)

        docs = vectorstore.max_marginal_relevance_search_by_vector(
            embedding=fake_embedding,
            k=2,
            fetch_k=3,
            kind=AzureDocumentDBVectorSearchType.VECTOR_IVF,
            with_embedding=True,
        )

        assert len(docs) == 2
        # Embeddings MUST be present in metadata when with_embedding=True
        for doc in docs:
            assert EMBEDDING_KEY in doc.metadata

    def test_mmr_search_default_no_embedding(self) -> None:
        """Regression test: max_marginal_relevance_search (high-level) must not
        raise KeyError with default parameters (with_embedding=False)."""
        vectorstore, mock_collection = _make_vectorstore()
        results = [
            _make_search_result("foo", [1.0] * 9 + [float(i)], i) for i in range(3)
        ]

        mock_collection.aggregate.return_value = iter(results)

        # The high-level search goes through max_marginal_relevance_search_by_vector
        docs = vectorstore.max_marginal_relevance_search(
            query="test query",
            k=2,
            fetch_k=3,
            kind=AzureDocumentDBVectorSearchType.VECTOR_IVF,
        )

        assert len(docs) == 2
        for doc in docs:
            assert EMBEDDING_KEY not in doc.metadata

    def test_mmr_user_metadata_under_embedding_key_preserved(self) -> None:
        """User metadata stored under embedding_key must not be clobbered or
        dropped by the internal MMR embedding fetch when with_embedding=False."""
        vectorstore, mock_collection = _make_vectorstore()
        fake_embedding = [1.0] * 9 + [0.0]
        user_value = "user_label"
        results = [
            _make_search_result(
                "foo",
                [1.0] * 9 + [float(i)],
                i,
                extra_metadata={EMBEDDING_KEY: user_value},
            )
            for i in range(3)
        ]

        mock_collection.aggregate.return_value = iter(results)

        docs = vectorstore.max_marginal_relevance_search_by_vector(
            embedding=fake_embedding,
            k=2,
            fetch_k=3,
            kind=AzureDocumentDBVectorSearchType.VECTOR_IVF,
            with_embedding=False,
        )

        assert len(docs) == 2
        # The user's original metadata value must be preserved, not clobbered
        for doc in docs:
            assert doc.metadata.get(EMBEDDING_KEY) == user_value
