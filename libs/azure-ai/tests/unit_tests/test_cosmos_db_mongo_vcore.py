"""Unit tests for AzureCosmosDBMongoVCoreVectorSearch."""

from typing import Dict, List, Optional
from unittest.mock import MagicMock
from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
    AzureCosmosDBMongoVCoreVectorSearch,
    CosmosDBVectorSearchType,
)
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

EMBEDDING_KEY = "vectorContent"
TEXT_KEY = "textContent"


def _make_vectorstore() -> AzureCosmosDBMongoVCoreVectorSearch:
    """Create a vectorstore instance with a mocked collection."""
    mock_collection = MagicMock()
    embeddings = FakeEmbeddings()
    return AzureCosmosDBMongoVCoreVectorSearch(
        collection=mock_collection,
        embedding=embeddings,
        text_key=TEXT_KEY,
        embedding_key=EMBEDDING_KEY,
    )


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


class TestMMRWithoutEmbedding:
    """Tests that max_marginal_relevance_search works even when with_embedding=False."""

    def test_mmr_default_no_embedding_in_metadata(self) -> None:
        """Regression test: MMR search must not raise KeyError when with_embedding
        is False (the default). Previously accessing doc.metadata[embedding_key]
        in the MMR step would raise KeyError because embeddings were not stored."""
        vectorstore = _make_vectorstore()
        fake_embedding = [1.0] * 9 + [0.0]
        results = [
            _make_search_result("foo", [1.0] * 9 + [float(i)], i)
            for i in range(3)
        ]

        vectorstore._collection.aggregate.return_value = iter(results)

        # with_embedding=False is the default; must not raise KeyError
        docs = vectorstore.max_marginal_relevance_search_by_vector(
            embedding=fake_embedding,
            k=2,
            fetch_k=3,
            kind=CosmosDBVectorSearchType.VECTOR_IVF,
            with_embedding=False,
        )

        assert len(docs) == 2
        # Embeddings must NOT be present in metadata when with_embedding=False
        for doc in docs:
            assert EMBEDDING_KEY not in doc.metadata

    def test_mmr_with_embedding_true_keeps_embedding(self) -> None:
        """When with_embedding=True, embeddings should remain in doc metadata."""
        vectorstore = _make_vectorstore()
        fake_embedding = [1.0] * 9 + [0.0]
        results = [
            _make_search_result("foo", [1.0] * 9 + [float(i)], i)
            for i in range(3)
        ]

        vectorstore._collection.aggregate.return_value = iter(results)

        docs = vectorstore.max_marginal_relevance_search_by_vector(
            embedding=fake_embedding,
            k=2,
            fetch_k=3,
            kind=CosmosDBVectorSearchType.VECTOR_IVF,
            with_embedding=True,
        )

        assert len(docs) == 2
        # Embeddings MUST be present in metadata when with_embedding=True
        for doc in docs:
            assert EMBEDDING_KEY in doc.metadata

    def test_mmr_search_default_no_embedding(self) -> None:
        """Regression test: max_marginal_relevance_search (high-level) must not
        raise KeyError with default parameters (with_embedding=False)."""
        vectorstore = _make_vectorstore()
        results = [
            _make_search_result("foo", [1.0] * 9 + [float(i)], i)
            for i in range(3)
        ]

        vectorstore._collection.aggregate.return_value = iter(results)

        # The high-level search goes through max_marginal_relevance_search_by_vector
        docs = vectorstore.max_marginal_relevance_search(
            query="test query",
            k=2,
            fetch_k=3,
            kind=CosmosDBVectorSearchType.VECTOR_IVF,
        )

        assert len(docs) == 2
        for doc in docs:
            assert EMBEDDING_KEY not in doc.metadata

    def test_mmr_user_metadata_under_embedding_key_preserved(self) -> None:
        """User metadata stored under embedding_key must not be clobbered or
        dropped by the internal MMR embedding fetch when with_embedding=False."""
        vectorstore = _make_vectorstore()
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

        vectorstore._collection.aggregate.return_value = iter(results)

        docs = vectorstore.max_marginal_relevance_search_by_vector(
            embedding=fake_embedding,
            k=2,
            fetch_k=3,
            kind=CosmosDBVectorSearchType.VECTOR_IVF,
            with_embedding=False,
        )

        assert len(docs) == 2
        # The user's original metadata value must be preserved, not clobbered
        for doc in docs:
            assert doc.metadata.get(EMBEDDING_KEY) == user_value
