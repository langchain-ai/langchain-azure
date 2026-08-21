from typing import TYPE_CHECKING, Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from pytest_socket import SocketBlockedError

from langchain_azure_ai.vectorstores.azuresearch import AzureSearch
from tests.integration_tests.vectorstores.fake_embeddings import FakeEmbeddings

DEFAULT_VECTOR_DIMENSION = 4

if TYPE_CHECKING:
    from azure.search.documents.indexes.models import SearchIndex


class FakeEmbeddingsWithDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    def __init__(self, dimension: int = DEFAULT_VECTOR_DIMENSION):
        super().__init__()
        self.dimension = dimension

    def embed_documents(self, embedding_texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (self.dimension - 1) + [float(i)]
            for i in range(len(embedding_texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (self.dimension - 1) + [float(0.0)]


DEFAULT_INDEX_NAME = "langchain-index"
DEFAULT_ENDPOINT = "https://my-search-service.search.windows.net"
DEFAULT_KEY = "mykey"
DEFAULT_ACCESS_TOKEN = "myaccesstoken1"
DEFAULT_EMBEDDING_MODEL = FakeEmbeddingsWithDimension()


def mock_default_index(*args: Any, **kwargs: Any) -> "SearchIndex":
    from azure.search.documents.indexes.models import (
        ExhaustiveKnnAlgorithmConfiguration,
        ExhaustiveKnnParameters,
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        VectorSearch,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )

    return SearchIndex(
        name=DEFAULT_INDEX_NAME,
        fields=[
            SearchField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                hidden=False,
                searchable=False,
                filterable=True,
                sortable=False,
                facetable=False,
            ),
            SearchField(
                name="content",
                type=SearchFieldDataType.String,
                key=False,
                hidden=False,
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False,
            ),
            SearchField(
                name="content_vector",
                type="Collection(Edm.Single)",
                searchable=True,
                vector_search_dimensions=4,
                vector_search_profile_name="myHnswProfile",
            ),
            SearchField(
                name="metadata",
                type="Edm.String",
                key=False,
                hidden=False,
                searchable=True,
                filterable=False,
                sortable=False,
                facetable=False,
            ),
        ],
        vector_search=VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile", algorithm_configuration_name="default"
                ),
                VectorSearchProfile(
                    name="myExhaustiveKnnProfile",
                    algorithm_configuration_name="default_exhaustive_knn",
                ),
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="default",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
                ExhaustiveKnnAlgorithmConfiguration(
                    name="default_exhaustive_knn",
                    parameters=ExhaustiveKnnParameters(
                        metric=VectorSearchAlgorithmMetric.COSINE
                    ),
                ),
            ],
        ),
    )


def create_vector_store(
    additional_search_client_options: Optional[Dict[str, Any]] = None,
) -> AzureSearch:
    return AzureSearch(
        azure_search_endpoint=DEFAULT_ENDPOINT,
        azure_search_key=DEFAULT_KEY,
        azure_ad_access_token=DEFAULT_ACCESS_TOKEN,
        index_name=DEFAULT_INDEX_NAME,
        embedding_function=DEFAULT_EMBEDDING_MODEL,
        additional_search_client_options=additional_search_client_options,
    )


@pytest.mark.requires("azure.search.documents")
def test_init_existing_index() -> None:
    from azure.search.documents.indexes import SearchIndexClient

    def mock_create_index() -> None:
        pytest.fail("Should not create index in this test")

    with patch.multiple(
        SearchIndexClient, get_index=mock_default_index, create_index=mock_create_index
    ):
        vector_store = create_vector_store()
        assert vector_store.client is not None


@pytest.mark.requires("azure.search.documents")
def test_init_new_index() -> None:
    from azure.core.exceptions import ResourceNotFoundError
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import SearchIndex

    def no_index(self: SearchIndexClient, name: str) -> SearchIndex:
        raise ResourceNotFoundError

    created_index: Optional[SearchIndex] = None

    def mock_create_index(self: SearchIndexClient, index: SearchIndex) -> None:
        nonlocal created_index
        created_index = index

    with patch.multiple(
        SearchIndexClient, get_index=no_index, create_index=mock_create_index
    ):
        vector_store = create_vector_store()
        assert vector_store.client is not None
        assert created_index is not None
        assert created_index.as_dict() == mock_default_index().as_dict()


@pytest.mark.requires("azure.search.documents")
def test_additional_search_options() -> None:
    from azure.search.documents.indexes import SearchIndexClient

    def mock_create_index() -> None:
        pytest.fail("Should not create index in this test")

    with patch.multiple(
        SearchIndexClient, get_index=mock_default_index, create_index=mock_create_index
    ):
        vector_store = create_vector_store(
            additional_search_client_options={"api_version": "test"}
        )
        assert vector_store.client is not None
        assert vector_store.client._config.api_version == "test"


@pytest.mark.requires("azure.search.documents")
def test_additional_search_options_retry_policy() -> None:
    """
    Reproduces bug captured in:
    https://github.com/langchain-ai/langchain-community/issues/76
    """
    from azure.core.exceptions import HttpResponseError, ServiceRequestError
    from azure.core.pipeline.policies import RetryPolicy
    from azure.search.documents.indexes import SearchIndexClient

    def mock_create_index() -> None:
        pytest.fail("Should not create index in this test")

    with patch.multiple(
        SearchIndexClient, get_index=mock_default_index, create_index=mock_create_index
    ):
        vector_store = create_vector_store(
            additional_search_client_options={
                "retry_policy": RetryPolicy(
                    total_retries=3,
                    backoff_factor=0.5,
                    timeout=5,
                ),
            }
        )
        assert vector_store.client is not None

        # Bug previously raised an:
        #  AttributeError: 'coroutine' object has no attribute 'http_response'.
        # Expect a network connection to be made (and blocked or refused).
        # ServiceRequestError covers DNS/connection failures in environments
        # where sockets are not blocked by pytest-socket.
        with pytest.raises(
            (HttpResponseError, ServiceRequestError, SocketBlockedError)
        ):
            list(vector_store.client.search())


@pytest.mark.requires("azure.search.documents")
def test_ids_used_correctly() -> None:
    """Check whether vector store uses the document ids when provided with them."""
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from langchain_core.documents import Document

    class Response:
        def __init__(self) -> None:
            self.succeeded: bool = True

    def mock_upload_documents(
        self: SearchClient, documents: List[object]
    ) -> List[Response]:
        # assume all documents uploaded successfully
        response = [Response() for _ in documents]
        return response

    documents = [
        Document(
            page_content="page zero Lorem Ipsum",
            metadata={"source": "document.pdf", "page": 0, "id": "ID-document-1"},
        ),
        Document(
            page_content="page one Lorem Ipsum",
            metadata={"source": "document.pdf", "page": 1, "id": "ID-document-2"},
        ),
    ]
    ids_provided = [i.metadata.get("id") for i in documents]

    with (
        patch.object(SearchClient, "upload_documents", mock_upload_documents),
        patch.object(SearchIndexClient, "get_index", mock_default_index),
    ):
        vector_store = create_vector_store()
        ids_used_at_upload = vector_store.add_documents(documents, ids=ids_provided)
        assert len(ids_provided) == len(ids_used_at_upload)
        assert ids_provided == ids_used_at_upload


@pytest.mark.requires("azure.search.documents")
def test_custom_field_names_set_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AZURESEARCH_FIELDS_* env vars must be honored even when set after the
    `azuresearch` module has already been imported.

    Regression test for https://github.com/langchain-ai/langchain-azure/issues/207:
    field names used to be resolved once into module-level globals at import
    time, so setting the env vars afterwards (the normal order of operations
    for any application) silently had no effect.
    """
    from azure.core.exceptions import ResourceNotFoundError
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import SearchIndex

    monkeypatch.setenv("AZURESEARCH_FIELDS_ID", "chunk_id")
    monkeypatch.setenv("AZURESEARCH_FIELDS_CONTENT", "chunk")
    monkeypatch.setenv("AZURESEARCH_FIELDS_CONTENT_VECTOR", "vector")
    monkeypatch.setenv("AZURESEARCH_FIELDS_TAG", "meta")

    def no_index(self: SearchIndexClient, name: str) -> SearchIndex:
        raise ResourceNotFoundError

    created_index: Optional[SearchIndex] = None

    def mock_create_index(self: SearchIndexClient, index: SearchIndex) -> None:
        nonlocal created_index
        created_index = index

    with patch.multiple(
        SearchIndexClient, get_index=no_index, create_index=mock_create_index
    ):
        vector_store = create_vector_store()

    assert vector_store._field_names.id == "chunk_id"
    assert vector_store._field_names.content == "chunk"
    assert vector_store._field_names.content_vector == "vector"
    assert vector_store._field_names.metadata == "meta"

    assert created_index is not None
    assert {f.name for f in created_index.fields} == {
        "chunk_id",
        "chunk",
        "vector",
        "meta",
    }


@pytest.mark.requires("azure.search.documents")
def test_add_texts_uses_custom_field_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """`add_texts` must upload documents keyed by the resolved field names,
    not by the hardcoded defaults.
    """
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    monkeypatch.setenv("AZURESEARCH_FIELDS_ID", "chunk_id")
    monkeypatch.setenv("AZURESEARCH_FIELDS_CONTENT", "chunk")
    monkeypatch.setenv("AZURESEARCH_FIELDS_CONTENT_VECTOR", "vector")
    monkeypatch.setenv("AZURESEARCH_FIELDS_TAG", "meta")

    class Response:
        def __init__(self) -> None:
            self.succeeded: bool = True

    uploaded: List[Dict[str, Any]] = []

    def mock_upload_documents(
        self: SearchClient, documents: List[Dict[str, Any]]
    ) -> List[Response]:
        uploaded.extend(documents)
        return [Response() for _ in documents]

    with (
        patch.object(SearchClient, "upload_documents", mock_upload_documents),
        patch.object(SearchIndexClient, "get_index", mock_default_index),
    ):
        vector_store = create_vector_store()
        vector_store.add_texts(["hello world"])

    assert len(uploaded) == 1
    assert {"chunk_id", "chunk", "vector", "meta"}.issubset(uploaded[0].keys())
    assert "content_vector" not in uploaded[0]


@pytest.mark.requires("azure.search.documents")
def test_field_names_are_isolated_per_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two `AzureSearch` instances configured with different field names must
    not clobber each other.

    Previously the field names were resolved once into shared module-level
    globals, so the most-recently-constructed instance would silently win for
    every instance in the process.
    """
    from azure.search.documents.indexes import SearchIndexClient

    with patch.object(SearchIndexClient, "get_index", mock_default_index):
        monkeypatch.setenv("AZURESEARCH_FIELDS_CONTENT_VECTOR", "vector_a")
        store_a = create_vector_store()

        monkeypatch.setenv("AZURESEARCH_FIELDS_CONTENT_VECTOR", "vector_b")
        store_b = create_vector_store()

        assert store_a._field_names.content_vector == "vector_a"
        assert store_b._field_names.content_vector == "vector_b"


@pytest.mark.requires("azure.search.documents")
def test_semantic_hybrid_search_returns_matching_answer() -> None:
    """`semantic_hybrid_search_with_score_and_rerank` must attach the semantic
    answer for a document's own id.

    Regression test: the id was previously popped off the raw result before
    being looked up in the semantic-answers map, so `answers` always came
    back empty regardless of whether the service returned one.
    """
    import json

    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    class FakeAnswer:
        def __init__(self, key: str, text: str, highlights: str) -> None:
            self.key = key
            self.text = text
            self.highlights = highlights

    class FakeSearchResults:
        def __init__(
            self, items: List[Dict[str, Any]], answers: List[FakeAnswer]
        ) -> None:
            self._items = items
            self._answers = answers

        def __iter__(self) -> Any:
            return iter(self._items)

        def get_answers(self) -> List[FakeAnswer]:
            return self._answers

    result_item = {
        "id": "doc-1",
        "content": "hello world",
        "content_vector": [1.0, 1.0, 1.0, 0.0],
        "metadata": json.dumps({"source": "x"}),
        "@search.score": 0.9,
        "@search.reranker_score": 2.5,
    }
    answer = FakeAnswer(key="doc-1", text="the answer", highlights="the answer")

    def mock_search(self: SearchClient, **kwargs: Any) -> FakeSearchResults:
        return FakeSearchResults([dict(result_item)], [answer])

    with (
        patch.object(SearchClient, "search", mock_search),
        patch.object(SearchIndexClient, "get_index", mock_default_index),
    ):
        vector_store = create_vector_store()
        docs = vector_store.semantic_hybrid_search_with_score_and_rerank("query")

    assert len(docs) == 1
    doc, _score, _reranker_score = docs[0]
    assert doc.metadata["id"] == "doc-1"
    assert doc.metadata["answers"] == {"text": "the answer", "highlights": "the answer"}
