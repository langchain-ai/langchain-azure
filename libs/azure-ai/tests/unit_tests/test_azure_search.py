import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional
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
def test_reorder_mmr_returns_empty_for_no_results() -> None:
    """The sync MMR reorder helper returns an empty list when the search
    yielded no candidates."""
    import numpy as np

    from langchain_azure_ai.vectorstores.azuresearch import (
        _reorder_results_with_maximal_marginal_relevance,
    )

    result = _reorder_results_with_maximal_marginal_relevance(
        iter([]),  # type: ignore[arg-type]
        query_embedding=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert result == []


@pytest.mark.requires("azure.search.documents")
def test_areorder_mmr_returns_empty_for_no_results() -> None:
    """The async MMR reorder helper must also return an empty list, mirroring
    the sync path, instead of raising a ValueError when unpacking an empty
    ``zip`` of results."""
    import numpy as np

    from langchain_azure_ai.vectorstores.azuresearch import (
        _areorder_results_with_maximal_marginal_relevance,
    )

    async def _empty_results() -> AsyncIterator[Dict[Any, Any]]:
        return
        yield  # pragma: no cover  -- makes this an (empty) async generator

    async def _run() -> List[Any]:
        return await _areorder_results_with_maximal_marginal_relevance(
            _empty_results(),  # type: ignore[arg-type]
            query_embedding=np.array([1.0, 0.0, 0.0, 0.0]),
        )

    assert asyncio.run(_run()) == []
