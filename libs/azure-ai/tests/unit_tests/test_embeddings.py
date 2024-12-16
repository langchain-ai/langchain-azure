from typing import Any
from unittest import mock

# import aiohttp to force Pants to include it in the required dependencies
import aiohttp  # noqa
import pytest
from azure.ai.inference.models import EmbeddingItem, EmbeddingsResult
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel


@pytest.fixture()
def test_embed_model() -> AzureAIEmbeddingsModel:
    with mock.patch(
        "langchain_azure_ai.embeddings.inference.EmbeddingsClient", autospec=True
    ):
        embed_model = AzureAIEmbeddingsModel(
            endpoint="https://my-endpoint.inference.ai.azure.com",
            credential="my-api-key",
            model_name="my_model_name",
        )
    embed_model._client.embed.return_value = EmbeddingsResult(  # type: ignore
        data=[EmbeddingItem(embedding=[1.0, 2.0, 3.0], index=0)]
    )
    return embed_model


def test_embed(test_embed_model: AzureAIEmbeddingsModel) -> None:
    """Test the basic embedding functionality."""
    # In case the endpoint being tested serves more than one model
    documents = [
        Document(
            id="1",
            page_content="Before college the two main things I worked on, "
            "outside of school, were writing and programming.",
        )
    ]
    vector_store = InMemoryVectorStore(test_embed_model)
    vector_store.add_documents(documents=documents)

    results = vector_store.similarity_search(query="Before college", k=1)

    assert len(results) == len(documents)
    assert results[0].page_content == documents[0].page_content


def test_get_metadata(test_embed_model: AzureAIEmbeddingsModel, caplog: Any) -> None:
    """Tests if we can get model metadata back from the endpoint. If so,
    model_name should not be 'unknown'. Some endpoints may not support this
    and in those cases a warning should be logged.
    """
    assert (
        test_embed_model.model_name != "unknown"
        or "does not support model metadata retrieval" in caplog.text
    )
