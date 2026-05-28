"""Unit tests for AzureAIMemoryRetrieverTool."""

from unittest.mock import Mock, patch

from langchain_core.documents import Document

from langchain_azure_ai.tools import AzureAIMemoryRetrieverTool


def test_memory_retriever_tool_initializes_retriever() -> None:
    """Tool should initialize the underlying retriever with provided args."""
    mock_retriever = Mock()
    with patch(
        "langchain_azure_ai.tools.azure_ai_memory.AzureAIMemoryRetriever",
        return_value=mock_retriever,
    ) as retriever_cls:
        AzureAIMemoryRetrieverTool(
            project_endpoint="https://test.api.azureml.ms",
            store_name="test_store",
            scope="user:test",
            k=7,
        )

    retriever_cls.assert_called_once_with(
        project_endpoint="https://test.api.azureml.ms",
        credential=None,
        store_name="test_store",
        scope="user:test",
        k=7,
    )


def test_memory_retriever_tool_formats_results() -> None:
    """Tool should return a readable bullet list for found memories."""
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="Prefers dark roast coffee",
            metadata={"memory_id": "m1"},
        ),
        Document(page_content="Likes cappuccino", metadata={}),
    ]

    with patch(
        "langchain_azure_ai.tools.azure_ai_memory.AzureAIMemoryRetriever",
        return_value=mock_retriever,
    ):
        tool = AzureAIMemoryRetrieverTool(
            project_endpoint="https://test.api.azureml.ms",
            store_name="test_store",
            scope="user:test",
        )

    result = tool.invoke({"query": "coffee preferences"})
    assert "- [m1] Prefers dark roast coffee" in result
    assert "- Likes cappuccino" in result
    mock_retriever.invoke.assert_called_once_with("coffee preferences")


def test_memory_retriever_tool_returns_empty_message() -> None:
    """Tool should return a default message when no memories are found."""
    mock_retriever = Mock()
    mock_retriever.invoke.return_value = []

    with patch(
        "langchain_azure_ai.tools.azure_ai_memory.AzureAIMemoryRetriever",
        return_value=mock_retriever,
    ):
        tool = AzureAIMemoryRetrieverTool(
            project_endpoint="https://test.api.azureml.ms",
            store_name="test_store",
            scope="user:test",
        )

    assert tool.invoke({"query": "unknown"}) == "No relevant memories found."
