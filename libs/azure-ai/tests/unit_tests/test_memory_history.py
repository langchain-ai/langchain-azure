"""Unit tests for AzureAIMemoryChatMessageHistory."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_azure_ai.chat_message_histories import AzureAIMemoryChatMessageHistory

try:
    import azure.ai.projects  # noqa: F401
except (ImportError, SyntaxError) as _exc:
    pytest.skip(
        f"azure-ai-projects 2.0.0b4+ is required for memory history tests: {_exc}",
        allow_module_level=True,
    )


class TestRoleMapping:
    """Test role mapping from LangChain messages to Foundry items."""

    def test_human_message_mapping(self) -> None:
        """Test that human messages map to user message item param."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock()

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = HumanMessage(content="Hello")
        item = history._map_lc_message_to_foundry_item(msg)

        # Verify item has correct content and role
        assert "content" in item
        assert item["content"] == "Hello"
        assert item["role"] == "user"

    def test_ai_message_mapping(self) -> None:
        """Test that AI messages map to assistant message item param."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock()

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = AIMessage(content="Hi there!")
        item = history._map_lc_message_to_foundry_item(msg)

        # Verify item has correct content and role
        assert "content" in item
        assert item["content"] == "Hi there!"
        assert item["role"] == "assistant"

    def test_system_message_mapping(self) -> None:
        """Test that system messages map to system message item param."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock()

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_id",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = SystemMessage(content="System instruction")
        item = history._map_lc_message_to_foundry_item(msg)

        # Verify item has correct content and role
        assert "content" in item
        assert item["content"] == "System instruction"
        assert item["role"] == "system"

    def test_tool_message_mapping(self) -> None:
        """Test that tool messages map to assistant message item param."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock()

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = ToolMessage(content="Tool result", tool_call_id="tool_123")
        item = history._map_lc_message_to_foundry_item(msg)

        # Verify item has correct content and role
        assert "content" in item
        assert item["content"] == "Tool result"
        # Verify it's treated as assistant message (tool results are assistant output)
        assert item["role"] == "assistant"

    def test_ai_message_chunk_mapping(self) -> None:
        """Test AIMessageChunk maps to assistant message item param."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock()

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = AIMessageChunk(content="Streaming response")
        item = history._map_lc_message_to_foundry_item(msg)

        # Verify item has correct content and role
        assert "content" in item
        assert item["content"] == "Streaming response"
        # Verify it's treated as assistant message
        assert item["role"] == "assistant"


class TestChatMessageHistory:
    """Test AzureAIMemoryChatMessageHistory functionality."""

    def test_add_message_updates_base_history(self) -> None:
        """Test that adding a message updates the underlying base history."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock(return_value=Mock())

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = HumanMessage(content="Test message")
        history.add_message(msg)

        # Verify message was added to base history
        assert len(history.messages) == 1
        assert history.messages[0].content == "Test message"

    def test_add_message_calls_begin_update_memories(self) -> None:
        """Test that adding a message triggers Foundry update."""
        mock_client = Mock()
        mock_poller = Mock()
        mock_client.memory_stores.begin_update_memories = Mock(return_value=mock_poller)

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = HumanMessage(content="Test message")
        history.add_message(msg)

        # Verify begin_update_memories was called
        mock_client.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = mock_client.memory_stores.begin_update_memories.call_args[1]
        assert call_kwargs["name"] == "test_store"
        assert call_kwargs["scope"] == "user:test"
        assert len(call_kwargs["items"]) == 1

    def test_add_message_swallows_exceptions(self) -> None:
        """Test that exceptions during memory update don't break the chat flow."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock(
            side_effect=Exception("Network error")
        )

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        msg = HumanMessage(content="Test message")
        # Should not raise exception
        history.add_message(msg)

        # Message should still be in base history
        assert len(history.messages) == 1
        assert history.messages[0].content == "Test message"

    def test_clear_only_clears_base_history(self) -> None:
        """Test that clear() only affects the base history, not Foundry memories."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock(return_value=Mock())
        # Ensure these methods are NOT called
        mock_client.memory_stores.delete = Mock()
        mock_client.memory_stores.delete_scope = Mock()

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        history.add_message(HumanMessage(content="Test"))
        assert len(history.messages) == 1

        history.clear()

        # Base history should be cleared
        assert len(history.messages) == 0
        # Foundry delete methods should NOT be called
        mock_client.memory_stores.delete.assert_not_called()
        mock_client.memory_stores.delete_scope.assert_not_called()

    def test_properties(self) -> None:
        """Test that properties return correct values."""
        mock_client = Mock()

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test123",
                session_id="session_abc",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        assert history.store_name == "test_store"
        assert history.scope == "user:test123"
        assert history.session_id == "session_abc"

    def test_custom_role_mapper(self) -> None:
        """Test that custom role mapper is used when provided."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock(return_value=Mock())

        custom_item = Mock()
        custom_mapper = Mock(return_value=custom_item)

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
                role_mapper=custom_mapper,
            )

        msg = HumanMessage(content="Test")
        history.add_message(msg)

        # Custom mapper should have been called
        custom_mapper.assert_called_once_with(msg)

    def test_add_messages_multiple(self) -> None:
        """Test adding multiple messages at once."""
        mock_client = Mock()
        mock_client.memory_stores.begin_update_memories = Mock(return_value=Mock())

        with patch("azure.ai.projects.AIProjectClient", return_value=mock_client):
            history = AzureAIMemoryChatMessageHistory(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:test",
                session_id="session_1",
                base_history_factory=lambda _: InMemoryChatMessageHistory(),
            )

        messages = [
            HumanMessage(content="Message 1"),
            AIMessage(content="Response 1"),
            HumanMessage(content="Message 2"),
        ]

        history.add_messages(messages)

        # All messages should be in base history
        assert len(history.messages) == 3
        # begin_update_memories should be called for each message
        assert mock_client.memory_stores.begin_update_memories.call_count == 3
