"""Unit tests for AzureAIMemoryRetrieveTool and AzureAIMemorySaveTool."""

from unittest.mock import Mock, patch

import pytest

from langchain_azure_ai.tools import AzureAIMemoryRetrieveTool, AzureAIMemorySaveTool

try:
    import azure.ai.projects  # noqa: F401
except (ImportError, SyntaxError) as _exc:
    pytest.skip(
        f"azure-ai-projects 2.0.0b4+ is required for memory tools tests: {_exc}",
        allow_module_level=True,
    )


class TestAzureAIMemoryRetrieveTool:
    """Tests for AzureAIMemoryRetrieveTool."""

    def _make_tool(self, mock_client: Mock) -> AzureAIMemoryRetrieveTool:
        """Helper to create a tool with a mocked AIProjectClient."""
        with patch(
            "langchain_azure_ai.tools.services.memory.AIProjectClient",
            return_value=mock_client,
        ):
            return AzureAIMemoryRetrieveTool(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:alice",
                k=5,
            )

    def test_tool_name_and_description(self) -> None:
        """Test that name and description are set correctly."""
        mock_client = Mock()
        tool = self._make_tool(mock_client)

        assert tool.name == "azure_ai_memory_retrieve"
        assert "memories" in tool.description.lower()

    def test_tool_properties(self) -> None:
        """Test that constructor parameters are stored correctly."""
        mock_client = Mock()
        tool = self._make_tool(mock_client)

        assert tool.store_name == "test_store"
        assert tool.scope == "user:alice"
        assert tool.k == 5

    def test_run_returns_formatted_memories(self) -> None:
        """Test that _run returns a formatted list of memories."""
        mock_client = Mock()
        mock_memory_item = Mock()
        mock_memory_item.content = "Alice prefers dark roast coffee"
        mock_entry = Mock()
        mock_entry.memory_item = mock_memory_item

        mock_result = Mock()
        mock_result.memories = [mock_entry]
        mock_client.beta.memory_stores.search_memories = Mock(return_value=mock_result)

        tool = self._make_tool(mock_client)
        result = tool._run("What are my coffee preferences?")

        assert "Alice prefers dark roast coffee" in result
        assert result.startswith("- ")

    def test_run_with_multiple_memories(self) -> None:
        """Test that _run returns all memories in the result."""
        mock_client = Mock()

        def _make_entry(content: str) -> Mock:
            mem = Mock()
            mem.content = content
            entry = Mock()
            entry.memory_item = mem
            return entry

        mock_result = Mock()
        mock_result.memories = [
            _make_entry("Prefers dark roast"),
            _make_entry("Allergic to nuts"),
        ]
        mock_client.beta.memory_stores.search_memories = Mock(return_value=mock_result)

        tool = self._make_tool(mock_client)
        result = tool._run("User preferences")

        assert "Prefers dark roast" in result
        assert "Allergic to nuts" in result

    def test_run_returns_no_memories_message(self) -> None:
        """Test that _run returns a 'no memories' message when result is empty."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.memories = []
        mock_client.beta.memory_stores.search_memories = Mock(return_value=mock_result)

        tool = self._make_tool(mock_client)
        result = tool._run("anything")

        assert "No relevant memories found" in result

    def test_run_passes_correct_arguments(self) -> None:
        """Test that _run calls search_memories with the right parameters."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.memories = []
        mock_client.beta.memory_stores.search_memories = Mock(return_value=mock_result)

        tool = self._make_tool(mock_client)
        tool._run("coffee preferences")

        mock_client.beta.memory_stores.search_memories.assert_called_once()
        call_kwargs = mock_client.beta.memory_stores.search_memories.call_args[1]
        assert call_kwargs["name"] == "test_store"
        assert call_kwargs["scope"] == "user:alice"
        assert call_kwargs["items"] == "coffee preferences"

    def test_run_passes_max_memories_option(self) -> None:
        """Test that _run passes the k value as max_memories."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.memories = []
        mock_client.beta.memory_stores.search_memories = Mock(return_value=mock_result)

        with patch(
            "langchain_azure_ai.tools.services.memory.AIProjectClient",
            return_value=mock_client,
        ):
            tool = AzureAIMemoryRetrieveTool(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:alice",
                k=3,
            )

        tool._run("query")

        call_kwargs = mock_client.beta.memory_stores.search_memories.call_args[1]
        assert call_kwargs["options"].max_memories == 3

    def test_run_handles_partial_parse_error(self) -> None:
        """Test that _run handles errors in memory item parsing gracefully."""
        mock_client = Mock()
        mock_result = Mock()
        # Simulate an entry whose memory_item.content raises an AttributeError
        bad_entry = Mock()
        bad_entry.memory_item = None
        mock_result.memories = [bad_entry]
        mock_client.beta.memory_stores.search_memories = Mock(return_value=mock_result)

        tool = self._make_tool(mock_client)
        # Should not raise; returns no-memories message after failed parse
        result = tool._run("query")
        assert isinstance(result, str)

    def test_missing_project_endpoint_raises(self) -> None:
        """Test that omitting project_endpoint raises ValueError."""
        with pytest.raises(ValueError, match="project_endpoint"):
            AzureAIMemoryRetrieveTool(
                store_name="test_store",
                scope="user:alice",
            )

    def test_agent_scoped_retrieval(self) -> None:
        """Test that agent-scoped scope is forwarded correctly."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.memories = []
        mock_client.beta.memory_stores.search_memories = Mock(return_value=mock_result)

        with patch(
            "langchain_azure_ai.tools.services.memory.AIProjectClient",
            return_value=mock_client,
        ):
            tool = AzureAIMemoryRetrieveTool(
                project_endpoint="https://test.api.azureml.ms",
                store_name="agent_store",
                scope="agent:support-bot",
            )

        tool._run("open tickets")

        call_kwargs = mock_client.beta.memory_stores.search_memories.call_args[1]
        assert call_kwargs["scope"] == "agent:support-bot"


class TestAzureAIMemorySaveTool:
    """Tests for AzureAIMemorySaveTool."""

    def _make_tool(self, mock_client: Mock) -> AzureAIMemorySaveTool:
        """Helper to create a tool with a mocked AIProjectClient."""
        with patch(
            "langchain_azure_ai.tools.services.memory.AIProjectClient",
            return_value=mock_client,
        ):
            return AzureAIMemorySaveTool(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:alice",
            )

    def test_tool_name_and_description(self) -> None:
        """Test that name and description are set correctly."""
        mock_client = Mock()
        tool = self._make_tool(mock_client)

        assert tool.name == "azure_ai_memory_save"
        assert (
            "save" in tool.description.lower() or "persist" in tool.description.lower()
        )

    def test_tool_properties(self) -> None:
        """Test that constructor parameters are stored correctly."""
        mock_client = Mock()
        tool = self._make_tool(mock_client)

        assert tool.store_name == "test_store"
        assert tool.scope == "user:alice"
        assert tool.update_delay == 0

    def test_run_returns_success_message(self) -> None:
        """Test that _run returns a success message."""
        mock_client = Mock()
        mock_client.beta.memory_stores.begin_update_memories = Mock(return_value=Mock())

        tool = self._make_tool(mock_client)
        result = tool._run("Alice prefers dark roast coffee.")

        assert "successfully" in result.lower()

    def test_run_passes_correct_arguments(self) -> None:
        """Test that _run calls begin_update_memories with the right parameters."""
        mock_client = Mock()
        mock_client.beta.memory_stores.begin_update_memories = Mock(return_value=Mock())

        tool = self._make_tool(mock_client)
        tool._run("Alice prefers dark roast coffee.")

        mock_client.beta.memory_stores.begin_update_memories.assert_called_once()
        call_kwargs = mock_client.beta.memory_stores.begin_update_memories.call_args[1]
        assert call_kwargs["name"] == "test_store"
        assert call_kwargs["scope"] == "user:alice"
        assert call_kwargs["items"] == "Alice prefers dark roast coffee."
        assert call_kwargs["update_delay"] == 0

    def test_run_with_custom_update_delay(self) -> None:
        """Test that update_delay is forwarded to begin_update_memories."""
        mock_client = Mock()
        mock_client.beta.memory_stores.begin_update_memories = Mock(return_value=Mock())

        with patch(
            "langchain_azure_ai.tools.services.memory.AIProjectClient",
            return_value=mock_client,
        ):
            tool = AzureAIMemorySaveTool(
                project_endpoint="https://test.api.azureml.ms",
                store_name="test_store",
                scope="user:alice",
                update_delay=60,
            )

        tool._run("Some content")

        call_kwargs = mock_client.beta.memory_stores.begin_update_memories.call_args[1]
        assert call_kwargs["update_delay"] == 60

    def test_run_returns_error_message_on_failure(self) -> None:
        """Test that _run returns an error message when the API call fails."""
        mock_client = Mock()
        mock_client.beta.memory_stores.begin_update_memories = Mock(
            side_effect=Exception("Network error")
        )

        tool = self._make_tool(mock_client)
        result = tool._run("Some content")

        assert "Failed" in result
        assert "Network error" in result

    def test_missing_project_endpoint_raises(self) -> None:
        """Test that omitting project_endpoint raises ValueError."""
        with pytest.raises(ValueError, match="project_endpoint"):
            AzureAIMemorySaveTool(
                store_name="test_store",
                scope="user:alice",
            )

    def test_agent_scoped_save(self) -> None:
        """Test that agent-scoped scope is forwarded correctly."""
        mock_client = Mock()
        mock_client.beta.memory_stores.begin_update_memories = Mock(return_value=Mock())

        with patch(
            "langchain_azure_ai.tools.services.memory.AIProjectClient",
            return_value=mock_client,
        ):
            tool = AzureAIMemorySaveTool(
                project_endpoint="https://test.api.azureml.ms",
                store_name="agent_store",
                scope="agent:support-bot",
            )

        tool._run("Ticket #4821 resolved.")

        call_kwargs = mock_client.beta.memory_stores.begin_update_memories.call_args[1]
        assert call_kwargs["scope"] == "agent:support-bot"
