"""Unit tests for AzureAIMemoryMiddleware."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_azure_ai.agents.middleware import AzureAIMemoryMiddleware

try:
    import azure.ai.projects  # noqa: F401
except (ImportError, SyntaxError) as _exc:
    pytest.skip(
        f"azure-ai-projects 2.0.0b4+ is required for memory middleware tests: {_exc}",
        allow_module_level=True,
    )


def test_memory_middleware_batches_user_and_assistant_messages() -> None:
    """Middleware should batch only user/assistant messages and flush on cadence."""
    mock_client = Mock()
    first_poller = Mock(update_id="update_1")
    second_poller = Mock(update_id="update_2")
    mock_client.beta.memory_stores.begin_update_memories = Mock(
        side_effect=[first_poller, second_poller]
    )

    with patch(
        "langchain_azure_ai.agents.middleware.azure_ai_memory.AIProjectClient",
        return_value=mock_client,
    ):
        middleware = AzureAIMemoryMiddleware(
            store_name="test_store",
            scope="user:test",
            project_endpoint="https://test.api.azureml.ms",
            update_every_n_turns=2,
        )

    state1 = {
        "messages": [
            HumanMessage(content="user 1"),
            AIMessage(content="assistant 1"),
            SystemMessage(content="system"),
            ToolMessage(content="tool output", tool_call_id="tool_1"),
        ]
    }
    state2 = {
        "messages": [
            *state1["messages"],
            HumanMessage(content="user 2"),
            AIMessage(content="assistant 2"),
        ]
    }
    state3 = {
        "messages": [
            *state2["messages"],
            HumanMessage(content="user 3"),
            AIMessage(content="assistant 3"),
        ]
    }
    state4 = {
        "messages": [
            *state3["messages"],
            HumanMessage(content="user 4"),
            AIMessage(content="assistant 4"),
        ]
    }

    middleware.after_agent(state1, Mock())
    mock_client.beta.memory_stores.begin_update_memories.assert_not_called()

    middleware.after_agent(state2, Mock())
    assert mock_client.beta.memory_stores.begin_update_memories.call_count == 1
    first_call = mock_client.beta.memory_stores.begin_update_memories.call_args_list[0][1]
    assert first_call["name"] == "test_store"
    assert first_call["scope"] == "user:test"
    assert first_call["previous_update_id"] is None
    assert [item["content"] for item in first_call["items"]] == [
        "user 1",
        "assistant 1",
        "user 2",
        "assistant 2",
    ]

    middleware.after_agent(state3, Mock())
    assert mock_client.beta.memory_stores.begin_update_memories.call_count == 1

    middleware.after_agent(state4, Mock())
    assert mock_client.beta.memory_stores.begin_update_memories.call_count == 2
    second_call = mock_client.beta.memory_stores.begin_update_memories.call_args_list[1][1]
    assert second_call["previous_update_id"] == "update_1"


def test_memory_middleware_retries_after_failed_flush() -> None:
    """Middleware should keep pending items when update fails."""
    mock_client = Mock()
    successful_poller = Mock(update_id="update_1")
    mock_client.beta.memory_stores.begin_update_memories = Mock(
        side_effect=[Exception("network"), successful_poller]
    )

    with patch(
        "langchain_azure_ai.agents.middleware.azure_ai_memory.AIProjectClient",
        return_value=mock_client,
    ):
        middleware = AzureAIMemoryMiddleware(
            store_name="test_store",
            scope="user:test",
            project_endpoint="https://test.api.azureml.ms",
            update_every_n_turns=1,
        )

    middleware.after_agent({"messages": [HumanMessage(content="first")]}, Mock())
    first_call = mock_client.beta.memory_stores.begin_update_memories.call_args_list[0][1]
    assert [item["content"] for item in first_call["items"]] == ["first"]

    middleware.after_agent(
        {"messages": [HumanMessage(content="first"), HumanMessage(content="second")]},
        Mock(),
    )
    second_call = mock_client.beta.memory_stores.begin_update_memories.call_args_list[1][1]
    assert [item["content"] for item in second_call["items"]] == ["first", "second"]


def test_memory_middleware_validates_turn_interval() -> None:
    """Middleware should reject invalid update cadence values."""
    with pytest.raises(ValueError, match="update_every_n_turns must be >= 1"):
        AzureAIMemoryMiddleware(
            store_name="test_store",
            scope="user:test",
            project_endpoint="https://test.api.azureml.ms",
            update_every_n_turns=0,
        )
