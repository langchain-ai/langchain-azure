"""Unit tests for AzureAIMemoryMiddleware."""

from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

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
            roles=["user", "assistant"],
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

    middleware.after_agent(cast(Any, state1), Mock())
    mock_client.beta.memory_stores.begin_update_memories.assert_not_called()

    middleware.after_agent(cast(Any, state2), Mock())
    assert mock_client.beta.memory_stores.begin_update_memories.call_count == 1
    first_call = mock_client.beta.memory_stores.begin_update_memories.call_args_list[0][
        1
    ]
    assert first_call["name"] == "test_store"
    assert first_call["scope"] == "user:test"
    assert first_call["previous_update_id"] is None
    assert [item["content"] for item in first_call["items"]] == [
        "user 1",
        "assistant 1",
        "user 2",
        "assistant 2",
    ]

    middleware.after_agent(cast(Any, state3), Mock())
    assert mock_client.beta.memory_stores.begin_update_memories.call_count == 1

    middleware.after_agent(cast(Any, state4), Mock())
    assert mock_client.beta.memory_stores.begin_update_memories.call_count == 2
    update_calls = mock_client.beta.memory_stores.begin_update_memories.call_args_list
    second_call = update_calls[1][1]
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

    update = middleware.after_agent(
        {"messages": [HumanMessage(content="first")]}, Mock()
    )
    first_call = mock_client.beta.memory_stores.begin_update_memories.call_args_list[0][
        1
    ]
    assert [item["content"] for item in first_call["items"]] == ["first"]

    middleware.after_agent(
        cast(
            Any,
            {
                **(update or {}),
                "messages": [
                    HumanMessage(content="first"),
                    HumanMessage(content="second"),
                ],
            },
        ),
        Mock(),
    )
    update_calls = mock_client.beta.memory_stores.begin_update_memories.call_args_list
    second_call = update_calls[1][1]
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


def test_memory_middleware_defaults_to_user_role() -> None:
    """Middleware should remember only user messages by default."""
    mock_client = Mock()
    mock_client.beta.memory_stores.begin_update_memories = Mock(return_value=Mock())

    with patch(
        "langchain_azure_ai.agents.middleware.azure_ai_memory.AIProjectClient",
        return_value=mock_client,
    ):
        middleware = AzureAIMemoryMiddleware(
            store_name="test_store",
            scope="user:test",
            project_endpoint="https://test.api.azureml.ms",
        )

    middleware.after_agent(
        {"messages": [HumanMessage(content="user"), AIMessage(content="assistant")]},
        Mock(),
    )
    update_memory_calls = (
        mock_client.beta.memory_stores.begin_update_memories.call_args_list
    )
    call_kwargs = update_memory_calls[0][1]
    assert len(call_kwargs["items"]) == 1
    assert call_kwargs["items"][0]["role"] == "user"


def test_memory_middleware_resolves_project_endpoint_from_env_alias() -> None:
    """Middleware should resolve endpoint from FOUNDRY_PROJECT_ENDPOINT."""
    mock_client = Mock()
    mock_client.beta.memory_stores.begin_update_memories = Mock(return_value=Mock())

    with (
        patch.dict(
            "os.environ",
            {"FOUNDRY_PROJECT_ENDPOINT": "https://foundry.endpoint"},
            clear=True,
        ),
        patch(
            "langchain_azure_ai.agents.middleware.azure_ai_memory.AIProjectClient",
            return_value=mock_client,
        ) as mock_project_client,
    ):
        AzureAIMemoryMiddleware(
            store_name="test_store",
            scope="user:test",
        )

    assert (
        mock_project_client.call_args_list[0][1]["endpoint"]
        == "https://foundry.endpoint"
    )


def _create_memory_agent(
    responses: list[str], checkpointer: Any = None, **kwargs: Any
) -> tuple[Any, Mock]:
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    mock_client = Mock()
    mock_client.beta.memory_stores.begin_update_memories = Mock(return_value=Mock())

    with patch(
        "langchain_azure_ai.agents.middleware.azure_ai_memory.AIProjectClient",
        return_value=mock_client,
    ):
        middleware = AzureAIMemoryMiddleware(
            store_name="test_store",
            scope="user:test",
            project_endpoint="https://test.api.azureml.ms",
            roles=["user", "assistant"],
            **kwargs,
        )
    agent = create_agent(
        model=FakeListChatModel(responses=responses),
        middleware=[middleware],
        checkpointer=checkpointer,
    )
    return agent, mock_client


def _summarize_messages(agent: Any, config: Any) -> None:
    from langgraph.graph.message import REMOVE_ALL_MESSAGES

    agent.update_state(
        config,
        {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                SystemMessage(content="summary"),
            ]
        },
    )


def _sent_contents(mock_client: Mock) -> list[str]:
    return [
        item["content"]
        for call in mock_client.beta.memory_stores.begin_update_memories.call_args_list
        for item in call.kwargs["items"]
    ]


def test_memory_middleware_tracks_each_thread() -> None:
    """Messages from every thread should be sent, not only the first one."""
    from langgraph.checkpoint.memory import InMemorySaver

    agent, mock_client = _create_memory_agent(
        ["reply a1", "reply b1", "reply a2", "reply b2"], InMemorySaver()
    )

    for content, thread_id in [("a1", "a"), ("b1", "b"), ("a2", "a"), ("b2", "b")]:
        agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            {"configurable": {"thread_id": thread_id}},
        )

    assert _sent_contents(mock_client) == [
        "a1",
        "reply a1",
        "b1",
        "reply b1",
        "a2",
        "reply a2",
        "b2",
        "reply b2",
    ]


def test_memory_middleware_does_not_resend_checkpointed_messages() -> None:
    """A new middleware instance shouldn't resend a thread's earlier turns."""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    config: Any = {"configurable": {"thread_id": "thread"}}
    agent, _ = _create_memory_agent(["reply 1"], checkpointer)
    agent.invoke({"messages": [HumanMessage(content="user 1")]}, config)

    agent, mock_client = _create_memory_agent(["reply 2"], checkpointer)
    agent.invoke({"messages": [HumanMessage(content="user 2")]}, config)

    assert _sent_contents(mock_client) == ["user 2", "reply 2"]


def test_memory_middleware_without_checkpointer_sends_each_turn() -> None:
    """Each invocation's turn should be sent when there is no checkpointer."""
    agent, mock_client = _create_memory_agent(["reply 1", "reply 2", "reply 3"])

    for content in ["user 1", "user 2", "user 3"]:
        agent.invoke({"messages": [HumanMessage(content=content)]})

    assert _sent_contents(mock_client) == [
        "user 1",
        "reply 1",
        "user 2",
        "reply 2",
        "user 3",
        "reply 3",
    ]


def test_memory_middleware_without_checkpointer_skips_passed_history() -> None:
    """History passed back in by the caller shouldn't be sent again."""
    agent, mock_client = _create_memory_agent(["reply 1", "reply 2", "reply 3"])

    messages: list[Any] = []
    for content in ["user 1", "user 2", "user 3"]:
        result = agent.invoke({"messages": [*messages, HumanMessage(content=content)]})
        messages = result["messages"]

    assert _sent_contents(mock_client) == [
        "user 1",
        "reply 1",
        "user 2",
        "reply 2",
        "user 3",
        "reply 3",
    ]


def test_memory_middleware_without_checkpointer_skips_passed_chat_messages() -> None:
    """History passed back as ``ChatMessage`` shouldn't be sent again."""
    agent, mock_client = _create_memory_agent(["reply 1", "reply 2", "reply 3"])

    messages: list[Any] = []
    for content in ["user 1", "user 2", "user 3"]:
        result = agent.invoke(
            {"messages": [*messages, ChatMessage(role="user", content=content)]}
        )
        messages = result["messages"]

    assert _sent_contents(mock_client) == [
        "user 1",
        "reply 1",
        "user 2",
        "reply 2",
        "user 3",
        "reply 3",
    ]


def test_memory_middleware_sends_queued_turns_after_recreation() -> None:
    """Turns queued by an earlier instance should be sent by the new one."""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    config: Any = {"configurable": {"thread_id": "thread"}}
    agent, first_client = _create_memory_agent(
        ["reply 1"], checkpointer, update_every_n_turns=2
    )
    agent.invoke({"messages": [HumanMessage(content="user 1")]}, config)
    first_client.beta.memory_stores.begin_update_memories.assert_not_called()

    agent, second_client = _create_memory_agent(
        ["reply 2", "reply 3"], checkpointer, update_every_n_turns=2
    )
    agent.invoke({"messages": [HumanMessage(content="user 2")]}, config)
    agent.invoke({"messages": [HumanMessage(content="user 3")]}, config)

    assert _sent_contents(second_client) == [
        "user 1",
        "reply 1",
        "user 2",
        "reply 2",
        "user 3",
        "reply 3",
    ]

    agent, third_client = _create_memory_agent(["reply 4"], checkpointer)
    agent.invoke({"messages": [HumanMessage(content="user 4")]}, config)

    assert _sent_contents(third_client) == ["user 4", "reply 4"]


def test_memory_middleware_sends_failed_turns_after_recreation() -> None:
    """Turns from a failed update should be sent by a new instance."""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    config: Any = {"configurable": {"thread_id": "thread"}}
    agent, first_client = _create_memory_agent(["reply 1"], checkpointer)
    first_client.beta.memory_stores.begin_update_memories.side_effect = Exception(
        "network"
    )
    agent.invoke({"messages": [HumanMessage(content="user 1")]}, config)

    agent, second_client = _create_memory_agent(["reply 2"], checkpointer)
    agent.invoke({"messages": [HumanMessage(content="user 2")]}, config)

    assert _sent_contents(second_client) == [
        "user 1",
        "reply 1",
        "user 2",
        "reply 2",
    ]


def test_memory_middleware_does_not_resend_turns_sent_with_another_thread() -> None:
    """Turns sent in a batch flushed by another thread shouldn't be sent again."""
    from langgraph.checkpoint.memory import InMemorySaver

    agent, mock_client = _create_memory_agent(
        ["reply a1", "reply b1", "reply a2", "reply b2"],
        InMemorySaver(),
        update_every_n_turns=2,
    )

    for content, thread_id in [("a1", "a"), ("b1", "b"), ("a2", "a"), ("b2", "b")]:
        agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            {"configurable": {"thread_id": thread_id}},
        )

    update_calls = mock_client.beta.memory_stores.begin_update_memories.call_args_list
    assert [
        [item["content"] for item in call.kwargs["items"]] for call in update_calls
    ] == [
        ["a1", "reply a1", "b1", "reply b1"],
        ["a2", "reply a2", "b2", "reply b2"],
    ]


def test_memory_middleware_sends_latest_turn_after_failed_turns_are_summarized() -> (
    None
):
    """Unsent turns that were summarized away shouldn't hide the latest turn."""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    config: Any = {"configurable": {"thread_id": "thread"}}
    agent, first_client = _create_memory_agent(["reply 1", "reply 2"], checkpointer)
    first_client.beta.memory_stores.begin_update_memories.side_effect = [
        Mock(),
        Exception("network"),
    ]
    for content in ["user 1", "user 2"]:
        agent.invoke({"messages": [HumanMessage(content=content)]}, config)
    _summarize_messages(agent, config)

    agent, second_client = _create_memory_agent(["reply 3"], checkpointer)
    agent.invoke({"messages": [HumanMessage(content="user 3")]}, config)

    assert _sent_contents(second_client) == ["user 3", "reply 3"]


def test_memory_middleware_sends_turns_queued_after_summarization() -> None:
    """Turns queued after a summarization should be sent by a new instance."""
    from langgraph.checkpoint.memory import InMemorySaver

    checkpointer = InMemorySaver()
    config: Any = {"configurable": {"thread_id": "thread"}}
    agent, first_client = _create_memory_agent(
        ["reply 1", "reply 2", "reply 3", "reply 4", "reply 5"],
        checkpointer,
        update_every_n_turns=3,
    )
    for content in ["user 1", "user 2", "user 3", "user 4"]:
        agent.invoke({"messages": [HumanMessage(content=content)]}, config)
    _summarize_messages(agent, config)
    agent.invoke({"messages": [HumanMessage(content="user 5")]}, config)
    assert first_client.beta.memory_stores.begin_update_memories.call_count == 1

    agent, second_client = _create_memory_agent(["reply 6"], checkpointer)
    agent.invoke({"messages": [HumanMessage(content="user 6")]}, config)

    assert _sent_contents(second_client) == [
        "user 5",
        "reply 5",
        "user 6",
        "reply 6",
    ]
