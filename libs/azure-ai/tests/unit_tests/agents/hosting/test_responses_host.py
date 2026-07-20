# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``ResponsesHostServer`` via Starlette TestClient."""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses import ResponseObject  # noqa: E402
from azure.ai.agentserver.responses.models import (  # noqa: E402
    ItemMessage,
    MessageContentInputTextContent,
)
from langchain_core.runnables import RunnableConfig  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import (  # noqa: E402
    ResponsesHostServer,
    ResponsesServerOptions,
)
from langchain_azure_ai.agents.hosting._responses import (  # noqa: E402
    CONVERSATION_METADATA_CHECKPOINT_ID,
    CONVERSATION_METADATA_NAMESPACE,
    CONVERSATION_METADATA_THREAD_ID,
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    ConversationChainStorageManager,
    HostingRunnableConfig,
)
from langchain_azure_ai.agents.hosting._responses_host import (  # noqa: E402
    METADATA_STEERABLE_CONVERSATION,
)

from .conftest import (  # noqa: E402
    make_checkpointed_echo_graph,
    make_checkpointed_two_node_graph,
    make_custom_state_graph,
    make_echo_graph,
    make_streaming_graph,
)


def _client(server: ResponsesHostServer) -> TestClient:
    return TestClient(server.app)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_type = ""
    for line in body.splitlines():
        if line.startswith("event:"):
            current_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {"raw": data}
            events.append((current_type, payload))
    return events


def _message_item(text: str) -> ItemMessage:
    return ItemMessage(
        role="user",
        content=[MessageContentInputTextContent({"type": "input_text", "text": text})],
    )


def _request(**kwargs: object) -> MagicMock:
    request = MagicMock()
    request.instructions = kwargs.get("instructions")
    request.previous_response_id = kwargs.get("previous_response_id")
    request.conversation = kwargs.get("conversation")
    return request


def _response_object(
    response_id: str,
    *,
    previous_response_id: str | None = None,
    conversation_id: str | None = None,
    internal_metadata: dict[str, object] | None = None,
) -> ResponseObject:
    return cast(
        ResponseObject,
        SimpleNamespace(
            id=response_id,
            previous_response_id=previous_response_id,
            conversation=(
                SimpleNamespace(id=conversation_id) if conversation_id else None
            ),
            internal_metadata=internal_metadata,
        ),
    )


class _ConversationChainMetadata(dict[str, object]):
    def __init__(
        self,
        namespaces: dict[str, "_ConversationChainMetadata"] | None = None,
    ) -> None:
        super().__init__()
        self._namespaces = namespaces if namespaces is not None else {}
        self.flush_count = 0

    def __call__(self, name: str | None = None) -> "_ConversationChainMetadata":
        if name is None:
            return self
        return self._namespaces.setdefault(
            name,
            _ConversationChainMetadata(self._namespaces),
        )

    async def flush(self) -> None:
        self.flush_count += 1


def _context(
    *,
    response_id: str = "resp-current",
    conversation_id: str | None = "conv-test",
    current_text: str = "hello",
    history: list[object] | None = None,
    conversation_chain_metadata: _ConversationChainMetadata | None = None,
) -> MagicMock:
    context = MagicMock()
    context.response_id = response_id
    context.conversation_id = conversation_id
    context.conversation_chain_id = conversation_id or f"resp-{response_id}"
    context.is_recovery = False
    context.persisted_response = None
    context.client_cancelled = False
    context._cancellation_signal = asyncio.Event()
    context.platform_context = object()
    context.get_input_items = AsyncMock(return_value=[_message_item(current_text)])
    context.get_history = AsyncMock(return_value=history or [])
    context.conversation_chain_metadata = (
        conversation_chain_metadata
        if conversation_chain_metadata is not None
        else _ConversationChainMetadata()
    )
    return context


def test_non_streaming_request_returns_completed_response() -> None:
    server = ResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/responses", json={"input": "hello", "model": "test"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "completed"
    output = payload["output"]
    assert any(item.get("type") == "message" for item in output)
    text = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
    )
    assert "Echo: hello" in text


def test_streaming_request_emits_sse_lifecycle_events() -> None:
    server = ResponsesHostServer(make_streaming_graph())
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={"input": "ignored", "stream": True, "model": "test"},
        )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    types = [t for t, _ in events]
    assert "response.created" in types
    assert "response.in_progress" in types
    assert "response.output_text.delta" in types
    assert "response.completed" in types
    deltas = [p["delta"] for t, p in events if t == "response.output_text.delta"]
    assert "".join(deltas) == "Hello, world!"


def test_steerable_capability_metadata_is_true_when_enabled() -> None:
    server = ResponsesHostServer(
        make_streaming_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={
                "input": "hello",
                "metadata": {
                    "client.key": "kept",
                },
            },
        )

    metadata = resp.json()["metadata"]
    assert metadata[METADATA_STEERABLE_CONVERSATION] == "true"
    assert metadata["client.key"] == "kept"


def test_steerable_capability_metadata_is_false_when_disabled() -> None:
    server = ResponsesHostServer(make_streaming_graph())
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={
                "input": "hello",
                "metadata": {
                    "client.key": "kept",
                },
            },
        )

    metadata = resp.json()["metadata"]
    assert metadata[METADATA_STEERABLE_CONVERSATION] == "false"
    assert metadata["client.key"] == "kept"


async def test_steering_pressure_completes_superseded_response() -> None:
    server = ResponsesHostServer(make_streaming_graph())
    context = _context(current_text="original turn")
    context.client_cancelled = False
    cancellation_signal = asyncio.Event()
    cancellation_signal.set()

    events = [
        event
        async for event in server.handle_create(
            _request(),
            context,
            cancellation_signal,
        )
    ]

    event_types = [event.type for event in events if hasattr(event, "type")]
    assert event_types[-1] == "response.completed"
    assert "response.failed" not in event_types


async def test_handle_create_passes_cancellation_signal_to_graph() -> None:
    captured_config: dict[str, object] = {}
    server = ResponsesHostServer(make_streaming_graph(captured_config))
    context = _context()
    cancellation_signal = asyncio.Event()

    _ = [
        event
        async for event in server.handle_create(
            _request(),
            context,
            cancellation_signal,
        )
    ]

    assert (
        captured_config["configurable"]["response_cancellation_signal"]  # type: ignore[index]
        is cancellation_signal
    )


async def test_handle_create_checkpoints_admission_before_config_resolution() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None)
    events = server.handle_create(
        _request(previous_response_id="resp-parent"), context, asyncio.Event()
    )

    assert (await anext(events)).type == "response.created"
    assert (await anext(events)).type == "response.in_progress"
    admission_checkpoint = await anext(events)

    assert type(admission_checkpoint).__name__ == "ResponseCheckpointEvent"
    assert dict(admission_checkpoint.response.internal_metadata) == {}
    await events.aclose()


async def test_handle_create_persists_conversation_metadata_once_per_turn() -> None:
    chain_metadata = _ConversationChainMetadata()
    server = ResponsesHostServer(make_checkpointed_echo_graph())

    _ = [
        event
        async for event in server.handle_create(
            _request(),
            _context(conversation_chain_metadata=chain_metadata),
            asyncio.Event(),
        )
    ]

    langgraph_metadata = chain_metadata(CONVERSATION_METADATA_NAMESPACE)
    assert langgraph_metadata[CONVERSATION_METADATA_THREAD_ID] == "resp-current"
    assert isinstance(
        langgraph_metadata[CONVERSATION_METADATA_CHECKPOINT_ID],
        str,
    )
    assert langgraph_metadata.flush_count == 1


async def test_recovery_replays_input_without_current_response_checkpoint() -> None:
    chain_metadata = _ConversationChainMetadata()
    langgraph_metadata = chain_metadata(CONVERSATION_METADATA_NAMESPACE)
    langgraph_metadata[CONVERSATION_METADATA_CHECKPOINT_ID] = "checkpoint-parent"
    langgraph_metadata[CONVERSATION_METADATA_THREAD_ID] = "resp-root"
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        current_text="replay me",
        conversation_chain_metadata=chain_metadata,
    )
    context.is_recovery = True
    context.persisted_response = _response_object("resp-current")
    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"),
        context,
    )

    graph_input = await server._resume_graph_input(
        _request(previous_response_id="resp-parent"), context
    )

    assert config["configurable"]["checkpoint_id"] == "checkpoint-parent"
    assert [message.content for message in graph_input["messages"]] == ["replay me"]


async def test_recovery_resumes_from_current_response_checkpoint() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, current_text="do not replay")
    context.is_recovery = True
    context.persisted_response = SimpleNamespace(
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-current",
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        }
    )

    graph_input = await server._resume_graph_input(_request(), context)

    assert graph_input is None


def test_readiness_endpoint_is_available() -> None:
    server = ResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_constructor_accepts_reexported_response_options() -> None:
    options = ResponsesServerOptions(default_model="test")
    server = ResponsesHostServer(make_echo_graph(), options=options)

    with _client(server) as client:
        resp = client.post("/responses", json={"input": "hello"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


def test_constructor_rejects_resilient_background_without_checkpointer() -> None:
    options = ResponsesServerOptions(resilient_background=True)

    with pytest.raises(
        ValueError,
        match="requires a LangGraph checkpointer when resilient_background=True",
    ):
        ResponsesHostServer(make_echo_graph(), options=options)


def test_constructor_accepts_resilient_background_with_checkpointer() -> None:
    options = ResponsesServerOptions(resilient_background=True)

    ResponsesHostServer(make_checkpointed_echo_graph(), options=options)


def test_constructor_rejects_non_messages_state_schema() -> None:
    with pytest.raises(ValueError, match="messages"):
        ResponsesHostServer(make_custom_state_graph())


async def test_checkpointed_graph_uses_current_input_only() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context()

    result = await server.build_input(_request(), context)

    context.get_history.assert_not_called()
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "hello"


async def test_checkpointed_previous_response_id_does_not_duplicate_history() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        history=[_message_item("from responses transcript")],
        current_text="current turn",
    )

    result = await server.build_input(
        _request(previous_response_id="resp-previous"),
        context,
    )

    context.get_history.assert_not_called()
    assert [message.content for message in result["messages"]] == ["current turn"]


async def test_non_checkpointed_graph_uses_responses_history() -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(history=[_message_item("from history")])

    result = await server.build_input(_request(), context)

    context.get_history.assert_awaited_once()
    assert [message.content for message in result["messages"]] == [
        "from history",
        "hello",
    ]


async def test_non_checkpointed_previous_response_id_includes_history_once() -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(
        conversation_id=None,
        history=[_message_item("turn one"), _message_item("turn two")],
        current_text="turn three",
    )

    result = await server.build_input(
        _request(previous_response_id="resp-2"),
        context,
    )

    context.get_history.assert_awaited_once()
    assert [message.content for message in result["messages"]] == [
        "turn one",
        "turn two",
        "turn three",
    ]


async def test_root_response_id_is_thread_id() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(response_id="resp-1", conversation_id=None)

    config = await server.build_runnable_config(_request(), context)

    assert config["configurable"]["thread_id"] == "resp-1"
    assert config["configurable"]["response_context"] is context


async def test_explicit_conversation_id_is_thread_id() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(response_id="resp-1", conversation_id="conv-context")

    config = await server.build_runnable_config(
        _request(conversation="conv-request"),
        context,
    )

    assert config["configurable"]["thread_id"] == "conv-request"
    assert config["configurable"]["response_context"] is context


async def test_previous_response_without_chain_metadata_is_rejected() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None)

    with pytest.raises(
        RuntimeError,
        match="Conversation chain metadata is required for a follow-up response",
    ):
        await server.build_runnable_config(
            _request(previous_response_id="resp-2"),
            context,
        )


async def test_previous_response_uses_conversation_chain_metadata() -> None:
    chain_metadata = _ConversationChainMetadata()
    langgraph_metadata = chain_metadata(CONVERSATION_METADATA_NAMESPACE)
    langgraph_metadata[CONVERSATION_METADATA_CHECKPOINT_ID] = "checkpoint-2"
    langgraph_metadata[CONVERSATION_METADATA_THREAD_ID] = "resp-1"
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        conversation_chain_metadata=chain_metadata,
    )

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-1"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"
    assert config["configurable"]["checkpoint_ns"] == ""


async def test_explicit_conversation_uses_conversation_chain_metadata() -> None:
    chain_metadata = _ConversationChainMetadata()
    langgraph_metadata = chain_metadata(CONVERSATION_METADATA_NAMESPACE)
    langgraph_metadata[CONVERSATION_METADATA_CHECKPOINT_ID] = "checkpoint-2"
    langgraph_metadata[CONVERSATION_METADATA_THREAD_ID] = "conv-api"
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id="conv-api",
        conversation_chain_metadata=chain_metadata,
    )

    config = await server.build_runnable_config(
        _request(conversation="conv-api"),
        context,
    )

    assert config["configurable"]["thread_id"] == "conv-api"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"


async def test_recovery_uses_persisted_thread_and_checkpoint() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        response_id="resp-current",
        conversation_id=None,
    )
    context.is_recovery = True
    context.persisted_response = SimpleNamespace(
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-committed",
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        }
    )

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-root"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-committed"
    assert config["configurable"]["checkpoint_ns"] == ""


async def test_recovery_ignores_checkpoint_newer_than_response_snapshot() -> None:
    server = ResponsesHostServer(make_checkpointed_two_node_graph())
    request = _request()
    context = _context(
        response_id="resp-root",
        conversation_id=None,
        current_text="turn one",
    )
    config = await server.build_runnable_config(request, context)
    graph_input = await server.build_input(request, context)
    committed_checkpoint_id: str | None = None
    graph_stream = server.graph.astream(
        graph_input,
        config,
        stream_mode=["updates", "checkpoints"],
        durability="sync",
    )
    saw_plan = False
    async for mode, payload in graph_stream:
        if mode == "updates" and "plan" in payload:
            saw_plan = True
        elif mode == "checkpoints" and saw_plan and committed_checkpoint_id is None:
            committed_checkpoint_id = payload["config"]["configurable"]["checkpoint_id"]

    assert committed_checkpoint_id is not None
    latest_snapshot = await server.graph.aget_state(config)
    assert (
        latest_snapshot.config["configurable"]["checkpoint_id"]
        != committed_checkpoint_id
    )

    recovery_context = _context(
        response_id="resp-root",
        conversation_id=None,
        current_text="turn one",
    )
    recovery_context.is_recovery = True
    recovery_context.persisted_response = SimpleNamespace(
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: committed_checkpoint_id,
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        }
    )
    recovery_config = await server.build_runnable_config(
        request,
        recovery_context,
    )
    assert recovery_config["configurable"]["checkpoint_id"] == committed_checkpoint_id
    recovered_nodes = [
        node
        async for update in server.graph.astream(
            None,
            recovery_config,
            stream_mode="updates",
            durability="sync",
        )
        for node in update
    ]

    assert recovered_nodes == ["research"]


async def test_conversation_metadata_supports_linear_response_chain() -> None:
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    chain_metadata = _ConversationChainMetadata()

    async def invoke(
        response_id: str,
        text: str,
        previous_response_id: str | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        request = _request(previous_response_id=previous_response_id)
        context = _context(
            response_id=response_id,
            conversation_id=None,
            current_text=text,
            conversation_chain_metadata=chain_metadata,
        )
        config = await server.build_runnable_config(request, context)
        graph_input = await server.build_input(request, context)
        state: dict[str, object] = {}
        graph_stream = server.graph.astream(
            graph_input,
            config=config,
            stream_mode=["values", "checkpoints"],
            durability="sync",
        )
        async for mode, payload in graph_stream:
            if mode == "values":
                state = payload
            elif mode == "checkpoints":
                checkpoint_ref = HostingRunnableConfig(
                    cast(RunnableConfig, payload["config"])
                ).checkpoint_ref
                assert checkpoint_ref is not None
                await ConversationChainStorageManager(
                    chain_metadata
                ).persist_checkpoint_ref(checkpoint_ref)
        return state, config

    root_state, root_config = await invoke("resp-root", "root")
    child_state, child_config = await invoke("resp-child", "child", "resp-root")
    grandchild_state, grandchild_config = await invoke(
        "resp-grandchild",
        "grandchild",
        "resp-child",
    )

    root_checkpoint_id = child_config["configurable"]["checkpoint_id"]
    assert child_config["configurable"]["checkpoint_id"] == root_checkpoint_id
    assert grandchild_config["configurable"]["checkpoint_id"] != root_checkpoint_id
    assert "checkpoint_id" not in root_config["configurable"]
    assert [message.content for message in root_state["messages"]] == [
        "root",
        "Echo: root",
    ]
    assert [message.content for message in child_state["messages"]] == [
        "root",
        "Echo: root",
        "child",
        "Echo: child",
    ]
    assert [message.content for message in grandchild_state["messages"]] == [
        "root",
        "Echo: root",
        "child",
        "Echo: child",
        "grandchild",
        "Echo: grandchild",
    ]
    assert chain_metadata(CONVERSATION_METADATA_NAMESPACE).flush_count > 0


async def test_conversation_management_debug_log_has_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(history=[_message_item("from history")])

    caplog.set_level(
        logging.DEBUG,
        logger="langchain_azure_ai.agents.hosting._responses_host",
    )
    await server.build_input(_request(), context)

    assert "mode=responses_history" in caplog.text
    assert "history_items=1" in caplog.text
    assert "history_messages=1" in caplog.text
    assert "current_items=1" in caplog.text
    assert "current_messages=1" in caplog.text
