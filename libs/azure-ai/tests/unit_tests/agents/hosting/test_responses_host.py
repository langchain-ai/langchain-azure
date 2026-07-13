# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``ResponsesHostServer`` via Starlette TestClient."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses.models import (  # noqa: E402
    ItemMessage,
    MessageContentInputTextContent,
)
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import (  # noqa: E402
    ResponsesHostServer,
    ResponsesServerOptions,
)
from langchain_azure_ai.agents.hosting.utils import (  # noqa: E402
    METADATA_LANGGRAPH_CHECKPOINT_ID,
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


def _context(
    *,
    response_id: str = "resp-current",
    conversation_id: str | None = "conv-test",
    current_text: str = "hello",
    history: list[object] | None = None,
    provider: object | None = None,
) -> MagicMock:
    context = MagicMock()
    context.response_id = response_id
    context.conversation_id = conversation_id
    context.conversation_chain_id = conversation_id or f"resp-{response_id}"
    context.is_recovery = False
    context.persisted_response = None
    context.platform_context = object()
    context.get_input_items = AsyncMock(return_value=[_message_item(current_text)])
    context.get_history = AsyncMock(return_value=history or [])
    context._provider = provider
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
    assert config["configurable"]["responses_context"] is context


async def test_explicit_conversation_id_is_thread_id() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(response_id="resp-1", conversation_id="conv-context")

    config = await server.build_runnable_config(
        _request(conversation="conv-request"),
        context,
    )

    assert config["configurable"]["thread_id"] == "conv-request"
    assert config["configurable"]["responses_context"] is context


async def test_previous_response_chain_resolves_root_response_id() -> None:
    provider = MagicMock()
    provider.get_response = AsyncMock(
        side_effect=[
            {"id": "resp-2", "previous_response_id": "resp-1"},
            {"id": "resp-1", "previous_response_id": None},
        ]
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-1"
    assert [call.args[0] for call in provider.get_response.await_args_list] == [
        "resp-2",
        "resp-1",
    ]
    assert all(
        call.kwargs == {"context": context.platform_context}
        for call in provider.get_response.await_args_list
    )


async def test_previous_response_uses_parent_checkpoint_metadata() -> None:
    provider = MagicMock()
    provider.get_response = AsyncMock(
        side_effect=[
            {
                "id": "resp-2",
                "previous_response_id": "resp-1",
                "internal_metadata": {
                    METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-2"
                },
            },
            {"id": "resp-1", "previous_response_id": None},
        ]
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-1"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"
    assert config["configurable"]["checkpoint_ns"] == ""


async def test_response_chain_prefers_root_conversation_id() -> None:
    provider = MagicMock()
    provider.get_response = AsyncMock(
        return_value={
            "id": "resp-2",
            "previous_response_id": "resp-1",
            "conversation": {"id": "conv-api"},
        }
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "conv-api"
    provider.get_response.assert_awaited_once()


async def test_response_chain_lookup_failure_uses_immediate_parent() -> None:
    provider = MagicMock()
    provider.get_response = AsyncMock(side_effect=KeyError("missing"))
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-2"


async def test_recovery_resolves_thread_and_uses_persisted_checkpoint() -> None:
    provider = MagicMock()
    provider.get_response = AsyncMock(
        return_value={"id": "resp-parent", "previous_response_id": None}
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        response_id="resp-current",
        conversation_id=None,
        provider=provider,
    )
    context.is_recovery = True
    context.persisted_response = SimpleNamespace(
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-committed",
        }
    )

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-parent"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-committed"
    assert config["configurable"]["checkpoint_ns"] == ""
    provider.get_response.assert_awaited_once()


def test_sync_config_uses_immediate_parent_response_id() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(response_id="resp-current", conversation_id=None)

    config = server.build_runnable_config_sync(
        _request(previous_response_id="resp-parent"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-parent"
    assert config["configurable"]["responses_context"] is context


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
            committed_checkpoint_id = payload["config"]["configurable"][
                "checkpoint_id"
            ]

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


async def test_response_checkpoint_metadata_supports_sibling_fork() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    responses: dict[str, dict[str, object]] = {}
    provider = MagicMock()

    async def get_response(
        response_id: str,
        **kwargs: object,
    ) -> dict[str, object]:
        del kwargs
        return responses[response_id]

    provider.get_response = AsyncMock(side_effect=get_response)

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
            provider=provider,
        )
        config = await server.build_runnable_config(request, context)
        graph_input = await server.build_input(request, context)
        state: dict[str, object] = {}
        stream = SimpleNamespace(internal_metadata={})
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
                    checkpoint_id = payload["config"]["configurable"][
                        "checkpoint_id"
                    ]
                    stream.internal_metadata[
                        METADATA_LANGGRAPH_CHECKPOINT_ID
                    ] = checkpoint_id
        responses[response_id] = {
            "id": response_id,
            "previous_response_id": previous_response_id,
            "internal_metadata": dict(stream.internal_metadata),
        }
        return state, config

    root_state, root_config = await invoke("resp-root", "root")
    child_state, child_config = await invoke("resp-child", "child", "resp-root")
    fork_state, fork_config = await invoke("resp-fork", "fork", "resp-root")

    root_checkpoint_id = responses["resp-root"]["internal_metadata"][  # type: ignore[index]
        METADATA_LANGGRAPH_CHECKPOINT_ID
    ]
    assert child_config["configurable"]["checkpoint_id"] == root_checkpoint_id
    assert fork_config["configurable"]["checkpoint_id"] == root_checkpoint_id
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
    assert [message.content for message in fork_state["messages"]] == [
        "root",
        "Echo: root",
        "fork",
        "Echo: fork",
    ]


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
