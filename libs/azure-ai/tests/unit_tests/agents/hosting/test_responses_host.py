# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``ResponsesHostServer`` via Starlette TestClient."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.core.streaming import streams
from azure.ai.agentserver.responses import CreateResponse, ResponseObject
from azure.ai.agentserver.responses.models import (
    ItemMessage,
    MessageContentInputTextContent,
)
from langchain_core.runnables import RunnableConfig
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import (
    HostingFeature,
    ResponsesHostServer,
    ResponsesServerOptions,
)
from langchain_azure_ai.agents.hosting._responses import (
    CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY,
    CONVERSATION_STATE_STORE_PREFIX,
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    ConversationChainStorageManager,
    HostingRunnableConfig,
)
from langchain_azure_ai.agents.hosting._responses_host import (
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


def test_constructor_registers_responses_features() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._responses_host."
        "_add_process_hosting_features"
    ) as add_process_features:
        server = ResponsesHostServer(make_checkpointed_echo_graph())

    assert server._hosting_features == HostingFeature.RESPONSES
    add_process_features.assert_called_once_with(HostingFeature.RESPONSES)


def test_constructor_registers_resilient_background_feature() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._responses_host."
        "_add_process_hosting_features"
    ) as add_process_features:
        server = ResponsesHostServer(
            make_checkpointed_echo_graph(),
            options=ResponsesServerOptions(resilient_background=True),
        )

    expected = HostingFeature.RESPONSES | HostingFeature.RESILIENT_BACKGROUND
    assert server._hosting_features == expected
    add_process_features.assert_called_once_with(expected)


def test_constructor_registers_steerable_conversations_feature() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._responses_host."
        "_add_process_hosting_features"
    ) as add_process_features:
        server = ResponsesHostServer(
            make_checkpointed_echo_graph(),
            options=ResponsesServerOptions(steerable_conversations=True),
        )

    expected = HostingFeature.RESPONSES | HostingFeature.STEERABLE_CONVERSATIONS
    assert server._hosting_features == expected
    add_process_features.assert_called_once_with(expected)


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
        type="message",
        role="user",
        content=[MessageContentInputTextContent({"type": "input_text", "text": text})],
    )


def _request(**kwargs: object) -> CreateResponse:
    return cast(
        CreateResponse,
        {key: value for key, value in kwargs.items() if value is not None},
    )


def _response_object(
    response_id: str,
    *,
    previous_response_id: str | None = None,
    conversation_id: str | None = None,
    internal_metadata: dict[str, object] | None = None,
) -> ResponseObject:
    response: dict[str, object] = {"id": response_id, "output": []}
    if previous_response_id is not None:
        response["previous_response_id"] = previous_response_id
    if conversation_id is not None:
        response["conversation"] = {"id": conversation_id}
    if internal_metadata is not None:
        response["metadata"] = {"_internal_metadata": internal_metadata}
    return cast(ResponseObject, response)


async def _capture_graph_call(
    server: ResponsesHostServer,
    request: CreateResponse,
    context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, RunnableConfig]:
    captured: dict[str, Any] = {}

    async def graph_stream(
        graph_input: Any,
        *,
        config: RunnableConfig,
        **_: Any,
    ) -> AsyncGenerator[Any, None]:
        captured["input"] = graph_input
        captured["config"] = config
        if False:
            yield None

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    _ = [
        event
        async for event in server.handle_create(
            request,
            context,
            asyncio.Event(),
        )
    ]
    return captured["input"], cast(RunnableConfig, captured["config"])


def _state_store_name(conversation_chain_id: str) -> str:
    return f"{CONVERSATION_STATE_STORE_PREFIX}/{conversation_chain_id}"


def _seed_conversation_checkpoint(
    foundry_state_stores: dict[str, dict[str, Any]],
    conversation_chain_id: str,
    *,
    thread_id: str,
    checkpoint_id: str,
) -> None:
    foundry_state_stores[_state_store_name(conversation_chain_id)] = {
        CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY: {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }


def _context(
    *,
    response_id: str = "resp-current",
    conversation_id: str | None = "conv-test",
    conversation_chain_id: str | None = None,
    current_text: str = "hello",
    history: list[object] | None = None,
    provider: object | None = None,
    user_id_key: str | None = None,
) -> MagicMock:
    context = MagicMock()
    context.response_id = response_id
    context.conversation_id = conversation_id
    context.conversation_chain_id = (
        conversation_chain_id or conversation_id or f"resp-{response_id}"
    )
    context._provider = provider
    context.isolation = None
    context.is_recovery = False
    context.persisted_response = None
    context.client_cancelled = False
    context._cancellation_signal = asyncio.Event()
    context.shutdown = asyncio.Event()
    context.exit_for_recovery = AsyncMock()
    context.platform_context = SimpleNamespace(user_id_key=user_id_key)
    context.get_input_items = AsyncMock(return_value=[_message_item(current_text)])
    context.get_history = AsyncMock(return_value=history or [])
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

    event_types = [event["type"] for event in events if "type" in event]
    assert event_types[-1] == "response.completed"
    assert "response.failed" not in event_types


async def test_shutdown_before_graph_execution_defers_for_recovery() -> None:
    class _ExitForRecovery(BaseException):
        pass

    server = ResponsesHostServer(make_streaming_graph())
    context = _context()
    context.shutdown.set()
    context.exit_for_recovery.side_effect = _ExitForRecovery

    events = server.handle_create(_request(), context, asyncio.Event())

    assert (await anext(events))["type"] == "response.created"
    with pytest.raises(_ExitForRecovery):
        await anext(events)
    context.exit_for_recovery.assert_awaited_once_with()


async def test_shutdown_after_stream_event_defers_for_recovery() -> None:
    class _ExitForRecovery(BaseException):
        pass

    server = ResponsesHostServer(make_streaming_graph())
    context = _context()
    context.exit_for_recovery.side_effect = _ExitForRecovery
    events = server.handle_create(_request(), context, asyncio.Event())

    while True:
        event = await anext(events)
        if (
            isinstance(event, dict)
            and event.get("type") == "response.output_text.delta"
        ):
            context.shutdown.set()
            break

    with pytest.raises(_ExitForRecovery):
        while True:
            await anext(events)
    context.exit_for_recovery.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_recovered_response_reclaims_stale_replay_stream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(resilient_background=True),
    )
    response_id = f"stale-lock-{uuid.uuid4()}"
    stream_path = tmp_path / "streams" / f"{response_id}.jsonl"
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.touch()
    lock_path = stream_path.with_suffix(".jsonl.lock")
    lock_path.touch()
    attempts = 0

    class BodyObserved(Exception):
        pass

    async def fail_then_continue(_candidate_id: str) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            cause = FileExistsError(errno.EEXIST, "File exists", str(lock_path))
            raise RuntimeError("replay stream lock contention") from cause
        if attempts == 2:
            return object()
        raise BodyObserved

    monkeypatch.setattr(streams, "get_or_create", fail_then_continue)
    context = _context(response_id=response_id)
    context.is_recovery = True
    context.platform_context = SimpleNamespace(user_id_key=None, call_id=None)
    record = SimpleNamespace(input_items=[], previous_response_id=None)
    orchestrator = server.app._orchestrator
    assert orchestrator is not None

    with pytest.raises(BodyObserved):
        await orchestrator._run_resilient_stream_body(
            parsed=_request(stream=True),
            context=context,
            cancellation_signal=asyncio.Event(),
            record=record,
            response_id=response_id,
            agent_reference={"type": "agent_reference", "name": "test"},
            model="test",
            store=True,
            agent_session_id=None,
            conversation_id="conv-test",
            background=True,
        )

    assert attempts == 3
    assert not lock_path.exists()


@pytest.mark.asyncio
async def test_fresh_response_does_not_reclaim_replay_stream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(resilient_background=True),
    )
    response_id = f"live-lock-{uuid.uuid4()}"
    stream_path = tmp_path / "streams" / f"{response_id}.jsonl"
    stream_path.touch()
    lock_path = stream_path.with_suffix(".jsonl.lock")
    lock_path.touch()
    attempts = 0

    async def locked(_candidate_id: str) -> object:
        nonlocal attempts
        attempts += 1
        cause = FileExistsError(errno.EEXIST, "File exists", str(lock_path))
        raise RuntimeError("replay stream lock contention") from cause

    monkeypatch.setattr(streams, "get_or_create", locked)
    context = _context(response_id=response_id)
    context.platform_context = SimpleNamespace(user_id_key=None, call_id=None)
    record = SimpleNamespace(input_items=[], previous_response_id=None)
    orchestrator = server.app._orchestrator
    assert orchestrator is not None

    with pytest.raises(RuntimeError, match="lock contention"):
        await orchestrator._run_resilient_stream_body(
            parsed=_request(stream=True),
            context=context,
            cancellation_signal=asyncio.Event(),
            record=record,
            response_id=response_id,
            agent_reference={"type": "agent_reference", "name": "test"},
            model="test",
            store=True,
            agent_session_id=None,
            conversation_id="conv-test",
            background=True,
        )

    assert attempts == 1
    assert lock_path.exists()


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

    assert (await anext(events))["type"] == "response.created"
    assert (await anext(events))["type"] == "response.in_progress"
    admission_checkpoint = await anext(events)

    assert type(admission_checkpoint).__name__ == "ResponseCheckpointEvent"
    assert "_internal_metadata" not in admission_checkpoint.response["metadata"]
    await cast(AsyncGenerator[object, None], events).aclose()


async def test_handle_create_persists_conversation_checkpoint(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())

    _ = [
        event
        async for event in server.handle_create(
            _request(),
            _context(conversation_id=None, conversation_chain_id="chain-1"),
            asyncio.Event(),
        )
    ]

    checkpoint = foundry_state_stores[_state_store_name("chain-1")][
        CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY
    ]
    assert checkpoint["thread_id"] == "resp-current"
    assert isinstance(
        checkpoint["checkpoint_id"],
        str,
    )


async def test_recovery_replays_input_without_current_response_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "chain-1",
        thread_id="resp-root",
        checkpoint_id="checkpoint-parent",
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        conversation_chain_id="chain-1",
        current_text="replay me",
    )
    context.is_recovery = True
    context.persisted_response = _response_object("resp-current")
    request = _request(previous_response_id="resp-parent")
    graph_input, config = await _capture_graph_call(
        server,
        request,
        context,
        monkeypatch,
    )

    assert graph_input is not None
    assert config["configurable"]["checkpoint_id"] == "checkpoint-parent"
    assert [message.content for message in graph_input["messages"]] == ["replay me"]


async def test_recovery_resumes_from_current_response_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, current_text="do not replay")
    context.is_recovery = True
    context.persisted_response = _response_object(
        "resp-current",
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-current",
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        },
    )
    request = _request()
    graph_input, config = await _capture_graph_call(
        server,
        request,
        context,
        monkeypatch,
    )

    assert graph_input is None
    assert config["configurable"]["thread_id"] == "resp-root"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-current"


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

    with patch.object(
        ConversationChainStorageManager,
        "get_checkpoint_ref",
        new_callable=AsyncMock,
    ) as get_checkpoint_ref:
        config = await server.build_runnable_config(_request(), context)

    get_checkpoint_ref.assert_not_awaited()
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


async def test_checkpointed_conversation_is_isolated_by_user() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())

    with _client(server) as client:
        first = client.post(
            "/responses",
            headers={"x-agent-user-id": "user-a"},
            json={"input": "secret-a", "conversation": {"id": "shared"}},
        )
        second = client.post(
            "/responses",
            headers={"x-agent-user-id": "user-b"},
            json={"input": "hello-b", "conversation": {"id": "shared"}},
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    first_config = await server.build_runnable_config(
        _request(),
        _context(conversation_id="shared", user_id_key="user-a"),
    )
    second_config = await server.build_runnable_config(
        _request(),
        _context(conversation_id="shared", user_id_key="user-b"),
    )
    assert (
        first_config["configurable"]["thread_id"]
        != second_config["configurable"]["thread_id"]
    )
    repeated_first_config = await server.build_runnable_config(
        _request(),
        _context(conversation_id="shared", user_id_key="user-a"),
    )
    assert (
        first_config["configurable"]["thread_id"]
        == repeated_first_config["configurable"]["thread_id"]
    )

    first_state = await server.graph.aget_state(first_config)
    second_state = await server.graph.aget_state(second_config)
    assert [message.content for message in first_state.values["messages"]] == [
        "secret-a",
        "Echo: secret-a",
    ]
    assert [message.content for message in second_state.values["messages"]] == [
        "hello-b",
        "Echo: hello-b",
    ]


async def test_checkpoint_reference_is_isolated_by_user(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    first_context = _context(conversation_id="shared", user_id_key="user-a")
    second_context = _context(conversation_id="shared", user_id_key="user-b")

    first_initial = await server.build_runnable_config(_request(), first_context)
    second_initial = await server.build_runnable_config(_request(), second_context)
    first_thread_id = first_initial["configurable"]["thread_id"]
    second_thread_id = second_initial["configurable"]["thread_id"]
    _seed_conversation_checkpoint(
        foundry_state_stores,
        first_thread_id,
        thread_id=first_thread_id,
        checkpoint_id="checkpoint-user-a",
    )
    _seed_conversation_checkpoint(
        foundry_state_stores,
        second_thread_id,
        thread_id=second_thread_id,
        checkpoint_id="checkpoint-user-b",
    )

    first_config = await server.build_runnable_config(_request(), first_context)
    second_config = await server.build_runnable_config(_request(), second_context)

    assert first_config["configurable"]["checkpoint_id"] == "checkpoint-user-a"
    assert second_config["configurable"]["checkpoint_id"] == "checkpoint-user-b"


async def test_previous_response_id_chain_resolves_root_thread_id() -> None:
    class _Provider:
        async def get_response(
            self,
            response_id: str,
            *,
            context: object = None,
        ) -> dict[str, str | None]:
            del context
            responses: dict[str, dict[str, str | None]] = {
                "resp-2": {"previous_response_id": "resp-1"},
                "resp-1": {"previous_response_id": None},
            }
            return responses[response_id]

    server = ResponsesHostServer(make_checkpointed_echo_graph())
    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        _context(conversation_id=None, provider=_Provider()),
    )

    assert config["configurable"]["thread_id"] == "resp-1"


async def test_previous_response_uses_conversation_state_store(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "chain-1",
        thread_id="resp-1",
        checkpoint_id="checkpoint-2",
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        conversation_chain_id="chain-1",
    )

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-1"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"
    assert config["configurable"]["checkpoint_ns"] == ""


async def test_explicit_conversation_uses_conversation_state_store(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "conv-api",
        thread_id="conv-api",
        checkpoint_id="checkpoint-2",
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id="conv-api")

    config = await server.build_runnable_config(
        _request(conversation="conv-api"),
        context,
    )

    assert config["configurable"]["thread_id"] == "conv-api"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"


async def test_recovery_uses_persisted_thread_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        response_id="resp-current",
        conversation_id=None,
    )
    context.is_recovery = True
    context.persisted_response = _response_object(
        "resp-current",
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-committed",
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        },
    )

    request = _request(previous_response_id="resp-parent")
    _, config = await _capture_graph_call(
        server,
        request,
        context,
        monkeypatch,
    )

    assert config["configurable"]["thread_id"] == "resp-root"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-committed"
    assert config["configurable"]["checkpoint_ns"] == ""


async def test_recovery_ignores_checkpoint_newer_than_response_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            checkpoint = cast(dict[str, Any], payload)
            committed_checkpoint_id = checkpoint["config"]["configurable"][
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
    recovery_context.persisted_response = _response_object(
        "resp-root",
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: committed_checkpoint_id,
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        },
    )
    with monkeypatch.context() as recovery_patch:
        _, recovery_config = await _capture_graph_call(
            server,
            request,
            recovery_context,
            recovery_patch,
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


async def test_conversation_state_store_supports_linear_response_chain(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    conversation_chain_id = "chain-1"

    async def invoke(
        response_id: str,
        text: str,
        previous_response_id: str | None = None,
    ) -> tuple[dict[str, Any], RunnableConfig]:
        request = _request(previous_response_id=previous_response_id)
        context = _context(
            response_id=response_id,
            conversation_id=None,
            conversation_chain_id=conversation_chain_id,
            current_text=text,
        )
        config = await server.build_runnable_config(request, context)
        graph_input = await server.build_input(request, context)
        state: dict[str, Any] = {}
        graph_stream = server.graph.astream(
            graph_input,
            config=config,
            stream_mode=["values", "checkpoints"],
            durability="sync",
        )
        async for mode, payload in graph_stream:
            if mode == "values":
                state = cast(dict[str, Any], payload)
            elif mode == "checkpoints":
                checkpoint = cast(dict[str, Any], payload)
                checkpoint_ref = HostingRunnableConfig(
                    cast(RunnableConfig, checkpoint["config"])
                ).checkpoint_ref
                assert checkpoint_ref is not None
                await ConversationChainStorageManager(
                    conversation_chain_id
                ).persist_checkpoint_ref(checkpoint_ref)
        return state, config

    root_state, root_config = await invoke("resp-root", "root")
    child_state, child_config = await invoke("resp-child", "child", "resp-root")
    grandchild_state, grandchild_config = await invoke(
        "resp-grandchild",
        "grandchild",
        "resp-child",
    )

    root_configurable = root_config.get("configurable")
    child_configurable = child_config.get("configurable")
    grandchild_configurable = grandchild_config.get("configurable")
    assert root_configurable is not None
    assert child_configurable is not None
    assert grandchild_configurable is not None
    root_checkpoint_id = child_configurable["checkpoint_id"]
    assert child_configurable["checkpoint_id"] == root_checkpoint_id
    assert grandchild_configurable["checkpoint_id"] != root_checkpoint_id
    assert "checkpoint_id" not in root_configurable
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
    assert (
        CONVERSATION_STATE_CHECKPOINT_REFERENCE_KEY
        in foundry_state_stores[_state_store_name(conversation_chain_id)]
    )


async def test_previous_response_id_thread_is_scoped_by_user() -> None:
    class _Provider:
        async def get_response(
            self,
            response_id: str,
            *,
            context: object = None,
        ) -> dict[str, str | None]:
            del context
            responses: dict[str, dict[str, str | None]] = {
                "resp-2": {"previous_response_id": "resp-1"},
                "resp-1": {"previous_response_id": None},
            }
            return responses[response_id]

    server = ResponsesHostServer(make_checkpointed_echo_graph())
    request = _request(previous_response_id="resp-2")
    first_config = await server.build_runnable_config(
        request,
        _context(conversation_id=None, provider=_Provider(), user_id_key="user-a"),
    )
    second_config = await server.build_runnable_config(
        request,
        _context(conversation_id=None, provider=_Provider(), user_id_key="user-b"),
    )

    assert (
        first_config["configurable"]["thread_id"]
        != second_config["configurable"]["thread_id"]
    )


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
