# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Instruction handling at Responses request and graph-completion boundaries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from operator import add
from pathlib import Path
from typing import Annotated, Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponseContext,
    ResponseEventStream,
)
from azure.ai.agentserver.responses.hosting._orchestrator import ResponseModeFlags
from azure.ai.agentserver.responses.models import CreateResponse, ResponseObject
from azure.ai.agentserver.responses.streaming._checkpoint import ResponseCheckpointEvent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.func import task
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import Messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, PrivateAttr
from starlette.testclient import TestClient
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import (
    FoundryCheckpointSaver,
    ResponsesHostServer,
)
from langchain_azure_ai.agents.hosting._instructions import (
    _is_request_instruction,
    _ResponseInstructions,
)
from langchain_azure_ai.agents.hosting._responses import (
    CONVERSATION_CHECKPOINT_KEY,
    TaskStorageManager,
)

from .test_foundry_checkpoint_saver import _credential, _FakeStateStore


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))


class _ActualInputModel(FakeListChatModel):
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self._calls.append(deepcopy(messages))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)


class _MappingState(TypedDict):
    messages: list[AnyMessage] | dict[str, str]


class _ParallelState(MessagesState):
    completed: Annotated[list[str], add]


class _Payload(BaseModel):
    value: str


class _PayloadState(MessagesState):
    payload: _Payload


class _CommandUpdate(BaseModel):
    messages: list[AnyMessage]
    omitted: str | None = None


def _graph(
    model: _ActualInputModel,
    saver: BaseCheckpointSaver | None,
) -> CompiledStateGraph:
    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=saver)


def _create(
    client: TestClient,
    *,
    stream: bool = False,
    expected_status: str = "completed",
    **body: Any,
) -> ResponseObject:
    result = client.post("/responses", json={"model": "test", **body, "stream": stream})
    assert result.status_code == 200, result.text
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in result.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        response = events[-1]["response"]
    else:
        response = result.json()
    assert response["status"] == expected_status, response
    return cast(ResponseObject, response)


def _config(response: ResponseObject) -> RunnableConfig:
    reference = TaskStorageManager.from_stream(
        ResponseEventStream(response_id=response["id"], response=response)
    ).checkpoint_ref
    assert reference is not None
    return {"configurable": {**reference.to_dict(), "checkpoint_ns": ""}}


def _contents(messages: Sequence[BaseMessage]) -> list[Any]:
    return [message.content for message in messages]


def _stored_config(
    client: TestClient, provider: InMemoryResponseProvider, response: ResponseObject
) -> RunnableConfig:
    assert client.portal is not None
    stored = client.portal.call(provider.get_response, response["id"])
    assert stored is not None
    return _config(stored)


def _recover(
    client: TestClient,
    server: ResponsesHostServer,
    provider: InMemoryResponseProvider,
    response: ResponseObject,
    **body: Any,
) -> tuple[ResponseObject, list[Any]]:
    assert client.portal is not None
    stored = client.portal.call(provider.get_response, response["id"])
    assert stored is not None
    request = cast(CreateResponse, {"model": "test", **body, "stream": False})
    context = ResponseContext(
        response_id=stored["id"],
        mode_flags=ResponseModeFlags(stream=False, store=True, background=False),
        request=request,
        provider=provider,
    )
    context.is_recovery = True
    context.persisted_response = deepcopy(stored)

    async def recover() -> list[Any]:
        return [
            event
            async for event in server._handle_create_async_gen(
                request, context, asyncio.Event()
            )
        ]

    events = client.portal.call(recover)
    checkpoint = next(
        event.response
        for event in reversed(events)
        if isinstance(event, ResponseCheckpointEvent)
    )
    return checkpoint, events


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("conversation", [False, True])
def test_current_request_instructions_do_not_inherit_previous_instructions(
    stream: bool, conversation: bool
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    server = ResponsesHostServer(
        _graph(model, InMemorySaver()), store=InMemoryResponseProvider()
    )
    linkage: dict[str, Any] = (
        {"conversation": {"id": "conversation"}} if conversation else {}
    )
    with TestClient(server.app) as client:
        first = _create(
            client, stream=stream, input="first", instructions="turn-A", **linkage
        )
        second = _create(
            client,
            stream=stream,
            input="second",
            instructions="turn-B",
            **(linkage or {"previous_response_id": first["id"]}),
        )
        _create(
            client,
            stream=stream,
            input="third",
            **(linkage or {"previous_response_id": second["id"]}),
        )
    assert [_contents(call) for call in model._calls[:2]] == [
        ["turn-A", "first"],
        ["first", "answer", "turn-B", "second"],
    ]
    assert _contents(model._calls[2]) == [
        "first",
        "answer",
        "second",
        "answer",
        "third",
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_history_without_checkpointer_uses_only_current_instructions(
    stream: bool,
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    server = ResponsesHostServer(_graph(model, None), store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        first = _create(client, stream=stream, input="first", instructions="turn-A")
        _create(client, stream=stream, input="second", previous_response_id=first["id"])
    assert [_contents(call) for call in model._calls] == [
        ["turn-A", "first"],
        ["first", "answer", "second"],
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_intermediate_checkpoints_keep_instructions_and_final_reference_is_clean(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    saver = InMemorySaver()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(_graph(model, saver), store=provider)
    publish = AsyncMock(wraps=server._conversation_chain_store.set)
    monkeypatch.setattr(server._conversation_chain_store, "set", publish)
    with TestClient(server.app) as client:
        response = _create(
            client, stream=stream, input="first", instructions="request-only"
        )
        final_config = _stored_config(client, provider, response)
        assert client.portal is not None
        final = client.portal.call(server.graph.aget_state, final_config)
        assert final.next == ()
        assert final.tasks == ()
        assert final.interrupts == ()
        assert not any(_is_request_instruction(m) for m in final.values["messages"])
        assert publish.await_args is not None
        chain = client.portal.call(
            partial(
                server._conversation_chain_store.get,
                publish.await_args.args[0],
                CONVERSATION_CHECKPOINT_KEY,
            )
        )
        assert chain is not None
        assert chain["checkpoint_id"] == final_config["configurable"]["checkpoint_id"]
        publish.assert_awaited_once()
    checkpoints = list(saver.list(None))
    assert any("request-only" in repr(checkpoint) for checkpoint in checkpoints)
    assert saver.get_tuple(final_config) is not None


@pytest.mark.parametrize("instructions", [None, "", "turn-B"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "reply_kind", ["mcp_approval_response", "function_call_output"]
)
def test_interrupt_continuation_replaces_or_removes_previous_instructions(
    instructions: str | None, stream: bool, reply_kind: str
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)

    def pause(state: MessagesState) -> dict[str, Any]:
        for question in ("first?", "second?", "third?"):
            interrupt({"question": question})
        return {}

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_node("answer", answer)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", "answer")
    builder.add_edge("answer", END)
    saver = InMemorySaver()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(builder.compile(checkpointer=saver), store=provider)
    with TestClient(server.app) as client:
        response = _create(
            client, stream=stream, input="first", instructions="turn-A"
        )
        assert client.portal is not None
        paused = client.portal.call(
            server.graph.aget_state, _stored_config(client, provider, response)
        )
        assert paused.next
        assert _contents(paused.values["messages"]) == ["turn-A", "first"]
        for _ in range(3):
            approval = next(
                item
                for item in response["output"]
                if item["type"] == "mcp_approval_request"
            )
            function_call = next(
                item for item in response["output"] if item["type"] == "function_call"
            )
            item = (
                {
                    "type": reply_kind,
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
                if reply_kind == "mcp_approval_response"
                else {
                    "type": reply_kind,
                    "call_id": function_call["call_id"],
                    "output": "approved",
                }
            )
            response = _create(
                client,
                stream=stream,
                previous_response_id=response["id"],
                instructions=instructions,
                input=[item],
            )
        final = client.portal.call(
            server.graph.aget_state, _stored_config(client, provider, response)
        )
    assert _contents(model._calls[0]) == [
        "first",
        *([instructions] if instructions else []),
    ]
    assert not any(_is_request_instruction(m) for m in final.values["messages"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "", "turn-B"])
@pytest.mark.parametrize(
    "message_update",
    ["correction", {"role": "user", "content": "correction"}],
    ids=["string", "dict"],
)
def test_interrupt_continuation_preserves_single_message_updates(
    stream: bool,
    instructions: str | None,
    message_update: str | dict[str, str],
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)

    def pause(state: MessagesState) -> dict[str, Any]:
        interrupt({"question": "continue?"})
        return {}

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_node("answer", answer)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", "answer")
    builder.add_edge("answer", END)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()), store=provider
    )
    with TestClient(server.app) as client:
        first = _create(client, stream=stream, input="first", instructions="turn-A")
        function_call = next(
            item for item in first["output"] if item["type"] == "function_call"
        )
        second = _create(
            client,
            stream=stream,
            previous_response_id=first["id"],
            instructions=instructions,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": function_call["call_id"],
                    "output": json.dumps(
                        {"resume": True, "update": {"messages": message_update}}
                    ),
                }
            ],
        )
        assert client.portal is not None
        final = client.portal.call(
            server.graph.aget_state, _stored_config(client, provider, second)
        )
    assert _contents(model._calls[0]) == [
        "first",
        *([instructions] if instructions else []),
        "correction",
    ]
    assert _contents(final.values["messages"]) == ["first", "correction", "answer"]


def test_explicit_messages_and_developer_roles_are_preserved() -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(_graph(model, InMemorySaver()), store=provider)
    with TestClient(server.app) as client:
        first = _create(
            client,
            instructions="request-only",
            input=[
                {"role": "system", "content": "explicit-system"},
                {"role": "developer", "content": "explicit-developer"},
                {"role": "user", "content": "first"},
            ],
        )
        second = _create(client, input="second", previous_response_id=first["id"])
        assert client.portal is not None
        final = client.portal.call(
            server.graph.aget_state, _stored_config(client, provider, second)
        )
        items = client.get(
            f"/responses/{first['id']}/input_items", params={"order": "asc"}
        )
        assert items.status_code == 200
    assert [item["role"] for item in convert_to_openai_messages(model._calls[0])] == [
        "developer",
        "system",
        "developer",
        "user",
    ]
    assert _contents(model._calls[1]) == [
        "explicit-system",
        "explicit-developer",
        "first",
        "answer",
        "second",
    ]
    assert _contents(final.values["messages"]) == [
        "explicit-system",
        "explicit-developer",
        "first",
        "answer",
        "second",
        "answer",
    ]
    assert first["instructions"] == "request-only"
    assert [item["role"] for item in items.json()["data"]] == [
        "system",
        "developer",
        "user",
    ]


def test_same_response_recovery_keeps_instructions_and_completed_tasks() -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    calls: list[str] = []
    failed = False

    @task
    def effect() -> str:
        calls.append("effect")
        return "done"

    async def answer(state: MessagesState) -> dict[str, Any]:
        nonlocal failed
        assert await effect() == "done"
        if not failed:
            failed = True
            raise RuntimeError("recover this response")
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    saver = InMemorySaver()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(builder.compile(checkpointer=saver), store=provider)
    with TestClient(server.app) as client:
        first = _create(
            client, expected_status="failed", input="first", instructions="request-only"
        )
        assert client.portal is not None
        recovered, events = _recover(
            client, server, provider, first, input="first", instructions="request-only"
        )
        assert events[-1]["type"] == "response.completed"
        final = client.portal.call(server.graph.aget_state, _config(recovered))
    assert calls == ["effect"]
    assert _contents(model._calls[0]) == ["request-only", "first"]
    assert not any(_is_request_instruction(m) for m in final.values["messages"])


def test_host_does_not_wrap_graph_channels_child_graphs_or_caches() -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    child = _graph(model, None)
    builder = StateGraph(MessagesState)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    cache: InMemoryCache[Any] = InMemoryCache()
    graph = builder.compile(checkpointer=InMemorySaver(), cache=cache)
    root_messages = graph.channels["messages"]
    child_channels = child.channels
    server = ResponsesHostServer(graph)
    assert server.graph.channels["messages"] is root_messages
    assert child.channels is child_channels
    assert server.graph.cache is cache
    assert child.cache is None


@pytest.mark.parametrize("stream", [False, True])
def test_completion_cleanup_does_not_rerun_conditional_routing(stream: bool) -> None:
    routed: list[str] = []

    def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": []}

    def route(state: MessagesState) -> str:
        routed.append("route")
        return END

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_conditional_edges("answer", route)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()), store=provider
    )
    with TestClient(server.app) as client:
        response = _create(
            client, stream=stream, input="first", instructions="temporary"
        )
        assert client.portal is not None
        final = client.portal.call(
            server.graph.aget_state, _stored_config(client, provider, response)
        )
    assert routed == ["route"]
    assert final.next == ()
    assert _contents(final.values["messages"]) == ["first"]


@pytest.mark.parametrize("backend", ["memory", "sqlite", "foundry"])
async def test_branches_use_clean_final_checkpoints_without_changing_parent(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if backend == "foundry":
        monkeypatch.setattr(
            "langchain_azure_ai.agents.hosting._checkpoint."
            "_foundry_checkpoint_saver.FoundryStateStore.get_or_create",
            AsyncMock(return_value=_FakeStateStore()),
        )
    storage = (
        AsyncSqliteSaver.from_conn_string(str(tmp_path / "instructions.sqlite"))
        if backend == "sqlite"
        else nullcontext(
            FoundryCheckpointSaver(_credential())
            if backend == "foundry"
            else InMemorySaver()
        )
    )
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    async with storage as saver:
        provider = InMemoryResponseProvider()
        server = ResponsesHostServer(
            _graph(model, saver),
            store=provider,
            enable_response_branching=True,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            parent = await client.post(
                "/responses", json={"input": "first", "instructions": "turn-A"}
            )
            assert parent.status_code == 200
            stored = await provider.get_response(parent.json()["id"])
            assert stored is not None
            parent_config = _config(stored)
            parent_state = await server.graph.aget_state(parent_config)
            for text, instruction in (("second", "turn-B"), ("sibling", None)):
                response = await client.post(
                    "/responses",
                    json={
                        "input": text,
                        "instructions": instruction,
                        "previous_response_id": stored["id"],
                    },
                )
                assert response.status_code == 200
                assert response.json()["status"] == "completed", response.text
                child = await provider.get_response(response.json()["id"])
                assert child is not None
                child_state = await server.graph.aget_state(_config(child))
                assert child_state.next == child_state.tasks == ()
                assert not any(
                    _is_request_instruction(message)
                    for message in child_state.values["messages"]
                )
            restored_parent = await server.graph.aget_state(parent_config)
        assert restored_parent.values == parent_state.values
        assert not any(
            _is_request_instruction(message)
            for message in parent_state.values["messages"]
        )
        assert _contents(model._calls[1]) == [
            "first",
            "answer",
            "turn-B",
            "second",
        ]
        assert _contents(model._calls[2]) == ["first", "answer", "sibling"]
        historical = [
            item
            async for item in saver.alist(
                {
                    "configurable": {
                        "thread_id": parent_config["configurable"]["thread_id"]
                    }
                }
            )
        ]
        assert any("turn-A" in repr(item) for item in historical)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("resume", [False, True])
def test_parallel_completion_cleans_only_messages(
    stream: bool, resume: bool
) -> None:
    seen: list[list[Any]] = []

    def left(state: _ParallelState) -> dict[str, Any]:
        seen.append(_contents(state["messages"]))
        return {"messages": [AIMessage(content="left")], "completed": ["left"]}

    def right(state: _ParallelState) -> dict[str, Any]:
        if resume:
            interrupt("continue?")
        seen.append(_contents(state["messages"]))
        return {"messages": [AIMessage(content="right")], "completed": ["right"]}

    builder = StateGraph(_ParallelState)
    for name, node in (("left", left), ("right", right)):
        builder.add_node(name, node)
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    saver = InMemorySaver()
    provider = InMemoryResponseProvider()
    graph = builder.compile(checkpointer=saver)
    server = ResponsesHostServer(graph, store=provider)
    with TestClient(server.app) as client:
        response = _create(
            client, stream=stream, input="first", instructions="request-only"
        )
        if resume:
            approval = next(
                item
                for item in response["output"]
                if item["type"] == "mcp_approval_request"
            )
            response = _create(
                client,
                stream=stream,
                previous_response_id=response["id"],
                instructions="replacement",
                input=[
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval["id"],
                        "approve": True,
                    }
                ],
            )
        config = _stored_config(client, provider, response)
        assert client.portal is not None
        final = client.portal.call(server.graph.aget_state, config)
    assert seen == [
        ["request-only", "first"],
        ["first", "replacement"] if resume else ["request-only", "first"],
    ]
    assert final.next == final.tasks == ()
    assert sorted(final.values["completed"]) == ["left", "right"]
    assert sorted(_contents(final.values["messages"])) == ["first", "left", "right"]
    saved = saver.get_tuple(config)
    assert saved is not None and saved.parent_config is not None
    parent = saver.get_tuple(saved.parent_config)
    assert parent is not None
    assert saved.checkpoint["versions_seen"] == parent.checkpoint["versions_seen"]
    assert {
        key: value
        for key, value in saved.checkpoint["channel_values"].items()
        if key != "messages"
    } == {
        key: value
        for key, value in parent.checkpoint["channel_values"].items()
        if key != "messages"
    }


@pytest.mark.parametrize("stream", [False, True])
def test_cleanup_failure_is_reported_and_recovery_does_not_rerun_the_graph(
    stream: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    saver = InMemorySaver()
    write = saver.aput

    async def fail_cleanup(
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        if metadata["source"] == "update":
            raise RuntimeError("final instruction cleanup failed")
        return await write(config, checkpoint, metadata, new_versions)

    monkeypatch.setattr(saver, "aput", fail_cleanup)
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(_graph(model, saver), store=provider)
    publish = AsyncMock(wraps=server._conversation_chain_store.set)
    monkeypatch.setattr(server._conversation_chain_store, "set", publish)
    with TestClient(server.app) as client:
        response = _create(
            client,
            stream=stream,
            expected_status="failed",
            input="first",
            instructions="request-only",
        )
        publish.assert_not_awaited()
        assert "final instruction cleanup failed" in caplog.text
        assert len(model._calls) == 1
        monkeypatch.setattr(saver, "aput", write)
        recovered, events = _recover(
            client,
            server,
            provider,
            response,
            input="first",
            instructions="request-only",
        )
        assert events[-1]["type"] == "response.completed"
        assert client.portal is not None
        state = client.portal.call(server.graph.aget_state, _config(recovered))
    publish.assert_awaited_once()
    assert len(model._calls) == 1
    assert _contents(state.values["messages"]) == ["first", "answer"]


def test_cleanup_preserves_typed_state_with_strict_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("langgraph.graph.state._serde.STRICT_MSGPACK_ENABLED", True)
    seen: list[list[Any]] = []

    def answer(state: _PayloadState) -> dict[str, Any]:
        seen.append(_contents(state["messages"]))
        return {"payload": _Payload(value="keep"), "messages": []}

    builder = StateGraph(_PayloadState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=None))
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(builder.compile(checkpointer=saver), store=provider)
    with TestClient(server.app) as client:
        response = _create(client, input="first", instructions="request-only")
        assert client.portal is not None
        final = client.portal.call(
            server.graph.aget_state, _stored_config(client, provider, response)
        )
        second = _create(
            client,
            input="second",
            instructions="replacement",
            previous_response_id=response["id"],
        )
        continued = client.portal.call(
            server.graph.aget_state, _stored_config(client, provider, second)
        )
    assert final.values["payload"] == _Payload(value="keep")
    assert _contents(final.values["messages"]) == ["first"]
    assert continued.values["payload"] == _Payload(value="keep")
    assert _contents(continued.values["messages"]) == ["first", "second"]
    assert seen == [
        ["request-only", "first"],
        ["first", "replacement", "second"],
    ]


async def test_command_preparation_preserves_resume_and_other_updates() -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    graph = _graph(model, InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "command"}}
    previous = _ResponseInstructions.create("previous", "previous-response")
    assert previous.message is not None
    await graph.ainvoke(
        {"messages": previous.prepend([HumanMessage(content="old")])}, config
    )
    replacement = [HumanMessage(content="replacement")]
    update = {"messages": replacement, "other": "keep"}
    command = Command(resume={"approval": "yes"}, update=update, goto="answer")
    prepared = await _ResponseInstructions.create("current", "response").prepare(
        graph, config, command
    )
    assert isinstance(prepared, Command)
    assert prepared.resume == command.resume
    assert prepared.goto == command.goto
    assert prepared.graph == command.graph
    assert isinstance(prepared.update, list)
    assert prepared.update[1:] == list(update.items())
    channel, messages = prepared.update[0]
    assert channel == "messages"
    assert isinstance(messages[0], RemoveMessage)
    assert messages[0].id == previous.message.id
    assert _contents(messages[1:]) == ["current"]
    assert command.update is update
    assert _contents(replacement) == ["replacement"]


@pytest.mark.parametrize(
    "message_update",
    [
        "correction",
        {"role": "user", "content": "correction"},
        ("user", "correction"),
        HumanMessage(content="correction"),
    ],
    ids=["string", "dict", "tuple", "message"],
)
async def test_preparation_preserves_single_message_updates(
    message_update: Messages,
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    graph = _graph(model, InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "single-message"}}
    await graph.ainvoke({"messages": [HumanMessage(content="old")]}, config)
    update = {"messages": deepcopy(message_update)}
    graph_input: Command = Command(resume="approved", update=update)
    prepared = await _ResponseInstructions.create("current", "response").prepare(
        graph, config, graph_input
    )
    assert isinstance(prepared, Command)
    assert prepared.resume == "approved"
    assert isinstance(prepared.update, list)
    assert _contents(prepared.update[0][1]) == ["current"]
    assert prepared.update[1:] == list(update.items())
    assert graph_input.update is update


@pytest.mark.parametrize("update_shape", ["pairs", "model"])
async def test_command_preparation_preserves_repeated_writes_and_unset_fields(
    update_shape: str,
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    graph = _graph(model, InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "command-updates"}}
    await graph.ainvoke({"messages": [HumanMessage(content="old")]}, config)
    update = (
        [
            ("other", "first"),
            ("messages", [HumanMessage(content="one")]),
            ("other", "second"),
            ("messages", [HumanMessage(content="two")]),
        ]
        if update_shape == "pairs"
        else _CommandUpdate(messages=[HumanMessage(content="one")])
    )
    command: Command = Command(resume="approved", update=update)
    prepared = await _ResponseInstructions.create("current", "response").prepare(
        graph, config, command
    )
    assert isinstance(prepared, Command)
    assert isinstance(prepared.update, list)
    assert [pair for pair in prepared.update if pair[0] != "messages"] == (
        [("other", "first"), ("other", "second")] if update_shape == "pairs" else []
    )
    assert _contents(prepared.update[0][1]) == ["current"]
    assert prepared.update[1:] == list(command._update_as_tuples())


def test_requests_without_instructions_preserve_custom_message_channels() -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)

    async def answer(state: _MappingState) -> dict[str, Any]:
        messages = state["messages"]
        assert isinstance(messages, list)
        reply = await model.ainvoke(messages)
        return {"messages": {"reply": str(reply.content)}}

    builder = StateGraph(_MappingState)
    builder.add_node("answer", answer, input_schema=_MappingState)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(client, input="first")
        _create(client, input="second", previous_response_id=first["id"])
    assert [_contents(messages) for messages in model._calls] == [["first"], ["second"]]


@pytest.mark.parametrize("instructions", [False, 1, [], {}])
def test_invalid_instructions_fail_explicitly(instructions: Any) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    server = ResponsesHostServer(_graph(model, None))
    with TestClient(server.app) as client:
        result = client.post(
            "/responses", json={"input": "first", "instructions": instructions}
        )
    assert result.status_code in (400, 422)
    assert model._calls == []
