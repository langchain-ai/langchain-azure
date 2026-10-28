# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public Responses branching contracts with real graphs and checkpoint history."""

from __future__ import annotations

import asyncio
import hashlib
import json
import operator
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses import (  # noqa: E402
    FileResponseStore,
    InMemoryResponseProvider,
    PlatformContext,
    ResponseContext,
    ResponseEventStream,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.models import CreateResponse, ResponseObject  # noqa: E402
from azure.ai.agentserver.responses.models.runtime import ResponseModeFlags  # noqa: E402
from azure.ai.agentserver.responses.streaming._checkpoint import (  # noqa: E402
    ResponseCheckpointEvent,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig  # noqa: E402
from langgraph.checkpoint.base import BaseCheckpointSaver  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.graph.state import CompiledStateGraph  # noqa: E402
from langgraph.types import interrupt  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from langchain_azure_ai.agents.hosting import ResponsesHostServer  # noqa: E402
from langchain_azure_ai.agents.hosting._responses import (  # noqa: E402
    TaskStorageManager,
)
from langchain_azure_ai.agents.hosting._responses.branching import (  # noqa: E402
    BRANCHING_MODE,
    METADATA_RESPONSE_BRANCHING,
    response_branching_enabled,
)
from langchain_azure_ai.agents.hosting._responses.task_storage_manager import (  # noqa: E402
    METADATA_RESPONSE_EXECUTION,
)


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))


def _graph(
    calls: list[list[str]], *, saver: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    async def answer(state: MessagesState) -> dict[str, Any]:
        texts = [
            str(message.content)
            for message in state["messages"]
            if isinstance(message, (HumanMessage, SystemMessage))
        ]
        calls.append(texts)
        return {"messages": [AIMessage(content="|".join(texts))]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=saver if saver is not None else InMemorySaver())


def _text(response: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in response["output"]
        if item["type"] == "message"
        for part in item["content"]
    )


def _config(response: dict[str, Any]) -> RunnableConfig:
    reference = TaskStorageManager.from_stream(
        ResponseEventStream(response_id=response["id"], response=cast(Any, response))
    ).checkpoint_ref
    assert reference is not None
    return {
        "configurable": {
            "thread_id": reference.thread_id,
            "checkpoint_id": reference.checkpoint_id,
            "checkpoint_ns": "",
        }
    }


def _create(
    client: TestClient, *, headers: dict[str, str] | None = None, **body: Any
) -> dict[str, Any]:
    result = client.post("/responses", json=body, headers=headers)
    assert result.status_code == 200, result.text
    response = result.json()
    assert response["status"] == "completed", response
    return cast(dict[str, Any], response)


def test_siblings_restore_the_exact_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    graph = _graph(calls)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        get_response = provider.get_response
        stored_parent = client.portal.call(get_response, parent["id"])
        before = graph.get_state(_config(cast(Any, stored_parent)))
        lookup = AsyncMock(wraps=get_response)
        monkeypatch.setattr(provider, "get_response", lookup)
        first = _create(client, input="B", previous_response_id=parent["id"])
        assert lookup.await_count == 1
        stored_first = client.portal.call(get_response, first["id"])
        lookup.reset_mock()
        second = _create(client, input="C", previous_response_id=parent["id"])
        assert lookup.await_count == 1
        stored_second = client.portal.call(get_response, second["id"])
        continued = _create(client, input="D", previous_response_id=first["id"])

    assert [_text(item) for item in (parent, first, second, continued)] == [
        "A",
        "A|B",
        "A|C",
        "A|B|D",
    ]
    assert calls == [["A"], ["A", "B"], ["A", "C"], ["A", "B", "D"]]
    assert graph.get_state(_config(cast(Any, stored_parent))).values == before.values
    configs = [
        _config(cast(Any, item))["configurable"]
        for item in (stored_parent, stored_first, stored_second)
    ]
    assert len({config["thread_id"] for config in configs}) == 1
    assert len({config["checkpoint_id"] for config in configs}) == 3
    assert graph.checkpointer.__class__ is InMemorySaver


def test_default_file_provider_and_unretained_child() -> None:
    calls: list[list[str]] = []
    server = ResponsesHostServer(_graph(calls), enable_response_branching=True)
    assert isinstance(server.app._orchestrator._provider, FileResponseStore)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        child = _create(
            client, input="B", previous_response_id=parent["id"], store=False
        )
        assert client.get(f"/responses/{child['id']}").status_code == 404
        assert client.get(f"/responses/{parent['id']}").status_code == 200
        rejected = client.post(
            "/responses", json={"input": "C", "previous_response_id": child["id"]}
        )
        assert rejected.status_code == 400
        assert rejected.json()["error"]["code"] == "previous_response_not_found"
        assert rejected.json()["error"]["param"] == "previous_response_id"
    assert calls == [["A"], ["A", "B"]]


@pytest.mark.parametrize("instructions", [None, "child instructions"])
def test_request_instructions_do_not_leak_into_children(
    instructions: str | None,
) -> None:
    calls: list[list[str]] = []
    server = ResponsesHostServer(
        _graph(calls), store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _create(
            client,
            instructions="parent instructions",
            input=[
                {"role": "system", "content": "history system"},
                {"role": "developer", "content": "history developer"},
                {"role": "user", "content": "A"},
            ],
        )
        child = _create(
            client,
            input="B",
            previous_response_id=parent["id"],
            instructions=instructions,
        )
    expected = ["history system", "history developer", "A"]
    if instructions is not None:
        expected.append(instructions)
    expected.append("B")
    assert calls[-1] == expected
    assert "parent instructions" not in _text(child)
    assert child.get("instructions") == instructions


@pytest.mark.parametrize(
    "body",
    [
        {"response_id": None},
        {"previous_response_id": ""},
        {"previous_response_id": False},
        {"previous_response_id": "malformed"},
        {"previous_response_id": "parent", "conversation": "conversation"},
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_invalid_linkage_fails_before_execution(
    body: dict[str, Any], stream: bool
) -> None:
    calls: list[list[str]] = []
    server = ResponsesHostServer(_graph(calls), enable_response_branching=True)
    with TestClient(server.app) as client:
        response = client.post(
            "/responses", json={"input": "invalid", "stream": stream, **body}
        )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert calls == []


@pytest.mark.parametrize(
    "extra",
    [
        {"app": MagicMock()},
        {"options": ResponsesServerOptions(steerable_conversations=True)},
        {"store": True},
        {"store": False},
    ],
)
def test_unsupported_host_combinations_fail_at_startup(extra: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ResponsesHostServer(_graph([]), enable_response_branching=True, **extra)


def test_branching_requires_async_checkpoint_support() -> None:
    with pytest.raises(ValueError, match="checkpointer"):
        ResponsesHostServer(
            _graph([]).copy({"checkpointer": None}), enable_response_branching=True
        )
    with pytest.raises(ValueError, match="asynchronous"):
        ResponsesHostServer(
            _graph([], saver=BaseCheckpointSaver()), enable_response_branching=True
        )


@pytest.mark.parametrize("status", ["in_progress", "failed", "cancelled"])
@pytest.mark.parametrize("stream", [False, True])
def test_unfinished_response_is_not_a_parent(status: str, stream: bool) -> None:
    calls: list[list[str]] = []
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        _graph(calls), store=provider, enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, parent["id"])
        stored["status"] = cast(Any, status)
        client.portal.call(provider.update_response, stored)
        response = client.post(
            "/responses",
            json={"input": "B", "previous_response_id": parent["id"], "stream": stream},
        )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "previous_response_id"
    assert calls == [["A"]]


@pytest.mark.parametrize("static_pause", [False, True])
def test_paused_graph_is_not_a_branch_parent(static_pause: bool) -> None:
    runs = []

    async def pause(state: MessagesState) -> dict[str, Any]:
        runs.append(True)
        answer = interrupt("Your answer?")
        return {"messages": [AIMessage(content=str(answer))]}

    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    graph = builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_before=["pause"] if static_pause else None,
    )
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        original_runs = list(runs)
        rejected = client.post(
            "/responses", json={"input": "B", "previous_response_id": parent["id"]}
        )
        assert rejected.status_code == 400
        assert runs == original_runs
        if not static_pause:
            call_id = next(
                item["call_id"]
                for item in parent["output"]
                if item["type"] == "function_call"
            )
            resumed = _create(
                client,
                previous_response_id=parent["id"],
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"resume": "answered"}),
                    }
                ],
            )
            assert _text(resumed) == "answered"


def test_matching_external_tool_result_is_not_dropped() -> None:
    from langchain_core.messages import ToolMessage

    async def answer(state: MessagesState) -> dict[str, Any]:
        last = state["messages"][-1]
        if isinstance(last, ToolMessage):
            return {"messages": [AIMessage(content=f"result:{last.content}")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_external", "name": "lookup", "args": {}}],
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="lookup")
        invalid = client.post(
            "/responses",
            json={
                "previous_response_id": parent["id"],
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "unrelated",
                        "output": "wrong",
                    }
                ],
            },
        )
        assert invalid.status_code == 400
        child = _create(
            client,
            previous_response_id=parent["id"],
            input=[
                {
                    "type": "function_call_output",
                    "call_id": "call_external",
                    "output": "found",
                }
            ],
        )
    assert _text(child) == "result:found"


@pytest.mark.parametrize("damage", ["reference", "checkpoint"])
def test_missing_parent_state_does_not_fall_back(damage: str) -> None:
    calls: list[list[str]] = []
    saver = InMemorySaver()
    graph = _graph(calls, saver=saver)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, parent["id"])
        if damage == "reference":
            seeded = ResponseEventStream(response_id=parent["id"], response=stored)
            seeded.internal_metadata["langgraph_checkpoint_id"] = None
            client.portal.call(provider.update_response, cast(Any, seeded.response))
        else:
            config = _config(cast(Any, stored))["configurable"]
            saver.storage[config["thread_id"]][""].pop(config["checkpoint_id"])
        result = client.post(
            "/responses", json={"input": "B", "previous_response_id": parent["id"]}
        )
    assert result.status_code == 500
    assert result.json()["error"]["code"] == "server_error"
    assert calls == [["A"]]


def test_exact_parent_from_an_ordinary_conversation_survives_newer_turns() -> None:
    calls: list[list[str]] = []
    graph = _graph(calls)
    provider = InMemoryResponseProvider()
    ordinary = ResponsesHostServer(graph, store=provider)
    with TestClient(ordinary.app) as client:
        parent = _create(
            client,
            input="A",
            instructions="old request instructions",
            conversation="existing-conversation",
        )
        _create(client, input="newer", conversation="existing-conversation")
    branching = ResponsesHostServer(
        graph, store=provider, enable_response_branching=True
    )
    with TestClient(branching.app) as client:
        child = _create(client, input="B", previous_response_id=parent["id"])
    assert _text(child) == "A|B"
    assert calls[-1] == ["A", "B"]


def test_hooks_and_bound_configuration_cannot_redirect_a_branch() -> None:
    class ConfiguredHost(ResponsesHostServer):
        async def build_runnable_config(
            self, request: CreateResponse, context: ResponseContext
        ) -> RunnableConfig:
            config = await super().build_runnable_config(request, context)
            config["configurable"].update(
                thread_id="hook-thread",
                checkpoint_ns="hook:private",
                checkpoint_id="hook-checkpoint",
            )
            return config

    graph = _graph([]).with_config(
        {
            "configurable": {
                "thread_id": "bound-thread",
                "checkpoint_ns": "bound:private",
                "checkpoint_id": "bound-checkpoint",
            }
        }
    )
    provider = InMemoryResponseProvider()
    server = ConfiguredHost(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        child = _create(client, input="B", previous_response_id=parent["id"])
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, parent["id"])
    assert _text(child) == "A|B"
    assert _config(cast(Any, stored))["configurable"]["thread_id"] == parent["id"]
    assert graph.config["configurable"]["thread_id"] == "bound-thread"


def test_provider_scope_and_scoped_thread_identity() -> None:
    from azure.ai.agentserver.core import get_request_context

    class ScopedSaver(InMemorySaver):
        async def aget_tuple(self, config: RunnableConfig) -> Any:
            platform = get_request_context()
            assert platform.user_id == "alice"
            assert platform.call_id == "opaque-call"
            return await super().aget_tuple(config)

    calls: list[list[str]] = []
    provider = FileResponseStore()
    server = ResponsesHostServer(
        _graph(calls, saver=ScopedSaver()),
        store=provider,
        enable_response_branching=True,
    )
    headers = {"x-agent-user-id": "alice", "x-agent-foundry-call-id": "opaque-call"}
    with TestClient(server.app) as client:
        parent = _create(client, input="A", headers=headers)
        denied = client.post(
            "/responses",
            headers={"x-agent-user-id": "bob"},
            json={"input": "denied", "previous_response_id": parent["id"]},
        )
        assert denied.status_code == 400
        assert denied.json()["error"]["code"] == "previous_response_not_found"
        child = _create(
            client, input="B", previous_response_id=parent["id"], headers=headers
        )
        assert client.portal is not None
        stored = client.portal.call(
            partial(
                provider.get_response,
                parent["id"],
                context=PlatformContext(user_id_key="alice", call_id="opaque-call"),
            )
        )
    thread = _config(cast(Any, stored))["configurable"]["thread_id"]
    assert thread == f"user-{hashlib.sha256(b'alice').hexdigest()}:{parent['id']}"
    assert _text(child) == "A|B"
    assert calls == [["A"], ["A", "B"]]


@pytest.mark.parametrize("enabled", [False, True])
def test_client_cannot_forge_the_admitted_mode(enabled: bool) -> None:
    seen = []

    async def answer(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        context = config["configurable"]["response_context"]
        seen.append(context.client_headers["x-client-langgraph-response-branching"])
        return {"messages": [AIMessage(content="done")]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        _create(
            client,
            input="A",
            headers={
                "x-client-langgraph-response-branching": "off"
                if enabled
                else BRANCHING_MODE
            },
        )
    assert seen == [BRANCHING_MODE if enabled else "off"]


@pytest.mark.parametrize("user_id", ["bob", None])
def test_parent_scope_is_checked_with_an_unpartitioned_provider(
    user_id: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        _graph(calls), store=provider, enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A", headers={"x-agent-user-id": "alice"})
        assert client.portal is not None
        saved = client.portal.call(
            partial(
                provider.get_response,
                parent["id"],
                context=PlatformContext(user_id_key="alice"),
            )
        )
        monkeypatch.setattr(provider, "get_response", AsyncMock(return_value=saved))
        denied = client.post(
            "/responses",
            json={"input": "B", "previous_response_id": parent["id"]},
            headers={"x-agent-user-id": user_id} if user_id else {},
        )
    assert denied.status_code == 400
    assert denied.json()["error"]["code"] == "previous_response_not_found"
    assert calls == [["A"]]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_concurrent_siblings_keep_full_graph_state(
    backend: str, tmp_path: Path
) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    class State(MessagesState):
        turns: Annotated[list[str], operator.add]

    barrier = asyncio.Barrier(2)

    async def run(saver: BaseCheckpointSaver) -> None:
        async def answer(state: State) -> dict[str, Any]:
            texts = [
                str(message.content)
                for message in state["messages"]
                if isinstance(message, HumanMessage)
            ]
            if texts[-1] != "A":
                await barrier.wait()
            return {
                "messages": [AIMessage(content="|".join(texts))],
                "turns": [texts[-1]],
            }

        builder = StateGraph(State)
        builder.add_node("answer", answer)
        builder.add_edge(START, "answer")
        builder.add_edge("answer", END)
        graph = builder.compile(checkpointer=saver)
        provider = InMemoryResponseProvider()
        chain = SimpleNamespace(get=AsyncMock(), set=AsyncMock())
        server = ResponsesHostServer(
            graph,
            store=provider,
            conversation_chain_store=chain,
            enable_response_branching=True,
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        ) as client:
            parent = (await client.post("/responses", json={"input": "A"})).json()
            parent_snapshot = await provider.get_response(parent["id"])
            before = await graph.aget_state(_config(cast(Any, parent_snapshot)))
            results = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        client.post(
                            "/responses",
                            json={"input": text, "previous_response_id": parent["id"]},
                        )
                        for text in ("B", "C")
                    )
                ),
                timeout=10,
            )
        assert [_text(result.json()) for result in results] == ["A|B", "A|C"]
        assert before.values["turns"] == ["A"]
        assert (
            await graph.aget_state(_config(cast(Any, parent_snapshot)))
        ).values == before.values
        for result, text in zip(results, ("B", "C")):
            stored = await provider.get_response(result.json()["id"])
            state = await graph.aget_state(_config(cast(Any, stored)))
            assert state.values["turns"] == ["A", text]
        chain.get.assert_not_awaited()
        chain.set.assert_not_awaited()

    if backend == "sqlite":
        async with AsyncSqliteSaver.from_conn_string(
            str(tmp_path / "checkpoints.sqlite")
        ) as saver:
            await run(saver)
    else:
        await run(InMemorySaver())


def _context(
    request: CreateResponse,
    provider: InMemoryResponseProvider,
    response_id: str,
    *,
    snapshot: ResponseObject | None = None,
) -> ResponseContext:
    context = ResponseContext(
        response_id=response_id,
        request=request,
        previous_response_id=request.get("previous_response_id"),
        mode_flags=ResponseModeFlags(stream=False, store=True, background=True),
        provider=provider,
        client_headers={"x-client-langgraph-response-branching": BRANCHING_MODE},
    )
    context.is_recovery = snapshot is not None
    context.persisted_response = snapshot
    return context


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("stage", ["admission", "progress", "finished"])
async def test_recovery_keeps_original_mode_and_response_owned_progress(
    child: bool, stage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs: list[str] = []

    async def plan(state: MessagesState) -> dict[str, Any]:
        texts = [
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]
        runs.append("|".join(texts))
        return {"messages": [AIMessage(content=f"plan:{'|'.join(texts)}")]}

    async def finish(state: MessagesState) -> dict[str, Any]:
        runs.append("finish")
        return {"messages": [AIMessage(content="done")]}

    builder = StateGraph(MessagesState)
    builder.add_node("plan", plan)
    builder.add_node("finish", finish)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "finish")
    builder.add_edge("finish", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    request = cast(CreateResponse, {"input": "B" if child else "A"})
    if child:
        parent_request = cast(CreateResponse, {"input": "A"})
        parent_events = [
            event
            async for event in server._handle_create_async_gen(
                parent_request,
                _context(parent_request, provider, "resp-parent"),
                asyncio.Event(),
            )
        ]
        await provider.create_response(parent_events[-1]["response"], None, None)
        request["previous_response_id"] = "resp-parent"
    events = server._handle_create_async_gen(
        request, _context(request, provider, "resp-current"), asyncio.Event()
    )
    snapshots = []
    async for event in events:
        if isinstance(event, ResponseCheckpointEvent):
            snapshots.append(deepcopy(event.response))
            if stage == "admission":
                break
    await events.aclose()
    if stage == "admission":
        saved = snapshots[0]
        assert (
            TaskStorageManager.from_stream(
                ResponseEventStream(response_id="resp-current", response=saved)
            ).checkpoint_ref
            is None
        )
    else:
        saved = (
            next(
                snapshot
                for snapshot in snapshots
                if "plan:" in _text(cast(Any, snapshot))
                and "done" not in _text(cast(Any, snapshot))
            )
            if stage == "progress"
            else snapshots[-1]
        )
        # A newer checkpoint in the same thread must not redirect native resume.
        await graph.ainvoke(
            {"messages": [HumanMessage(content="newer")]},
            _config(cast(Any, saved)),
        )
    before = list(runs)
    lookup = AsyncMock(wraps=provider.get_response)
    monkeypatch.setattr(provider, "get_response", lookup)
    recovered = ResponsesHostServer(
        graph, store=provider, enable_response_branching=False
    )
    result = [
        event
        async for event in recovered._handle_create_async_gen(
            request,
            _context(request, provider, "resp-current", snapshot=saved),
            asyncio.Event(),
        )
    ][-1]
    assert result["type"] == "response.completed"
    assert _text(result["response"]) == f"plan:{'A|B' if child else 'A'}done"
    assert result["response"].get("previous_response_id") == request.get(
        "previous_response_id"
    )
    if stage == "admission":
        assert runs == before + ["A|B" if child else "A", "finish"]
        assert lookup.await_count == int(child)
    else:
        assert runs == before
        lookup.assert_not_awaited()


@pytest.mark.parametrize(
    "damage", ["malformed", "missing", "identity", "ownership", "mode"]
)
async def test_invalid_recovery_progress_never_restarts_or_looks_up_parent(
    damage: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    graph = _graph(calls)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    request = cast(CreateResponse, {"input": "A"})
    saved = None
    async for event in server._handle_create_async_gen(
        request, _context(request, provider, "resp-current"), asyncio.Event()
    ):
        if isinstance(event, ResponseCheckpointEvent):
            saved = deepcopy(event.response)
    assert saved is not None
    stream = ResponseEventStream(response_id="resp-current", response=saved)
    if damage == "malformed":
        stream.internal_metadata["langgraph_checkpoint_id"] = None
    elif damage == "missing":
        stream.internal_metadata["langgraph_checkpoint_id"] = "missing"
    elif damage == "identity":
        stream.internal_metadata["langgraph_thread_id"] = "another-thread"
    elif damage == "mode":
        stream.internal_metadata[METADATA_RESPONSE_BRANCHING] = "unknown"
    else:
        config = _config(cast(Any, saved))["configurable"]
        saver = cast(InMemorySaver, graph.checkpointer)
        checkpoint, metadata, parent = saver.storage[config["thread_id"]][""][
            config["checkpoint_id"]
        ]
        decoded = saver.serde.loads_typed(metadata)
        decoded[METADATA_RESPONSE_EXECUTION] = json.dumps(["different-response", None])
        saver.storage[config["thread_id"]][""][config["checkpoint_id"]] = (
            checkpoint,
            saver.serde.dumps_typed(decoded),
            parent,
        )
    saved = cast(ResponseObject, stream.response)
    lookup = AsyncMock(side_effect=AssertionError("must not resolve a parent"))
    monkeypatch.setattr(provider, "get_response", lookup)
    result = [
        event
        async for event in server._handle_create_async_gen(
            request,
            _context(request, provider, "resp-current", snapshot=saved),
            asyncio.Event(),
        )
    ][-1]
    assert result["type"] == "response.failed"
    assert result["response"]["error"]["code"] == "server_error"
    assert calls == [["A"]]
    lookup.assert_not_awaited()


def test_legacy_recovery_does_not_enable_branching_after_flag_change() -> None:
    request = cast(CreateResponse, {"input": "A"})
    saved = ResponseEventStream(response_id="legacy", request=request).response
    context = _context(request, InMemoryResponseProvider(), "legacy", snapshot=saved)
    context.client_headers.clear()
    assert not response_branching_enabled(context, configured=True)
    context.client_headers["x-client-langgraph-response-branching"] = "off"
    assert not response_branching_enabled(context, configured=True)


async def test_tool_continuation_recovery_still_advances_ordinary_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def pause(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=str(interrupt("Answer?")))]}

    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    provider = InMemoryResponseProvider()
    chain = SimpleNamespace(get=AsyncMock(return_value=None), set=AsyncMock())
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=provider,
        conversation_chain_store=chain,
        enable_response_branching=True,
    )
    request = cast(CreateResponse, {"input": "A"})
    parent = [
        event
        async for event in server._handle_create_async_gen(
            request, _context(request, provider, "resp-parent"), asyncio.Event()
        )
    ][-1]["response"]
    await provider.create_response(parent, None, None)
    call_id = next(
        item["call_id"] for item in parent["output"] if item["type"] == "function_call"
    )
    reply = cast(
        CreateResponse,
        {
            "previous_response_id": "resp-parent",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"resume": "answered"}),
                }
            ],
        },
    )
    saved = None
    async for event in server._handle_create_async_gen(
        reply, _context(reply, provider, "resp-reply"), asyncio.Event()
    ):
        if isinstance(event, ResponseCheckpointEvent):
            saved = deepcopy(event.response)
    assert saved is not None
    chain.set.assert_awaited_once()
    chain.set.reset_mock()
    lookup = AsyncMock(side_effect=AssertionError("must not resolve a parent"))
    monkeypatch.setattr(provider, "get_response", lookup)
    result = [
        event
        async for event in server._handle_create_async_gen(
            reply,
            _context(reply, provider, "resp-reply", snapshot=saved),
            asyncio.Event(),
        )
    ][-1]
    assert result["type"] == "response.completed"
    assert _text(result["response"]) == "answered"
    chain.set.assert_awaited_once()
    lookup.assert_not_awaited()


@pytest.mark.parametrize("streaming", [False, True])
def test_failed_checkpoint_write_never_completes_or_changes_parent(
    streaming: bool,
) -> None:
    class FailingSaver(InMemorySaver):
        fail = False

        async def aput(self, *args: Any, **kwargs: Any) -> RunnableConfig:
            if self.fail:
                raise OSError("checkpoint backend unavailable")
            return await super().aput(*args, **kwargs)

    calls: list[list[str]] = []
    saver = FailingSaver()
    graph = _graph(calls, saver=saver)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        saved_parent = client.portal.call(provider.get_response, parent["id"])
        before = graph.get_state(_config(cast(Any, saved_parent)))
        saver.fail = True
        response = client.post(
            "/responses",
            json={
                "input": "B",
                "previous_response_id": parent["id"],
                "stream": streaming,
            },
        )
        if streaming:
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            assert events[-1]["type"] == "response.failed"
            assert not any(event["type"] == "response.completed" for event in events)
            failed = events[-1]["response"]
        else:
            failed = response.json()
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "server_error"
        stored = client.portal.call(provider.get_response, failed["id"])
        assert (
            TaskStorageManager.from_stream(
                ResponseEventStream(response_id=failed["id"], response=stored)
            ).checkpoint_ref
            is None
        )
        assert client.portal.call(provider.get_response, parent["id"]) == saved_parent
    assert graph.get_state(_config(cast(Any, saved_parent))).values == before.values


def test_streamed_child_and_null_linkage_on_prefixed_routes() -> None:
    server = ResponsesHostServer(
        _graph([]), prefix="/v1", enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = client.post(
            "/v1/responses",
            json={"input": "A", "previous_response_id": None, "conversation": None},
        ).json()
        response = client.post(
            "/v1/responses",
            json={"input": "B", "previous_response_id": parent["id"], "stream": True},
        )
    assert response.status_code == 200
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert events[-1]["type"] == "response.completed"
    assert _text(events[-1]["response"]) == "A|B"


def test_hosted_default_provider_remains_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "azure.ai.agentserver.core.tasks._enablement._RESILIENT_TASKS_ENABLED", False
    )
    provider = InMemoryResponseProvider()
    factory = MagicMock(return_value=provider)
    monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "test")
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://test.services.ai.azure.com/api/projects/test",
    )
    monkeypatch.setenv("FOUNDRY_AGENT_SESSION_GUID", "a" * 32)
    monkeypatch.setattr(
        "azure.ai.agentserver.responses.store._foundry_provider.FoundryStorageProvider",
        factory,
    )
    monkeypatch.setattr("azure.identity.aio.DefaultAzureCredential", MagicMock())
    server = ResponsesHostServer(_graph([]), enable_response_branching=True)
    factory.assert_called_once()
    assert server.app._orchestrator._provider is provider
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        child = _create(client, input="B", previous_response_id=parent["id"])
    assert _text(child) == "A|B"
