# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Completed-parent branching through the public Responses endpoint."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from multiprocessing.connection import Connection
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from azure.ai.agentserver.core.platform_headers import USER_ID
from azure.ai.agentserver.core.tasks import (
    resilient_tasks_enabled,
    set_resilient_tasks_enabled,
)
from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    PlatformContext,
    ResponseContext,
    ResponseEventStream,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses._id_generator import IdGenerator
from azure.ai.agentserver.responses.models import (
    CreateResponse,
    ItemMessage,
    MessageContentInputTextContent,
    ResponseObject,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot, interrupt
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._checkpoint._checkpoint_reuse import (
    _RequestCheckpointSaver,
)
from langchain_azure_ai.agents.hosting._responses import (
    METADATA_LANGGRAPH_ROOT_RESPONSE_ID,
    CheckpointRef,
    TaskStorageManager,
)
from langchain_azure_ai.agents.hosting._responses.task_storage_manager import (
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
)


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))


@pytest.fixture
def graph() -> CompiledStateGraph:
    async def answer(state: MessagesState) -> dict[str, Any]:
        await asyncio.sleep(0)
        text = "|".join(
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        )
        return {"messages": [AIMessage(content=text)]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.fixture
def recovery_context() -> MagicMock:
    return MagicMock(
        spec=ResponseContext,
        response_id="resp-recovery",
        conversation_id=None,
        conversation_chain_id="resp-recovery",
        is_recovery=True,
        persisted_response={"id": "resp-recovery", "output": []},
        client_cancelled=False,
        shutdown=asyncio.Event(),
        platform_context=SimpleNamespace(user_id_key=None),
        get_input_items=AsyncMock(
            return_value=[
                ItemMessage(
                    type="message",
                    role="user",
                    content=[
                        MessageContentInputTextContent(
                            {"type": "input_text", "text": "B"}
                        )
                    ],
                )
            ]
        ),
    )


def _config(response: ResponseObject) -> RunnableConfig:
    reference = TaskStorageManager.from_stream(
        ResponseEventStream(response_id=response["id"], response=response)
    ).checkpoint_ref
    assert reference is not None
    return {
        "configurable": {
            "thread_id": reference.thread_id,
            "checkpoint_id": reference.checkpoint_id,
            "checkpoint_ns": "",
        }
    }


def _text(response: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in response["output"]
        if item["type"] == "message"
        for part in item["content"]
    )


def _create(client: TestClient, *, stream: bool = False, **body: Any) -> dict[str, Any]:
    result = client.post("/responses", json={**body, "stream": stream})
    assert result.status_code == 200, result.text
    if stream:
        events = [
            json.loads(line.removeprefix("data: "))
            for line in result.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        return cast(dict[str, Any], events[-1]["response"])
    return cast(dict[str, Any], result.json())


def _run_branch_crash_host(
    state_root: str,
    stream: bool,
    recovery_response_id: str | None,
    connection: Connection,
) -> None:
    os.environ["AGENTSERVER_STATE_ROOT"] = state_root
    os.environ.pop("FOUNDRY_PROJECT_ENDPOINT", None)
    os.environ.pop("FOUNDRY_AGENT_NAME", None)
    os.environ.pop("FOUNDRY_AGENT_VERSION", None)
    set_resilient_tasks_enabled(True)

    async def run() -> None:
        branch_started = Event()
        branch_ids: list[str] = []
        prepared: list[str] = []
        recovered: list[bool] = []

        async def prepare(state: MessagesState) -> dict[str, Any]:
            prepared.append(str(state["messages"][-1].content))
            return {}

        async def answer(
            state: MessagesState, config: RunnableConfig
        ) -> dict[str, Any]:
            text = "|".join(
                str(message.content)
                for message in state["messages"]
                if isinstance(message, HumanMessage)
            )
            if text == "A|B":
                context = config["configurable"]["response_context"]
                if recovery_response_id is None:
                    branch_ids.append(context.response_id)
                    branch_started.set()
                    await asyncio.Event().wait()
                recovered.append(context.is_recovery)
            return {"messages": [AIMessage(content=text)]}

        async with AsyncSqliteSaver.from_conn_string(
            str(Path(state_root) / "checkpoints.sqlite")
        ) as saver:
            builder = StateGraph(MessagesState)
            builder.add_node("prepare", prepare)
            builder.add_node("answer", answer)
            builder.add_edge(START, "prepare")
            builder.add_edge("prepare", "answer")
            builder.add_edge("answer", END)
            graph = builder.compile(checkpointer=saver)
            server = ResponsesHostServer(
                graph,
                options=ResponsesServerOptions(resilient_background=True),
                enable_response_branching=True,
            )
            provider = server.app._endpoint._provider

            async def wait_for_response(response_id: str) -> ResponseObject:
                async with asyncio.timeout(20):
                    while True:
                        response = await provider.get_response(response_id)
                        if response is not None:
                            if recovery_response_id is not None:
                                if response["status"] not in {"queued", "in_progress"}:
                                    return response
                            else:
                                reference = TaskStorageManager.from_stream(
                                    ResponseEventStream(
                                        response_id=response_id, response=response
                                    )
                                ).recovery_checkpoint_ref
                                if reference is not None:
                                    state = await graph.aget_state(_config(response))
                                    if state.next == ("answer",):
                                        return response
                        await asyncio.sleep(0)

            with TestClient(server.app) as client:
                assert client.portal is not None
                if recovery_response_id is None:
                    parent = _create(client, input="A")
                    stored_parent = client.portal.call(
                        provider.get_response, parent["id"]
                    )
                    with ThreadPoolExecutor(max_workers=1) as requests:
                        requests.submit(
                            _create,
                            client,
                            input="B",
                            previous_response_id=parent["id"],
                            background=True,
                            store=True,
                            stream=stream,
                        )
                        assert branch_started.wait(10)
                        progress = client.portal.call(wait_for_response, branch_ids[0])
                        sibling = _create(
                            client, input="C", previous_response_id=parent["id"]
                        )
                        stored_sibling = client.portal.call(
                            provider.get_response, sibling["id"]
                        )
                        connection.send((stored_parent, progress, stored_sibling))
                        await asyncio.Event().wait()
                else:
                    response = client.portal.call(
                        wait_for_response, recovery_response_id
                    )
                    stored = client.get(f"/responses/{recovery_response_id}")
                    assert stored.status_code == 200, stored.text
                    state = await graph.aget_state(_config(response))
                    connection.send(
                        (
                            stored.json(),
                            [
                                message.content
                                for message in state.values["messages"]
                                if isinstance(message, HumanMessage)
                            ],
                            prepared,
                            recovered,
                        )
                    )

    with patch("azure.ai.agentserver.core._tracing._configure_tracing"):
        asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("file_provider", [False, True])
@pytest.mark.parametrize(
    "failure", ["response", "reference", "checkpoint", "status", "paused", "backend"]
)
def test_branch_validation_precedes_stream(
    graph: CompiledStateGraph,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    file_provider: bool,
    stream: bool,
) -> None:
    if failure == "paused":

        def pause(state: MessagesState) -> dict[str, Any]:
            interrupt("Approval needed")
            return {}

        builder = StateGraph(MessagesState)
        builder.add_node("pause", pause)
        builder.add_edge(START, "pause")
        builder.add_edge("pause", END)
        graph = builder.compile(checkpointer=InMemorySaver())
    server = ResponsesHostServer(
        graph,
        store=None if file_provider else InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    provider = server.app._endpoint._provider
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, parent["id"])
        if failure == "response":
            assert client.delete(f"/responses/{parent['id']}").status_code == 200
        elif failure == "checkpoint":
            saver = graph.checkpointer
            assert isinstance(saver, InMemorySaver)
            client.portal.call(
                saver.adelete_thread, _config(stored)["configurable"]["thread_id"]
            )
        elif failure == "backend":
            monkeypatch.setattr(
                provider,
                "get_response",
                AsyncMock(side_effect=RuntimeError("private storage details")),
            )
        elif failure in {"reference", "status"}:
            if failure == "reference":
                parent_stream = ResponseEventStream(
                    response_id=parent["id"], response=stored
                )
                parent_stream.internal_metadata.pop(METADATA_LANGGRAPH_CHECKPOINT_ID)
                stored = parent_stream.response
            else:
                stored["status"] = "failed"
            monkeypatch.setattr(
                provider, "get_response", AsyncMock(return_value=stored)
            )
        execute = MagicMock(wraps=server.graph.astream)
        monkeypatch.setattr(server.graph, "astream", execute)
        response = client.post(
            "/responses",
            json={"input": "B", "previous_response_id": parent["id"], "stream": stream},
            headers={"x-request-id": "branch-validation-request"},
        )
    expected_status = {"response": 404, "backend": 500}.get(failure, 400)
    assert response.status_code == expected_status, response.text
    assert response.headers["content-type"] == "application/json"
    error = response.json()["error"]
    assert error["additionalInfo"]["request_id"] == response.headers["x-request-id"]
    if failure == "backend":
        assert error["code"] == "server_error"
        assert "private storage details" not in response.text
    else:
        assert error["param"] == "previous_response_id"
        assert error["code"] == (
            "previous_response_not_found"
            if failure == "response"
            else "invalid_request_error"
        )
    execute.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enable_response_branching", [False, True])
def test_siblings_and_descendants_use_the_selected_parent(
    graph: CompiledStateGraph,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    enable_response_branching: bool,
) -> None:
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        graph, store=provider, enable_response_branching=enable_response_branching
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored_parent = client.portal.call(provider.get_response, parent["id"])
        parent_state = graph.get_state(_config(stored_parent))
        chain_get = AsyncMock()
        chain_set = AsyncMock()
        monkeypatch.setattr(server._conversation_chain_store, "get", chain_get)
        monkeypatch.setattr(server._conversation_chain_store, "set", chain_set)

        first = _create(
            client, input="B", previous_response_id=parent["id"], stream=stream
        )
        second = _create(
            client, input="C", previous_response_id=parent["id"], stream=stream
        )
        descendant = _create(
            client, input="D", previous_response_id=first["id"], stream=stream
        )

        assert all(
            response["status"] == "completed"
            for response in (first, second, descendant)
        )
        assert [_text(response) for response in (first, second, descendant)] == [
            "A|B",
            "A|C",
            "A|B|D",
        ]
        stored_first = client.portal.call(provider.get_response, first["id"])
        stored_second = client.portal.call(provider.get_response, second["id"])

    configs = [
        _config(item)["configurable"]
        for item in (stored_parent, stored_first, stored_second)
    ]
    assert len({config["thread_id"] for config in configs}) == 1
    assert len({config["checkpoint_id"] for config in configs}) == 3
    assert graph.get_state(_config(stored_parent)).values == parent_state.values
    chain_get.assert_not_awaited()
    chain_set.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("file_provider", [False, True])
@pytest.mark.parametrize("branching", [False, True])
def test_continuation_reuses_parent_and_selected_checkpoint(
    graph: CompiledStateGraph,
    monkeypatch: pytest.MonkeyPatch,
    branching: bool,
    file_provider: bool,
    stream: bool,
) -> None:
    server = ResponsesHostServer(
        graph,
        store=None if file_provider else InMemoryResponseProvider(),
        enable_response_branching=branching,
    )
    provider = server.app._endpoint._provider
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored_parent = client.portal.call(provider.get_response, parent["id"])
        selected = _config(stored_parent)["configurable"]
        parent_reads = AsyncMock(wraps=provider.get_response)
        checkpoint_reads = 0
        get_tuple = InMemorySaver.aget_tuple

        async def read_checkpoint(
            saver: InMemorySaver, config: RunnableConfig
        ) -> CheckpointTuple | None:
            nonlocal checkpoint_reads
            configurable = config.get("configurable") or {}
            if (
                configurable.get("thread_id") == selected["thread_id"]
                and configurable.get("checkpoint_ns", "") == ""
                and configurable.get("checkpoint_id") == selected["checkpoint_id"]
            ):
                checkpoint_reads += 1
            return await get_tuple(saver, config)

        monkeypatch.setattr(provider, "get_response", parent_reads)
        monkeypatch.setattr(InMemorySaver, "aget_tuple", read_checkpoint)
        child = _create(
            client, input="B", previous_response_id=parent["id"], stream=stream
        )

    assert child["status"] == "completed"
    assert _text(child) == "A|B"
    parent_reads.assert_awaited_once()
    assert parent_reads.await_args is not None
    assert parent_reads.await_args.args == (parent["id"],)
    if branching:
        assert checkpoint_reads == 1


@pytest.mark.parametrize("stream", [False, True])
def test_response_history_selects_the_parent_without_a_checkpointer(
    graph: CompiledStateGraph, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    server = ResponsesHostServer(
        graph.copy({"checkpointer": None}), store=InMemoryResponseProvider()
    )
    chain_get = AsyncMock(side_effect=AssertionError("Unexpected chain read"))
    chain_set = AsyncMock(side_effect=AssertionError("Unexpected chain write"))
    monkeypatch.setattr(server._conversation_chain_store, "get", chain_get)
    monkeypatch.setattr(server._conversation_chain_store, "set", chain_set)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        _create(client, input="B", previous_response_id=parent["id"])
        child = _create(
            client, input="C", previous_response_id=parent["id"], stream=stream
        )
    assert child["status"] == "completed"
    assert _text(child) == "A|C"
    chain_get.assert_not_awaited()
    chain_set.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
def test_response_linkage_reemits_and_resumes_interrupts_without_chain_access(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    def pause(state: MessagesState) -> dict[str, Any]:
        decision = interrupt("Approval needed")
        return {"messages": [AIMessage(content=f"Approved: {decision}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()), store=provider
    )
    chain_get = AsyncMock(side_effect=AssertionError("Unexpected chain read"))
    chain_set = AsyncMock(side_effect=AssertionError("Unexpected chain write"))
    monkeypatch.setattr(server._conversation_chain_store, "get", chain_get)
    monkeypatch.setattr(server._conversation_chain_store, "set", chain_set)
    with TestClient(server.app) as client:
        parent = _create(client, input="A", stream=stream)
        repeated = _create(
            client,
            input="still waiting",
            previous_response_id=parent["id"],
            stream=stream,
        )
        approval = next(
            item
            for item in repeated["output"]
            if item["type"] == "mcp_approval_request"
        )
        child = _create(
            client,
            previous_response_id=repeated["id"],
            stream=stream,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
            ],
        )
        assert client.portal is not None
        stored_parent = client.portal.call(provider.get_response, parent["id"])
        stored_repeated = client.portal.call(provider.get_response, repeated["id"])
    assert parent["status"] == repeated["status"] == child["status"] == "completed"
    assert (
        _config(stored_parent)["configurable"]
        == _config(stored_repeated)["configurable"]
    )
    assert _text(child) == "Approved: Approval needed"
    chain_get.assert_not_awaited()
    chain_set.assert_not_awaited()


@pytest.mark.parametrize("enable_response_branching", [False, True])
def test_response_linkage_leaves_the_parent_conversation_head_unchanged(
    graph: CompiledStateGraph, enable_response_branching: bool
) -> None:
    server = ResponsesHostServer(
        graph,
        store=InMemoryResponseProvider(),
        enable_response_branching=enable_response_branching,
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A", conversation="conv-original")
        child = _create(client, input="B", previous_response_id=parent["id"])
        continued = _create(client, input="C", conversation="conv-original")
    assert child["status"] == continued["status"] == "completed"
    assert _text(child) == "A|B"
    assert _text(continued) == "A|C"


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("before_graph_start", [False, True])
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("parent_completed", [False, True])
def test_steering_preserves_the_selected_context(
    monkeypatch: pytest.MonkeyPatch,
    checkpointed: bool,
    before_graph_start: bool,
    conversation: bool,
    parent_completed: bool,
) -> None:
    # shortcut: SDK drain guard reads old input; remove after its fix.
    monkeypatch.setattr(
        "azure.ai.agentserver.responses.hosting._response_task_lifecycle._task_input_deleted",
        AsyncMock(return_value=False),
    )
    started = asyncio.Event()
    seen: dict[str, list[tuple[str, str]]] = {}

    async def answer(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
        human = [
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]
        seen[human[-1]] = [
            (message.type, str(message.content)) for message in state["messages"]
        ]
        if human[-1] == "B":
            started.set()
            await asyncio.wait_for(
                config["configurable"]["response_cancellation_signal"].wait(), 5
            )
        return {"messages": [AIMessage(content="|".join(human))]}

    class SteeringServer(ResponsesHostServer):
        async def handle_create(
            self,
            request: CreateResponse,
            context: ResponseContext,
            cancellation_signal: asyncio.Event,
        ) -> AsyncIterator[Any]:
            async for event in super().handle_create(
                request, context, cancellation_signal
            ):
                yield event
                if (
                    before_graph_start
                    and request.get("input") == "B"
                    and isinstance(event, dict)
                    and event.get("type") == "response.created"
                ):
                    started.set()
                    await asyncio.wait_for(cancellation_signal.wait(), 5)

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=InMemorySaver() if checkpointed else None)
    provider = InMemoryResponseProvider()
    server = SteeringServer(
        graph,
        store=provider,
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    monkeypatch.setattr(
        provider,
        "delete_response",
        AsyncMock(side_effect=AssertionError("This scenario never deletes responses")),
    )
    chain_get = AsyncMock(wraps=server._conversation_chain_store.get)
    chain_set = AsyncMock(wraps=server._conversation_chain_store.set)
    monkeypatch.setattr(server._conversation_chain_store, "get", chain_get)
    monkeypatch.setattr(server._conversation_chain_store, "set", chain_set)

    async def scenario() -> tuple[
        dict[str, Any] | None, dict[str, Any], dict[str, Any]
    ]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://testserver",
        ) as client:

            async def create(
                text: str,
                parent: str | None = None,
                response_id: str | None = None,
                stream: bool = False,
            ) -> dict[str, Any]:
                body: dict[str, Any] = {
                    "input": text,
                    "stream": stream,
                    "agent_session_id": "steering-test",
                }
                if response_id is not None:
                    body["response_id"] = response_id
                if conversation:
                    body["conversation"] = "conv-steering"
                elif parent is not None:
                    body["previous_response_id"] = parent
                response = await asyncio.wait_for(
                    client.post("/responses", json=body), 5
                )
                assert response.status_code == 200, response.text
                if stream:
                    events = [
                        json.loads(line.removeprefix("data: "))
                        for line in response.text.splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    ]
                    return cast(dict[str, Any], events[-1]["response"])
                return cast(dict[str, Any], response.json())

            parent = await create("A") if parent_completed else None
            parent_id = parent["id"] if parent is not None else None
            superseded_id = IdGenerator.new_response_id(parent_id)
            superseded = asyncio.create_task(
                create("B", parent_id, superseded_id, stream=True)
            )
            try:
                await asyncio.wait_for(started.wait(), 5)
                child = await create("C", superseded_id, stream=True)
                return parent, await superseded, child
            finally:
                if not superseded.done():
                    superseded.cancel()
                await asyncio.gather(superseded, return_exceptions=True)

    was_enabled = resilient_tasks_enabled()
    set_resilient_tasks_enabled(True)
    try:
        with TestClient(server.app) as client:
            assert client.portal is not None
            parent, superseded, child = client.portal.call(scenario)
            assert superseded["status"] == "completed", superseded
            assert child["status"] == "completed", child
            expected_human = (["A"] if parent_completed else []) + ["B", "C"]
            if checkpointed and before_graph_start:
                expected_human.remove("B")
            assert [
                text for role, text in seen["C"] if role == "human"
            ] == expected_human
            assert seen["C"].count(("ai", "A")) == int(parent_completed)
            assert ("B" not in seen) == before_graph_start
            stored_superseded = client.portal.call(
                provider.get_response, superseded["id"]
            )
            if not conversation:
                metadata = ResponseEventStream(
                    response_id=superseded["id"], response=stored_superseded
                ).internal_metadata
                assert metadata[METADATA_LANGGRAPH_ROOT_RESPONSE_ID] == (
                    parent["id"] if parent is not None else superseded["id"]
                )
            if parent is not None and checkpointed and before_graph_start:
                stored_parent = client.portal.call(provider.get_response, parent["id"])
                assert _config(stored_parent) == _config(stored_superseded)
            if not conversation:
                chain_get.assert_not_awaited()
                chain_set.assert_not_awaited()
    finally:
        set_resilient_tasks_enabled(was_enabled)


def test_concurrent_siblings_do_not_share_new_input(graph: CompiledStateGraph) -> None:
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")

        async def siblings() -> list[httpx.Response]:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server.app),
                base_url="http://test",
            ) as requests:
                return await asyncio.gather(
                    *(
                        requests.post(
                            "/responses",
                            json={"input": text, "previous_response_id": parent["id"]},
                        )
                        for text in ("B", "C")
                    )
                )

        assert client.portal is not None
        responses = client.portal.call(siblings)
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["status"] == "completed" for response in responses)
    assert [_text(response.json()) for response in responses] == ["A|B", "A|C"]


@pytest.mark.parametrize("stream", [False, True])
def test_branch_recovers_after_process_crash(tmp_path: Path, stream: bool) -> None:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    processes = []
    try:
        crashing = context.Process(
            target=_run_branch_crash_host,
            args=(str(tmp_path), stream, None, sender),
        )
        processes.append(crashing)
        crashing.start()
        assert receiver.poll(30), "Branch did not persist its progress before crash"
        parent, progress, sibling = receiver.recv()
        assert progress["status"] == "in_progress"
        assert _text(sibling) == "A|C"
        assert (
            _config(parent)["configurable"]["thread_id"]
            == _config(progress)["configurable"]["thread_id"]
        )
        assert _config(progress) != _config(sibling)
        crashing.kill()
        crashing.join(10)
        assert not crashing.is_alive()
        assert crashing.exitcode != 0

        recovering = context.Process(
            target=_run_branch_crash_host,
            args=(str(tmp_path), stream, progress["id"], sender),
        )
        processes.append(recovering)
        recovering.start()
        assert receiver.poll(30), "SDK did not automatically recover the branch"
        response, messages, prepared, recovered = receiver.recv()
        assert response["id"] == progress["id"]
        assert response["previous_response_id"] == parent["id"]
        assert response["status"] == "completed"
        assert _text(response) == "A|B"
        assert messages == ["A", "B"]
        assert prepared == []
        assert recovered == [True]
        recovering.join(10)
        assert recovering.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(10)
            process.close()
        receiver.close()
        sender.close()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("saved_progress", [False, True])
async def test_branch_recovery_uses_selected_checkpoint(
    graph: CompiledStateGraph,
    recovery_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    saved_progress: bool,
) -> None:
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored_parent = client.portal.call(provider.get_response, parent["id"])
        parent_config = _config(stored_parent)
        parent_state = await graph.aget_state(parent_config)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="B")]},
            parent_config,
            interrupt_before=["answer"],
            durability="sync",
        )
        progress = await graph.aget_state(
            {"configurable": {"thread_id": parent_config["configurable"]["thread_id"]}}
        )
        assert progress.next == ("answer",)
        sibling = _create(client, input="C", previous_response_id=parent["id"])
        assert _text(sibling) == "A|C"

        recovery_context._provider = provider
        if saved_progress:
            reference = CheckpointRef.from_dict(progress.config["configurable"])
            assert reference is not None
            metadata: dict[str, Any] = {}
            TaskStorageManager(metadata).store_checkpoint_ref(reference)
            recovery_context.persisted_response["metadata"] = {
                "_internal_metadata": json.dumps(metadata)
            }
            lookup = AsyncMock(side_effect=KeyError(parent["id"]))
        else:
            lookup = AsyncMock(wraps=provider.get_response)
        monkeypatch.setattr(provider, "get_response", lookup)
        chain_get = AsyncMock()
        chain_set = AsyncMock()
        monkeypatch.setattr(server._conversation_chain_store, "get", chain_get)
        monkeypatch.setattr(server._conversation_chain_store, "set", chain_set)
        events = [
            event
            async for event in server._handle_create_async_gen(
                cast(
                    CreateResponse,
                    {
                        "input": "B",
                        "previous_response_id": parent["id"],
                        "stream": stream,
                        "background": True,
                        "store": True,
                    },
                ),
                recovery_context,
                asyncio.Event(),
            )
        ]

    response = events[-1]["response"]
    assert response["status"] == "completed"
    assert _text(response) == "A|B"
    state = await graph.aget_state(_config(response))
    assert [
        message.content
        for message in state.values["messages"]
        if isinstance(message, HumanMessage)
    ] == ["A", "B"]
    assert (await graph.aget_state(parent_config)).values == parent_state.values
    if saved_progress:
        recovery_context.get_input_items.assert_not_awaited()
        lookup.assert_not_awaited()
    else:
        recovery_context.get_input_items.assert_awaited_once()
        lookup.assert_awaited_once_with(
            parent["id"], context=recovery_context.platform_context
        )
    chain_get.assert_not_awaited()
    chain_set.assert_not_awaited()


async def test_branch_recovery_from_finished_checkpoint_preserves_saved_output(
    recovery_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node_calls: list[str] = []

    async def answer(state: MessagesState) -> dict[str, Any]:
        text = "|".join(
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        )
        node_calls.append(text)
        return {"messages": [AIMessage(content=text)]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        child = _create(client, input="B", previous_response_id=parent["id"])
        assert _text(child) == "A|B"
        assert client.portal is not None
        stored_child = client.portal.call(provider.get_response, child["id"])
        child_config = _config(stored_child)
        child_state = await graph.aget_state(child_config)
        assert not child_state.next
        assert not child_state.tasks
        assert not child_state.interrupts

        snapshot = deepcopy(stored_child)
        snapshot["status"] = "in_progress"
        saved_output = deepcopy(snapshot["output"])
        sibling = _create(client, input="C", previous_response_id=parent["id"])
        assert _text(sibling) == "A|C"
        latest = await graph.aget_state(
            {"configurable": {"thread_id": child_config["configurable"]["thread_id"]}}
        )
        assert latest.values["messages"][-1].content == "A|C"

        recovery_context.response_id = child["id"]
        recovery_context.conversation_chain_id = child["id"]
        recovery_context.persisted_response = snapshot
        recovery_context._provider = provider
        lookup = AsyncMock(side_effect=KeyError(parent["id"]))
        monkeypatch.setattr(provider, "get_response", lookup)
        build_input = AsyncMock(wraps=server.build_input)
        monkeypatch.setattr(server, "build_input", build_input)
        execute = MagicMock(wraps=server.graph.astream)
        monkeypatch.setattr(server.graph, "astream", execute)
        events = [
            event
            async for event in server._handle_create_async_gen(
                cast(
                    CreateResponse,
                    {
                        "input": "B",
                        "previous_response_id": parent["id"],
                        "background": True,
                        "store": True,
                    },
                ),
                recovery_context,
                asyncio.Event(),
            )
        ]

    response = events[-1]["response"]
    assert response["status"] == "completed"
    assert response["output"] == saved_output
    assert node_calls == ["A", "A|B", "A|C"]
    execute.assert_called_once()
    assert execute.call_args.args[0] is None
    assert (
        execute.call_args.kwargs["config"]["configurable"]["checkpoint_id"]
        == child_config["configurable"]["checkpoint_id"]
    )
    build_input.assert_not_awaited()
    recovery_context.get_input_items.assert_not_awaited()
    lookup.assert_not_awaited()


@pytest.mark.parametrize("branching", [False, True])
@pytest.mark.parametrize("damage", ["partial", "malformed"])
async def test_recovery_rejects_damaged_progress_without_replaying_input(
    graph: CompiledStateGraph,
    recovery_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    branching: bool,
) -> None:
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        graph, store=provider, enable_response_branching=branching
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored_parent = client.portal.call(provider.get_response, parent["id"])
        parent_config = _config(stored_parent)
        await graph.ainvoke(
            {"messages": [HumanMessage(content="B")]},
            parent_config,
            interrupt_before=["answer"],
            durability="sync",
        )
        progress = await graph.aget_state(
            {"configurable": {"thread_id": parent_config["configurable"]["thread_id"]}}
        )
        assert progress.next == ("answer",)
        reference = CheckpointRef.from_dict(progress.config["configurable"])
        assert reference is not None
        metadata: dict[str, Any] = {}
        TaskStorageManager(metadata).store_checkpoint_ref(reference)
        if damage == "partial":
            metadata.pop("langgraph_checkpoint_id")
        else:
            metadata["langgraph_checkpoint_id"] = 42
        recovery_context.persisted_response["metadata"] = {
            "_internal_metadata": json.dumps(metadata)
        }
        recovery_context._provider = provider
        lookup = AsyncMock(wraps=provider.get_response)
        monkeypatch.setattr(provider, "get_response", lookup)
        execute = AsyncMock(wraps=server.build_input)
        monkeypatch.setattr(server, "build_input", execute)
        chain_get = AsyncMock()
        chain_set = AsyncMock()
        monkeypatch.setattr(server._conversation_chain_store, "get", chain_get)
        monkeypatch.setattr(server._conversation_chain_store, "set", chain_set)
        events = [
            event
            async for event in server._handle_create_async_gen(
                cast(
                    CreateResponse,
                    {
                        "input": "B",
                        "previous_response_id": parent["id"],
                        "background": True,
                        "store": True,
                    },
                ),
                recovery_context,
                asyncio.Event(),
            )
        ]

    assert events[-1]["response"]["status"] == "failed"
    execute.assert_not_awaited()
    recovery_context.get_input_items.assert_not_awaited()
    lookup.assert_not_awaited()
    chain_get.assert_not_awaited()
    chain_set.assert_not_awaited()


@pytest.mark.parametrize("recovering", [False, True])
@pytest.mark.parametrize("failure", ["response", "reference", "checkpoint", "status"])
async def test_unavailable_parent_fails_without_executing_a_new_root(
    graph: CompiledStateGraph,
    recovery_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    recovering: bool,
) -> None:
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, parent["id"])
        if failure == "response":
            lookup = AsyncMock(side_effect=KeyError(parent["id"]))
        else:
            if failure == "reference":
                stored["metadata"] = {}
            elif failure == "checkpoint":
                reference = _config(stored)["configurable"]
                saver = graph.checkpointer
                assert isinstance(saver, InMemorySaver)
                client.portal.call(saver.adelete_thread, reference["thread_id"])
            else:
                stored["status"] = "failed"
            lookup = AsyncMock(return_value=stored)
        monkeypatch.setattr(provider, "get_response", lookup)
        execute = AsyncMock(wraps=server.build_input)
        monkeypatch.setattr(server, "build_input", execute)
        if recovering:
            recovery_context._provider = provider
            events = [
                event
                async for event in server._handle_create_async_gen(
                    cast(
                        CreateResponse,
                        {"input": "B", "previous_response_id": parent["id"]},
                    ),
                    recovery_context,
                    asyncio.Event(),
                )
            ]
            assert events[-1]["response"]["status"] == "failed"
        else:
            response = client.post(
                "/responses",
                json={"input": "B", "previous_response_id": parent["id"]},
            )
            assert response.status_code == (
                404 if failure in {"response", "reference"} else 400
            )
            assert response.json()["error"]["param"] == "previous_response_id"

    execute.assert_not_awaited()


@pytest.mark.parametrize("cleanup_phase", ["config", "validation"])
def test_checkpoint_cleanup_never_starts_a_new_root(
    graph: CompiledStateGraph, monkeypatch: pytest.MonkeyPatch, cleanup_phase: str
) -> None:
    saver = graph.checkpointer
    assert isinstance(saver, InMemorySaver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert parent["status"] == "completed"
        if cleanup_phase == "config":
            build_config = server.build_runnable_config
            config_builds = 0

            async def cleanup_after_config(
                request: CreateResponse, context: ResponseContext
            ) -> RunnableConfig:
                nonlocal config_builds
                config = await build_config(request, context)
                config_builds += 1
                if config_builds == 2:
                    await saver.adelete_thread(config["configurable"]["thread_id"])
                return config

            monkeypatch.setattr(server, "build_runnable_config", cleanup_after_config)
        else:
            get_state = server._graph.aget_state
            validation_reads = 0

            async def cleanup_after_validation(config: RunnableConfig) -> StateSnapshot:
                nonlocal validation_reads
                state = await get_state(config)
                validation_reads += 1
                if validation_reads == 2:
                    await saver.adelete_thread(config["configurable"]["thread_id"])
                return state

            monkeypatch.setattr(server._graph, "aget_state", cleanup_after_validation)
        response = _create(client, input="B", previous_response_id=parent["id"])

    if cleanup_phase == "config":
        assert response["status"] == "failed"
    else:
        assert response["status"] == "completed"
        assert _text(response) == "A|B"


@pytest.mark.parametrize("enabled", [False, True])
def test_custom_saver_config_is_available_before_checkpoint_reads(
    graph: CompiledStateGraph, enabled: bool
) -> None:
    class PartitionSaver(InMemorySaver):
        @property
        def config_specs(self) -> list[ConfigurableFieldSpec]:
            return [ConfigurableFieldSpec(id="partition", annotation=str)]

        async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            if config["configurable"].get("partition") != "tenant-a":
                raise ValueError("The checkpoint partition is required.")
            return await super().aget_tuple(config)

    class PartitionServer(ResponsesHostServer):
        async def build_runnable_config(
            self, request: CreateResponse, context: ResponseContext
        ) -> RunnableConfig:
            config = await super().build_runnable_config(request, context)
            config["configurable"]["partition"] = "tenant-a"
            return config

    server = PartitionServer(
        graph.copy({"checkpointer": PartitionSaver()}),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        response = _create(client, input="B", previous_response_id=parent["id"])
    assert parent["status"] == "completed"
    assert response["status"] == "completed"
    assert _text(response) == "A|B"


@pytest.mark.parametrize("recovering", [False, True])
async def test_paused_parent_cannot_be_branched(
    recovery_context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    recovering: bool,
) -> None:
    def pause(state: MessagesState) -> dict[str, Any]:
        interrupt("Approval needed")
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=provider,
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        assert parent["status"] == "completed"
        assert client.portal is not None
        stored_parent = client.portal.call(provider.get_response, parent["id"])
        state = await server.graph.aget_state(_config(stored_parent))
        assert state.next == ("pause",)
        assert state.interrupts
        build_input = AsyncMock(wraps=server.build_input)
        monkeypatch.setattr(server, "build_input", build_input)
        execute = MagicMock(wraps=server.graph.astream)
        monkeypatch.setattr(server.graph, "astream", execute)
        if recovering:
            recovery_context._provider = provider
            events = [
                event
                async for event in server._handle_create_async_gen(
                    cast(
                        CreateResponse,
                        {
                            "input": "B",
                            "previous_response_id": parent["id"],
                            "background": True,
                            "store": True,
                        },
                    ),
                    recovery_context,
                    asyncio.Event(),
                )
            ]
            assert events[-1]["response"]["status"] == "failed"
            recovery_context.get_input_items.assert_not_awaited()
        else:
            response = client.post(
                "/responses",
                json={"input": "B", "previous_response_id": parent["id"]},
            )
            assert response.status_code == 400
            assert response.json()["error"]["param"] == "previous_response_id"

    build_input.assert_not_awaited()
    execute.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("caller", ["user-a", "user-b", None])
@pytest.mark.parametrize(
    ("parent_status", "reference_state"),
    [
        ("completed", "complete"),
        ("failed", "complete"),
        ("in_progress", "complete"),
        ("completed", "missing"),
        ("completed", "thread_only"),
        ("completed", "checkpoint_only"),
        ("completed", "empty_checkpoint"),
        ("completed", "invalid_checkpoint"),
        ("completed", "invalid_thread"),
    ],
)
def test_parent_checkpoint_remains_user_scoped(
    graph: CompiledStateGraph,
    monkeypatch: pytest.MonkeyPatch,
    caller: str | None,
    parent_status: Literal["completed", "failed", "in_progress"],
    reference_state: str,
    stream: bool,
) -> None:
    server = ResponsesHostServer(
        graph,
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        parent = client.post(
            "/responses", json={"input": "A"}, headers={USER_ID: "user-a"}
        ).json()
        provider = server.app._endpoint._provider
        assert client.portal is not None
        stored_parent = client.portal.call(
            partial(
                provider.get_response,
                parent["id"],
                context=PlatformContext(user_id_key="user-a"),
            )
        )
        assert stored_parent is not None
        parent_stream = ResponseEventStream(
            response_id=parent["id"], response=stored_parent
        )
        metadata = parent_stream.internal_metadata
        if reference_state == "missing":
            metadata.clear()
        elif reference_state == "thread_only":
            metadata.pop(METADATA_LANGGRAPH_CHECKPOINT_ID)
        elif reference_state == "checkpoint_only":
            metadata.pop(METADATA_LANGGRAPH_THREAD_ID)
        elif reference_state == "empty_checkpoint":
            metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = ""
        elif reference_state == "invalid_checkpoint":
            metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = 123
        elif reference_state == "invalid_thread":
            metadata[METADATA_LANGGRAPH_THREAD_ID] = 123
        stored_parent = parent_stream.response
        stored_parent["status"] = parent_status
        monkeypatch.setattr(
            provider, "get_response", AsyncMock(return_value=stored_parent)
        )
        execute = MagicMock(wraps=server.graph.astream)
        monkeypatch.setattr(server.graph, "astream", execute)
        response = client.post(
            "/responses",
            json={"input": "B", "previous_response_id": parent["id"], "stream": stream},
            headers={USER_ID: caller} if caller is not None else {},
        )
    assert parent["status"] == "completed"
    expected_status = 404
    if caller == "user-a" and reference_state not in {
        "missing",
        "checkpoint_only",
        "invalid_thread",
    }:
        expected_status = (
            200
            if reference_state == "complete" and parent_status == "completed"
            else 400
        )
    assert response.status_code == expected_status, response.text
    if expected_status == 200:
        if stream:
            assert "event: response.completed" in response.text
        else:
            assert response.json()["status"] == "completed"
        return
    execute.assert_not_called()
    assert response.headers["content-type"] == "application/json"
    error = response.json()["error"]
    assert error["param"] == "previous_response_id"
    if expected_status == 404:
        assert error["code"] == "previous_response_not_found"
        assert error["message"] == "The previous response was not found."
    else:
        assert error["code"] == "invalid_request_error"
        assert error["message"] == (
            "The previous response must be completed."
            if reference_state == "complete"
            else "The previous response has no saved graph checkpoint."
        )


def test_branching_requires_a_checkpointer_and_disabled_steering(
    graph: CompiledStateGraph,
) -> None:
    with pytest.raises(ValueError, match="checkpointer"):
        ResponsesHostServer(
            graph.copy({"checkpointer": None}), enable_response_branching=True
        )
    with pytest.raises(ValueError, match="steerable_conversations=False"):
        ResponsesHostServer(
            graph,
            options=ResponsesServerOptions(steerable_conversations=True),
            enable_response_branching=True,
        )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("method_name", ["aget_tuple", "aput", "aput_writes"])
def test_branching_rejects_unimplemented_async_checkpoint_methods(
    graph: CompiledStateGraph,
    monkeypatch: pytest.MonkeyPatch,
    wrapped: bool,
    method_name: str,
) -> None:
    class IncompleteSaver(InMemorySaver):
        pass

    monkeypatch.setattr(
        IncompleteSaver, method_name, getattr(BaseCheckpointSaver, method_name)
    )
    saver = IncompleteSaver()
    checkpointer = _RequestCheckpointSaver(saver) if wrapped else saver
    with pytest.raises(ValueError, match=method_name):
        ResponsesHostServer(
            graph.copy({"checkpointer": checkpointer}),
            enable_response_branching=True,
        )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_sync_sqlite_saver_is_rejected_only_when_branching_is_enabled(
    graph: CompiledStateGraph, wrapped: bool, enabled: bool
) -> None:
    with SqliteSaver.from_conn_string(":memory:") as saver:
        checkpointer = _RequestCheckpointSaver(saver) if wrapped else saver
        with (
            pytest.raises(ValueError, match="asynchronous.*aput_writes")
            if enabled
            else nullcontext()
        ):
            ResponsesHostServer(
                graph.copy({"checkpointer": checkpointer}),
                enable_response_branching=enabled,
            )


@pytest.mark.parametrize("wrapped", [False, True])
async def test_branching_accepts_async_sqlite_saver(
    graph: CompiledStateGraph, wrapped: bool
) -> None:
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        checkpointer = _RequestCheckpointSaver(saver) if wrapped else saver
        server = ResponsesHostServer(
            graph.copy({"checkpointer": checkpointer}),
            enable_response_branching=True,
        )
        assert isinstance(server.graph.checkpointer, _RequestCheckpointSaver)
        assert server.graph.checkpointer.__wrapped__ is saver


@pytest.mark.parametrize("steerable_conversations", [False, True])
@pytest.mark.parametrize("provide_options", [False, True])
def test_branching_rejects_external_host(
    graph: CompiledStateGraph,
    steerable_conversations: bool,
    provide_options: bool,
) -> None:
    app = ResponsesAgentServerHost(
        options=ResponsesServerOptions(steerable_conversations=steerable_conversations),
        store=InMemoryResponseProvider(),
    )
    options = ResponsesServerOptions() if provide_options else None
    with pytest.raises(ValueError, match="app=None"):
        ResponsesHostServer(
            graph,
            app=app,
            options=options,
            enable_response_branching=True,
        )


@pytest.mark.parametrize("steerable_conversations", [False, True])
def test_external_host_remains_supported_without_branching(
    graph: CompiledStateGraph, steerable_conversations: bool
) -> None:
    app = ResponsesAgentServerHost(
        options=ResponsesServerOptions(steerable_conversations=steerable_conversations),
        store=InMemoryResponseProvider(),
    )
    server = ResponsesHostServer(graph, app=app)
    assert server.app is app
    with TestClient(server.app) as client:
        response = _create(client, input="A")
    assert _text(response) == "A"


@pytest.mark.parametrize("caller", ["user-b", None])
def test_ordinary_parent_checkpoint_remains_user_scoped(
    graph: CompiledStateGraph, caller: str | None
) -> None:
    server = ResponsesHostServer(
        graph,
        store=InMemoryResponseProvider(),
        enable_response_branching=False,
    )
    with TestClient(server.app) as client:
        parent = client.post(
            "/responses", json={"input": "A"}, headers={USER_ID: "user-a"}
        ).json()
        response = client.post(
            "/responses",
            json={"input": "B", "previous_response_id": parent["id"]},
            headers={USER_ID: caller} if caller is not None else {},
        ).json()
    assert parent["status"] == "completed"
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "previous_response_not_found"
