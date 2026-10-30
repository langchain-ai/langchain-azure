# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Completed-parent branching through the public Responses endpoint."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from azure.ai.agentserver.core.platform_headers import USER_ID
from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponseContext,
    ResponseEventStream,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.models import CreateResponse, ResponseObject
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
from langchain_azure_ai.agents.hosting._responses import TaskStorageManager


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


@pytest.mark.parametrize("failure", ["response", "reference", "checkpoint", "status"])
def test_unavailable_parent_fails_without_executing_a_new_root(
    graph: CompiledStateGraph, monkeypatch: pytest.MonkeyPatch, failure: str
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
        response = _create(client, input="B", previous_response_id=parent["id"])

    assert response["status"] == "failed"
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

            async def cleanup_after_config(
                request: CreateResponse, context: ResponseContext
            ) -> RunnableConfig:
                config = await build_config(request, context)
                await saver.adelete_thread(config["configurable"]["thread_id"])
                return config

            monkeypatch.setattr(server, "build_runnable_config", cleanup_after_config)
        else:
            get_state = server._graph.aget_state

            async def cleanup_after_validation(config: RunnableConfig) -> StateSnapshot:
                state = await get_state(config)
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


def test_paused_parent_cannot_be_branched() -> None:
    def pause(state: MessagesState) -> dict[str, Any]:
        interrupt("Approval needed")
        return {}

    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        parent = _create(client, input="A")
        response = _create(client, input="B", previous_response_id=parent["id"])
    assert parent["status"] == "completed"
    assert response["status"] == "failed"


@pytest.mark.parametrize("caller", ["user-b", None])
@pytest.mark.parametrize("enable_response_branching", [False, True])
def test_parent_checkpoint_remains_user_scoped(
    graph: CompiledStateGraph, caller: str | None, enable_response_branching: bool
) -> None:
    server = ResponsesHostServer(
        graph,
        store=InMemoryResponseProvider(),
        enable_response_branching=enable_response_branching,
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
    assert response["status"] == "failed"


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
