# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Completed-parent branching through the public Responses endpoint."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import httpx
import pytest
from azure.ai.agentserver.core.platform_headers import USER_ID
from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponseEventStream,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.models import ResponseObject
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from starlette.testclient import TestClient

from langchain_azure_ai.agents.hosting import ResponsesHostServer
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
def test_siblings_and_descendants_use_the_selected_parent(
    graph: CompiledStateGraph, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
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
def test_parent_checkpoint_remains_user_scoped(
    graph: CompiledStateGraph, caller: str | None
) -> None:
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
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
