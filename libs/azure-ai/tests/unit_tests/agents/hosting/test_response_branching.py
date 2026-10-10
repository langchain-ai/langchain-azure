"""Opt-in response checkpoint branching and strict restoration tests."""

from __future__ import annotations

import asyncio
import json
import operator
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses.store._memory import InMemoryResponseProvider
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, before_model
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from starlette.testclient import TestClient
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._responses import (
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    HostingRunnableConfig,
)
from langchain_azure_ai.agents.hosting._responses.branching import (
    BRANCH_BOUNDARY_KEY,
    BRANCH_MODE,
    BRANCH_MODE_HEADER,
    BRANCH_MODE_METADATA,
    BRANCH_ORIGIN_KEY,
    BranchingAdmissionMiddleware,
    ResponseBranchStore,
    ResponseCheckpointSaver,
)

from .hitl.graphs import (
    build_parallel_interrupt_graph,
    build_sequential_interrupt_graph,
    build_simple_interrupt_graph,
)
from .test_responses_host import _context, _parse_sse, _request, _response_object


@pytest.fixture(autouse=True)
def _isolate_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "azure.ai.agentserver.core._tracing._configure_tracing", lambda *_, **__: None
    )


def _post(
    client: TestClient,
    text: Any,
    *,
    previous_response_id: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    request = {"model": "test", "input": text, "stream": stream, "store": True}
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    response = client.post("/responses", json={**request, **kwargs})
    assert response.status_code == 200, response.text
    if stream:
        events = _parse_sse(response.text)
        for _, payload in events:
            if "response" in payload:
                assert (
                    payload["response"].get("previous_response_id")
                    == previous_response_id
                )
        return next(
            payload["response"]
            for kind, payload in reversed(events)
            if kind in {"response.completed", "response.failed"}
        )
    return response.json()


def _text(response: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in response["output"]
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


class _BranchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ledger: Annotated[list[str], operator.add]


class _ModelInputCapture(BaseCallbackHandler):
    def __init__(self) -> None:
        self.inputs: list[list[BaseMessage]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        self.inputs.extend(messages)


def _branch_graph() -> tuple[CompiledStateGraph, list[str]]:
    executions: list[str] = []

    async def record(state: _BranchState) -> dict[str, Any]:
        text = next(
            str(message.content)
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        executions.append(text)
        ledger = [*state.get("ledger", []), text]
        return {"ledger": [text], "messages": [AIMessage(content=",".join(ledger))]}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=InMemorySaver()), executions


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "", "next-request"])
def test_checkpoint_instructions_preserve_existing_messages(
    enabled: bool, stream: bool, instructions: str | None
) -> None:
    observed: list[list[str]] = []

    async def record(state: _BranchState) -> dict[str, Any]:
        observed.append(
            [
                str(message.content)
                for message in state["messages"]
                if isinstance(message, SystemMessage)
            ]
        )
        return {"messages": [AIMessage(content="ok")]}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        root = _post(
            client,
            [
                {"role": "system", "content": "application-prompt"},
                {"role": "user", "content": "A"},
            ],
            instructions="initial-request",
            stream=stream,
        )
        child = _post(
            client,
            "B",
            previous_response_id=root["id"],
            instructions=instructions,
            stream=stream,
        )

    assert root["status"] == child["status"] == "completed"
    initial = ["initial-request", "application-prompt"]
    assert observed == [initial, [*initial, *([instructions] if instructions else [])]]


async def test_legacy_checkpoint_without_instruction_metadata_remains_usable() -> None:
    graph, executions = _branch_graph()
    await graph.ainvoke(
        {"messages": [SystemMessage(content="application-prompt"), HumanMessage("A")]},
        {"configurable": {"thread_id": "legacy"}},
    )
    executions.clear()
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    events = [
        event
        async for event in server.handle_create(
            _request(conversation={"id": "legacy"}),
            _context(conversation_id="legacy", current_text="B"),
            asyncio.Event(),
        )
    ]

    assert events[-1]["response"]["status"] == "completed", events[-1]
    assert executions == ["B"]
    snapshot = await graph.aget_state({"configurable": {"thread_id": "legacy"}})
    assert [
        message.content
        for message in snapshot.values["messages"]
        if isinstance(message, SystemMessage)
    ] == ["application-prompt"]


@pytest.mark.parametrize(
    "fields",
    [
        {"previous_response_id": "parent", "conversation": {"id": "conversation"}},
        {"previous_response_id": ""},
        {"response_id": "client-value"},
    ],
)
async def test_disabled_branching_leaves_request_validation_to_sdk(
    fields: dict[str, Any],
) -> None:
    app = AsyncMock()
    middleware = BranchingAdmissionMiddleware(app, enabled=False)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/responses",
        "headers": [
            (BRANCH_MODE_HEADER.encode("ascii"), BRANCH_MODE.encode("ascii")),
            (b"content-type", b"application/json"),
        ],
    }
    receive = AsyncMock(
        return_value={"type": "http.request", "body": json.dumps(fields).encode()}
    )
    send = AsyncMock()

    await middleware(scope, receive, send)

    app.assert_awaited_once()
    receive.assert_not_awaited()
    send.assert_not_awaited()
    assert app.await_args is not None
    assert app.await_args.args[0]["headers"] == [(b"content-type", b"application/json")]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("conversation_parent", [False, True])
def test_branching_rejects_legacy_parent(
    stream: bool, conversation_parent: bool
) -> None:
    graph, executions = _branch_graph()
    provider = InMemoryResponseProvider()
    legacy_server = ResponsesHostServer(
        graph, store=provider, enable_response_branching=conversation_parent
    )
    with TestClient(legacy_server.app) as client:
        parent = _post(
            client,
            "A",
            conversation={"id": "legacy"} if conversation_parent else None,
        )
        assert parent["status"] == "completed", parent
        assert _text(parent) == "A"

    saver = cast(InMemorySaver, graph.checkpointer)
    checkpoint = next(saver.list(None))
    assert checkpoint.checkpoint["channel_values"]["ledger"] == ["A"]
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        rejected = _post(client, "B", previous_response_id=parent["id"], stream=stream)
        assert rejected["status"] == "failed", rejected
        assert rejected["error"]["code"] == "server_error"
        assert rejected["error"]["message"] == (
            "The parent response was not created with response branching."
        )
        assert executions == ["A"]
        root = _post(client, "C", stream=stream)
        child = _post(client, "D", previous_response_id=root["id"], stream=stream)
        assert child["status"] == "completed", child
        assert _text(child) == "C,D"

    assert executions == ["A", "C", "D"]


def test_background_parent_requires_completion_and_preserves_branch_state() -> None:
    started = threading.Event()
    release = threading.Event()
    executions: list[str] = []

    async def record(state: _BranchState) -> dict[str, Any]:
        text = next(
            str(message.content)
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        executions.append(text)
        if text == "B":
            started.set()
            assert await asyncio.to_thread(release.wait, 10)
        ledger = [*state.get("ledger", []), text]
        return {"ledger": [text], "messages": [AIMessage(content=",".join(ledger))]}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()), enable_response_branching=True
    )
    response_id = "caresp_" + "a" * 18 + "d" * 32

    with TestClient(server.app) as client:
        root = _post(client, "A")
        assert root["status"] == "completed", root
        with ThreadPoolExecutor(max_workers=1) as executor:
            background = executor.submit(
                client.post,
                "/responses",
                json={
                    "model": "test",
                    "input": "B",
                    "previous_response_id": root["id"],
                    "background": True,
                    "stream": True,
                    "store": True,
                },
                headers={"x-agent-response-id": response_id},
            )
            try:
                assert started.wait(timeout=10)
                running = client.get(f"/responses/{response_id}")
                assert running.status_code == 200, running.text
                assert running.json()["status"] == "in_progress", running.text
                assert not background.done()
                rejected = _post(client, "rejected", previous_response_id=response_id)
                assert rejected["status"] == "failed", rejected
                assert rejected["error"]["message"] == (
                    "The parent response must be stored and completed."
                )
                assert executions == ["A", "B"]
            finally:
                release.set()
            completed = background.result(timeout=10)

        assert completed.status_code == 200, completed.text
        events = _parse_sse(completed.text)
        parent = next(
            payload["response"]
            for kind, payload in reversed(events)
            if kind in {"response.completed", "response.failed"}
        )
        assert parent["status"] == "completed", parent
        assert parent["background"] is True
        assert parent["id"] == response_id
        assert parent["previous_response_id"] == root["id"]
        assert _text(parent) == "A,B"
        stored = client.get(f"/responses/{response_id}")
        assert stored.status_code == 200, stored.text
        assert stored.json()["status"] == "completed", stored.text

        continuation = _post(client, "C", previous_response_id=response_id)
        sibling = _post(client, "D", previous_response_id=response_id)
        unchanged = client.get(f"/responses/{response_id}")

    assert continuation["status"] == "completed", continuation
    assert sibling["status"] == "completed", sibling
    assert _text(continuation) == "A,B,C"
    assert _text(sibling) == "A,B,D"
    assert unchanged.status_code == 200, unchanged.text
    assert _text(unchanged.json()) == "A,B"
    assert executions == ["A", "B", "C", "D"]


def test_branching_rejects_unimplemented_async_saver() -> None:
    graph, _ = _branch_graph()
    graph.checkpointer = BaseCheckpointSaver()
    with pytest.raises(ValueError, match="asynchronous checkpoint"):
        ResponsesHostServer(
            graph, store=InMemoryResponseProvider(), enable_response_branching=True
        )


def test_hosted_branching_response_store_uses_lazy_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_azure_ai._user_agent import get_user_agent
    from langchain_azure_ai.agents.hosting import _responses_host

    graph, _ = _branch_graph()
    agent_config = MagicMock()
    agent_config.from_env.return_value.is_hosted = True
    agent_config.from_env.return_value.project_endpoint = (
        "https://example.services.ai.azure.com/api/projects/test"
    )
    credential = MagicMock()
    settings = MagicMock()
    provider = MagicMock(return_value=InMemoryResponseProvider())
    monkeypatch.setattr(_responses_host, "AgentConfig", agent_config)
    monkeypatch.setattr(_responses_host, "DefaultAzureCredential", credential)
    monkeypatch.setattr(_responses_host, "FoundryStorageSettings", settings)
    monkeypatch.setattr(_responses_host, "FoundryStorageProvider", provider)

    ResponsesHostServer(graph, enable_response_branching=True)

    settings.from_endpoint.assert_called_once_with(
        agent_config.from_env.return_value.project_endpoint
    )
    provider.assert_called_once_with(
        credential.return_value,
        settings.from_endpoint.return_value,
        get_server_version=get_user_agent,
    )


@pytest.mark.parametrize(
    ("fields", "parameter"),
    [
        (
            {"previous_response_id": "parent", "conversation": "conversation"},
            "previous_response_id",
        ),
        (
            {"previous_response_id": "parent", "conversation": {"id": "conversation"}},
            "previous_response_id",
        ),
        ({"previous_response_id": ""}, "previous_response_id"),
        ({"previous_response_id": "   "}, "previous_response_id"),
        ({"previous_response_id": False}, "previous_response_id"),
        ({"previous_response_id": 12}, "previous_response_id"),
        ({"previous_response_id": []}, "previous_response_id"),
        ({"previous_response_id": {}}, "previous_response_id"),
        ({"response_id": "caresp_" + "a" * 18 + "b" * 32}, "response_id"),
    ],
)
def test_invalid_linkage_rejected_before_execution(
    fields: dict[str, Any], parameter: str
) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        response = client.post(
            "/responses",
            json={"model": "test", "input": "A", "stream": True, **fields},
        )

    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.json()["error"]["param"] == parameter
    assert executions == []


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_checkpoint_saver_read_policy(enabled: bool, asynchronous: bool) -> None:
    from langchain_azure_ai.agents.hosting._responses.branching import BranchingError

    saver = ResponseCheckpointSaver(InMemorySaver(), branching=enabled)
    config: RunnableConfig = {"configurable": {"thread_id": "missing"}}

    async def read() -> Any:
        return (
            await saver.aget_tuple(config) if asynchronous else saver.get_tuple(config)
        )

    assert await read() is None
    config["configurable"]["checkpoint_id"] = "explicit-missing"
    if enabled:
        with pytest.raises(BranchingError, match="required checkpoint"):
            await read()
    else:
        assert await read() is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_checkpoint_saver_delegates_thread_deletion(
    enabled: bool, asynchronous: bool
) -> None:
    adapter = ResponseCheckpointSaver(InMemorySaver(), branching=enabled)
    deleted: RunnableConfig = {
        "configurable": {"thread_id": "deleted", "checkpoint_ns": ""}
    }
    retained: RunnableConfig = {
        "configurable": {"thread_id": "retained", "checkpoint_ns": ""}
    }
    metadata: CheckpointMetadata = {"source": "input", "step": 0, "parents": {}}
    checkpoint = empty_checkpoint()
    deleted_checkpoint = await adapter.aput(deleted, checkpoint, metadata, {})
    retained_checkpoint = await adapter.aput(retained, empty_checkpoint(), metadata, {})
    await adapter.aput_writes(
        deleted_checkpoint, [("messages", "deleted-write")], "task"
    )
    await adapter.aput_writes(
        retained_checkpoint, [("messages", "retained-write")], "task"
    )
    original = await adapter.aget_tuple(retained_checkpoint)
    assert original is not None and original.pending_writes

    if asynchronous:
        await adapter.adelete_thread("deleted")
    else:
        adapter.delete_thread("deleted")

    assert await adapter.aget_tuple(deleted) is None
    assert [saved async for saved in adapter.alist(deleted)] == []
    assert await adapter.aget_tuple(retained_checkpoint) == original
    recreated = await adapter.aput(deleted, checkpoint, metadata, {})
    saved = await adapter.aget_tuple(recreated)
    assert saved is not None
    assert saved.pending_writes == []


@pytest.mark.parametrize("operation", ["state", "execute"])
async def test_strict_saver_rejects_checkpoint_deleted_after_preflight(
    operation: str,
) -> None:
    from langchain_azure_ai.agents.hosting._responses.branching import (
        BranchingError,
    )

    graph, executions = _branch_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "parent"}}
    await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent = (await graph.aget_state(config)).config
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    guarded = graph.copy(
        {"checkpointer": ResponseCheckpointSaver(saver, branching=True)}
    )
    assert (await guarded.aget_state(parent)).values["ledger"] == ["A"]
    await saver.adelete_thread("parent")

    with pytest.raises(BranchingError, match="checkpoint") as failure:
        if operation == "state":
            await guarded.aget_state(parent)
        else:
            await guarded.ainvoke({"messages": [HumanMessage(content="C")]}, parent)

    assert failure.value.code == "checkpoint_unavailable"
    assert executions == ["A"]
    assert graph.checkpointer is saver


async def test_strict_saver_retains_parent_and_isolates_graph_copy() -> None:
    graph, _ = _branch_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    guarded = graph.copy(
        {"checkpointer": ResponseCheckpointSaver(saver, branching=True)}
    )
    config: RunnableConfig = {"configurable": {"thread_id": "parent"}}
    await guarded.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent = (await guarded.aget_state(config)).config
    await guarded.ainvoke({"messages": [HumanMessage(content="B")]}, parent)
    fork = await guarded.ainvoke({"messages": [HumanMessage(content="C")]}, parent)

    assert fork["ledger"] == ["A", "C"]
    assert (await guarded.aget_state(parent)).values["ledger"] == ["A"]
    assert graph.checkpointer is saver


@pytest.mark.parametrize("stream", [False, True])
def test_response_branches_preserve_non_message_state(stream: bool) -> None:
    graph, _ = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        root = _post(client, "A", stream=stream)
        original = _post(client, "B", previous_response_id=root["id"], stream=stream)
        fork = _post(client, "C", previous_response_id=root["id"], stream=stream)
        original_next = _post(
            client, "D", previous_response_id=original["id"], stream=stream
        )
        fork_next = _post(client, "E", previous_response_id=fork["id"], stream=stream)
        regenerated = _post(client, "B", previous_response_id=root["id"], stream=stream)
        for response, parent_id in (
            (root, None),
            (original, root["id"]),
            (fork, root["id"]),
            (original_next, original["id"]),
            (fork_next, fork["id"]),
            (regenerated, root["id"]),
        ):
            retrieved = client.get(f"/responses/{response['id']}")
            assert retrieved.status_code == 200
            for representation in (response, retrieved.json()):
                assert representation["id"] == response["id"]
                assert representation.get("previous_response_id") == parent_id
                assert representation.get("conversation") is None
                assert "_internal_metadata" not in representation.get("metadata", {})

    assert _text(root) == "A"
    assert _text(original) == "A,B"
    assert _text(fork) == "A,C"
    assert _text(original_next) == "A,B,D"
    assert _text(fork_next) == "A,C,E"
    assert _text(regenerated) == "A,B"
    assert regenerated["id"] != original["id"]
    for response in (root, original, fork, original_next, fork_next, regenerated):
        assert response["status"] == "completed"
        assert "_internal_metadata" not in response.get("metadata", {})


@pytest.mark.parametrize("stream", [False, True])
def test_branches_preserve_parent_messages(stream: bool) -> None:
    observed: list[list[str]] = []

    async def record(state: _BranchState) -> dict[str, Any]:
        observed.append(
            [
                str(message.content)
                for message in state["messages"]
                if isinstance(message, SystemMessage)
            ]
        )
        messages: list[AnyMessage] = [AIMessage(content="ok")]
        if len(observed) == 1:
            messages.append(SystemMessage(content="application-owned"))
        return {"messages": messages}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(
            client,
            [
                {"role": "system", "content": "shared-text"},
                {"role": "developer", "content": "developer-context"},
                {"role": "user", "content": "A"},
            ],
            instructions="shared-text",
            stream=stream,
        )
        omitted = _post(client, "B", previous_response_id=root["id"], stream=stream)
        cleared = _post(
            client,
            "C",
            previous_response_id=root["id"],
            instructions=None,
            stream=stream,
        )
        replaced = _post(
            client,
            "D",
            previous_response_id=root["id"],
            instructions="child-only",
            stream=stream,
        )

    assert all(
        response["status"] == "completed"
        for response in (root, omitted, cleared, replaced)
    )
    assert observed == [
        ["shared-text", "shared-text", "developer-context"],
        ["shared-text", "shared-text", "developer-context", "application-owned"],
        ["shared-text", "shared-text", "developer-context", "application-owned"],
        [
            "shared-text",
            "shared-text",
            "developer-context",
            "application-owned",
            "child-only",
        ],
    ]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mutation", ["summarization", "trim"])
def test_application_message_management_remains_compatible(
    enabled: bool, stream: bool, mutation: str
) -> None:
    model_inputs = _ModelInputCapture()
    summary_inputs = _ModelInputCapture()
    saver = InMemorySaver()

    @before_model
    def trim_history(state: Any, runtime: Any) -> dict[str, Any]:
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *trim_messages(
                    state["messages"],
                    max_tokens=1,
                    token_counter=len,
                    start_on="human",
                ),
            ]
        }

    graph = create_agent(
        FakeListChatModel(responses=["ok"], callbacks=[model_inputs]),
        tools=[],
        system_prompt="application-owned",
        middleware=[
            SummarizationMiddleware(
                FakeListChatModel(
                    responses=["Earlier dialog summary"], callbacks=[summary_inputs]
                ),
                trigger=("messages", 2),
                keep=("messages", 1),
            )
            if mutation == "summarization"
            else trim_history,
        ],
        checkpointer=saver,
    )
    server = ResponsesHostServer(
        graph,
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        root = _post(client, "A", instructions="root-only", stream=stream)
        assert root["status"] == "completed", root
        parent = next(saver.list(None))
        response = root
        for text, fields in (
            ("B", {}),
            ("C", {"instructions": None}),
            ("D", {"instructions": ""}),
            ("E", {"instructions": "child-only"}),
            ("F", {}),
        ):
            response = _post(
                client,
                text,
                previous_response_id=response["id"],
                stream=stream,
                **fields,
            )
            assert response["status"] == "completed", response

    assert len(model_inputs.inputs) == 6
    assert bool(summary_inputs.inputs) == (mutation == "summarization")
    for batch in model_inputs.inputs:
        assert any(message.content == "application-owned" for message in batch)
    assert all(
        not any(
            key.startswith("langchain_response_instructions") for key in saved.metadata
        )
        for saved in saver.list(None)
    )
    unchanged = saver.get_tuple(parent.config)
    assert unchanged is not None
    assert unchanged.checkpoint == parent.checkpoint
    assert unchanged.metadata == parent.metadata
    assert graph.checkpointer is saver


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "shared-text"])
def test_formatted_messages_remain_compatible(
    enabled: bool, stream: bool, instructions: str | None
) -> None:
    class FormattedState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages(format="langchain-openai")]

    observed: list[list[str]] = []

    async def record(state: FormattedState) -> dict[str, Any]:
        observed.append(
            [
                str(message.content)
                for message in state["messages"]
                if isinstance(message, SystemMessage)
            ]
        )
        messages: list[AnyMessage] = [AIMessage(content="ok")]
        if len(observed) == 1:
            messages.append(SystemMessage(content="application-owned"))
        return {"messages": messages}

    builder = StateGraph(FormattedState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        root = _post(
            client,
            [
                {"role": "system", "content": "shared-text"},
                {"role": "developer", "content": "developer-context"},
                {"role": "user", "content": "A"},
            ],
            instructions=instructions,
            stream=stream,
        )
        child = _post(client, "B", previous_response_id=root["id"], stream=stream)

    assert root["status"] == child["status"] == "completed"
    initial = [
        *([instructions] if instructions else []),
        "shared-text",
        "developer-context",
    ]
    assert observed == [initial, [*initial, "application-owned"]]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("progress", ["none", "saved", "missing", "malformed"])
@pytest.mark.parametrize("boundary_published", [False, True])
async def test_recovery_uses_confirmed_origin_and_recorded_progress(
    enabled: bool, progress: str, boundary_published: bool
) -> None:
    graph, executions = _branch_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "root"}}
    await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent_config = (await graph.aget_state(config)).config
    parent_ref = HostingRunnableConfig(parent_config).checkpoint_ref
    assert parent_ref is not None
    saved_ref = parent_ref
    metadata: dict[str, Any] = {BRANCH_MODE_METADATA: BRANCH_MODE}
    if progress == "saved" or boundary_published:
        await graph.ainvoke({"messages": [HumanMessage(content="B")]}, parent_config)
        completed_ref = HostingRunnableConfig(
            (await graph.aget_state(config)).config
        ).checkpoint_ref
        assert completed_ref is not None
        saved_ref = completed_ref
    if progress == "saved":
        metadata[METADATA_LANGGRAPH_THREAD_ID] = saved_ref.thread_id
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = saved_ref.checkpoint_id
    elif progress == "missing":
        metadata[METADATA_LANGGRAPH_THREAD_ID] = "root"
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = "deleted-checkpoint"
    elif progress == "malformed":
        metadata[METADATA_LANGGRAPH_THREAD_ID] = "root"
    await graph.ainvoke({"messages": [HumanMessage(content="C")]}, parent_config)
    executions.clear()

    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    await server._conversation_chain_store.set(
        "child",
        BRANCH_ORIGIN_KEY,
        {
            **ResponseBranchStore._record(parent_ref, paused=False),
            "mode": BRANCH_MODE,
            "parent_response_id": "parent",
        },
    )
    if boundary_published:
        await server._conversation_chain_store.set(
            "child",
            BRANCH_BOUNDARY_KEY,
            ResponseBranchStore._record(saved_ref, paused=False),
        )
    context = _context(response_id="child", conversation_id=None, current_text="B")
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    context.is_recovery = True
    context.persisted_response = _response_object(
        "child", previous_response_id="parent", internal_metadata=metadata
    )
    provider = AsyncMock()
    context._provider = provider
    events = [
        event
        async for event in server.handle_create(
            _request(previous_response_id="parent"), context, asyncio.Event()
        )
    ]

    terminal = events[-1]["response"]
    if boundary_published or progress in {"missing", "malformed"}:
        assert terminal["status"] == "failed"
        assert terminal["error"]["code"] == "server_error"
        assert executions == []
    else:
        assert terminal["status"] == "completed", terminal
        assert executions == (["B"] if progress == "none" else [])
    provider.get_response.assert_not_awaited()


@pytest.mark.parametrize("enabled", [False, True])
async def test_recovery_preserves_application_messages(enabled: bool) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    request = _request(
        previous_response_id="parent" if enabled else None,
        conversation=None if enabled else {"id": "root"},
        instructions="task-only",
    )
    context = _context(
        response_id="child",
        conversation_id=None if enabled else "root",
        current_text="B",
    )
    graph_input = await server.build_input(request, context)
    graph_input["messages"].append(SystemMessage(content="application-owned"))
    config: RunnableConfig = {"configurable": {"thread_id": "root"}}
    await graph.ainvoke(graph_input, config, interrupt_before=["record"])
    paused = await graph.aget_state(config)
    assert paused.next == ("record",)
    saved_ref = HostingRunnableConfig(paused.config).checkpoint_ref
    assert saved_ref is not None
    metadata: dict[str, Any] = {
        METADATA_LANGGRAPH_THREAD_ID: saved_ref.thread_id,
        METADATA_LANGGRAPH_CHECKPOINT_ID: saved_ref.checkpoint_id,
    }
    if enabled:
        metadata[BRANCH_MODE_METADATA] = BRANCH_MODE
        context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
        await server._conversation_chain_store.set(
            "child",
            BRANCH_ORIGIN_KEY,
            {
                **ResponseBranchStore._record(saved_ref, paused=False),
                "mode": BRANCH_MODE,
                "parent_response_id": "parent",
            },
        )
    context.is_recovery = True
    context.persisted_response = _response_object("child", internal_metadata=metadata)
    events = [
        event async for event in server.handle_create(request, context, asyncio.Event())
    ]
    assert events[-1]["response"]["status"] == "completed", events[-1]
    assert executions == ["B"]
    recovered = await graph.aget_state(config)
    assert [
        message.content
        for message in recovered.values["messages"]
        if isinstance(message, SystemMessage)
    ] == ["task-only", "application-owned"]

    next_events = [
        event
        async for event in server.handle_create(
            _request(conversation={"id": "root"}),
            _context(response_id="next", conversation_id="root", current_text="C"),
            asyncio.Event(),
        )
    ]
    assert next_events[-1]["response"]["status"] == "completed", next_events[-1]
    assert executions == ["B", "C"]
    continued = await graph.aget_state(config)
    assert [
        message.content
        for message in continued.values["messages"]
        if isinstance(message, SystemMessage)
    ] == ["task-only", "application-owned"]


@pytest.mark.parametrize("shutdown", [False, True])
async def test_interrupted_root_is_not_replayed_or_deferred(shutdown: bool) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=False
    )
    context = _context(response_id="root", conversation_id=None)
    context.is_recovery = True
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    if shutdown:
        context.shutdown.set()
    context.persisted_response = _response_object(
        "root",
        internal_metadata={
            BRANCH_MODE_METADATA: BRANCH_MODE,
            METADATA_LANGGRAPH_THREAD_ID: "root",
            METADATA_LANGGRAPH_CHECKPOINT_ID: "partial-root",
        },
    )

    events = [
        event
        async for event in server.handle_create(_request(), context, asyncio.Event())
    ]

    assert events[-1]["response"]["error"]["code"] == "server_error"
    assert executions == []
    context.exit_for_recovery.assert_not_awaited()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_backend_failure_has_safe_consistent_response_error(
    enabled: bool, stream: bool
) -> None:
    class FailingSaver(InMemorySaver):
        async def aget_tuple(self, config: Any) -> Any:
            raise RuntimeError("private backend connection details")

    graph, executions = _branch_graph()
    graph.checkpointer = FailingSaver()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    with TestClient(server.app) as client:
        failed = _post(client, "A", stream=stream)
        retrieved = client.get(f"/responses/{failed['id']}")

    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "server_error"
    assert "private backend" not in failed["error"]["message"]
    assert retrieved.status_code == 200
    assert retrieved.json()["error"] == failed["error"]
    assert executions == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("user_id", [None, "test-user"])
def test_approval_branches_accept_independent_answers(
    stream: bool,
    user_id: str | None,
) -> None:
    server = ResponsesHostServer(
        build_simple_interrupt_graph(),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    headers = {"x-agent-user-id": user_id} if user_id else {}
    with TestClient(server.app, headers=headers) as client:
        paused = _post(client, "What is my name?", stream=stream)
        waiting = _post(
            client, "still waiting", previous_response_id=paused["id"], stream=stream
        )
        pending = next(
            item for item in paused["output"] if item["type"] == "function_call"
        )
        answer = {
            "type": "function_call_output",
            "call_id": pending["call_id"],
            "output": json.dumps({"resume": "Alice"}),
        }
        approved = _post(
            client, [answer], previous_response_id=waiting["id"], stream=stream
        )
        waiting_again = _post(
            client, "still waiting", previous_response_id=paused["id"], stream=stream
        )
        second_answer = {**answer, "output": json.dumps({"resume": "Bob"})}
        historical = _post(
            client, [second_answer], previous_response_id=paused["id"], stream=stream
        )

    assert paused["status"] == waiting["status"] == approved["status"] == "completed"
    assert _text(approved) == "ok:Alice"
    assert waiting_again["status"] == "completed", waiting_again
    assert any(item["type"] == "function_call" for item in waiting_again["output"])
    assert historical["status"] == "completed", historical
    assert _text(historical) == "ok:Bob"


@pytest.mark.parametrize("failed_read", [1, 2, 3])
@pytest.mark.parametrize("stream", [False, True])
def test_approval_can_retry_after_pre_execution_checkpoint_failure(
    failed_read: int, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions: list[str] = []

    def ask(state: _BranchState) -> dict[str, Any]:
        executions.append("ask")
        return {"messages": [AIMessage(content=f"ok:{interrupt('name?')}")]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    saver = InMemorySaver()
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    original_get = saver.aget_tuple
    reads = 0

    async def fail_once(config: RunnableConfig) -> Any:
        nonlocal reads
        reads += 1
        if reads == failed_read:
            raise TimeoutError("temporary checkpoint read failure")
        return await original_get(config)

    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        pending = next(
            item for item in paused["output"] if item["type"] == "function_call"
        )
        answers = [
            {
                "type": "function_call_output",
                "call_id": pending["call_id"],
                "output": json.dumps({"resume": "Alice"}),
            }
        ]
        executions.clear()
        monkeypatch.setattr(saver, "aget_tuple", fail_once)
        failed = _post(
            client, answers, previous_response_id=paused["id"], stream=stream
        )
        assert failed["status"] == "failed", failed
        assert failed["error"]["code"] == "server_error"
        assert executions == []
        retried = _post(
            client, answers, previous_response_id=paused["id"], stream=stream
        )
        assert retried["status"] == "completed", retried
        assert retried["id"] != failed["id"]
        assert _text(retried) == "ok:Alice"
        historical = _post(
            client, answers, previous_response_id=paused["id"], stream=stream
        )

    assert historical["status"] == "completed", historical
    assert _text(historical) == "ok:Alice"
    assert executions == ["ask", "ask"]


@pytest.mark.parametrize("failure", ["missing", "mismatch", "checkpoint", "writes"])
@pytest.mark.parametrize("stream", [False, True])
def test_paused_checkpoint_copy_fails_before_graph_execution(
    failure: str, stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions: list[str] = []

    def ask(state: _BranchState) -> dict[str, Any]:
        answer = interrupt("name?")
        executions.append(answer)
        return {"messages": [AIMessage(content=f"ok:{answer}")]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    saver = InMemorySaver()
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        pending = next(
            item for item in paused["output"] if item["type"] == "function_call"
        )
        saved = saver.get_tuple({"configurable": {"thread_id": paused["id"]}})
        assert saved is not None
        if failure in {"missing", "mismatch"}:
            invalid = (
                None
                if failure == "missing"
                else saved._replace(
                    config={
                        "configurable": {
                            **saved.config["configurable"],
                            "thread_id": "wrong-thread",
                        }
                    }
                )
            )
            monkeypatch.setattr(saver, "aget_tuple", AsyncMock(return_value=invalid))
        else:
            monkeypatch.setattr(
                saver,
                "aput" if failure == "checkpoint" else "aput_writes",
                AsyncMock(side_effect=OSError("private copy details")),
            )
        failed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps({"resume": "Alice"}),
                }
            ],
            previous_response_id=paused["id"],
            stream=stream,
        )
        assert client.portal is not None
        origin = client.portal.call(
            server._conversation_chain_store.get, failed["id"], BRANCH_ORIGIN_KEY
        )

    assert failed["status"] == "failed", failed
    assert failed["error"]["code"] == "server_error"
    assert "private copy details" not in json.dumps(failed)
    assert origin is None
    assert executions == []
    assert saver.get_tuple(saved.config) == saved


async def test_paused_origin_recovery_reuses_private_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = build_simple_interrupt_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    config: RunnableConfig = {"configurable": {"thread_id": "parent"}}
    await graph.ainvoke({"messages": [HumanMessage(content="start")]}, config)
    parent = await graph.aget_state(config)
    parent_ref = HostingRunnableConfig(parent.config).checkpoint_ref
    assert parent_ref is not None
    boundary = ResponseBranchStore._record(parent_ref, paused=True)
    records = {("parent", BRANCH_BOUNDARY_KEY): boundary}
    store = MagicMock()
    store.get = AsyncMock(side_effect=lambda key, kind: records.get((key, kind)))
    store.set = AsyncMock(
        side_effect=lambda key, kind, value: records.__setitem__((key, kind), value)
    )
    provider = MagicMock()
    provider.get_response = AsyncMock(
        return_value={
            "status": "completed",
            "metadata": {"_internal_metadata": {BRANCH_BOUNDARY_KEY: boundary}},
        }
    )
    branches = ResponseBranchStore(store, provider)
    context = _context(response_id="child", conversation_id=None)
    origin = await branches.prepare(
        response_key="child",
        parent_key="parent",
        parent_id="parent",
        context=context,
        saver=saver,
    )
    assert origin.thread_id == "child"
    assert origin.checkpoint_id == parent_ref.checkpoint_id
    assert records["child", BRANCH_ORIGIN_KEY] == {
        **ResponseBranchStore._record(origin, paused=True),
        "mode": BRANCH_MODE,
        "parent_response_id": "parent",
    }
    copied = await graph.aget_state(
        {"configurable": {**origin.to_dict(), "checkpoint_ns": ""}}
    )
    assert [task.interrupts for task in copied.tasks] == [
        task.interrupts for task in parent.tasks
    ]
    assert await graph.aget_state(parent.config) == parent
    context.is_recovery = True
    monkeypatch.setattr(saver, "aput", AsyncMock(side_effect=AssertionError("recopy")))
    restored = await branches.prepare(
        response_key="child",
        parent_key="parent",
        parent_id="parent",
        context=context,
        saver=saver,
    )
    assert restored == origin
    provider.get_response.assert_awaited_once()
    store.set.assert_awaited_once()


@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_approval_branches_have_independent_answers(stream: bool) -> None:
    executions: list[str] = []
    barrier = asyncio.Barrier(2)

    async def ask(state: _BranchState) -> dict[str, Any]:
        answer = interrupt("name?")
        await asyncio.wait_for(barrier.wait(), timeout=5)
        executions.append(answer)
        return {"messages": [AIMessage(content=f"ok:{answer}")]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        pending = next(
            item for item in paused["output"] if item["type"] == "function_call"
        )
        assert executions == []

        def approve(answer: str) -> dict[str, Any]:
            return _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": pending["call_id"],
                        "output": json.dumps({"resume": answer}),
                    }
                ],
                previous_response_id=paused["id"],
                stream=stream,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(approve, ["Alice", "Bob"]))

    assert all(response["status"] == "completed" for response in responses)
    assert [_text(response) for response in responses] == ["ok:Alice", "ok:Bob"]
    assert sorted(executions) == ["Alice", "Bob"]


@pytest.mark.parametrize(
    "builder", [build_sequential_interrupt_graph, build_parallel_interrupt_graph]
)
@pytest.mark.parametrize("user_id", [None, "test-user"])
def test_partial_approvals_continue_from_the_new_pause(
    builder: Any, user_id: str | None
) -> None:
    server = ResponsesHostServer(
        builder(), store=InMemoryResponseProvider(), enable_response_branching=True
    )
    headers = {"x-agent-user-id": user_id} if user_id else {}
    with TestClient(server.app, headers=headers) as client:
        paused = _post(client, "start")
        original_pause = paused
        for answer in ("Alice", "Paris"):
            pending = next(
                item for item in paused["output"] if item["type"] == "function_call"
            )
            approved = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": pending["call_id"],
                        "output": json.dumps({"resume": answer}),
                    }
                ],
                previous_response_id=paused["id"],
            )
            assert approved["status"] == "completed", approved
            paused = approved
        old = _post(client, "waiting", previous_response_id=original_pause["id"])

    assert not any(item["type"] == "function_call" for item in paused["output"])
    assert old["status"] == "completed", old
    assert any(item["type"] == "function_call" for item in old["output"])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_parallel_approval_updates_preserve_messages(
    enabled: bool, stream: bool
) -> None:
    graph = build_parallel_interrupt_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        pending = [item for item in paused["output"] if item["type"] == "function_call"]
        answers = [
            {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": json.dumps(
                    {
                        "resume": answer,
                        "update": {
                            "messages": [
                                {"type": "system", "content": f"explicit:{answer}"}
                            ]
                        },
                    }
                ),
            }
            for item, answer in zip(pending, ("Alice", "Paris"), strict=True)
        ]
        approved = _post(
            client,
            answers,
            previous_response_id=paused["id"],
            stream=stream,
        )

    assert approved["status"] == "completed", approved
    assert "a=Alice" in _text(approved)
    assert "b=Paris" in _text(approved)
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    checkpoint = (
        saver.get_tuple({"configurable": {"thread_id": approved["id"]}})
        if enabled
        else next(saver.list(None))
    )
    assert checkpoint is not None
    system_messages = [
        message.content
        for message in checkpoint.checkpoint["channel_values"]["messages"]
        if isinstance(message, SystemMessage)
    ]
    assert system_messages == ["explicit:Alice", "explicit:Paris"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "message_update",
    [
        {"role": "user", "content": "edited-input"},
        "edited-input",
        [{"role": "user", "content": "edited-input"}],
    ],
    ids=["object", "text", "list"],
)
def test_approval_preserves_message_update_shapes(
    enabled: bool,
    stream: bool,
    message_update: Any,
) -> None:
    graph = build_simple_interrupt_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        pending = next(
            item for item in paused["output"] if item["type"] == "function_call"
        )
        approved = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps(
                        {"resume": "Alice", "update": {"messages": message_update}}
                    ),
                }
            ],
            previous_response_id=paused["id"],
            stream=stream,
        )

    assert approved["status"] == "completed", approved
    assert _text(approved) == "ok:Alice"
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    checkpoint = (
        saver.get_tuple({"configurable": {"thread_id": approved["id"]}})
        if enabled
        else next(saver.list(None))
    )
    assert checkpoint is not None
    messages = checkpoint.checkpoint["channel_values"]["messages"]
    assert [message.content for message in messages] == [
        "start",
        "edited-input",
        "ok:Alice",
    ]


def test_sdk_validation_allows_same_identity_retry() -> None:
    graph, executions = _branch_graph()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    request = {"model": "test", "input": "A", "store": False, "background": True}
    headers = {"x-agent-response-id": "caresp_" + "a" * 18 + "b" * 32}

    with TestClient(server.app) as client:
        rejected = client.post("/responses", json=request, headers=headers)
        assert rejected.status_code == 400, rejected.text
        assert executions == []
        retried = client.post(
            "/responses",
            json={"model": "test", "input": "A", "store": True},
            headers=headers,
        )

    assert retried.status_code == 200, retried.text
    assert retried.json()["status"] == "completed", retried.text
    assert executions == ["A"]


def test_repeated_response_identity_uses_sdk_admission() -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        root = _post(client, "A")
        duplicate = client.post(
            "/responses",
            json={"input": "B", "model": "test", "store": True},
            headers={"x-agent-response-id": root["id"]},
        )

    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["id"] == root["id"]
    assert executions == ["A", "B"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_failed_response_retry_is_passed_to_sdk(stream: bool, stored: bool) -> None:
    executions: list[str] = []

    async def fail(state: _BranchState) -> dict[str, Any]:
        executions.append("side-effect")
        raise RuntimeError("private execution details")

    builder = StateGraph(_BranchState)
    builder.add_node("fail", fail)
    builder.add_edge(START, "fail")
    builder.add_edge("fail", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    headers = {"x-agent-response-id": "caresp_" + "a" * 18 + "b" * 32}
    payload = {"model": "test", "input": "A", "store": stored, "stream": stream}
    with TestClient(server.app) as client:
        failed = client.post("/responses", json=payload, headers=headers)
        assert executions == ["side-effect"]
        retry = client.post("/responses", json=payload, headers=headers)

    assert failed.status_code == 200, failed.text
    if stream:
        assert any(kind == "response.failed" for kind, _ in _parse_sse(failed.text))
    else:
        assert failed.json()["status"] == "failed", failed.text
    assert retry.status_code == 200, retry.text
    assert "private execution details" not in failed.text


def test_inferred_conversation_does_not_override_explicit_parent() -> None:
    graph, _ = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    context = _context(conversation_id="inferred-conversation")

    assert server._uses_response_branching(
        _request(previous_response_id="parent"), context
    )
    assert not server._uses_response_branching(
        _request(conversation={"id": "explicit-conversation"}), context
    )


@pytest.mark.parametrize("stored", [None, False, True])
def test_storage_policy_controls_branch_publication(stored: bool | None) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        request: dict[str, Any] = {"model": "test", "input": "A"}
        if stored is not None:
            request["store"] = stored
        response = client.post("/responses", json=request)
        assert response.status_code == 200, response.text
        root = response.json()
        assert root["status"] == "completed", root
        retrieved = client.get(f"/responses/{root['id']}")
        assert client.portal is not None
        boundary = client.portal.call(
            server._conversation_chain_store.get, root["id"], BRANCH_BOUNDARY_KEY
        )

    assert executions == ["A"]
    if stored is False:
        assert retrieved.status_code == 404
        assert boundary is None
    else:
        assert retrieved.status_code == 200
        assert boundary is not None


@pytest.mark.parametrize("mode", [None, "unknown-mode"])
async def test_recovery_cannot_fall_back_when_recorded_mode_is_invalid(
    mode: str | None,
) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    context = _context(response_id="child", conversation_id=None)
    context.is_recovery = True
    context.client_headers = {}
    if mode is not None:
        context.client_headers[BRANCH_MODE_HEADER] = mode
    context.persisted_response = _response_object(
        "child",
        previous_response_id="parent",
        internal_metadata={BRANCH_MODE_METADATA: BRANCH_MODE},
    )
    events = [
        event
        async for event in server.handle_create(
            _request(previous_response_id="parent"), context, asyncio.Event()
        )
    ]

    assert events[-1]["response"]["status"] == "failed"
    assert executions == []


def test_foundry_identity_preserves_platform_context_and_trusted_mode() -> None:
    from azure.ai.agentserver.core import get_request_context
    from azure.ai.agentserver.responses import FoundryResourceNotFoundError

    observed: list[tuple[str | None, str | None]] = []

    class FoundryLikeProvider(InMemoryResponseProvider):
        async def get_response(self, response_id: str, *, context: Any = None) -> Any:
            observed.append(
                (get_request_context().user_id, get_request_context().call_id)
            )
            try:
                return await super().get_response(response_id, context=context)
            except KeyError as exc:
                raise FoundryResourceNotFoundError("Not found") from exc

    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=FoundryLikeProvider(), enable_response_branching=True
    )
    response_id = "caresp_" + "c" * 18 + "d" * 32
    with TestClient(server.app) as client:
        response = client.post(
            "/responses",
            json={"model": "test", "input": "A"},
            headers={
                "x-agent-response-id": response_id,
                "x-agent-user-id": "test-user",
                "x-agent-foundry-call-id": "test-call",
                BRANCH_MODE_HEADER: "forged-mode",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "completed"
        child = client.post(
            "/responses",
            json={"model": "test", "input": "B", "previous_response_id": response_id},
            headers={
                "x-agent-user-id": "test-user",
                "x-agent-foundry-call-id": "test-call",
                BRANCH_MODE_HEADER: "forged-mode",
            },
        )
        assert child.status_code == 200, child.text
        assert child.json()["status"] == "completed", child.text

    assert observed and all(
        context == ("test-user", "test-call") for context in observed
    )
    assert executions == ["A", "B"]


@pytest.mark.parametrize("failed_key", [BRANCH_ORIGIN_KEY, BRANCH_BOUNDARY_KEY])
def test_failed_branch_writes_do_not_publish_a_usable_parent(
    failed_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    original_set = server._conversation_chain_store.set
    failed_once = False

    async def fail_once(identity: str, key: str, data: dict[str, str]) -> None:
        nonlocal failed_once
        if key == failed_key and not failed_once:
            failed_once = True
            raise OSError("private storage details")
        await original_set(identity, key, data)

    with TestClient(server.app) as client:
        root = _post(client, "A")
        monkeypatch.setattr(server._conversation_chain_store, "set", fail_once)
        failed = _post(client, "B", previous_response_id=root["id"])
        assert failed["status"] == "failed"
        assert failed["error"]["code"] == "server_error"
        assert executions == (["A"] if failed_key == BRANCH_ORIGIN_KEY else ["A", "B"])
        unavailable = _post(client, "bad-child", previous_response_id=failed["id"])
        assert unavailable["status"] == "failed"
        fork = _post(client, "C", previous_response_id=root["id"])

    assert _text(fork) == "A,C"
    assert "bad-child" not in executions


@pytest.mark.parametrize("stream", [False, True])
def test_failed_terminal_persistence_does_not_make_indexed_response_a_parent(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, executions = _branch_graph()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    terminal_writes: list[str] = []

    def fail_terminal_write(
        write: Callable[..., Awaitable[None]],
    ) -> Callable[..., Awaitable[None]]:
        async def persist(response: Any, *args: Any, **kwargs: Any) -> None:
            if response.get("status") == "completed" and not terminal_writes:
                response_id = str(response["id"])
                boundary = await server._conversation_chain_store.get(
                    response_id, BRANCH_BOUNDARY_KEY
                )
                assert boundary is not None
                terminal_writes.append(response_id)
                raise OSError("private storage details")
            await write(response, *args, **kwargs)

        return persist

    with TestClient(server.app) as client:
        root = _post(client, "A")
        for method in ("create_response", "update_response"):
            monkeypatch.setattr(
                provider, method, fail_terminal_write(getattr(provider, method))
            )
        failed = client.post(
            "/responses",
            json={
                "model": "test",
                "input": "B",
                "previous_response_id": root["id"],
                "store": True,
                "stream": stream,
            },
        )
        assert failed.status_code in {200, 500}, failed.text
        assert "private storage details" not in failed.text
        assert len(terminal_writes) == 1
        failed_id = terminal_writes[0]
        assert executions == ["A", "B"]
        retrieved = client.get(f"/responses/{failed_id}")
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["status"] == "failed"
        assert retrieved.json()["error"]["code"] == "storage_error"
        assert "private storage details" not in retrieved.text
        unavailable = _post(client, "bad-child", previous_response_id=failed_id)
        assert unavailable["status"] == "failed"
        fork = _post(client, "C", previous_response_id=root["id"])

    assert _text(fork) == "A,C"
    assert executions == ["A", "B", "C"]


async def test_persisted_parent_survives_host_and_saver_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._responses."
        "conversation_chain_store.FoundryStateStore",
        FoundryStateStore,
    )
    database = str(tmp_path / "checkpoints.sqlite")
    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        graph, _ = _branch_graph()
        graph.checkpointer = saver
        server = ResponsesHostServer(graph, enable_response_branching=True)
        with TestClient(server.app) as client:
            root = _post(client, "A")
            original = _post(client, "B", previous_response_id=root["id"])
            assert _text(original) == "A,B"

    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        graph, executions = _branch_graph()
        graph.checkpointer = saver
        recreated = ResponsesHostServer(graph, enable_response_branching=True)
        with TestClient(recreated.app) as client:
            fork = _post(client, "C", previous_response_id=root["id"])
            next_original = _post(client, "D", previous_response_id=original["id"])

    assert _text(fork) == "A,C"
    assert _text(next_original) == "A,B,D"
    assert executions == ["C", "D"]
