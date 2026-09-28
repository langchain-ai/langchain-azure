"""Opt-in response checkpoint branching and strict restoration tests."""

from __future__ import annotations

import asyncio
import json
import operator
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses.store._memory import InMemoryResponseProvider
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
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
    BRANCH_OWNER_HEADER,
    ResponseBranchStore,
    ResponseExecutionStore,
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


def test_branching_rejects_unimplemented_async_saver() -> None:
    graph, _ = _branch_graph()
    graph.checkpointer = BaseCheckpointSaver()
    with pytest.raises(ValueError, match="asynchronous checkpoint"):
        ResponsesHostServer(
            graph, store=InMemoryResponseProvider(), enable_response_branching=True
        )


@pytest.mark.parametrize("enabled", [False, True])
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
    enabled: bool, fields: dict[str, Any], parameter: str
) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
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


@pytest.mark.parametrize("operation", ["state", "execute"])
async def test_strict_saver_rejects_checkpoint_deleted_after_preflight(
    operation: str,
) -> None:
    from langchain_azure_ai.agents.hosting._responses.branching import (
        BranchingError,
        StrictCheckpointSaver,
    )

    graph, executions = _branch_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "parent"}}
    await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent = (await graph.aget_state(config)).config
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    guarded = graph.copy({"checkpointer": StrictCheckpointSaver(saver)})
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
    from langchain_azure_ai.agents.hosting._responses.branching import (
        StrictCheckpointSaver,
    )

    graph, _ = _branch_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    guarded = graph.copy({"checkpointer": StrictCheckpointSaver(saver)})
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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_instructions_are_request_local(enabled: bool, stream: bool) -> None:
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
        ["shared-text", "developer-context", "application-owned"],
        ["shared-text", "developer-context", "application-owned"],
        ["shared-text", "developer-context", "application-owned", "child-only"],
    ]


async def test_ambiguous_legacy_instructions_fail_without_graph_execution() -> None:
    graph, executions = _branch_graph()
    await graph.ainvoke(
        {"messages": [SystemMessage(content="ambiguous"), HumanMessage(content="A")]},
        {"configurable": {"thread_id": "legacy"}},
    )
    executions.clear()
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    context = _context(conversation_id="legacy", current_text="B")
    events = [
        event
        async for event in server.handle_create(
            _request(conversation={"id": "legacy"}), context, asyncio.Event()
        )
    ]

    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["code"] == "server_error"
    assert executions == []


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("progress", ["none", "saved", "missing", "malformed"])
async def test_recovery_uses_confirmed_origin_and_recorded_progress(
    enabled: bool, progress: str
) -> None:
    graph, executions = _branch_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "root"}}
    await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent_config = (await graph.aget_state(config)).config
    parent_ref = HostingRunnableConfig(parent_config).checkpoint_ref
    assert parent_ref is not None
    metadata: dict[str, Any] = {BRANCH_MODE_METADATA: BRANCH_MODE}
    if progress == "saved":
        await graph.ainvoke({"messages": [HumanMessage(content="B")]}, parent_config)
        saved_ref = HostingRunnableConfig(
            (await graph.aget_state(config)).config
        ).checkpoint_ref
        assert saved_ref is not None
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
    context = _context(response_id="child", conversation_id=None, current_text="B")
    context.client_headers = {
        BRANCH_MODE_HEADER: BRANCH_MODE,
        BRANCH_OWNER_HEADER: "admitted-child",
    }
    await server._branch_executions.claim(
        "response",
        server._branch_executions.response_identity("child", context.platform_context),
        "admitted-child",
    )
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
    if progress in {"missing", "malformed"}:
        assert terminal["status"] == "failed"
        assert terminal["error"]["code"] == "server_error"
        assert executions == []
    else:
        assert terminal["status"] == "completed", terminal
        assert executions == (["B"] if progress == "none" else [])
    provider.get_response.assert_not_awaited()


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
def test_normal_approval_and_waiting_preserved_but_second_answer_rejected(
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
        second_answer = {**answer, "output": json.dumps({"resume": "Bob"})}
        historical = _post(
            client, [second_answer], previous_response_id=paused["id"], stream=stream
        )

    assert paused["status"] == waiting["status"] == approved["status"] == "completed"
    assert _text(approved) == "ok:Alice"
    assert historical["status"] == "failed"
    assert historical["error"]["code"] == "server_error"


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
    assert old["status"] == "failed"


def test_duplicate_response_identity_never_runs_graph_twice() -> None:
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

    assert duplicate.status_code in {200, 400, 409}
    assert executions == ["A"]
    if duplicate.status_code == 200:
        assert duplicate.json()["id"] == root["id"]
        assert _text(duplicate.json()) == "A"


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


def test_concurrent_response_identity_has_one_execution_owner() -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    response_id = "caresp_" + "a" * 18 + "b" * 32
    barrier = threading.Barrier(2)
    with TestClient(server.app) as client:

        def create(text: str) -> Any:
            barrier.wait(timeout=5)
            return client.post(
                "/responses",
                json={"input": text, "model": "test", "store": True},
                headers={"x-agent-response-id": response_id},
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(create, ["A", "B"]))
        saved = client.get(f"/responses/{response_id}")

    assert sorted(response.status_code for response in responses) == [200, 409]
    assert len(executions) == 1
    assert _text(saved.json()) == executions[0]


async def test_execution_claims_survive_instances_and_partition_users(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore
    from azure.ai.agentserver.responses import PlatformContext

    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._responses.branching.FoundryStateStore",
        FoundryStateStore,
    )
    first = ResponseExecutionStore()
    second = ResponseExecutionStore()
    identity = first.response_identity(
        "same-response", PlatformContext(user_id_key="A")
    )
    outcomes = await asyncio.gather(
        first.claim("response", identity, "first-owner"),
        second.claim("response", identity, "second-owner"),
    )
    assert sum(outcomes) == 1
    owner = "first-owner" if outcomes[0] else "second-owner"
    assert await ResponseExecutionStore().owner("response", identity) == owner
    assert await second.claim("response", identity, owner)
    other_identity = first.response_identity(
        "same-response", PlatformContext(user_id_key="B")
    )
    assert await second.claim("response", other_identity, "other-user")


@pytest.mark.parametrize("mode", [None, "unknown-mode"])
async def test_recovery_cannot_fall_back_when_admission_mode_is_invalid(
    mode: str | None,
) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    context = _context(response_id="child", conversation_id=None)
    context.is_recovery = True
    context.client_headers = {BRANCH_OWNER_HEADER: "owner"}
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


def test_foundry_identity_admission_preserves_platform_context() -> None:
    from azure.ai.agentserver.core import get_request_context
    from azure.ai.agentserver.responses import (
        FoundryResourceNotFoundError,
        PlatformContext,
    )

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
                BRANCH_OWNER_HEADER: "forged-owner",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "completed"
        assert client.portal is not None
        identity = server._branch_executions.response_identity(
            response_id, PlatformContext(user_id_key="test-user")
        )
        owner = client.portal.call(
            server._branch_executions.owner, "response", identity
        )

    assert observed[0] == ("test-user", "test-call")
    assert owner and owner != "forged-owner"
    assert executions == ["A"]


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


async def test_persisted_parent_survives_host_and_saver_recreation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    for module in ("branching", "conversation_chain_store"):
        monkeypatch.setattr(
            f"langchain_azure_ai.agents.hosting._responses.{module}.FoundryStateStore",
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
