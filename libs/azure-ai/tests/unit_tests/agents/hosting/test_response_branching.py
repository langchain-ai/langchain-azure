"""Completed-parent branching without approval or tool-result resume semantics."""

from __future__ import annotations

import asyncio
import json
import operator
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseEventStream,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.models import CreateResponse
from azure.ai.agentserver.responses.store._memory import InMemoryResponseProvider
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langgraph import channels
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    empty_checkpoint,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from starlette.testclient import TestClient
from starlette.types import Scope
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._responses import (
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    CheckpointRef,
    HostingRunnableConfig,
)
from langchain_azure_ai.agents.hosting._responses.branching import (
    BRANCH_BOUNDARY_KEY,
    BRANCH_MODE,
    BRANCH_MODE_HEADER,
    BRANCH_MODE_METADATA,
    BranchingAdmissionMiddleware,
    BranchingError,
    ResponseCheckpointSaver,
)
from langchain_azure_ai.agents.hosting._responses.task_storage_manager import (
    TaskStorageManager,
)

from .conftest import make_echo_graph
from .hitl.graphs import build_parallel_interrupt_graph
from .test_responses_host import _context, _parse_sse, _request, _response_object


@pytest.fixture(autouse=True)
def _isolate_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        "azure.ai.agentserver.core._tracing._configure_tracing", lambda *_, **__: None
    )


def _post(
    client: TestClient,
    value: Any,
    *,
    previous_response_id: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    request = {"model": "test", "input": value, "stream": stream, "store": True}
    if previous_response_id is not None:
        request["previous_response_id"] = previous_response_id
    response = client.post("/responses", json={**request, **kwargs})
    assert response.status_code == 200, response.text
    if stream:
        events = _parse_sse(response.text)
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


def _boundary(
    client: TestClient, server: ResponsesHostServer, response_id: str
) -> CheckpointRef:
    assert client.portal is not None
    assert server._response_provider is not None
    response = client.portal.call(server._response_provider.get_response, response_id)
    metadata = _metadata(response)
    assert BRANCH_BOUNDARY_KEY in metadata
    ref = TaskStorageManager(metadata).checkpoint_ref
    assert ref is not None
    return ref


def _metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    metadata = (response.get("metadata") or {}).get("_internal_metadata") or {}
    return json.loads(metadata) if isinstance(metadata, str) else dict(metadata)


def _config(ref: CheckpointRef) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": ref.thread_id,
            "checkpoint_id": ref.checkpoint_id,
            "checkpoint_ns": "",
        }
    }


class _BranchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ledger: Annotated[list[str], operator.add]


def _branch_graph(
    *, tool_results: bool | None = None
) -> tuple[CompiledStateGraph, list[str]]:
    executions: list[str] = []

    async def record(state: _BranchState) -> dict[str, Any]:
        text = next(
            str(message.content)
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        executions.append(text)
        messages: list[AnyMessage] = []
        if tool_results is not None:
            call_id = f"call-{len(executions)}"
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lookup", "args": {}, "id": call_id}],
                )
            )
            if tool_results:
                messages.append(ToolMessage(content="found", tool_call_id=call_id))
        if tool_results is not False:
            messages.append(
                AIMessage(content=",".join([*state.get("ledger", []), text]))
            )
        return {"ledger": [text], "messages": messages}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=InMemorySaver()), executions


@pytest.mark.parametrize("stream", [False, True])
def test_completed_branches_preserve_exact_parent_state(stream: bool) -> None:
    graph, executions = _branch_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        a = _post(client, "A", stream=stream)
        parent_ref = _boundary(client, server, a["id"])
        parent = saver.get_tuple(_config(parent_ref))
        assert parent is not None
        original_parent = deepcopy((parent.checkpoint, parent.pending_writes))
        b = _post(client, "B", previous_response_id=a["id"], stream=stream)
        c = _post(client, "C", previous_response_id=a["id"], stream=stream)
        d = _post(client, "D", previous_response_id=b["id"], stream=stream)
        e = _post(client, "E", previous_response_id=c["id"], stream=stream)
        regenerated = _post(client, "B", previous_response_id=a["id"], stream=stream)
        for response, parent_id in (
            (a, None),
            (b, a["id"]),
            (c, a["id"]),
            (d, b["id"]),
            (e, c["id"]),
            (regenerated, a["id"]),
        ):
            assert response["status"] == "completed", response
            assert response.get("previous_response_id") == parent_id
            assert _boundary(client, server, response["id"]).thread_id == (
                parent_ref.thread_id
            )
            retrieved = client.get(f"/responses/{response['id']}")
            assert retrieved.status_code == 200
            assert _text(retrieved.json()) == _text(response)
            assert "_internal_metadata" not in response.get("metadata", {})
            assert "_internal_metadata" not in retrieved.json().get("metadata", {})
        retained = saver.get_tuple(_config(parent_ref))
        assert retained is not None
        assert (retained.checkpoint, retained.pending_writes) == original_parent

    assert [_text(item) for item in (a, b, c, d, e, regenerated)] == [
        "A",
        "A,B",
        "A,C",
        "A,B,D",
        "A,C,E",
        "A,B",
    ]
    assert regenerated["id"] != b["id"]
    assert executions == ["A", "B", "C", "D", "E", "B"]


@pytest.mark.parametrize("stream", [False, True])
def test_branching_does_not_use_conversation_chain_storage(stream: bool) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with (
        patch.object(
            server._conversation_chain_store,
            "get",
            side_effect=AssertionError("Branching must use response metadata."),
        ) as reads,
        patch.object(
            server._conversation_chain_store,
            "set",
            side_effect=AssertionError("Branching must not persist a separate index."),
        ) as writes,
        TestClient(server.app) as client,
    ):
        a = _post(client, "A", stream=stream)
        b = _post(client, "B", previous_response_id=a["id"], stream=stream)
        c = _post(client, "C", previous_response_id=a["id"], stream=stream)
    assert [response["status"] for response in (a, b, c)] == ["completed"] * 3
    assert [_text(response) for response in (a, b, c)] == ["A", "A,B", "A,C"]
    assert executions == ["A", "B", "C"]
    reads.assert_not_awaited()
    writes.assert_not_awaited()


@pytest.mark.parametrize("boundary_saved", [False, True])
async def test_completed_branch_recovery_does_not_require_parent_or_branch_records(
    boundary_saved: bool,
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    graph, executions = _branch_graph()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _post(client, "A")
        child = _post(client, "C", previous_response_id=parent["id"])
        assert child["status"] == "completed", child
        persisted = await provider.get_response(child["id"])
        await provider.delete_response(parent["id"])
        persisted["status"] = "in_progress"
        persisted_stream = ResponseEventStream(
            response_id=child["id"], response=persisted
        )
        if not boundary_saved:
            persisted_stream.internal_metadata.pop(BRANCH_BOUNDARY_KEY)
        foundry_state_stores.clear()
        executions.clear()
        context = _context(
            response_id=child["id"], conversation_id=None, current_text="do not replay"
        )
        context.is_recovery = True
        context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
        context.persisted_response = persisted_stream.checkpoint().response
        with patch.object(
            provider,
            "get_response",
            side_effect=AssertionError("Owned progress must not require its parent."),
        ) as lookup:
            events = [
                event
                async for event in server.handle_create(
                    _request(previous_response_id=parent["id"]),
                    context,
                    asyncio.Event(),
                )
            ]
    terminal = events[-1]["response"]
    assert terminal["status"] == "completed", terminal
    assert _text(terminal) == "A,C"
    assert executions == []
    lookup.assert_not_awaited()
    context.get_input_items.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "reply",
    ["text", "partial_approval", "all_approvals", "partial_output", "all_outputs"],
)
def test_paused_parent_is_rejected_without_resuming(stream: bool, reply: str) -> None:
    graph = build_parallel_interrupt_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        assert parent["status"] == "completed", parent
        ref = _boundary(client, server, parent["id"])
        saved = saver.get_tuple(_config(ref))
        assert saved is not None
        original = deepcopy((saved.checkpoint, saved.pending_writes))
        approvals = [
            item for item in parent["output"] if item["type"] == "mcp_approval_request"
        ]
        calls = [item for item in parent["output"] if item["type"] == "function_call"]
        assert len(approvals) == len(calls) == 2
        value: Any = "do not process this text"
        if "approval" in reply:
            value = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": item["id"],
                    "approve": True,
                }
                for item in (
                    approvals[:1] if reply == "partial_approval" else approvals
                )
            ]
        elif "output" in reply:
            value = [
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": json.dumps({"resume": "answer"}),
                }
                for item in (calls[:1] if reply == "partial_output" else calls)
            ]
        child = _post(client, value, previous_response_id=parent["id"], stream=stream)
        assert child["status"] == "failed", child
        assert "paused parent" in child["error"]["message"]
        retained = saver.get_tuple(_config(ref))
        assert retained is not None
        assert (retained.checkpoint, retained.pending_writes) == original
        assert client.portal is not None
        assert server._response_provider is not None
        stored = client.portal.call(server._response_provider.get_response, child["id"])
        assert BRANCH_BOUNDARY_KEY not in _metadata(stored)


@pytest.mark.parametrize("stream", [False, True])
def test_static_pause_is_not_mistaken_for_a_finished_graph(stream: bool) -> None:
    graph, executions = _branch_graph()
    graph.interrupt_before_nodes = ["record"]
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "A", stream=stream)
        assert parent["status"] == "completed", parent
        assert not any(
            item["type"] == "mcp_approval_request" for item in parent["output"]
        )
        child = _post(client, "B", previous_response_id=parent["id"], stream=stream)
    assert child["status"] == "failed", child
    assert "paused parent" in child["error"]["message"]
    assert executions == []


@pytest.mark.parametrize("explicit_conversation", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_existing_partial_approval_path_is_unchanged(
    explicit_conversation: bool, stream: bool
) -> None:
    server = ResponsesHostServer(
        build_parallel_interrupt_graph(),
        store=InMemoryResponseProvider(),
        enable_response_branching=explicit_conversation,
    )
    kwargs: dict[str, Any] = (
        {"conversation": {"id": "conv-partial"}} if explicit_conversation else {}
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream, **kwargs)
        call = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        if not explicit_conversation:
            kwargs["previous_response_id"] = parent["id"]
        partial = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps({"resume": "answer"}),
                }
            ],
            stream=stream,
            **kwargs,
        )
    assert partial["status"] == "completed", partial
    assert any(item["type"] == "function_call" for item in partial["output"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("completed_tools", [False, True])
def test_tool_call_completion_is_checked_even_without_interrupts(
    stream: bool, completed_tools: bool
) -> None:
    graph, executions = _branch_graph(tool_results=completed_tools)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "A", stream=stream)
        assert parent["status"] == "completed", parent
        ref = _boundary(client, server, parent["id"])
        assert client.portal is not None
        assert not client.portal.call(graph.aget_state, _config(ref)).next
        child = _post(client, "B", previous_response_id=parent["id"], stream=stream)
    assert child["status"] == ("completed" if completed_tools else "failed"), child
    assert executions == (["A", "B"] if completed_tools else ["A"])
    if completed_tools:
        assert _text(child) == "A,B"
    else:
        assert "pending approvals or tool results" in child["error"]["message"]


@pytest.mark.parametrize("stream", [False, True])
def test_completed_branches_restore_delta_channel_history(stream: bool) -> None:
    pytest.importorskip("langgraph.channels.delta")

    class DeltaState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        ledger: Annotated[
            list[str],
            channels.DeltaChannel(
                lambda value, writes: (
                    value + [entry for write in writes for entry in write]
                )
            ),
        ]

    builder = StateGraph(DeltaState)
    builder.add_node(
        "record", lambda state: {"ledger": [str(state["messages"][-1].content)]}
    )
    builder.add_node(
        "finish",
        lambda state: {"messages": [AIMessage(content=",".join(state["ledger"]))]},
    )
    builder.add_edge(START, "record")
    builder.add_edge("record", "finish")
    builder.add_edge("finish", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        a = _post(client, "A", stream=stream)
        b = _post(client, "B", previous_response_id=a["id"], stream=stream)
        c = _post(client, "C", previous_response_id=a["id"], stream=stream)
    assert [_text(item) for item in (a, b, c)] == ["A", "A,B", "A,C"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("parent", [False, True])
@pytest.mark.parametrize("kind", ["function_call_output", "mcp_approval_response"])
def test_branching_does_not_accept_resume_inputs(
    stream: bool, parent: bool, kind: str
) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    item = (
        {"type": kind, "call_id": "unknown", "output": "result"}
        if kind == "function_call_output"
        else {"type": kind, "approval_request_id": "unknown", "approve": True}
    )
    with TestClient(server.app) as client:
        a = _post(client, "A") if parent else None
        response = _post(
            client,
            [item],
            previous_response_id=a["id"] if a else None,
            stream=stream,
        )
    assert response["status"] == "failed", response
    assert "not supported by response branching" in response["error"]["message"]
    assert executions == (["A"] if parent else [])


@pytest.mark.parametrize(
    "fields",
    [
        {"previous_response_id": ""},
        {"previous_response_id": 42},
        {"previous_response_id": "parent", "conversation": "conversation"},
        {"response_id": "client-selected"},
    ],
)
def test_invalid_linkage_is_rejected_before_execution(fields: dict[str, Any]) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        response = client.post("/responses", json={"input": "A", **fields})
    assert response.status_code == 400, response.text
    assert executions == []


@pytest.mark.parametrize("checkpointer", [None, False, True])
def test_branching_requires_a_real_checkpoint_saver(
    checkpointer: bool | None,
) -> None:
    graph = make_echo_graph()
    graph.checkpointer = checkpointer
    with pytest.raises(ValueError, match="requires.*checkpoint saver"):
        ResponsesHostServer(graph, enable_response_branching=True)


def test_branching_requires_implemented_async_checkpoint_methods() -> None:
    graph, _ = _branch_graph()
    graph.checkpointer = BaseCheckpointSaver()
    with pytest.raises(ValueError, match="asynchronous checkpoint"):
        ResponsesHostServer(graph, enable_response_branching=True)


@pytest.mark.parametrize("configuration", ["steering", "app"])
def test_branching_rejects_unsupported_host_configuration(
    configuration: str,
) -> None:
    graph, _ = _branch_graph()
    kwargs: dict[str, Any] = (
        {"options": ResponsesServerOptions(steerable_conversations=True)}
        if configuration == "steering"
        else {"app": object()}
    )
    with pytest.raises(ValueError, match="does not support"):
        ResponsesHostServer(graph, enable_response_branching=True, **kwargs)


@pytest.mark.parametrize("stream", [False, True])
def test_legacy_parent_is_not_silently_reinterpreted(stream: bool) -> None:
    graph, executions = _branch_graph()
    provider = InMemoryResponseProvider()
    legacy = ResponsesHostServer(graph, store=provider)
    with TestClient(legacy.app) as client:
        parent = _post(client, "A")
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        child = _post(client, "B", previous_response_id=parent["id"], stream=stream)
    assert child["status"] == "failed", child
    assert "response-branching boundary" in child["error"]["message"]
    assert executions == ["A"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "failure", ["missing", "wrong_id", "backend", "metadata", "conflicting_ref"]
)
def test_invalid_parent_checkpoint_fails_without_running_graph(
    stream: bool, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, executions = _branch_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _post(client, "A")
        if failure in {"metadata", "conflicting_ref"}:
            assert client.portal is not None
            stored = client.portal.call(provider.get_response, parent["id"])
            metadata = _metadata(stored)
            boundary = metadata[BRANCH_BOUNDARY_KEY]
            if failure == "metadata":
                boundary["version"] = "invalid"
            else:
                boundary.update(thread_id="other-thread", checkpoint_id="other-state")
            stored["metadata"]["_internal_metadata"] = json.dumps(metadata)
            client.portal.call(provider.update_response, stored)
        else:
            original_read = saver.aget_tuple

            async def read(config: RunnableConfig) -> CheckpointTuple | None:
                if not (config.get("configurable") or {}).get("checkpoint_id"):
                    return await original_read(config)
                if failure == "backend":
                    raise PermissionError("private checkpoint endpoint")
                if failure == "missing":
                    return None
                saved = await original_read(config)
                assert saved is not None
                return saved._replace(
                    checkpoint={**saved.checkpoint, "id": "not-the-selected-checkpoint"}
                )

            monkeypatch.setattr(saver, "aget_tuple", read)
        child = _post(client, "B", previous_response_id=parent["id"], stream=stream)
        assert client.portal is not None
        stored_child = client.portal.call(provider.get_response, child["id"])
        assert BRANCH_BOUNDARY_KEY not in _metadata(stored_child)
    assert child["status"] == "failed", child
    assert "private checkpoint" not in child["error"]["message"]
    assert executions == ["A"]


class _ConfiguredSaver(InMemorySaver):
    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [ConfigurableFieldSpec(id="tenant_id", annotation=str, is_shared=True)]

    @staticmethod
    def _require_config(config: RunnableConfig) -> None:
        assert (config.get("configurable") or {}).get("tenant_id") == "tenant"
        assert (config.get("metadata") or {}).get("graph_marker") == "bound"

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._require_config(config)
        return super().get_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._require_config(config)
        return super().put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._require_config(config)
        super().put_writes(config, writes, task_id, task_path)


class _ConfiguredHost(ResponsesHostServer):
    async def build_runnable_config(
        self, request: CreateResponse, context: ResponseContext
    ) -> RunnableConfig:
        config = await super().build_runnable_config(request, context)
        return {
            **config,
            "configurable": {
                **(config.get("configurable") or {}),
                "tenant_id": "tenant",
            },
        }


class _ReplacementConfiguredHost(ResponsesHostServer):
    async def build_runnable_config(
        self, request: CreateResponse, context: ResponseContext
    ) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": f"custom-{context.response_id}",
                "tenant_id": "tenant",
            }
        }


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "host_class", [ResponsesHostServer, _ConfiguredHost, _ReplacementConfiguredHost]
)
def test_child_branch_reads_parent_once_per_request(
    stream: bool, host_class: type[ResponsesHostServer]
) -> None:
    graph, _ = _branch_graph()
    provider = InMemoryResponseProvider()
    server = host_class(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _post(client, "A")
        with (
            patch.object(
                server._conversation_chain_store,
                "get",
                wraps=server._conversation_chain_store.get,
            ) as records,
            patch.object(
                provider, "get_response", wraps=provider.get_response
            ) as parents,
        ):
            for value in ("B", "C"):
                records.reset_mock()
                parents.reset_mock()
                child = _post(
                    client, value, previous_response_id=parent["id"], stream=stream
                )
                assert child["status"] == "completed", child
                assert _text(child) == f"A,{value}"
                records.assert_not_awaited()
                assert (
                    sum(
                        lookup.args[0] == parent["id"]
                        for lookup in parents.await_args_list
                    )
                    == 1
                )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("configuration", ["hook", "bound", "override", "replacement"])
def test_branching_preserves_application_checkpoint_configuration(
    stream: bool, configuration: str
) -> None:
    graph, _ = _branch_graph()
    graph.checkpointer = _ConfiguredSaver()
    config: RunnableConfig = {"metadata": {"graph_marker": "bound"}}
    if configuration != "hook":
        config["configurable"] = {
            "tenant_id": "tenant" if configuration == "bound" else "overridden"
        }
    graph = graph.with_config(config)
    original_config = deepcopy(graph.config)
    host_class = {
        "bound": ResponsesHostServer,
        "replacement": _ReplacementConfiguredHost,
    }.get(configuration, _ConfiguredHost)
    server = host_class(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        a = _post(client, "A", stream=stream)
        b = _post(client, "B", previous_response_id=a["id"], stream=stream)
        c = _post(client, "C", previous_response_id=a["id"], stream=stream)
    assert [_text(item) for item in (a, b, c)] == ["A", "A,B", "A,C"]
    assert graph.checkpointer is server.graph.checkpointer
    assert graph.config == original_config


@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_config_hooks_keep_branch_selections_isolated(stream: bool) -> None:
    ready = asyncio.Event()
    selected: list[str] = []

    class ConcurrentHost(ResponsesHostServer):
        async def build_runnable_config(
            self, request: CreateResponse, context: ResponseContext
        ) -> RunnableConfig:
            config = await super().build_runnable_config(request, context)
            if request.get("previous_response_id"):
                selected.append(context.response_id)
                if len(selected) == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=10)
            return config

    graph, _ = _branch_graph()
    provider = InMemoryResponseProvider()
    server = ConcurrentHost(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        a = _post(client, "A")
        d = _post(client, "D")
        with (
            patch.object(
                provider, "get_response", wraps=provider.get_response
            ) as parents,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            b = executor.submit(
                _post, client, "B", previous_response_id=a["id"], stream=stream
            )
            e = executor.submit(
                _post, client, "E", previous_response_id=d["id"], stream=stream
            )
            children = [future.result(timeout=15) for future in (b, e)]
            assert [child["status"] for child in children] == ["completed", "completed"]
            assert [_text(child) for child in children] == ["A,B", "D,E"]
            assert sorted(
                lookup.args[0] for lookup in parents.await_args_list
            ) == sorted((a["id"], d["id"]))


@pytest.mark.parametrize("stream", [False, True])
def test_sibling_can_start_while_another_branch_is_running(stream: bool) -> None:
    started = threading.Event()
    release = threading.Event()

    async def record(state: _BranchState) -> dict[str, Any]:
        text = str(state["messages"][-1].content)
        if text == "B":
            started.set()
            assert await asyncio.to_thread(release.wait, 10)
        return {
            "ledger": [text],
            "messages": [AIMessage(content=",".join([*state.get("ledger", []), text]))],
        }

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
        a = _post(client, "A", stream=stream)
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(
                _post, client, "B", previous_response_id=a["id"], stream=stream
            )
            try:
                assert started.wait(10)
                c = _post(client, "C", previous_response_id=a["id"], stream=stream)
                assert c["status"] == "completed", c
                assert _text(c) == "A,C"
                assert not running.done()
            finally:
                release.set()
            b = running.result(timeout=10)
    assert b["status"] == "completed", b
    assert _text(b) == "A,B"


@pytest.mark.parametrize("source", ["input", "loop"])
async def test_checkpoint_write_failure_never_publishes_missing_progress(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, _ = _branch_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    original_put = saver.aput

    async def failed_put(
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        await asyncio.sleep(0)
        if metadata["source"] == source:
            raise OSError("private checkpoint backend failure")
        return await original_put(config, checkpoint, metadata, new_versions)

    monkeypatch.setattr(saver, "aput", failed_put)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    context = _context(response_id="root", conversation_id=None, current_text="A")
    events: list[Any] = []
    async with asyncio.timeout(5):
        async for event in server.handle_create(_request(), context, asyncio.Event()):
            events.append(event)
            if not isinstance(event, dict):
                metadata = json.loads(
                    event.response.get("metadata", {}).get("_internal_metadata", "{}")
                )
                if metadata.get(METADATA_LANGGRAPH_CHECKPOINT_ID):
                    assert (
                        await saver.aget_tuple(
                            {
                                "configurable": {
                                    "thread_id": metadata[METADATA_LANGGRAPH_THREAD_ID],
                                    "checkpoint_id": metadata[
                                        METADATA_LANGGRAPH_CHECKPOINT_ID
                                    ],
                                }
                            }
                        )
                        is not None
                    )
    terminal = events[-1]["response"]
    assert terminal["status"] == "failed", terminal
    assert "private checkpoint" not in terminal["error"]["message"]
    assert all(
        BRANCH_BOUNDARY_KEY not in _metadata(event.response)
        for event in events
        if not isinstance(event, dict)
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "progress",
    [
        "none",
        "own",
        "sibling",
        "missing",
        "published",
        "malformed",
        "wrong_parent",
        "wrong_thread",
    ],
)
async def test_recovery_uses_exact_parent_or_owned_progress(
    enabled: bool, progress: str
) -> None:
    graph, executions = _branch_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "root"}}
    await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
    parent_config = (await graph.aget_state(config)).config
    ref = HostingRunnableConfig(parent_config).checkpoint_ref
    assert ref is not None
    metadata: dict[str, Any] = {BRANCH_MODE_METADATA: BRANCH_MODE}
    if progress in {"own", "sibling", "published", "wrong_parent", "wrong_thread"}:
        saver = graph.checkpointer
        assert isinstance(saver, BaseCheckpointSaver)
        execution = graph.copy(
            {
                "checkpointer": ResponseCheckpointSaver(
                    saver,
                    branching=True,
                    response_key="sibling" if progress == "sibling" else "child",
                    parent_response_id=(
                        "other-parent" if progress == "wrong_parent" else "parent"
                    ),
                )
            }
        )
        await execution.ainvoke(
            {"messages": [HumanMessage(content="B")]}, parent_config
        )
        saved_ref = HostingRunnableConfig(
            (await graph.aget_state(config)).config
        ).checkpoint_ref
        assert saved_ref is not None
        metadata[METADATA_LANGGRAPH_THREAD_ID] = saved_ref.thread_id
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = saved_ref.checkpoint_id
        if progress == "published":
            metadata[BRANCH_BOUNDARY_KEY] = {
                "version": "1",
                "checkpoint_ns": "",
                "paused": "false",
            }
        elif progress == "wrong_thread":
            metadata[METADATA_LANGGRAPH_THREAD_ID] = "other-thread"
    elif progress == "missing":
        metadata[METADATA_LANGGRAPH_THREAD_ID] = ref.thread_id
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = "deleted-checkpoint"
    elif progress == "malformed":
        metadata[METADATA_LANGGRAPH_THREAD_ID] = ref.thread_id
    await graph.ainvoke({"messages": [HumanMessage(content="C")]}, parent_config)
    executions.clear()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        graph, store=provider if enabled else None, enable_response_branching=enabled
    )
    if progress == "none":
        parent = _response_object(
            "parent",
            internal_metadata={
                METADATA_LANGGRAPH_THREAD_ID: ref.thread_id,
                METADATA_LANGGRAPH_CHECKPOINT_ID: ref.checkpoint_id,
                BRANCH_BOUNDARY_KEY: {
                    "version": "1",
                    "checkpoint_ns": "",
                    "paused": "false",
                },
            },
        )
        parent["status"] = "completed"
        await provider.create_response(parent, None, None)
    context = _context(
        response_id="child", conversation_id=None, current_text="B", provider=provider
    )
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    context.is_recovery = True
    context.persisted_response = _response_object(
        "child", previous_response_id="parent", internal_metadata=metadata
    )
    with patch.object(provider, "get_response", wraps=provider.get_response) as lookup:
        events = [
            event
            async for event in server.handle_create(
                _request(previous_response_id="parent"), context, asyncio.Event()
            )
        ]
        if progress == "none":
            lookup.assert_awaited_once_with("parent", context=context.platform_context)
        else:
            lookup.assert_not_awaited()
    terminal = events[-1]["response"]
    assert terminal["status"] == (
        "completed" if progress in {"none", "own", "published"} else "failed"
    ), terminal
    assert executions == (["B"] if progress == "none" else [])
    if terminal["status"] == "completed":
        completed_ref = TaskStorageManager(
            _metadata(
                next(
                    event.response
                    for event in reversed(events)
                    if not isinstance(event, dict)
                )
            )
        ).checkpoint_ref
        assert completed_ref is not None
        snapshot = await graph.aget_state(_config(completed_ref))
        assert snapshot.values["ledger"] == ["A", "B"]
        if progress == "none":
            assert _text(terminal) == "A,B"


async def test_recovery_restores_text_from_completed_parallel_pending_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saver = InMemorySaver()
    left_saved = asyncio.Event()
    left_emitted = asyncio.Event()
    release_right = asyncio.Event()
    executions: list[str] = []
    original_put_writes = saver.aput_writes

    async def put_writes(
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await original_put_writes(config, writes, task_id, task_path)
        if any(
            channel == "messages" and any(message.id == "left-B" for message in value)
            for channel, value in writes
        ):
            left_saved.set()

    monkeypatch.setattr(saver, "aput_writes", put_writes)

    async def left(state: MessagesState) -> dict[str, list[AIMessage]]:
        text = next(
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        executions.append(f"left-{text}")
        return {"messages": [AIMessage(content=f"left-{text}", id=f"left-{text}")]}

    async def right(state: MessagesState) -> dict[str, list[AIMessage]]:
        text = next(
            message.content
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        executions.append(f"right-{text}")
        if text == "B":
            await release_right.wait()
        return {"messages": [AIMessage(content=f"right-{text}", id=f"right-{text}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("left", left)
    builder.add_node("right", right)
    for node in ("left", "right"):
        builder.add_edge(START, node)
        builder.add_edge(node, END)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver),
        store=provider,
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        parent = _post(client, "A")
    assert parent["status"] == "completed", parent
    executions.clear()
    request = _request(previous_response_id=parent["id"], store=True, background=True)
    context = _context(
        response_id="child", conversation_id=None, current_text="B", provider=provider
    )
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    snapshots: list[Any] = []

    async def consume() -> None:
        async for event in server.handle_create(request, context, asyncio.Event()):
            if not isinstance(event, dict):
                snapshots.append(deepcopy(event.response))
            elif (
                event["type"] == "response.output_text.delta"
                and event["delta"] == "left-B"
            ):
                left_emitted.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(
            asyncio.gather(left_saved.wait(), left_emitted.wait()), timeout=5
        )
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert snapshots and _text(snapshots[-1]) == ""
    ref = TaskStorageManager(_metadata(snapshots[-1])).checkpoint_ref
    assert ref is not None
    saved = await saver.aget_tuple(_config(ref))
    assert saved is not None and saved.pending_writes
    assert any(
        channel == "messages" and any(message.id == "left-B" for message in value)
        for _, channel, value in saved.pending_writes
    )
    release_right.set()
    for _ in range(2):
        recovery = _context(
            response_id="child",
            conversation_id=None,
            current_text="B",
            provider=provider,
        )
        recovery.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
        recovery.is_recovery = True
        recovery.persisted_response = deepcopy(snapshots[-1])
        events = [
            event
            async for event in server.handle_create(request, recovery, asyncio.Event())
        ]
        terminal = events[-1]["response"]
        assert terminal["status"] == "completed", terminal
        assert _text(terminal) == "left-Bright-B"
        assert _text(snapshots[-1]) == ""
    assert executions.count("left-B") == 1
    assert executions.count("right-B") == 2


@pytest.mark.parametrize("mode", [None, "unknown-mode"])
async def test_recovery_cannot_silently_change_admitted_mode(
    mode: str | None,
) -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    context = _context(response_id="child", conversation_id=None)
    context.is_recovery = True
    context.client_headers = {} if mode is None else {BRANCH_MODE_HEADER: mode}
    context.persisted_response = _response_object(
        "child", internal_metadata={BRANCH_MODE_METADATA: BRANCH_MODE}
    )
    events = [
        event
        async for event in server.handle_create(
            _request(previous_response_id="parent"), context, asyncio.Event()
        )
    ]
    assert events[-1]["response"]["status"] == "failed"
    assert executions == []


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_checkpoint_saver_does_not_fall_back_to_latest(
    enabled: bool, asynchronous: bool
) -> None:
    saver = ResponseCheckpointSaver(InMemorySaver(), branching=enabled)
    config: RunnableConfig = {
        "configurable": {"thread_id": "thread", "checkpoint_id": "missing"}
    }
    if enabled:
        with pytest.raises(BranchingError, match="unavailable"):
            if asynchronous:
                await saver.aget_tuple(config)
            else:
                saver.get_tuple(config)
    elif asynchronous:
        assert await saver.aget_tuple(config) is None
    else:
        assert saver.get_tuple(config) is None


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_recovery_head_and_pending_writes_cannot_follow_a_sibling(
    asynchronous: bool,
) -> None:
    saver = InMemorySaver()
    config: RunnableConfig = {
        "configurable": {"thread_id": "root", "checkpoint_ns": ""}
    }
    metadata: CheckpointMetadata = {"source": "loop", "step": 0, "parents": {}}
    selected = await saver.aput(config, empty_checkpoint(), metadata, {})
    adapter = ResponseCheckpointSaver(saver, branching=True)
    execution_config = adapter.resume_config(selected)
    assert "checkpoint_id" in selected["configurable"]
    assert "checkpoint_id" not in execution_config["configurable"]
    sibling = await saver.aput(config, empty_checkpoint(), metadata, {})
    writes = [("result", "already-completed")]
    checkpoint = empty_checkpoint()
    if asynchronous:
        await adapter.aput_writes(execution_config, writes, "task")
        advanced = await adapter.aput(execution_config, checkpoint, metadata, {})
    else:
        adapter.put_writes(execution_config, writes, "task")
        advanced = adapter.put(execution_config, checkpoint, metadata, {})
    selected_checkpoint = await saver.aget_tuple(selected)
    sibling_checkpoint = await saver.aget_tuple(sibling)
    assert selected_checkpoint is not None and sibling_checkpoint is not None
    assert selected_checkpoint.pending_writes == [
        ("task", "result", "already-completed")
    ]
    assert sibling_checkpoint.pending_writes == []
    await saver.aput(config, empty_checkpoint(), metadata, {})

    async def read() -> CheckpointTuple | None:
        return (
            await adapter.aget_tuple(execution_config)
            if asynchronous
            else adapter.get_tuple(execution_config)
        )

    saved = await read()
    assert saved is not None
    assert saved.config == advanced
    assert saved.parent_config == selected
    await saver.adelete_thread("root")
    await saver.aput(config, empty_checkpoint(), metadata, {})
    with pytest.raises(BranchingError, match="required checkpoint"):
        await read()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_checkpoint_saver_delegates_thread_deletion(
    asynchronous: bool,
) -> None:
    adapter = ResponseCheckpointSaver(InMemorySaver(), branching=True)
    deleted: RunnableConfig = {
        "configurable": {"thread_id": "deleted", "checkpoint_ns": ""}
    }
    retained: RunnableConfig = {
        "configurable": {"thread_id": "retained", "checkpoint_ns": ""}
    }
    metadata: CheckpointMetadata = {"source": "input", "step": 0, "parents": {}}
    deleted_ref = await adapter.aput(deleted, empty_checkpoint(), metadata, {})
    retained_ref = await adapter.aput(retained, empty_checkpoint(), metadata, {})
    await adapter.aput_writes(deleted_ref, [("result", "deleted")], "task")
    await adapter.aput_writes(retained_ref, [("result", "retained")], "task")
    original = await adapter.aget_tuple(retained_ref)
    assert original is not None and original.pending_writes
    if asynchronous:
        await adapter.adelete_thread("deleted")
    else:
        adapter.delete_thread("deleted")
    assert await adapter.aget_tuple(deleted) is None
    assert list(adapter.list(deleted)) == []
    assert [item async for item in adapter.alist(deleted)] == []
    assert await adapter.aget_tuple(retained_ref) == original


@pytest.mark.parametrize("stream", [False, True])
def test_failed_terminal_persistence_does_not_publish_a_usable_parent(
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
                metadata = _metadata(response)
                assert BRANCH_BOUNDARY_KEY in metadata
                assert TaskStorageManager(metadata).checkpoint_ref is not None
                terminal_writes.append(response_id)
                raise OSError("private storage details")
            await write(response, *args, **kwargs)

        return persist

    with TestClient(server.app) as client:
        a = _post(client, "A")
        for method in ("create_response", "update_response"):
            monkeypatch.setattr(
                provider, method, fail_terminal_write(getattr(provider, method))
            )
        failed = client.post(
            "/responses",
            json={
                "input": "B",
                "previous_response_id": a["id"],
                "stream": stream,
                "store": True,
            },
        )
        assert failed.status_code in {200, 500}, failed.text
        assert "private storage details" not in failed.text
        assert len(terminal_writes) == 1
        failed_id = terminal_writes[0]
        retrieved = client.get(f"/responses/{failed_id}")
        assert retrieved.status_code == 200, retrieved.text
        assert retrieved.json()["status"] == "failed"
        assert retrieved.json()["error"]["code"] == "storage_error"
        unavailable = _post(client, "not-run", previous_response_id=failed_id)
        assert unavailable["status"] == "failed"
        c = _post(client, "C", previous_response_id=a["id"])
    assert _text(c) == "A,C"
    assert executions == ["A", "B", "C"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("client_mode", [None, "forged-mode"])
async def test_branching_admission_sanitizes_mode_header(
    enabled: bool, client_mode: str | None
) -> None:
    downstream = AsyncMock()
    headers = [(b"x-agent-user-id", b"test-user")]
    if client_mode is not None:
        headers.append(
            (b"x-client-langchain-response-branching", client_mode.encode("ascii"))
        )
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/responses",
        "headers": headers,
    }
    receive = AsyncMock(
        return_value={"type": "http.request", "body": b"{}", "more_body": False}
    )
    await BranchingAdmissionMiddleware(downstream, enabled=enabled)(
        scope, receive, AsyncMock()
    )

    expected_headers = [(b"x-agent-user-id", b"test-user")]
    if enabled:
        expected_headers.append(
            (b"x-client-langchain-response-branching", b"checkpoint-v1")
        )
    downstream.assert_awaited_once()
    assert downstream.await_args is not None
    assert downstream.await_args.args[0]["headers"] == expected_headers


def test_foundry_identity_and_trusted_mode_survive_admission() -> None:
    from azure.ai.agentserver.core import get_request_context

    observed: list[tuple[str | None, str | None]] = []

    class ObservedProvider(InMemoryResponseProvider):
        async def get_response(self, response_id: str, *, context: Any = None) -> Any:
            observed.append(
                (get_request_context().user_id, get_request_context().call_id)
            )
            return await super().get_response(response_id, context=context)

    graph, executions = _branch_graph()
    server = ResponsesHostServer(
        graph, store=ObservedProvider(), enable_response_branching=True
    )
    response_id = "caresp_" + "c" * 18 + "d" * 32
    headers = {
        "x-agent-user-id": "test-user",
        "x-agent-foundry-call-id": "test-call",
        BRANCH_MODE_HEADER: "forged-mode",
    }
    with TestClient(server.app, headers=headers) as client:
        a = client.post(
            "/responses",
            json={"input": "A"},
            headers={"x-agent-response-id": response_id},
        )
        assert a.status_code == 200, a.text
        assert a.json()["status"] == "completed", a.text
        b = _post(client, "B", previous_response_id=response_id)
        assert b["status"] == "completed", b
    assert observed and all(item == ("test-user", "test-call") for item in observed)
    assert executions == ["A", "B"]


async def test_durable_parent_survives_host_and_saver_recreation(
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
            a = _post(client, "A")
            b = _post(client, "B", previous_response_id=a["id"])
            assert _text(b) == "A,B"
    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        graph, executions = _branch_graph()
        graph.checkpointer = saver
        server = ResponsesHostServer(graph, enable_response_branching=True)
        with TestClient(server.app) as client:
            c = _post(client, "C", previous_response_id=a["id"])
            d = _post(client, "D", previous_response_id=b["id"])
    assert _text(c) == "A,C"
    assert _text(d) == "A,B,D"
    assert executions == ["C", "D"]
