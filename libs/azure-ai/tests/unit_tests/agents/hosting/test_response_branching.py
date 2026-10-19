"""Opt-in response checkpoint branching and strict restoration tests."""

from __future__ import annotations

import asyncio
import json
import operator
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses import ResponseContext, ResponseExitForRecovery
from azure.ai.agentserver.responses.models import CreateResponse
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
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langgraph import channels
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    PendingWrite,
    empty_checkpoint,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import task
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import CachePolicy, Command, Interrupt, PregelTask, Send, interrupt
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
    _remap_pause_writes,
    remap_resume_command,
)
from langchain_azure_ai.agents.hosting._responses.checkpoint_ref import CheckpointRef

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


def _boundary_ref(
    client: TestClient, server: ResponsesHostServer, response_id: str
) -> CheckpointRef:
    assert client.portal is not None
    record = client.portal.call(
        server._conversation_chain_store.get, response_id, BRANCH_BOUNDARY_KEY
    )
    ref = CheckpointRef.from_dict(record)
    assert ref is not None
    return ref


class _BranchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    ledger: Annotated[list[str], operator.add]


class _ConfiguredSaver(InMemorySaver):
    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [ConfigurableFieldSpec(id="tenant_id", annotation=str, is_shared=True)]

    @staticmethod
    def _require_tenant(config: RunnableConfig) -> None:
        if (config.get("configurable") or {}).get("tenant_id") != "tenant":
            raise ValueError("The checkpoint tenant configuration is missing.")

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        self._require_tenant(config)
        return super().get_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._require_tenant(config)
        return super().put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._require_tenant(config)
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


def _configured_server(
    graph: CompiledStateGraph,
    configuration: str,
    *,
    enabled: bool = True,
    provider: InMemoryResponseProvider | None = None,
) -> ResponsesHostServer:
    host_class: type[ResponsesHostServer] = _ConfiguredHost
    if configuration == "bound":
        graph = graph.with_config({"configurable": {"tenant_id": "tenant"}})
        host_class = ResponsesHostServer
    elif configuration == "override":
        graph = graph.with_config({"configurable": {"tenant_id": "overridden"}})
    elif configuration != "hook":
        raise ValueError(f"Unknown configuration source: {configuration}")
    return host_class(
        graph,
        store=provider if provider is not None else InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )


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


def _application_owned_subgraph_graph() -> tuple[CompiledStateGraph, dict[str, int]]:
    class ChildState(TypedDict):
        count: int

    class RootState(_BranchState, total=False):
        child_state_ref: str

    snapshots: dict[str, int] = {}
    child_builder = StateGraph(ChildState)
    child_builder.add_node("count", lambda state: {"count": state["count"] + 1})
    child_builder.add_edge(START, "count")
    child_builder.add_edge("count", END)
    child = child_builder.compile()

    async def call_child(state: RootState) -> dict[str, Any]:
        previous_ref = state.get("child_state_ref")
        previous_count = snapshots[previous_ref] if previous_ref is not None else 0
        result = await child.ainvoke({"count": previous_count})
        state_ref = uuid4().hex
        snapshots[state_ref] = result["count"]
        text = next(
            str(message.content)
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        )
        return {
            "messages": [AIMessage(content=f"child:{result['count']}")],
            "ledger": [text],
            "child_state_ref": state_ref,
        }

    builder = StateGraph(RootState)
    builder.add_node("child", call_child)
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    return builder.compile(checkpointer=InMemorySaver()), snapshots


def _partial_approval_graph(
    saver: BaseCheckpointSaver,
    operations: list[str],
    *,
    join: bool = True,
    functional: bool = False,
) -> CompiledStateGraph:
    class ApprovalState(_BranchState):
        a_result: int
        b_result: str

    def approve_a(state: ApprovalState) -> dict[str, Any]:
        interrupt("A")
        operations.append("A")
        return {"a_result": operations.count("A")}

    def approve_b(state: ApprovalState) -> dict[str, Any]:
        answer = interrupt("B")
        operations.append("B")
        return {"b_result": answer}

    approve_a_task = task(approve_a)
    approve_b_task = task(approve_b)

    def run_a(state: ApprovalState) -> dict[str, Any]:
        return approve_a_task(state).result()

    def run_b(state: ApprovalState) -> dict[str, Any]:
        return approve_b_task(state).result()

    def finish(state: ApprovalState) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content=f"{state['a_result']}:{state['b_result']}")]
        }

    builder = StateGraph(ApprovalState)
    builder.add_node("a", run_a if functional else approve_a)
    builder.add_node("b", run_b if functional else approve_b)
    builder.add_node("finish", finish)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    if join:
        builder.add_edge(["a", "b"], "finish")
    else:
        builder.add_edge("a", END)
        builder.add_edge("b", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.parametrize("id_keyed_resume", [False, True])
def test_remap_pause_writes_preserves_parent_and_task_data(
    id_keyed_resume: bool,
) -> None:
    parent_interrupt = Interrupt.from_ns({"question": "one"}, "ask:parent-one")
    parent_tasks = (
        PregelTask("parent-one", "ask", ("push", 0), interrupts=(parent_interrupt,)),
        PregelTask("parent-two", "ask", ("push", 1)),
    )
    fork_tasks = (
        PregelTask("fork-two", "ask", ("push", 1)),
        PregelTask("fork-one", "ask", ("push", 0)),
    )
    prior_resume = ["prior-answer"]
    completed_result = {"answer": "already done"}
    global_resume = (
        {parent_interrupt.id: "mapped-answer"}
        if id_keyed_resume
        else {"interrupt-id": "reply", parent_interrupt.id: "literal-key"}
    )
    pending_writes: list[PendingWrite] = [
        ("parent-one", "__interrupt__", (parent_interrupt,)),
        ("parent-one", "__resume__", prior_resume),
        ("parent-two", "result", completed_result),
        ("", "__resume__", global_resume),
    ]
    original_writes = deepcopy(pending_writes)
    fork_interrupt = Interrupt.from_ns(parent_interrupt.value, "ask:fork-one")

    resume_ids, writes_by_task = _remap_pause_writes(
        parent_tasks, fork_tasks, pending_writes
    )

    assert resume_ids == {parent_interrupt.id: fork_interrupt.id}
    assert writes_by_task == {
        "fork-one": [
            ("__interrupt__", (fork_interrupt,)),
            ("__resume__", prior_resume),
        ],
        "fork-two": [("result", completed_result)],
        "": [
            (
                "__resume__",
                {fork_interrupt.id: "mapped-answer"}
                if id_keyed_resume
                else global_resume,
            )
        ],
    }
    assert pending_writes == original_writes
    assert parent_tasks[0].interrupts == original_writes[0][2]
    assert writes_by_task["fork-one"][0][1][0] is not parent_interrupt


@pytest.mark.parametrize(
    ("resume", "expected"),
    [
        ("answer", {"b" * 32: "answer"}),
        ({"choice": "answer"}, {"b" * 32: {"choice": "answer"}}),
        (
            {"a" * 32: "answer", "c" * 32: "stale"},
            {"b" * 32: "answer", "c" * 32: "stale"},
        ),
        ({"c" * 32: "stale"}, {"c" * 32: "stale"}),
        (None, None),
    ],
)
def test_remap_resume_command_preserves_payload_and_fields(
    resume: Any, expected: Any
) -> None:
    command: Command[str] = Command(
        resume=resume, update={"ledger": ["updated"]}, goto="finish"
    )

    remapped = remap_resume_command(
        command,
        {"a" * 32: "b" * 32},
        pending=(Interrupt(value="question", id="a" * 32),),
    )

    assert remapped.resume == expected
    assert remapped.update == command.update
    assert remapped.goto == command.goto
    assert command.resume == resume


def test_remap_resume_command_leaves_unbound_payload_unchanged() -> None:
    command: Command[str] = Command(resume={"a" * 32: "literal", "choice": "answer"})

    assert remap_resume_command(command, {"a" * 32: "b" * 32}) is command


@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_branches_keep_parent_thread(paused: bool, stream: bool) -> None:
    graph = build_simple_interrupt_graph() if paused else _branch_graph()[0]
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        root = _post(client, "start", stream=stream)
        parent_ref = _boundary_ref(client, server, root["id"])
        parent_config: RunnableConfig = {
            "configurable": {**parent_ref.to_dict(), "checkpoint_ns": ""}
        }
        parent_checkpoint = saver.get_tuple(parent_config)
        assert parent_checkpoint is not None
        call_id = (
            next(
                item["call_id"]
                for item in root["output"]
                if item["type"] == "function_call"
            )
            if paused
            else None
        )
        children = [
            _post(
                client,
                (
                    [
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"resume": answer}),
                        }
                    ]
                    if paused
                    else answer
                ),
                previous_response_id=root["id"],
                stream=stream,
            )
            for answer in ("Alice", "Bob")
        ]
        refs = [_boundary_ref(client, server, child["id"]) for child in children]
        assert all(ref.thread_id == parent_ref.thread_id for ref in refs)
        assert len({ref.checkpoint_id for ref in refs}) == 2
        assert [_text(child) for child in children] == (
            ["ok:Alice", "ok:Bob"] if paused else ["start,Alice", "start,Bob"]
        )
        assert saver.get_tuple(parent_config) == parent_checkpoint
        assert {
            saved.config["configurable"]["thread_id"] for saved in saver.list(None)
        } == {parent_ref.thread_id}
        if paused:
            assert client.portal is not None
            origins = [
                client.portal.call(
                    server._conversation_chain_store.get, child["id"], BRANCH_ORIGIN_KEY
                )
                for child in children
            ]
            origin_ids = set()
            for origin in origins:
                assert origin is not None
                origin_ids.add(origin["checkpoint_id"])
                assert origin["checkpoint_id"] != parent_ref.checkpoint_id
            assert len(origin_ids) == 2


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("paused", [False, True])
@pytest.mark.parametrize("configuration", ["bound", "hook", "override"])
def test_branches_preserve_checkpoint_configuration(
    enabled: bool, stream: bool, paused: bool, configuration: str
) -> None:
    graph = build_simple_interrupt_graph() if paused else _branch_graph()[0]
    saver = _ConfiguredSaver()
    graph.checkpointer = saver
    server = _configured_server(graph, configuration, enabled=enabled)
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        assert parent["status"] == "completed", parent
        parent_config: RunnableConfig | None = None
        parent_checkpoint = None
        if enabled:
            parent_config = {
                "configurable": {
                    **_boundary_ref(client, server, parent["id"]).to_dict(),
                    "checkpoint_ns": "",
                    "tenant_id": "tenant",
                }
            }
            parent_checkpoint = saver.get_tuple(parent_config)
            assert parent_checkpoint is not None
        call_id = (
            next(
                item["call_id"]
                for item in parent["output"]
                if item["type"] == "function_call"
            )
            if paused
            else None
        )
        for answer in ("Alice", "Bob") if enabled else ("Alice",):
            child = _post(
                client,
                (
                    [
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"resume": answer}),
                        }
                    ]
                    if paused
                    else answer
                ),
                previous_response_id=parent["id"],
                stream=stream,
            )
            assert child["status"] == "completed", child
            assert _text(child) == (f"ok:{answer}" if paused else f"start,{answer}")
        if parent_config is not None:
            assert saver.get_tuple(parent_config) == parent_checkpoint


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("inside_task", [False, True])
@pytest.mark.parametrize("configuration", ["bound", "hook", "override"])
def test_functional_branches_preserve_checkpoint_configuration(
    stream: bool, inside_task: bool, configuration: str
) -> None:
    operations: list[str] = []
    saver = _ConfiguredSaver()
    graph = (
        _functional_interrupt_graph(saver, operations, "nested")
        if inside_task
        else _functional_graph(saver, operations, "nested")
    )
    server = _configured_server(graph, configuration)
    expected_operations = ["one"] if inside_task else ["prepare", "prepare"]
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        assert parent["status"] == "completed", parent
        waiting = _post(
            client, "wait", previous_response_id=parent["id"], stream=stream
        )
        assert waiting["status"] == "completed", waiting
        call = next(
            item for item in waiting["output"] if item["type"] == "function_call"
        )
        for answer in ("Alice", "Bob"):
            child = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps({"resume": answer}),
                    }
                ],
                previous_response_id=waiting["id"],
                stream=stream,
            )
            assert child["status"] == "completed", child
            assert _text(child) == (
                f"one-1:{answer}" if inside_task else f"1,2:{answer}"
            )
            assert operations == expected_operations


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("configuration", ["bound", "hook", "override"])
def test_branch_configuration_preserves_bound_metadata_and_callbacks(
    stream: bool, configuration: str
) -> None:
    calls: list[UUID] = []

    class Callback(BaseCallbackHandler):
        def on_chain_start(
            self,
            serialized: dict[str, Any] | None,
            inputs: Any,
            *,
            run_id: UUID,
            **kwargs: Any,
        ) -> None:
            calls.append(run_id)

    class MetadataSaver(_ConfiguredSaver):
        def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
            assert (config.get("metadata") or {}).get("graph_marker") == "bound"
            return super().get_tuple(config)

    graph = build_simple_interrupt_graph()
    graph.checkpointer = MetadataSaver()
    graph = graph.with_config(
        {"metadata": {"graph_marker": "bound"}, "callbacks": [Callback()]}
    )
    server = _configured_server(graph, configuration)
    bound_config = server._graph.config
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        assert parent["status"] == "completed", parent
        call = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        child = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "Alice",
                }
            ],
            previous_response_id=parent["id"],
            stream=stream,
        )
    assert child["status"] == "completed", child
    assert _text(child) == "ok:Alice"
    assert calls
    assert len(calls) == len(set(calls))
    assert server._graph.config is bound_config


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("kind", ["completed", "paused", "functional"])
@pytest.mark.parametrize("configuration", ["bound", "hook", "override"])
async def test_recovery_preserves_checkpoint_configuration(
    enabled: bool, kind: str, configuration: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    saver = _ConfiguredSaver()
    executions: list[str] = []
    if kind == "functional":
        graph = _functional_interrupt_graph(saver, executions, "nested")
    elif kind == "paused":
        graph = build_simple_interrupt_graph()
    else:
        graph, executions = _branch_graph()
    graph.checkpointer = saver
    provider = InMemoryResponseProvider()
    server = _configured_server(graph, configuration, provider=provider)
    with TestClient(server.app) as client:
        parent = _post(client, "start")
    context = _context(
        response_id="configured-child", conversation_id=None, current_text="next"
    )
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    request = _request(previous_response_id=parent["id"])
    if kind != "completed":
        call = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        context.get_input_items.return_value = [
            {
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps({"resume": "Alice"}),
            }
        ]
    monkeypatch.setattr(
        server._branch_store,
        "publish",
        AsyncMock(side_effect=ResponseExitForRecovery),
    )
    persisted = None
    with pytest.raises(ResponseExitForRecovery):
        async for event in server.handle_create(request, context, asyncio.Event()):
            if not isinstance(event, dict):
                metadata = json.loads(
                    event.response.get("metadata", {}).get("_internal_metadata", "{}")
                )
                if metadata.get(METADATA_LANGGRAPH_CHECKPOINT_ID):
                    persisted = deepcopy(event.response)
    assert persisted is not None

    recreated = _configured_server(
        graph, configuration, enabled=enabled, provider=provider
    )
    recovery = _context(response_id="configured-child", conversation_id=None)
    recovery.client_headers = context.client_headers
    recovery.is_recovery = True
    recovery.persisted_response = persisted
    events = [
        event
        async for event in recreated.handle_create(request, recovery, asyncio.Event())
    ]
    terminal = events[-1]["response"]
    assert terminal["status"] == "completed", terminal
    if kind == "functional":
        assert _text(terminal) == "one-1:Alice", terminal
        assert executions == ["one"]
    elif kind == "paused":
        assert _text(terminal) == "ok:Alice", terminal
    else:
        assert _text(terminal) == "start,next", terminal
        assert executions == ["start", "next"]
    recovery.get_input_items.assert_not_awaited()


@pytest.mark.parametrize("stream", [False, True])
def test_waiting_parallel_forks_keep_parent_reply_ids(stream: bool) -> None:
    builder = StateGraph(_BranchState)
    for index in range(8):
        name = f"ask_{index}"

        def ask(state: _BranchState, name: str = name) -> dict[str, Any]:
            return {"messages": [AIMessage(content=f"{name}={interrupt(name)}")]}

        builder.add_node(name, ask)
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    graph = builder.compile(checkpointer=InMemorySaver())
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app, headers={"x-agent-user-id": "test-user"}) as client:
        parent = _post(client, "start", stream=stream)
        waiting = _post(
            client, "waiting", previous_response_id=parent["id"], stream=stream
        )
        call_ids = [
            item["call_id"]
            for item in parent["output"]
            if item["type"] == "function_call"
        ]
        assert waiting["status"] == "completed", waiting
        assert {
            item["call_id"]
            for item in waiting["output"]
            if item["type"] == "function_call"
        } == set(call_ids)
        responses = [
            _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"resume": answer}),
                    }
                    for call_id in call_ids
                ],
                previous_response_id=waiting["id"],
                stream=stream,
            )
            for answer in ("Alice", "Bob")
        ]
    for response, answer in zip(responses, ("Alice", "Bob"), strict=True):
        assert response["status"] == "completed", response
        assert all(f"ask_{index}={answer}" in _text(response) for index in range(8))
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    assert (
        len({saved.config["configurable"]["thread_id"] for saved in saver.list(None)})
        == 1
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("functional", [False, True])
def test_mcp_approval_forks_use_parent_ids(stream: bool, functional: bool) -> None:
    def approve() -> str:
        result = interrupt(
            {
                "action_requests": [{"name": "review", "args": {}}],
                "review_configs": [
                    {
                        "action_name": "review",
                        "allowed_decisions": ["approve", "reject"],
                    }
                ],
            }
        )
        return result["decisions"][0]["type"]

    approval_task = task(approve)

    def review(state: _BranchState) -> dict[str, Any]:
        answer = approval_task().result() if functional else approve()
        return {"messages": [AIMessage(content=answer)]}

    builder = StateGraph(_BranchState)
    builder.add_node("review", review)
    builder.add_edge(START, "review")
    builder.add_edge("review", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        approval_id = next(
            item["id"]
            for item in parent["output"]
            if item["type"] == "mcp_approval_request"
        )
        responses = [
            _post(
                client,
                [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval_id,
                        "approve": approve,
                    }
                ],
                previous_response_id=parent["id"],
                stream=stream,
            )
            for approve in (True, False)
        ]
        refs = [_boundary_ref(client, server, response["id"]) for response in responses]
        assert all(ref.thread_id == parent["id"] for ref in refs)
    assert all(response["status"] == "completed" for response in responses)
    assert [_text(response) for response in responses] == ["approve", "reject"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("functional", ["none", "completed", "interrupt"])
def test_send_forks_match_tasks_by_path(stream: bool, functional: str) -> None:
    operations: list[str] = []

    @task
    def prepare(name: str) -> str:
        operations.append(name)
        return name

    @task
    def approve(name: str) -> str:
        return interrupt(name)

    def distribute(state: _BranchState) -> list[Send]:
        return [
            Send("ask", {"messages": [HumanMessage(content=name)]})
            for name in ("one", "two")
        ]

    def ask(state: _BranchState) -> dict[str, Any]:
        name = str(state["messages"][0].content)
        if functional != "none":
            name = prepare(name).result()
        answer = (
            approve(name).result() if functional == "interrupt" else interrupt(name)
        )
        return {"messages": [AIMessage(content=f"{name}={answer}")]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask)
    builder.add_conditional_edges(START, distribute, ["ask"])
    builder.add_edge("ask", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        calls = [item for item in parent["output"] if item["type"] == "function_call"]
        assert len(calls) == 2
        for prefix in ("A", "B"):
            response = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": item["call_id"],
                        "output": json.dumps(
                            {
                                "resume": (
                                    f"{prefix}.{json.loads(item['arguments'])['value']}"
                                )
                            }
                        ),
                    }
                    for item in calls
                ],
                previous_response_id=parent["id"],
                stream=stream,
            )
            assert response["status"] == "completed", response
            assert all(
                f"{name}={prefix}.{name}" in _text(response) for name in ("one", "two")
            )
            assert (
                _boundary_ref(client, server, response["id"]).thread_id == parent["id"]
            )
        assert sorted(operations) == (["one", "two"] if functional != "none" else [])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_functional_task_interrupt_accepts_approval(
    stream: bool, enabled: bool
) -> None:
    @task
    def approve() -> str:
        return interrupt("approve?")

    def ask(state: _BranchState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=approve().result())]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        pending = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        completed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps({"resume": "yes"}),
                }
            ],
            previous_response_id=parent["id"],
            stream=stream,
        )
        assert completed["status"] == "completed", completed
        assert _text(completed) == "yes"


def _functional_interrupt_graph(
    saver: BaseCheckpointSaver[Any], operations: list[str], kind: str
) -> CompiledStateGraph:
    @task
    def prepare(label: str) -> str:
        operations.append(label)
        return f"{label}-{operations.count(label)}"

    @task
    def approve(label: str) -> str:
        prepared = prepare(label).result()
        return f"{prepared}:{interrupt(label)}"

    @task
    def nested_approve(label: str) -> str:
        return approve(label).result()

    @task
    async def async_approve(label: str) -> str:
        prepared = await prepare(label)
        return f"{prepared}:{interrupt(label)}"

    def ask(state: _BranchState) -> dict[str, Any]:
        approval = nested_approve if kind == "nested" else approve
        return {"messages": [AIMessage(content=approval("one").result())]}

    async def ask_async(state: _BranchState) -> dict[str, Any]:
        labels = ("one", "two") if kind == "async_parallel" else ("one",)
        results = await asyncio.gather(*(async_approve(label) for label in labels))
        return {"messages": [AIMessage(content=",".join(results))]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask_async if kind.startswith("async") else ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=saver)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["single", "nested", "async", "async_parallel"])
def test_functional_task_interrupt_forks_reuse_completed_work(
    stream: bool, kind: str
) -> None:
    operations: list[str] = []
    saver = InMemorySaver()
    graph = _functional_interrupt_graph(saver, operations, kind)
    labels = ("one", "two") if kind == "async_parallel" else ("one",)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        parent_ref = _boundary_ref(client, server, parent["id"])
        parent_config: RunnableConfig = {
            "configurable": {**parent_ref.to_dict(), "checkpoint_ns": ""}
        }
        original = saver.get_tuple(parent_config)
        parent_calls = {
            item["call_id"]
            for item in parent["output"]
            if item["type"] == "function_call"
        }
        assert parent_calls
        waiting = _post(
            client, "wait", previous_response_id=parent["id"], stream=stream
        )
        assert waiting["status"] == "completed", waiting
        calls = [item for item in waiting["output"] if item["type"] == "function_call"]
        assert {item["call_id"] for item in calls} == parent_calls
        for prefix in ("left", "right"):
            completed = waiting
            answered: set[str] = set()
            for _ in labels:
                pending = [
                    item
                    for item in completed["output"]
                    if item["type"] == "function_call"
                ]
                questions = {json.loads(item["arguments"])["value"] for item in pending}
                assert questions
                assert questions.isdisjoint(answered)
                answered.update(questions)
                completed = _post(
                    client,
                    [
                        {
                            "type": "function_call_output",
                            "call_id": item["call_id"],
                            "output": json.dumps(
                                {
                                    "resume": (
                                        f"{prefix}.{json.loads(item['arguments'])['value']}"
                                    )
                                }
                            ),
                        }
                        for item in pending
                    ],
                    previous_response_id=completed["id"],
                    stream=stream,
                )
                assert completed["status"] == "completed", completed
                if not any(
                    item["type"] == "function_call" for item in completed["output"]
                ):
                    break
            assert answered == set(labels)
            assert _text(completed) == ",".join(
                f"{label}-1:{prefix}.{label}" for label in labels
            )
            assert not any(
                item["type"] == "function_call" for item in completed["output"]
            )
            assert _boundary_ref(client, server, completed["id"]).thread_id == (
                parent_ref.thread_id
            )
            assert sorted(operations) == sorted(labels)
        assert saver.get_tuple(parent_config) == original
        assert (
            client.get(f"/responses/{parent['id']}").json()["output"]
            == (parent["output"])
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("corruption", ["unknown", "unrelated"])
def test_functional_task_interrupt_rejects_invalid_namespace(
    stream: bool, corruption: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    operations: list[str] = []
    saver = InMemorySaver()
    graph = _partial_approval_graph(saver, operations, functional=True)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    original_read = saver.aget_tuple
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        ref = _boundary_ref(client, server, parent["id"])
        config: RunnableConfig = {
            "configurable": {**ref.to_dict(), "checkpoint_ns": ""}
        }
        saved = saver.get_tuple(config)
        snapshot = graph.get_state(config)
        source = next(item for item in snapshot.tasks if item.name == "a")
        other = next(item for item in snapshot.tasks if item.name == "b")
        replacement = Interrupt(
            source.interrupts[0].value,
            id="f" * 32 if corruption == "unknown" else other.interrupts[0].id,
        )

        async def read(requested: RunnableConfig) -> Any:
            result = await original_read(requested)
            if (
                result is not None
                and requested.get("configurable", {}).get("checkpoint_ns", "") == ""
                and requested.get("configurable", {}).get("checkpoint_id")
                == ref.checkpoint_id
            ):
                return result._replace(
                    pending_writes=[
                        (
                            task_id,
                            channel,
                            (replacement,)
                            if task_id == source.id and channel == "__interrupt__"
                            else value,
                        )
                        for task_id, channel, value in result.pending_writes or ()
                    ]
                )
            return result

        monkeypatch.setattr(saver, "aget_tuple", read)
        pending = next(
            item
            for item in parent["output"]
            if item["type"] == "function_call"
            and json.loads(item["arguments"])["value"] == "A"
        )
        failed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps({"resume": "yes"}),
                }
            ],
            previous_response_id=parent["id"],
            stream=stream,
        )
        assert failed["status"] == "failed", failed
        assert failed["error"]["message"] == "The paused checkpoint is invalid."
        assert operations == []
        assert saver.get_tuple(config) == saved
        assert client.portal is not None
        for key in (BRANCH_ORIGIN_KEY, BRANCH_BOUNDARY_KEY):
            assert (
                client.portal.call(
                    server._conversation_chain_store.get, failed["id"], key
                )
                is None
            )


def _functional_graph(
    saver: BaseCheckpointSaver[Any], operations: list[str], kind: str = "single"
) -> CompiledStateGraph:
    lock = threading.Lock()

    def execute() -> int:
        with lock:
            operations.append("prepare")
            return len(operations)

    def prepare() -> int:
        return execute()

    cache_policy: CachePolicy[Callable[[], str | bytes]] | None = (
        CachePolicy() if kind == "cached" else None
    )
    prepare_task = task(cache_policy=cache_policy)(prepare)

    @task
    def nested_prepare() -> list[int]:
        return [prepare_task().result(), prepare_task().result()]

    @task
    async def async_prepare() -> int:
        return execute()

    def finish(values: list[int]) -> dict[str, Any]:
        answer = interrupt("approve?")
        prepared = ",".join(str(value) for value in values)
        return {"messages": [AIMessage(content=f"{prepared}:{answer}")]}

    def ask(state: _BranchState) -> dict[str, Any]:
        if kind == "nested":
            return finish(nested_prepare().result())
        values = [prepare_task().result()]
        if kind == "repeated":
            values.append(prepare_task().result())
        return finish(values)

    async def ask_async(state: _BranchState) -> dict[str, Any]:
        if kind == "async_parallel":
            return finish(
                sorted(await asyncio.gather(async_prepare(), async_prepare()))
            )
        return finish([await async_prepare()])

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask_async if kind.startswith("async") else ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(
        checkpointer=saver, cache=InMemoryCache() if kind == "cached" else None
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "kind", ["single", "repeated", "nested", "async", "async_parallel", "cached"]
)
def test_paused_branches_reuse_completed_functional_tasks(
    stream: bool, kind: str
) -> None:
    operations: list[str] = []
    saver = InMemorySaver()
    graph = _functional_graph(saver, operations, kind)
    expected_operations = ["prepare"] * (
        2 if kind in {"repeated", "nested", "async_parallel"} else 1
    )
    expected_result = ",".join(
        str(value) for value in range(1, len(expected_operations) + 1)
    )
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        if kind == "cached":
            _post(client, "warm cache", stream=stream)
        parent = _post(client, "start", stream=stream)
        pending = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        parent_ref = _boundary_ref(client, server, parent["id"])
        config: RunnableConfig = {
            "configurable": {**parent_ref.to_dict(), "checkpoint_ns": ""}
        }
        before = saver.get_tuple(config)
        assert before is not None
        assert operations == expected_operations

        for answer in ("left", "right"):
            completed = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": pending["call_id"],
                        "output": json.dumps({"resume": answer}),
                    }
                ],
                previous_response_id=parent["id"],
                stream=stream,
            )
            assert completed["status"] == "completed", completed
            assert operations == expected_operations
            assert _text(completed) == f"{expected_result}:{answer}"
            assert _boundary_ref(client, server, completed["id"]).thread_id == (
                parent_ref.thread_id
            )

        assert saver.get_tuple(config) == before
        assert (
            client.get(f"/responses/{parent['id']}").json()["output"]
            == (parent["output"])
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("functional", [False, True])
def test_functional_results_survive_later_approvals_without_caching_new_work(
    stream: bool, functional: bool
) -> None:
    operations: list[str] = []

    @task
    def prepare(label: str) -> str:
        operations.append(label)
        return f"{label}-{operations.count(label)}"

    def approve() -> str:
        before = prepare("before").result()
        first = interrupt("first")
        after = prepare("after").result()
        second = interrupt("second")
        return f"{before}:{first}/{after}:{second}"

    approval_task = task(approve)

    def ask(state: _BranchState) -> dict[str, Any]:
        answer = approval_task().result() if functional else approve()
        return {"messages": [AIMessage(content=answer)]}

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
        parent = _post(client, "start", stream=stream)
        initial = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        assert operations == ["before"]
        for index, answer in enumerate(("left", "right"), start=1):
            partial = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": initial["call_id"],
                        "output": json.dumps({"resume": answer}),
                    }
                ],
                previous_response_id=parent["id"],
                stream=stream,
            )
            assert partial["status"] == "completed", partial
            pending = next(
                item for item in partial["output"] if item["type"] == "function_call"
            )
            completed = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": pending["call_id"],
                        "output": json.dumps({"resume": "done"}),
                    }
                ],
                previous_response_id=partial["id"],
                stream=stream,
            )
            assert completed["status"] == "completed", completed
            assert _text(completed) == f"before-1:{answer}/after-{index}:done"
            assert operations == ["before", *(["after"] * index)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_caught_functional_task_errors_preserve_retry_behavior(
    stream: bool, enabled: bool
) -> None:
    operations: list[str] = []

    @task
    def prepare() -> int:
        operations.append("attempt")
        raise RuntimeError("expected failure")

    def ask(state: _BranchState) -> dict[str, Any]:
        try:
            prepare().result()
        except RuntimeError:
            pass
        return {"messages": [AIMessage(content=f"recovered:{interrupt('approve?')}")]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        pending = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        completed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps({"resume": "yes"}),
                }
            ],
            previous_response_id=parent["id"],
            stream=stream,
        )
        assert completed["status"] == "completed", completed
        assert _text(completed) == "recovered:yes"
        assert operations == ["attempt", "attempt"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "malformed",
        "wrong_identity",
        "wrong_checkpoint",
        "version",
        "boolean_version",
        "read",
        "write",
    ],
)
def test_invalid_functional_identities_fail_before_replaying_work(
    stream: bool, interrupted: bool, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    operations: list[str] = []
    saver = InMemorySaver()
    graph = (
        _functional_interrupt_graph(saver, operations, "single")
        if interrupted
        else _functional_graph(saver, operations)
    )
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    original_read = saver.aget_tuple
    original_write = saver.aput

    async def read(config: RunnableConfig) -> Any:
        saved = await original_read(config)
        namespace = config.get("configurable", {}).get("checkpoint_ns", "")
        if not namespace.startswith("__langgraph_response_tasks__:"):
            return saved
        if failure == "read":
            raise OSError("private identity storage failure")
        if failure == "missing":
            return None
        if saved is not None and failure in {
            "malformed",
            "wrong_identity",
            "version",
            "boolean_version",
        }:
            value = saved.checkpoint["channel_values"]["identity"]
            if failure != "malformed":
                identity = json.loads(value)
                if failure == "wrong_identity":
                    identity["name"] = "another_task"
                else:
                    identity["version"] = 2 if failure == "version" else True
                value = json.dumps(identity)
            else:
                value = "{"
            return saved._replace(
                checkpoint={**saved.checkpoint, "channel_values": {"identity": value}}
            )
        if saved is not None and failure == "wrong_checkpoint":
            return saved._replace(
                checkpoint={**saved.checkpoint, "id": "another-checkpoint"}
            )
        return saved

    async def write(
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> RunnableConfig:
        if (
            config.get("configurable", {})
            .get("checkpoint_ns", "")
            .startswith("__langgraph_response_tasks__:")
        ):
            raise OSError("private identity storage failure")
        return await original_write(config, checkpoint, metadata, new_versions)

    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        pending = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        monkeypatch.setattr(saver, "aget_tuple", read)
        if failure == "write":
            monkeypatch.setattr(saver, "aput", write)
        completed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps({"resume": "yes"}),
                }
            ],
            previous_response_id=parent["id"],
            stream=stream,
        )
        assert completed["status"] == "failed", completed
        assert completed["error"]["code"] == "server_error"
        assert "private identity storage failure" not in json.dumps(completed)
        assert operations == (["one"] if interrupted else ["prepare"])
        assert client.portal is not None
        assert (
            client.portal.call(
                server._conversation_chain_store.get, completed["id"], BRANCH_ORIGIN_KEY
            )
            is None
        )
        assert (
            client.portal.call(
                server._conversation_chain_store.get,
                completed["id"],
                BRANCH_BOUNDARY_KEY,
            )
            is None
        )


@pytest.mark.parametrize("stream", [False, True])
def test_functional_identity_write_failure_prevents_parent_publication(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    operations: list[str] = []
    saver = InMemorySaver()
    original_write = saver.aput

    async def write(
        config: RunnableConfig,
        checkpoint: Any,
        metadata: Any,
        new_versions: Any,
    ) -> RunnableConfig:
        if (
            config.get("configurable", {})
            .get("checkpoint_ns", "")
            .startswith("__langgraph_response_tasks__:")
        ):
            raise OSError("private identity storage failure")
        return await original_write(config, checkpoint, metadata, new_versions)

    monkeypatch.setattr(saver, "aput", write)
    server = ResponsesHostServer(
        _functional_graph(saver, operations),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        failed = _post(client, "start", stream=stream)
        assert failed["status"] == "failed", failed
        assert failed["error"]["code"] == "server_error"
        assert "private identity storage failure" not in json.dumps(failed)
        assert operations == ["prepare"]
        assert client.portal is not None
        assert (
            client.portal.call(
                server._conversation_chain_store.get, failed["id"], BRANCH_BOUNDARY_KEY
            )
            is None
        )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("child_checkpointer", [None, True])
def test_completed_subgraph_does_not_block_root_approval(
    stream: bool, child_checkpointer: bool | None
) -> None:
    executions: list[str] = []

    def child_work(state: _BranchState) -> dict[str, Any]:
        executions.append("child")
        return {"ledger": ["child"]}

    def approve(state: _BranchState) -> dict[str, Any]:
        return {"ledger": [f"approval:{interrupt('approve?')}"]}

    def finish(state: _BranchState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=",".join(sorted(state["ledger"])))]}

    child = StateGraph(_BranchState)
    child.add_node("work", child_work)
    child.add_edge(START, "work")
    child.add_edge("work", END)
    builder = StateGraph(_BranchState)
    builder.add_node("child", child.compile(checkpointer=child_checkpointer))
    builder.add_node("approval", approve)
    builder.add_node("finish", finish)
    builder.add_edge(START, "child")
    builder.add_edge(START, "approval")
    builder.add_edge(["child", "approval"], "finish")
    builder.add_edge("finish", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        pending = next(
            item for item in parent["output"] if item["type"] == "function_call"
        )
        assert executions == ["child"]

        for answer in ("Alice", "Bob"):
            completed = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": pending["call_id"],
                        "output": json.dumps({"resume": answer}),
                    }
                ],
                previous_response_id=parent["id"],
                stream=stream,
            )
            assert completed["status"] == "completed", completed
            assert _text(completed) == f"approval:{answer},child"
            assert executions == ["child"]

        waiting = _post(
            client, "waiting", previous_response_id=parent["id"], stream=stream
        )
        assert waiting["status"] == "completed", waiting
        assert any(item["type"] == "function_call" for item in waiting["output"])
        assert executions == ["child"]
        assert (
            client.get(f"/responses/{parent['id']}").json()["output"]
            == parent["output"]
        )


@pytest.mark.parametrize("stream", [False, True])
def test_paused_subgraph_fork_fails_without_mutating_parent(stream: bool) -> None:
    builder = StateGraph(_BranchState)
    builder.add_node("child", build_simple_interrupt_graph())
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start", stream=stream)
        parent_ref = _boundary_ref(client, server, parent["id"])
        config: RunnableConfig = {
            "configurable": {**parent_ref.to_dict(), "checkpoint_ns": ""}
        }
        saved = saver.get_tuple(config)
        pending = next(
            item for item in parent["output"] if item["type"] == "function_call"
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
            previous_response_id=parent["id"],
            stream=stream,
        )
        assert client.portal is not None
        assert (
            client.portal.call(
                server._conversation_chain_store.get, failed["id"], BRANCH_ORIGIN_KEY
            )
            is None
        )
    assert failed["status"] == "failed", failed
    assert "Paused subgraph branching is not supported" in failed["error"]["message"]
    assert saver.get_tuple(config) == saved


@pytest.mark.parametrize("stream", [False, True])
def test_root_only_branching_leaves_private_subgraph_memory_application_owned(
    stream: bool,
) -> None:
    class ChildState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        count: int

    root_inputs: list[list[str]] = []
    child_counts: list[int] = []

    def record_root(state: _BranchState) -> dict[str, Any]:
        inputs = [
            str(message.content)
            for message in state["messages"]
            if isinstance(message, HumanMessage)
        ]
        root_inputs.append(inputs)
        return {"ledger": [inputs[-1]]}

    def record_child(state: ChildState) -> dict[str, Any]:
        count = state.get("count", 0) + 1
        child_counts.append(count)
        return {"count": count, "messages": [AIMessage(content=f"child:{count}")]}

    child_builder = StateGraph(ChildState)
    child_builder.add_node("record", record_child)
    child_builder.add_edge(START, "record")
    child_builder.add_edge("record", END)
    builder = StateGraph(_BranchState)
    builder.add_node("record", record_root)
    builder.add_node("child", child_builder.compile(checkpointer=True))
    builder.add_edge(START, "record")
    builder.add_edge("record", "child")
    builder.add_edge("child", END)
    saver = InMemorySaver()
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "A", stream=stream)
        root_ref = _boundary_ref(client, server, root["id"])
        root_config: RunnableConfig = {
            "configurable": {**root_ref.to_dict(), "checkpoint_ns": ""}
        }
        saved = saver.get_tuple(root_config)
        assert saved is not None
        original = _post(client, "B", previous_response_id=root["id"], stream=stream)
        fork = _post(client, "C", previous_response_id=root["id"], stream=stream)
        fork_ref = _boundary_ref(client, server, fork["id"])
        fork_state = saver.get_tuple(
            {"configurable": {**fork_ref.to_dict(), "checkpoint_ns": ""}}
        )

    assert all(response["status"] == "completed" for response in (root, original, fork))
    assert root_inputs == [["A"], ["A", "B"], ["A", "C"]]
    assert child_counts == [1, 2, 3]
    assert fork_state is not None
    assert fork_state.checkpoint["channel_values"]["ledger"] == ["A", "C"]
    assert "count" not in fork_state.checkpoint["channel_values"]
    assert saver.get_tuple(root_config) == saved


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
def test_application_owned_subgraph_snapshots_follow_root_branches(
    stream: bool, concurrent: bool
) -> None:
    graph, snapshots = _application_owned_subgraph_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        root = _post(client, "A", stream=stream)
        root_ref = _boundary_ref(client, server, root["id"])
        root_config: RunnableConfig = {
            "configurable": {**root_ref.to_dict(), "checkpoint_ns": ""}
        }
        saved = saver.get_tuple(root_config)
        assert saved is not None
        if concurrent:
            with ThreadPoolExecutor(max_workers=2) as executor:
                original_future = executor.submit(
                    _post, client, "B", previous_response_id=root["id"], stream=stream
                )
                fork_future = executor.submit(
                    _post, client, "C", previous_response_id=root["id"], stream=stream
                )
                original, fork = original_future.result(), fork_future.result()
        else:
            original = _post(
                client, "B", previous_response_id=root["id"], stream=stream
            )
            fork = _post(client, "C", previous_response_id=root["id"], stream=stream)
        original_next = _post(
            client, "D", previous_response_id=original["id"], stream=stream
        )
        fork_next = _post(client, "E", previous_response_id=fork["id"], stream=stream)
        for response, ledger in (
            (root, ["A"]),
            (original, ["A", "B"]),
            (fork, ["A", "C"]),
            (original_next, ["A", "B", "D"]),
            (fork_next, ["A", "C", "E"]),
        ):
            assert response["status"] == "completed", response
            ref = _boundary_ref(client, server, response["id"])
            checkpoint = saver.get_tuple(
                {"configurable": {**ref.to_dict(), "checkpoint_ns": ""}}
            )
            assert checkpoint is not None
            values = checkpoint.checkpoint["channel_values"]
            assert values["ledger"] == ledger
            assert snapshots[values["child_state_ref"]] == len(ledger)

    assert [_text(response) for response in (root, original, fork)] == [
        "child:1",
        "child:2",
        "child:2",
    ]
    assert _text(original_next) == _text(fork_next) == "child:3"
    assert len(snapshots) == 5
    assert snapshots[saved.checkpoint["channel_values"]["child_state_ref"]] == 1
    assert saver.get_tuple(root_config) == saved


@pytest.mark.parametrize("stream", [False, True])
def test_missing_application_owned_subgraph_snapshot_fails_explicitly(
    stream: bool,
) -> None:
    graph, snapshots = _application_owned_subgraph_graph()
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        root = _post(client, "A", stream=stream)
        root_ref = _boundary_ref(client, server, root["id"])
        root_config: RunnableConfig = {
            "configurable": {**root_ref.to_dict(), "checkpoint_ns": ""}
        }
        saved = saver.get_tuple(root_config)
        assert saved is not None
        del snapshots[saved.checkpoint["channel_values"]["child_state_ref"]]
        failed = _post(client, "B", previous_response_id=root["id"], stream=stream)

    assert failed["status"] == "failed"
    assert failed["error"]["code"] == "server_error"
    assert snapshots == {}
    assert saver.get_tuple(root_config) == saved


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
@pytest.mark.parametrize("namespace", ["", "child:task"])
async def test_checkpoint_saver_records_only_root_response_ownership(
    enabled: bool, asynchronous: bool, namespace: str
) -> None:
    from langchain_azure_ai.agents.hosting._responses.branching import BranchingError

    saver = InMemorySaver()
    origin = CheckpointRef("root", "origin")
    adapter = ResponseCheckpointSaver(
        saver, branching=enabled, response_key="user-a:child", origin=origin
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": "root", "checkpoint_ns": namespace}
    }
    metadata = get_checkpoint_metadata(
        {"metadata": {"langgraph_response_execution": "untrusted-owner"}},
        {"source": "loop", "step": 1, "parents": {}},
    )
    original_metadata = deepcopy(metadata)
    checkpoint = empty_checkpoint()
    saved_config = (
        await adapter.aput(config, checkpoint, metadata, {})
        if asynchronous
        else adapter.put(config, checkpoint, metadata, {})
    )
    saved = await saver.aget_tuple(saved_config)
    assert saved is not None
    assert metadata == original_metadata
    assert saved.metadata["source"] == "loop"
    assert saved.metadata["step"] == 1
    assert saved.metadata["parents"] == {}
    if not enabled or namespace:
        assert saved.metadata == original_metadata
        return

    recreated = ResponseCheckpointSaver(
        saver, branching=True, response_key="user-a:child", origin=origin
    )
    await recreated.validate_progress(saved_config)
    for response_key, selected_origin in (
        ("user-b:child", origin),
        ("user-a:child", CheckpointRef("root", "other-origin")),
    ):
        foreign = ResponseCheckpointSaver(
            saver, branching=True, response_key=response_key, origin=selected_origin
        )
        with pytest.raises(BranchingError, match="ownership could not be verified"):
            await foreign.validate_progress(saved_config)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_checkpoint_saver_persists_functional_task_identities(
    asynchronous: bool,
) -> None:
    saver = InMemorySaver()
    adapter = ResponseCheckpointSaver(saver, branching=True, response_key="functional")
    operations: list[str] = []
    graph = _functional_graph(adapter, operations, "nested")
    config: RunnableConfig = {
        "configurable": {"thread_id": "functional"},
        "callbacks": [adapter.task_recorder],
    }
    inputs = {"messages": [HumanMessage(content="start")]}
    if asynchronous:
        await graph.ainvoke(inputs, config)
    else:
        graph.invoke(inputs, config)
    saved = await saver.aget_tuple(config)
    assert saved is not None
    task_ids = {
        task_id
        for task_id, channel, _ in saved.pending_writes or ()
        if channel == "__return__"
    }
    indexes = [
        item
        for item in saver.list(config)
        if item.config["configurable"]["checkpoint_ns"].startswith(
            "__langgraph_response_tasks__:"
        )
    ]
    assert len(indexes) == len(task_ids) == 3
    assert operations == ["prepare", "prepare"]
    assert {
        item.config["configurable"]["checkpoint_ns"].rsplit(":", 1)[-1]
        for item in indexes
    } == task_ids
    for item in indexes:
        assert item.checkpoint["id"] == saved.checkpoint["id"]
        identity = json.loads(item.checkpoint["channel_values"]["identity"])
        assert set(identity) == {"name", "parent_ns", "path", "step", "version"}
        assert identity["version"] == 1
    if asynchronous:
        await adapter.adelete_thread("functional")
    else:
        adapter.delete_thread("functional")
    assert list(saver.list(config)) == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_resume_saver_keeps_branch_head_and_writes_isolated(
    asynchronous: bool,
) -> None:
    from langchain_azure_ai.agents.hosting._responses.branching import BranchingError

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
    assert selected_checkpoint is not None
    assert sibling_checkpoint is not None
    assert selected_checkpoint.pending_writes == [
        ("task", "result", "already-completed")
    ]
    assert sibling_checkpoint.pending_writes == []
    child_config: RunnableConfig = {
        "configurable": {"thread_id": "root", "checkpoint_ns": "child:task"}
    }
    await adapter.aput(child_config, empty_checkpoint(), metadata, {})
    await saver.aput(config, empty_checkpoint(), metadata, {})

    async def read() -> Any:
        return (
            await adapter.aget_tuple(execution_config)
            if asynchronous
            else adapter.get_tuple(execution_config)
        )

    saved = await read()
    assert saved.config == advanced
    assert saved.parent_config == selected
    with pytest.raises(BranchingError, match="thread does not match"):
        adapter.get_tuple({"configurable": {"thread_id": "other"}})
    await saver.adelete_thread("root")
    await saver.aput(config, empty_checkpoint(), metadata, {})
    with pytest.raises(BranchingError, match="required checkpoint"):
        await read()


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
@pytest.mark.parametrize(
    "progress",
    ["none", "saved", "unowned", "invalid_owner", "missing", "malformed"],
)
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
    if progress in {"saved", "unowned", "invalid_owner"} or boundary_published:
        execution_graph = graph
        execution_config = parent_config
        if progress == "invalid_owner":
            execution_config = {
                **parent_config,
                "metadata": {"langgraph_response_execution": 0},
            }
        elif progress != "unowned":
            saver = graph.checkpointer
            assert isinstance(saver, BaseCheckpointSaver)
            execution_graph = graph.copy(
                {
                    "checkpointer": ResponseCheckpointSaver(
                        saver,
                        branching=True,
                        response_key="child",
                        origin=parent_ref,
                    )
                }
            )
        await execution_graph.ainvoke(
            {"messages": [HumanMessage(content="B")]}, execution_config
        )
        completed_ref = HostingRunnableConfig(
            (await graph.aget_state(config)).config
        ).checkpoint_ref
        assert completed_ref is not None
        saved_ref = completed_ref
    if progress in {"saved", "unowned", "invalid_owner"}:
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
    if boundary_published or progress in {
        "unowned",
        "invalid_owner",
        "missing",
        "malformed",
    }:
        assert terminal["status"] == "failed"
        assert terminal["error"]["code"] == "server_error"
        assert executions == []
    else:
        assert terminal["status"] == "completed", terminal
        assert executions == (["B"] if progress == "none" else [])
    provider.get_response.assert_not_awaited()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("progress", ["own", "sibling", "parent", "different_origin"])
async def test_recovery_requires_progress_from_the_same_response_execution(
    enabled: bool, stream: bool, progress: str
) -> None:
    graph, executions = _branch_graph()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _post(client, "A")
        parent_ref = _boundary_ref(client, server, parent["id"])
        sibling = _post(client, "C", previous_response_id=parent["id"])
        sibling_ref = _boundary_ref(client, server, sibling["id"])

    response_id = parent["id"][:-32] + uuid4().hex
    context = _context(response_id=response_id, conversation_id=None, current_text="B")
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    request = _request(previous_response_id=parent["id"], stream=stream)
    interrupted = server.handle_create(request, context, asyncio.Event())
    assert isinstance(interrupted, AsyncGenerator)
    persisted = None
    try:
        async for event in interrupted:
            if isinstance(event, dict):
                continue
            metadata = json.loads(
                event.response.get("metadata", {}).get("_internal_metadata", "{}")
            )
            if metadata.get(METADATA_LANGGRAPH_CHECKPOINT_ID):
                assert (
                    metadata[METADATA_LANGGRAPH_CHECKPOINT_ID]
                    != parent_ref.checkpoint_id
                )
                persisted = deepcopy(event.response)
                break
    finally:
        await interrupted.aclose()
    assert persisted is not None
    executions_before = list(executions)
    if progress in {"sibling", "parent"}:
        foreign_ref = sibling_ref if progress == "sibling" else parent_ref
        metadata = json.loads(persisted["metadata"]["_internal_metadata"])
        assert metadata[METADATA_LANGGRAPH_THREAD_ID] == foreign_ref.thread_id
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = foreign_ref.checkpoint_id
        persisted["metadata"]["_internal_metadata"] = json.dumps(metadata)
    elif progress == "different_origin":
        origin = await server._conversation_chain_store.get(
            response_id, BRANCH_ORIGIN_KEY
        )
        assert origin is not None
        await server._conversation_chain_store.set(
            response_id,
            BRANCH_ORIGIN_KEY,
            {**origin, **sibling_ref.to_dict()},
        )

    recreated = ResponsesHostServer(
        graph, store=provider, enable_response_branching=enabled
    )
    recovery = _context(response_id=response_id, conversation_id=None)
    recovery.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    recovery.is_recovery = True
    recovery.persisted_response = persisted
    events = [
        event
        async for event in recreated.handle_create(request, recovery, asyncio.Event())
    ]
    terminal = events[-1]["response"]
    recovery.get_input_items.assert_not_awaited()
    if progress != "own":
        assert terminal["status"] == "failed", terminal
        assert terminal["error"]["code"] == "server_error"
        assert executions == executions_before
        assert (
            await recreated._conversation_chain_store.get(
                response_id, BRANCH_BOUNDARY_KEY
            )
            is None
        )
    else:
        assert terminal["status"] == "completed", terminal
        assert _text(terminal) == "A,B"
        assert executions == ["A", "C", "B"]


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
    execution_graph = graph
    execution_config = config
    origin_ref = None
    if enabled:
        await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
        execution_config = (await graph.aget_state(config)).config
        origin_ref = HostingRunnableConfig(execution_config).checkpoint_ref
        assert origin_ref is not None
        saver = graph.checkpointer
        assert isinstance(saver, BaseCheckpointSaver)
        execution_graph = graph.copy(
            {
                "checkpointer": ResponseCheckpointSaver(
                    saver, branching=True, response_key="child", origin=origin_ref
                )
            }
        )
        executions.clear()
    await execution_graph.ainvoke(
        graph_input, execution_config, interrupt_before=["record"]
    )
    paused = await graph.aget_state(config)
    assert paused.next == ("record",)
    saved_ref = HostingRunnableConfig(paused.config).checkpoint_ref
    assert saved_ref is not None
    metadata: dict[str, Any] = {
        METADATA_LANGGRAPH_THREAD_ID: saved_ref.thread_id,
        METADATA_LANGGRAPH_CHECKPOINT_ID: saved_ref.checkpoint_id,
    }
    if enabled:
        assert origin_ref is not None
        metadata[BRANCH_MODE_METADATA] = BRANCH_MODE
        context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
        await server._conversation_chain_store.set(
            "child",
            BRANCH_ORIGIN_KEY,
            {
                **ResponseBranchStore._record(origin_ref, paused=False),
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
def test_paused_fork_failure_leaves_parent_unchanged(
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


async def test_paused_origin_recovery_reuses_same_thread_checkpoint(
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
    parent_checkpoint = await saver.aget_tuple(parent.config)
    assert parent_checkpoint is not None
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
    origin, resume_ids = await branches.prepare(
        response_key="child",
        parent_key="parent",
        parent_id="parent",
        context=context,
        graph=graph,
    )
    assert origin.thread_id == parent_ref.thread_id
    assert origin.checkpoint_id != parent_ref.checkpoint_id
    assert records["child", BRANCH_ORIGIN_KEY] == {
        **ResponseBranchStore._record(origin, paused=True),
        "version": "2",
        "mode": BRANCH_MODE,
        "parent_response_id": "parent",
        "resume_ids": json.dumps(resume_ids),
    }
    fork = await graph.aget_state(
        {"configurable": {**origin.to_dict(), "checkpoint_ns": ""}}
    )
    fork_checkpoint = await saver.aget_tuple(fork.config)
    assert fork_checkpoint is not None
    assert fork_checkpoint.parent_config == parent_checkpoint.parent_config
    assert resume_ids == {parent.interrupts[0].id: fork.interrupts[0].id}
    assert fork.interrupts[0].value == parent.interrupts[0].value
    assert await graph.aget_state(parent.config) == parent
    monkeypatch.setattr(saver, "aput", AsyncMock(side_effect=AssertionError("recopy")))
    repeated = await branches.prepare(
        response_key="child",
        parent_key="parent",
        parent_id="parent",
        context=context,
        graph=graph,
    )
    assert repeated == (origin, resume_ids)
    context.is_recovery = True
    restored = await branches.prepare(
        response_key="child",
        parent_key="parent",
        parent_id="parent",
        context=context,
        graph=graph,
    )
    assert restored == (origin, resume_ids)
    assert provider.get_response.await_count == 2
    store.set.assert_awaited_once()


@pytest.mark.parametrize(
    "resume_ids",
    [
        "",
        "{invalid",
        "[]",
        '{"id": 1}',
        '{"id": ""}',
        '{"a": "same", "b": "same"}',
        "{}",
    ],
)
async def test_recovery_rejects_invalid_resume_mapping(resume_ids: str) -> None:
    origin = {
        **ResponseBranchStore._record(
            CheckpointRef("parent", "checkpoint"), paused=True
        ),
        "version": "2",
        "mode": BRANCH_MODE,
        "parent_response_id": "parent",
        "resume_ids": resume_ids,
    }
    store = MagicMock()
    store.get = AsyncMock(
        side_effect=lambda _, key: origin if key == BRANCH_ORIGIN_KEY else None
    )
    provider = MagicMock()
    provider.get_response = AsyncMock()
    context = _context(response_id="child", conversation_id=None)
    context.is_recovery = True
    with pytest.raises(ValueError, match="resume mapping is invalid"):
        await ResponseBranchStore(store, provider).prepare(
            response_key="child",
            parent_key="parent",
            parent_id="parent",
            context=context,
        )
    provider.get_response.assert_not_awaited()


async def test_recovery_preserves_legacy_origin_without_resume_mapping() -> None:
    ref = CheckpointRef("legacy-child", "checkpoint")
    origin = {
        **ResponseBranchStore._record(ref, paused=True),
        "mode": BRANCH_MODE,
        "parent_response_id": "parent",
    }
    store = MagicMock()
    store.get = AsyncMock(
        side_effect=lambda _, key: origin if key == BRANCH_ORIGIN_KEY else None
    )
    context = _context(response_id="child", conversation_id=None)
    context.is_recovery = True
    assert await ResponseBranchStore(store).prepare(
        response_key="child",
        parent_key="parent",
        parent_id="parent",
        context=context,
    ) == (ref, {})


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
def test_partial_approval_preserves_unanswered_approval_identity(
    enabled: bool, stream: bool
) -> None:
    operations: list[str] = []
    server = ResponsesHostServer(
        _partial_approval_graph(InMemorySaver(), operations),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start", stream=stream)
        approvals = {
            json.loads(item["arguments"])["value"]: item
            for item in root["output"]
            if item["type"] == "mcp_approval_request"
        }
        partial = _post(
            client,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approvals["A"]["id"],
                    "approve": True,
                }
            ],
            previous_response_id=root["id"],
            stream=stream,
        )
        remaining = next(
            item for item in partial["output"] if item["type"] == "mcp_approval_request"
        )
        if enabled:
            assert remaining == approvals["B"]
            function_call = next(
                item for item in partial["output"] if item["type"] == "function_call"
            )
            assert function_call["arguments"] == approvals["B"]["arguments"]
            assert (
                function_call["call_id"]
                == json.loads(approvals["B"]["arguments"])["interrupt_id"]
            )
        else:
            assert remaining["id"] == approvals["B"]["id"]
        completed = _post(
            client,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": remaining["id"],
                    "approve": True,
                }
            ],
            previous_response_id=partial["id"],
            stream=stream,
        )
        assert client.get(f"/responses/{root['id']}").json()["output"] == root["output"]

    assert _text(completed) == "1:B"
    assert operations == ["A", "B"]


@pytest.mark.parametrize("stream", [False, True])
def test_new_approval_in_same_task_does_not_reuse_answered_identity(
    stream: bool,
) -> None:
    def ask_a(state: _BranchState) -> dict[str, Any]:
        first = interrupt("same")
        second = interrupt("same")
        return {"ledger": [f"A:{first}/{second}"]}

    def ask_b(state: _BranchState) -> dict[str, Any]:
        return {"ledger": [f"B:{interrupt('B')}"]}

    def finish(state: _BranchState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=",".join(sorted(state["ledger"])))]}

    builder = StateGraph(_BranchState)
    builder.add_node("a", ask_a)
    builder.add_node("b", ask_b)
    builder.add_node("finish", finish)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge(["a", "b"], "finish")
    builder.add_edge("finish", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start", stream=stream)
        original = {
            json.loads(item["arguments"])["value"]: item
            for item in root["output"]
            if item["type"] == "mcp_approval_request"
        }
        answered_a = {
            "type": "mcp_approval_response",
            "approval_request_id": original["same"]["id"],
            "approve": True,
        }
        partial = _post(
            client, [answered_a], previous_response_id=root["id"], stream=stream
        )
        remaining = {
            json.loads(item["arguments"])["value"]: item
            for item in partial["output"]
            if item["type"] == "mcp_approval_request"
        }
        assert remaining["same"]["id"] != original["same"]["id"]
        assert remaining["B"] == original["B"]
        answered_b = _post(
            client,
            [
                answered_a,
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": original["B"]["id"],
                    "approve": True,
                },
            ],
            previous_response_id=partial["id"],
            stream=stream,
        )
        assert [
            item
            for item in answered_b["output"]
            if item["type"] == "mcp_approval_request"
        ] == [remaining["same"]]
        completed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": json.loads(remaining["same"]["arguments"])[
                        "interrupt_id"
                    ],
                    "output": "two",
                }
            ],
            previous_response_id=answered_b["id"],
            stream=stream,
        )

    assert _text(completed) == "A:same/two,B:B"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("resume_a", [False, True])
def test_changed_approval_payload_gets_new_identity(
    stream: bool, resume_a: bool
) -> None:
    def ask_a(state: _BranchState) -> dict[str, Any]:
        interrupt("A")
        return {}

    def ask_b(state: _BranchState) -> dict[str, Any]:
        answer = interrupt(
            {"operation": "send", "arguments": {"targets": state.get("ledger", [])}}
        )
        return {"messages": [AIMessage(content=str(answer))]}

    builder = StateGraph(_BranchState)
    builder.add_node("a", ask_a)
    builder.add_node("b", ask_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start", stream=stream)
        original = [
            item for item in root["output"] if item["type"] == "mcp_approval_request"
        ]
        a = next(
            item for item in original if json.loads(item["arguments"])["value"] == "A"
        )
        b = next(item for item in original if item != a)
        command: dict[str, Any] = {"update": {"ledger": ["changed"]}}
        if resume_a:
            command["resume"] = "done"
        partial = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": json.loads(a["arguments"])["interrupt_id"],
                    "output": json.dumps(command),
                }
            ],
            previous_response_id=root["id"],
            stream=stream,
        )
        remaining = [
            item for item in partial["output"] if item["type"] == "mcp_approval_request"
        ]
        if not resume_a:
            assert a in remaining
        changed = next(item for item in remaining if item != a)
        assert changed["id"] != b["id"]
        assert json.loads(changed["arguments"])["value"] == {
            "operation": "send",
            "arguments": {"targets": ["changed"]},
        }
        waiting = _post(
            client,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": b["id"],
                    "approve": True,
                }
            ],
            previous_response_id=partial["id"],
            stream=stream,
        )
        assert {
            item["id"]: item
            for item in waiting["output"]
            if item["type"] == "mcp_approval_request"
        } == {item["id"]: item for item in remaining}
        completed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": json.loads(item["arguments"])["interrupt_id"],
                    "output": "accepted",
                }
                for item in remaining
            ],
            previous_response_id=waiting["id"],
            stream=stream,
        )
        assert client.get(f"/responses/{root['id']}").json()["output"] == root["output"]

    assert _text(completed) == "accepted"


@pytest.mark.parametrize(
    "corruption",
    ["invalid-json", "not-object", "wrong-key", "wrong-arguments", "wrong-id"],
)
def test_corrupt_approval_snapshots_fail_before_execution(corruption: str) -> None:
    class CorruptStore:
        def __init__(self) -> None:
            self.records: dict[tuple[str, str], dict[str, str]] = {}

        async def get(self, chain: str, key: str) -> dict[str, str] | None:
            record = self.records.get((chain, key))
            if record is None or key != BRANCH_ORIGIN_KEY:
                return record
            if corruption == "invalid-json":
                snapshots = "{invalid"
            elif corruption == "not-object":
                snapshots = "[]"
            else:
                approvals = json.loads(record["approval_requests"])
                wire_id = next(iter(approvals))
                if corruption == "wrong-key":
                    approvals["unrelated"] = approvals.pop(wire_id)
                elif corruption == "wrong-arguments":
                    approvals[wire_id]["arguments"] = json.dumps(
                        {"interrupt_id": "unrelated", "value": "A"}
                    )
                else:
                    approvals[wire_id]["id"] = "mcpr_invalid"
                snapshots = json.dumps(approvals)
            return {**record, "approval_requests": snapshots}

        async def set(self, chain: str, key: str, data: dict[str, str]) -> None:
            self.records[chain, key] = data

    operations: list[str] = []
    server = ResponsesHostServer(
        _partial_approval_graph(InMemorySaver(), operations),
        store=InMemoryResponseProvider(),
        conversation_chain_store=CorruptStore(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start")
        approval = next(
            item for item in root["output"] if item["type"] == "mcp_approval_request"
        )
        failed = _post(
            client,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
            ],
            previous_response_id=root["id"],
        )
        assert failed["status"] == "failed", failed
        assert failed["error"]["code"] == "server_error"
        assert client.get(f"/responses/{root['id']}").json()["output"] == root["output"]

    assert operations == []


def test_approval_id_respects_an_overridden_response_storage_partition() -> None:
    operations: list[str] = []
    server = ResponsesHostServer(
        _partial_approval_graph(InMemorySaver(), operations),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start")
        approvals = {
            json.loads(item["arguments"])["value"]: item
            for item in root["output"]
            if item["type"] == "mcp_approval_request"
        }
        response_id = "caresp_0123456789abcdef00" + "a" * 32
        response = client.post(
            "/responses",
            headers={"x-agent-response-id": response_id},
            json={
                "model": "test",
                "previous_response_id": root["id"],
                "input": [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approvals["A"]["id"],
                        "approve": True,
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        partial = response.json()
        assert partial["id"] == response_id
        remaining = next(
            item for item in partial["output"] if item["type"] == "mcp_approval_request"
        )
        assert remaining["id"].startswith("mcpr_0123456789abcdef00")
        assert remaining["id"] != approvals["B"]["id"]
        assert remaining["arguments"] == approvals["B"]["arguments"]
        assert remaining["response_id"] == response_id
        completed = _post(
            client,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": remaining["id"],
                    "approve": True,
                }
            ],
            previous_response_id=partial["id"],
        )
        assert client.get(f"/responses/{root['id']}").json()["output"] == root["output"]

    assert _text(completed) == "1:B"
    assert operations == ["A", "B"]


@pytest.mark.parametrize("resume_a", [False, True])
def test_new_agent_defaults_do_not_overwrite_original_approval_items(
    resume_a: bool,
) -> None:
    operations: list[str] = []
    server = ResponsesHostServer(
        _partial_approval_graph(InMemorySaver(), operations),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start")
        original = {
            json.loads(item["arguments"])["value"]: item
            for item in root["output"]
            if item["type"] == "mcp_approval_request"
        }
        agent_reference = {
            "type": "agent_reference",
            "name": "new-agent",
            "version": "1",
        }
        input_items = (
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": original["A"]["id"],
                    "approve": True,
                }
            ]
            if resume_a
            else "waiting"
        )
        partial = _post(
            client,
            input_items,
            previous_response_id=root["id"],
            agent_reference=agent_reference,
        )
        remaining = [
            item for item in partial["output"] if item["type"] == "mcp_approval_request"
        ]
        b = next(
            item for item in remaining if json.loads(item["arguments"])["value"] == "B"
        )
        assert b["id"] != original["B"]["id"]
        assert b["agent_reference"] == agent_reference
        assert client.get(f"/responses/{root['id']}").json()["output"] == root["output"]
        completed = _post(
            client,
            [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": item["id"],
                    "approve": True,
                }
                for item in remaining
            ],
            previous_response_id=partial["id"],
        )

    assert _text(completed) == "1:B"
    assert sorted(operations) == ["A", "B"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mcp", [False, True])
@pytest.mark.parametrize("join", [False, True])
def test_partial_approval_reuses_completed_task_result(
    enabled: bool, stream: bool, mcp: bool, join: bool
) -> None:
    operations: list[str] = []
    server = ResponsesHostServer(
        _partial_approval_graph(InMemorySaver(), operations, join=join),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        for question in ("A", "B"):
            approval = next(
                item
                for item in paused["output"]
                if item["type"] == ("mcp_approval_request" if mcp else "function_call")
                and json.loads(item["arguments"])["value"] == question
            )
            reply = (
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
                if mcp
                else {
                    "type": "function_call_output",
                    "call_id": approval["call_id"],
                    "output": json.dumps({"resume": question}),
                }
            )
            paused = _post(
                client,
                [reply],
                previous_response_id=paused["id"],
                stream=stream,
            )
            assert paused["status"] == "completed", paused
            if question == "A":
                assert operations == ["A"]

    assert _text(paused) == "1:B"
    assert operations == ["A", "B"]


@pytest.mark.parametrize("stream", [False, True])
def test_partial_approval_applies_delta_channel_writes_once(stream: bool) -> None:
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

    operations: list[str] = []

    def approve(question: str) -> dict[str, Any]:
        interrupt(question)
        operations.append(question)
        return {"ledger": [question]}

    def finish(state: DeltaState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=",".join(state["ledger"]))]}

    builder = StateGraph(DeltaState)
    builder.add_node("seed", lambda _: {"ledger": ["seed"]})
    builder.add_node("a", lambda _: approve("A"))
    builder.add_node("b", lambda _: approve("B"))
    builder.add_node("finish", finish)
    builder.add_edge(START, "seed")
    builder.add_edge("seed", "a")
    builder.add_edge("seed", "b")
    builder.add_edge(["a", "b"], "finish")
    builder.add_edge("finish", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        for question in ("A", "B"):
            approval = next(
                item
                for item in paused["output"]
                if item["type"] == "function_call"
                and json.loads(item["arguments"])["value"] == question
            )
            paused = _post(
                client,
                [
                    {
                        "type": "function_call_output",
                        "call_id": approval["call_id"],
                        "output": question,
                    }
                ],
                previous_response_id=paused["id"],
                stream=stream,
            )
            assert paused["status"] == "completed", paused
            if question == "A":
                assert operations == ["A"]

    assert operations == ["A", "B"]
    assert _text(paused) == "seed,A,B"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mcp", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_partial_approval_maps_current_reply_with_stale_id(
    stream: bool, mcp: bool, stale: bool
) -> None:
    operations: list[str] = []
    server = ResponsesHostServer(
        _partial_approval_graph(InMemorySaver(), operations),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start", stream=stream)
        kind = "mcp_approval_request" if mcp else "function_call"
        approvals = {
            json.loads(item["arguments"])["value"]: item
            for item in root["output"]
            if item["type"] == kind
        }
        first = next(
            item
            for item in root["output"]
            if item["type"] == "function_call"
            and json.loads(item["arguments"])["value"] == "A"
        )
        partial = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": first["call_id"],
                    "output": "A",
                }
            ],
            previous_response_id=root["id"],
            stream=stream,
        )
        pending = next(item for item in partial["output"] if item["type"] == kind)
        identity = "id" if mcp else "call_id"
        assert pending[identity] == approvals["B"][identity]
        assert pending["arguments"] == approvals["B"]["arguments"]
        if mcp:
            reply = [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": item["id"],
                    "approve": True,
                }
                for item in ([approvals["A"], pending] if stale else [pending])
            ]
        else:
            resume = {pending["call_id"]: "B"}
            if stale:
                resume[approvals["A"]["call_id"]] = "stale A"
            reply = [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps({"resume": resume}),
                }
            ]
        completed = _post(
            client, reply, previous_response_id=partial["id"], stream=stream
        )

    assert operations == ["A", "B"]
    assert _text(completed) == "1:B"
    assert not any(item["type"] == "function_call" for item in completed["output"])


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mcp", [False, True])
@pytest.mark.parametrize("functional", [False, True])
def test_concurrent_partial_approval_branches_reuse_their_own_results(
    stream: bool, mcp: bool, functional: bool
) -> None:
    operations: list[str] = []
    barrier = asyncio.Barrier(2)
    partial_id: str | None = None

    class CoordinatedHost(ResponsesHostServer):
        async def build_runnable_config(
            self, request: Any, context: Any
        ) -> RunnableConfig:
            config = await super().build_runnable_config(request, context)
            if (
                partial_id is not None
                and request.get("previous_response_id") == partial_id
            ):
                await asyncio.wait_for(barrier.wait(), timeout=5)
            return config

    server = CoordinatedHost(
        _partial_approval_graph(InMemorySaver(), operations, functional=functional),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start", stream=stream)
        original_b = next(
            item
            for item in root["output"]
            if item["type"] == "mcp_approval_request"
            and json.loads(item["arguments"])["value"] == "B"
        )
        approval = next(
            item
            for item in root["output"]
            if item["type"] == "function_call"
            and json.loads(item["arguments"])["value"] == "A"
        )
        partial = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": approval["call_id"],
                    "output": "A",
                }
            ],
            previous_response_id=root["id"],
            stream=stream,
        )
        partial_id = partial["id"]
        pending = next(
            item for item in partial["output"] if item["type"] == "function_call"
        )
        assert (
            next(
                item
                for item in partial["output"]
                if item["type"] == "mcp_approval_request"
            )
            == original_b
        )

        def approve(answer: str) -> dict[str, Any]:
            reply = (
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": original_b["id"],
                    "approve": True,
                }
                if mcp
                else {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": answer,
                }
            )
            return _post(
                client,
                [reply],
                previous_response_id=partial["id"],
                stream=stream,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            completed = list(executor.map(approve, ["left", "right"]))
        assert (
            client.get(f"/responses/{partial['id']}").json()["output"]
            == partial["output"]
        )

    assert [_text(response) for response in completed] == (
        ["1:B", "1:B"] if mcp else ["1:left", "1:right"]
    )
    assert operations.count("A") == 1
    assert operations.count("B") == 2


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("answers", [("A", "B", "C"), ("C", "A", "B")])
def test_partial_send_approvals_reuse_completed_and_empty_results(
    stream: bool, answers: tuple[str, ...]
) -> None:
    operations: list[str] = []

    def distribute(state: _BranchState) -> list[Send]:
        return [
            Send("review", {"messages": [HumanMessage(content=name)]})
            for name in ("A", "B", "C")
        ]

    def review(state: _BranchState) -> dict[str, Any]:
        name = str(state["messages"][0].content)
        interrupt(name)
        operations.append(name)
        return {} if name == "B" else {"ledger": [f"{name}:{operations.count(name)}"]}

    def finish(state: _BranchState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=",".join(sorted(state["ledger"])))]}

    builder = StateGraph(_BranchState)
    builder.add_node("review", review)
    builder.add_node("finish", finish)
    builder.add_conditional_edges(START, distribute, ["review"])
    builder.add_edge("review", "finish")
    builder.add_edge("finish", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        original = {
            json.loads(item["arguments"])["value"]: item
            for item in paused["output"]
            if item["type"] == "mcp_approval_request"
        }
        for index, question in enumerate(answers):
            paused = _post(
                client,
                [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": original[question]["id"],
                        "approve": True,
                    }
                ],
                previous_response_id=paused["id"],
                stream=stream,
            )
            assert paused["status"] == "completed", paused
            expected = {
                question: original[question] for question in answers[index + 1 :]
            }
            assert {
                json.loads(item["arguments"])["value"]: item
                for item in paused["output"]
                if item["type"] == "mcp_approval_request"
            } == expected
            if expected:
                paused = _post(
                    client, "waiting", previous_response_id=paused["id"], stream=stream
                )
                assert {
                    json.loads(item["arguments"])["value"]: item
                    for item in paused["output"]
                    if item["type"] == "mcp_approval_request"
                } == expected

    assert _text(paused) == "A:1,C:1"
    assert operations == list(answers)


@pytest.mark.parametrize("functional", [False, True])
async def test_recovery_reuses_completed_partial_approval_results(
    functional: bool,
) -> None:
    operations: list[str] = []
    saver = InMemorySaver()
    graph = _partial_approval_graph(saver, operations, functional=functional)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=True
    )
    with TestClient(server.app) as client:
        parent = _post(client, "start")
    context = _context(response_id="child", conversation_id=None)
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    request = _request(previous_response_id=parent["id"])
    config = await server.build_runnable_config(request, context)
    ref, _ = await server._branch_store.prepare(
        response_key="child",
        parent_key=parent["id"],
        parent_id=parent["id"],
        context=context,
        graph=graph,
        config=config,
    )
    config = HostingRunnableConfig(config).with_checkpoint_ref(ref).runnable_config
    initial = await graph.aget_state(config)
    origin_ref = HostingRunnableConfig(initial.config).checkpoint_ref
    assert origin_ref is not None
    approval = next(item for item in initial.interrupts if item.value == "A")
    adapter = ResponseCheckpointSaver(
        saver, branching=True, response_key="child", origin=origin_ref
    )
    execution_graph = graph.copy({"checkpointer": adapter})
    await execution_graph.ainvoke(
        Command(resume={approval.id: "A"}), adapter.resume_config(config)
    )
    progress = await graph.aget_state(config)
    progress_ref = HostingRunnableConfig(progress.config).checkpoint_ref
    assert progress_ref == origin_ref
    context.is_recovery = True
    context.persisted_response = _response_object(
        "child",
        previous_response_id=parent["id"],
        internal_metadata={
            BRANCH_MODE_METADATA: BRANCH_MODE,
            METADATA_LANGGRAPH_THREAD_ID: progress_ref.thread_id,
            METADATA_LANGGRAPH_CHECKPOINT_ID: progress_ref.checkpoint_id,
        },
    )
    events = [
        event async for event in server.handle_create(request, context, asyncio.Event())
    ]
    recovered = events[-1]["response"]
    assert recovered["status"] == "completed", recovered
    assert [
        json.loads(item["arguments"])["value"]
        for item in recovered["output"]
        if item["type"] == "mcp_approval_request"
    ] == ["B"]
    assert operations == ["A"]
    context.get_input_items.assert_not_awaited()


@pytest.mark.parametrize("prompt", [None, "same", {"question": "same"}])
async def test_recovered_approval_preserves_pending_but_not_answered_identity(
    prompt: str | dict[str, str] | None,
) -> None:
    repeated = prompt is not None
    operations: list[str] = []
    if repeated:

        def ask(state: _BranchState) -> dict[str, Any]:
            first = interrupt(prompt)
            if isinstance(first, dict):
                first = first["question"]
            second = interrupt(prompt)
            return {"messages": [AIMessage(content=f"{first}/{second}")]}

        builder = StateGraph(_BranchState)
        builder.add_node("ask", ask)
        builder.add_edge(START, "ask")
        builder.add_edge("ask", END)
        graph = builder.compile(checkpointer=InMemorySaver())
    else:
        graph = _partial_approval_graph(InMemorySaver(), operations)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        graph,
        store=provider,
        enable_response_branching=True,
    )
    with TestClient(server.app) as client:
        root = _post(client, "start")
        approvals = [
            item for item in root["output"] if item["type"] == "mcp_approval_request"
        ]
        first_approval = next(
            item
            for item in approvals
            if json.loads(item["arguments"])["value"] == (prompt if repeated else "A")
        )
        original_pending = (
            first_approval
            if repeated
            else next(
                item
                for item in approvals
                if json.loads(item["arguments"])["value"] == "B"
            )
        )

    response_id = root["id"][:-32] + "a" * 32
    context = _context(response_id=response_id, conversation_id=None)
    context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    context.get_input_items.return_value = [
        {
            "type": "mcp_approval_response",
            "approval_request_id": first_approval["id"],
            "approve": True,
        }
    ]
    request = _request(previous_response_id=root["id"])
    persisted = None
    interrupted = server.handle_create(request, context, asyncio.Event())
    assert isinstance(interrupted, AsyncGenerator)
    try:
        async for event in interrupted:
            if not isinstance(event, dict):
                persisted = deepcopy(event.response)
            elif (
                event.get("type") == "response.output_item.added"
                and event["item"].get("type") == "function_call"
            ):
                break
    finally:
        await interrupted.aclose()
    assert persisted is not None
    assert operations == ([] if repeated else ["A"])
    recreated = ResponsesHostServer(
        graph,
        store=provider,
        enable_response_branching=True,
    )
    recovery = _context(response_id=response_id, conversation_id=None)
    recovery.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
    recovery.is_recovery = True
    recovery.persisted_response = persisted
    events = [
        event
        async for event in recreated.handle_create(request, recovery, asyncio.Event())
    ]
    recovered = events[-1]["response"]
    assert recovered["status"] == "completed", recovered
    recovery.get_input_items.assert_not_awaited()
    pending = [
        item for item in recovered["output"] if item["type"] == "mcp_approval_request"
    ]
    assert len(pending) == 1
    if repeated:
        assert pending[0]["id"] != original_pending["id"]
    else:
        assert pending == [original_pending]
    await provider.create_response(recovered, [], [])
    with TestClient(recreated.app) as client:
        completed = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": json.loads(pending[0]["arguments"])["interrupt_id"],
                    "output": "two" if repeated else "B",
                }
            ],
            previous_response_id=recovered["id"],
        )

    assert _text(completed) == ("same/two" if repeated else "1:B")
    assert operations == ([] if repeated else ["A", "B"])


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
        approved_ref = (
            _boundary_ref(client, server, approved["id"]) if enabled else None
        )

    assert approved["status"] == "completed", approved
    assert "a=Alice" in _text(approved)
    assert "b=Paris" in _text(approved)
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    checkpoint = (
        saver.get_tuple(
            {"configurable": {**approved_ref.to_dict(), "checkpoint_ns": ""}}
        )
        if approved_ref is not None
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
        approved_ref = (
            _boundary_ref(client, server, approved["id"]) if enabled else None
        )

    assert approved["status"] == "completed", approved
    assert _text(approved) == "ok:Alice"
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    checkpoint = (
        saver.get_tuple(
            {"configurable": {**approved_ref.to_dict(), "checkpoint_ns": ""}}
        )
        if approved_ref is not None
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


@pytest.mark.parametrize("progress", ["own", "sibling"])
async def test_progress_ownership_survives_host_and_saver_recreation(
    progress: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._responses."
        "conversation_chain_store.FoundryStateStore",
        FoundryStateStore,
    )
    database = str(tmp_path / "progress-checkpoints.sqlite")
    config: RunnableConfig = {"configurable": {"thread_id": "root"}}
    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        graph, _ = _branch_graph()
        graph.checkpointer = saver
        await graph.ainvoke({"messages": [HumanMessage(content="A")]}, config)
        parent_config = (await graph.aget_state(config)).config
        origin_ref = HostingRunnableConfig(parent_config).checkpoint_ref
        assert origin_ref is not None
        progress_refs = {}
        for response_key, text in (("child", "B"), ("sibling", "C")):
            execution_graph = graph.copy(
                {
                    "checkpointer": ResponseCheckpointSaver(
                        saver,
                        branching=True,
                        response_key=response_key,
                        origin=origin_ref,
                    )
                }
            )
            await execution_graph.ainvoke(
                {"messages": [HumanMessage(content=text)]}, parent_config
            )
            ref = HostingRunnableConfig(
                (await graph.aget_state(config)).config
            ).checkpoint_ref
            assert ref is not None
            progress_refs[response_key] = ref
        selected = progress_refs["child" if progress == "own" else "sibling"]
        server = ResponsesHostServer(
            graph, store=InMemoryResponseProvider(), enable_response_branching=True
        )
        await server._conversation_chain_store.set(
            "child",
            BRANCH_ORIGIN_KEY,
            {
                **ResponseBranchStore._record(origin_ref, paused=False),
                "mode": BRANCH_MODE,
                "parent_response_id": "parent",
            },
        )

    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        graph, executions = _branch_graph()
        graph.checkpointer = saver
        recreated = ResponsesHostServer(
            graph, store=InMemoryResponseProvider(), enable_response_branching=True
        )
        context = _context(response_id="child", conversation_id=None)
        context.client_headers = {BRANCH_MODE_HEADER: BRANCH_MODE}
        context.is_recovery = True
        context.persisted_response = _response_object(
            "child",
            previous_response_id="parent",
            internal_metadata={
                BRANCH_MODE_METADATA: BRANCH_MODE,
                METADATA_LANGGRAPH_THREAD_ID: selected.thread_id,
                METADATA_LANGGRAPH_CHECKPOINT_ID: selected.checkpoint_id,
            },
        )
        events = [
            event
            async for event in recreated.handle_create(
                _request(previous_response_id="parent"), context, asyncio.Event()
            )
        ]
        terminal = events[-1]["response"]
        assert terminal["status"] == ("completed" if progress == "own" else "failed")
        assert executions == []
        context.get_input_items.assert_not_awaited()
        boundary = await recreated._conversation_chain_store.get(
            "child", BRANCH_BOUNDARY_KEY
        )
        if progress == "own":
            assert CheckpointRef.from_dict(boundary) == selected
        else:
            assert boundary is None


@pytest.mark.parametrize(
    ("kind", "interrupted"),
    [
        ("nested", False),
        ("async_parallel", False),
        ("nested", True),
        ("async", True),
    ],
)
async def test_functional_task_results_survive_host_and_saver_recreation(
    kind: str, interrupted: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._responses."
        "conversation_chain_store.FoundryStateStore",
        FoundryStateStore,
    )
    database = str(tmp_path / "functional-checkpoints.sqlite")
    operations: list[str] = []
    build = _functional_interrupt_graph if interrupted else _functional_graph
    expected_operations = ["one"] if interrupted else ["prepare", "prepare"]
    expected_result = "one-1" if interrupted else "1,2"
    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        server = ResponsesHostServer(
            build(saver, operations, kind),
            enable_response_branching=True,
        )
        with TestClient(server.app) as client:
            parent = _post(client, "start")
            pending = next(
                item for item in parent["output"] if item["type"] == "function_call"
            )
            waiting = _post(client, "waiting", previous_response_id=parent["id"])
            assert waiting["status"] == "completed", waiting
            waiting_pending = next(
                item for item in waiting["output"] if item["type"] == "function_call"
            )
        assert operations == expected_operations

    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        recreated = ResponsesHostServer(
            build(saver, operations, kind),
            enable_response_branching=True,
        )
        with TestClient(recreated.app) as client:
            for selected, call, answer in (
                (parent, pending, "left"),
                (waiting, waiting_pending, "right"),
            ):
                completed = _post(
                    client,
                    [
                        {
                            "type": "function_call_output",
                            "call_id": call["call_id"],
                            "output": json.dumps({"resume": answer}),
                        }
                    ],
                    previous_response_id=selected["id"],
                )
                assert completed["status"] == "completed", completed
                assert _text(completed) == f"{expected_result}:{answer}"
                assert operations == expected_operations


@pytest.mark.parametrize("functional", [False, True])
async def test_persisted_partial_approval_reuses_completed_task_after_recreation(
    functional: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._responses."
        "conversation_chain_store.FoundryStateStore",
        FoundryStateStore,
    )
    database = str(tmp_path / "partial-checkpoints.sqlite")
    operations: list[str] = []
    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        server = ResponsesHostServer(
            _partial_approval_graph(saver, operations, functional=functional),
            enable_response_branching=True,
        )
        with TestClient(server.app) as client:
            root = _post(client, "start")
            original_b = next(
                item
                for item in root["output"]
                if item["type"] == "mcp_approval_request"
                and json.loads(item["arguments"])["value"] == "B"
            )
            approval = next(
                item
                for item in root["output"]
                if item["type"] == "mcp_approval_request"
                and json.loads(item["arguments"])["value"] == "A"
            )
            partial = _post(
                client,
                [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval["id"],
                        "approve": True,
                    }
                ],
                previous_response_id=root["id"],
            )
            assert operations == ["A"]
            assert (
                next(
                    item
                    for item in partial["output"]
                    if item["type"] == "mcp_approval_request"
                )
                == original_b
            )

    async with sqlite.AsyncSqliteSaver.from_conn_string(database) as saver:
        recreated = ResponsesHostServer(
            _partial_approval_graph(saver, operations, functional=functional),
            enable_response_branching=True,
        )
        with TestClient(recreated.app) as client:
            assert (
                client.get(f"/responses/{root['id']}").json()["output"]
                == root["output"]
            )
            assert (
                client.get(f"/responses/{partial['id']}").json()["output"]
                == partial["output"]
            )
            waiting = _post(client, "waiting", previous_response_id=partial["id"])
            assert (
                next(
                    item
                    for item in waiting["output"]
                    if item["type"] == "mcp_approval_request"
                )
                == original_b
            )
            history = client.get(f"/responses/{waiting['id']}/input_items")
            assert history.status_code == 200, history.text
            assert [
                item
                for item in history.json()["data"]
                if item.get("id") == original_b["id"]
            ] == [original_b]
            completed = _post(
                client,
                [
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": original_b["id"],
                        "approve": True,
                    }
                ],
                previous_response_id=waiting["id"],
            )

    assert _text(completed) == "1:B"
    assert operations == ["A", "B"]
