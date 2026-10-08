"""Opt-in response checkpoint branching and strict restoration tests."""

from __future__ import annotations

import asyncio
import json
import operator
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal, cast
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

from langchain_azure_ai.agents.hosting import (
    ResponsesHostServer,
    ResponsesInstructionsMiddleware,
    get_response_instructions,
)
from langchain_azure_ai.agents.hosting._response_instructions import (
    _INSTRUCTIONS_CONFIG_KEY,
    _INSTRUCTIONS_MODE_HEADER,
    _INSTRUCTIONS_PROVENANCE,
    _INSTRUCTIONS_SOURCE,
    _ResponseInstructions,
)
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
    ResponseCheckpointSaver,
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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "evidence", ["matching", "thread", "namespace", "checkpoint", "retained", "missing"]
)
async def test_instruction_removal_evidence_survives_saver_recreation(
    asynchronous: bool, evidence: str
) -> None:
    saver = InMemorySaver()
    producer = ResponseCheckpointSaver(saver, branching=True)
    instruction = SystemMessage(
        content="task-only",
        id="response-instructions-child",
        additional_kwargs={_INSTRUCTIONS_PROVENANCE: "child"},
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": "root", "checkpoint_ns": ""},
        "metadata": {_INSTRUCTIONS_PROVENANCE: "1", _INSTRUCTIONS_SOURCE: "child"},
    }
    metadata: CheckpointMetadata = {"source": "loop", "step": 0, "parents": {}}
    parent = empty_checkpoint()
    parent["channel_values"] = {"messages": [instruction]}
    parent["channel_versions"] = {"messages": 1}
    if asynchronous:
        saved_config = await producer.aput(config, parent, metadata, {"messages": 1})
    else:
        saved_config = producer.put(config, parent, metadata, {"messages": 1})
    write_config: RunnableConfig = {
        **saved_config,
        "configurable": {**saved_config["configurable"]},
    }
    if evidence in {"thread", "namespace", "checkpoint"}:
        field = {
            "thread": "thread_id",
            "namespace": "checkpoint_ns",
            "checkpoint": "checkpoint_id",
        }[evidence]
        write_config["configurable"][field] = "unrelated"
    updates: list[Any] = [{"type": "remove", "id": instruction.id, "content": ""}]
    if evidence == "retained":
        updates.append(instruction)
    if evidence != "missing":
        if asynchronous:
            await saver.aput_writes(
                write_config, [("messages", updates)], "remove-task"
            )
        else:
            saver.put_writes(write_config, [("messages", updates)], "remove-task")

    consumer = ResponseCheckpointSaver(saver, branching=True)
    child = empty_checkpoint()
    child["channel_values"] = {"messages": [HumanMessage(content="remaining")]}
    child["channel_versions"] = {"messages": 2}
    continued_config: RunnableConfig = {**saved_config, "metadata": config["metadata"]}
    if asynchronous:
        assert await consumer.aget_tuple(saved_config) is not None
        result_config = await consumer.aput(
            continued_config, child, metadata, {"messages": 2}
        )
        result = await saver.aget_tuple(result_config)
    else:
        assert consumer.get_tuple(saved_config) is not None
        result_config = consumer.put(continued_config, child, metadata, {"messages": 2})
        result = saver.get_tuple(result_config)
    assert result is not None
    assert result.metadata.get(_INSTRUCTIONS_SOURCE) == (
        "" if evidence == "matching" else "child"
    )
    unchanged = saver.get_tuple(saved_config)
    assert unchanged is not None
    assert unchanged.metadata.get(_INSTRUCTIONS_SOURCE) == "child"
    assert unchanged.checkpoint["channel_values"]["messages"] == [instruction]


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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mutation", ["summarization", "trim"])
def test_default_instructions_allow_explicit_removal(
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
    source_key = "langchain_response_instructions_source_v1"
    assert parent.metadata.get(source_key) == ""
    assert next(saver.list(None)).metadata.get(source_key) == ""
    unchanged = saver.get_tuple(parent.config)
    assert unchanged is not None
    assert unchanged.checkpoint == parent.checkpoint
    assert unchanged.metadata == parent.metadata
    assert graph.checkpointer is saver


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mutation", ["delete", "retain", "lose-identity"])
def test_instruction_removal_preserves_identity_checks(
    enabled: bool, mutation: str
) -> None:
    observed: list[list[str]] = []

    def reduce_messages(left: list[Any], right: list[Any]) -> list[AnyMessage]:
        messages = cast(list[AnyMessage], add_messages(left, right))
        if mutation == "lose-identity" and any(
            isinstance(message, RemoveMessage) and message.id == REMOVE_ALL_MESSAGES
            for message in right
        ):
            return [
                message.model_copy(update={"id": None, "additional_kwargs": {}})
                if _INSTRUCTIONS_PROVENANCE in message.additional_kwargs
                else message
                for message in messages
            ]
        return messages

    RemovalState = TypedDict(
        "RemovalState", {"messages": Annotated[list[AnyMessage], reduce_messages]}
    )

    def mutate(state: RemovalState) -> dict[str, Any]:
        if mutation == "delete":
            return {
                "messages": [
                    RemoveMessage(id=message.id)
                    for message in state["messages"]
                    if _INSTRUCTIONS_PROVENANCE in message.additional_kwargs
                    and message.id is not None
                ]
            }
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *state["messages"]]}

    def record(state: RemovalState) -> dict[str, Any]:
        observed.append(
            [
                str(message.content)
                for message in state["messages"]
                if isinstance(message, SystemMessage)
            ]
        )
        return {"messages": [AIMessage(content="ok")]}

    builder = StateGraph(RemovalState)
    builder.add_node("mutate", mutate)
    builder.add_node("record", record)
    builder.add_edge(START, "mutate")
    builder.add_edge("mutate", "record")
    builder.add_edge("record", END)
    saver = InMemorySaver()
    graph = builder.compile(checkpointer=saver)
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    with TestClient(server.app) as client:
        root = _post(
            client,
            [
                {"role": "system", "content": "explicit-system"},
                {"role": "developer", "content": "explicit-developer"},
                {"role": "user", "content": "A"},
            ],
            instructions="root-only",
        )
        assert root["status"] == "completed", root
        parent = next(saver.list(None))
        child = _post(client, "B", previous_response_id=root["id"])

    assert parent.metadata.get(_INSTRUCTIONS_SOURCE) == (
        "" if mutation == "delete" else root["id"]
    )
    if mutation == "lose-identity":
        assert child["status"] == "failed", child
        assert len(observed) == 1
    else:
        assert child["status"] == "completed", child
        assert observed[-1] == ["explicit-system", "explicit-developer"]
    assert all(
        "explicit-system" in messages and "explicit-developer" in messages
        for messages in observed
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("pending", [False, True])
async def test_recovery_applies_persisted_instruction_removal(
    enabled: bool, pending: bool
) -> None:
    class InterruptedSaver(InMemorySaver):
        fail_next_removal = pending

        async def aput(
            self, config: Any, checkpoint: Any, metadata: Any, new_versions: Any
        ) -> RunnableConfig:
            if self.fail_next_removal and checkpoint["channel_values"].get("ledger"):
                self.fail_next_removal = False
                raise OSError("checkpoint write interrupted")
            return await super().aput(config, checkpoint, metadata, new_versions)

    removals: list[str] = []
    executions: list[list[str]] = []

    def remove(state: _BranchState) -> dict[str, Any]:
        removals.append("removed")
        return {
            "ledger": ["removed"],
            "messages": [
                RemoveMessage(id=message.id)
                for message in state["messages"]
                if _INSTRUCTIONS_PROVENANCE in message.additional_kwargs
                and message.id is not None
            ],
        }

    def record(state: _BranchState) -> dict[str, Any]:
        executions.append(
            [
                str(message.content)
                for message in state["messages"]
                if isinstance(message, SystemMessage)
            ]
        )
        return {"messages": [AIMessage(content="ok")]}

    saver = InterruptedSaver()
    builder = StateGraph(_BranchState)
    builder.add_node("remove", remove)
    builder.add_node("record", record)
    builder.add_edge(START, "remove")
    builder.add_edge("remove", "record")
    builder.add_edge("record", END)
    graph = builder.compile(checkpointer=saver)
    producer = graph.copy(
        {"checkpointer": ResponseCheckpointSaver(saver, branching=enabled)}
    )
    config: RunnableConfig = {
        "configurable": {"thread_id": "root"},
        "metadata": {_INSTRUCTIONS_PROVENANCE: "1", _INSTRUCTIONS_SOURCE: "child"},
    }
    graph_input: _BranchState = {
        "messages": [
            SystemMessage(
                content="task-only",
                id="response-instructions-child",
                additional_kwargs={_INSTRUCTIONS_PROVENANCE: "child"},
            ),
            SystemMessage(content="application-owned"),
            HumanMessage(content="B"),
        ],
        "ledger": [],
    }
    if pending:
        with pytest.raises(OSError, match="checkpoint write interrupted"):
            await producer.ainvoke(graph_input, config, interrupt_before=["record"])
    else:
        await producer.ainvoke(graph_input, config, interrupt_before=["record"])
    saved = await saver.aget_tuple(config)
    assert saved is not None
    assert saved.metadata.get(_INSTRUCTIONS_SOURCE) == ("child" if pending else "")
    if pending:
        assert any(
            channel == "messages" for _, channel, _ in saved.pending_writes or []
        )
    ref = HostingRunnableConfig(saved.config).checkpoint_ref
    assert ref is not None
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    context = _context(response_id="child", conversation_id=None if enabled else "root")
    context.is_recovery = True
    metadata: dict[str, Any] = {
        METADATA_LANGGRAPH_THREAD_ID: ref.thread_id,
        METADATA_LANGGRAPH_CHECKPOINT_ID: ref.checkpoint_id,
    }
    if enabled:
        context.client_headers = {
            BRANCH_MODE_HEADER: BRANCH_MODE,
            BRANCH_OWNER_HEADER: "admitted-child",
        }
        metadata[BRANCH_MODE_METADATA] = BRANCH_MODE
        await server._branch_executions.claim(
            "response",
            server._branch_executions.response_identity(
                "child", context.platform_context
            ),
            "admitted-child",
        )
        await server._conversation_chain_store.set(
            "child",
            BRANCH_ORIGIN_KEY,
            {
                **ResponseBranchStore._record(ref, paused=False),
                "mode": BRANCH_MODE,
                "parent_response_id": "parent",
            },
        )
    context.persisted_response = _response_object("child", internal_metadata=metadata)
    events = [
        event
        async for event in server.handle_create(
            _request(
                previous_response_id="parent" if enabled else None,
                conversation=None if enabled else {"id": "root"},
                instructions="task-only",
            ),
            context,
            asyncio.Event(),
        )
    ]
    assert events[-1]["response"]["status"] == "completed", events[-1]
    assert removals == ["removed"] * (2 if pending else 1)
    assert executions == [["application-owned"]]
    recovered = await saver.aget_tuple(config)
    assert recovered is not None
    assert recovered.metadata.get(_INSTRUCTIONS_SOURCE) == ""


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mutation", ["summarization", "trim"])
def test_context_instructions_survive_summarization(
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
            ResponsesInstructionsMiddleware(),
            SummarizationMiddleware(
                FakeListChatModel(
                    responses=["Earlier dialog summary"], callbacks=[summary_inputs]
                ),
                trigger=("messages", 3),
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
        instructions_mode="context",
    )
    with TestClient(
        server.app, headers={_INSTRUCTIONS_MODE_HEADER: "messages"}
    ) as client:
        root = _post(client, "A", instructions="root-only", stream=stream)
        child = _post(
            client,
            "B",
            previous_response_id=root["id"],
            instructions="child-only",
            stream=stream,
        )
        continued = _post(client, "C", previous_response_id=child["id"], stream=stream)

    for response in (root, child, continued):
        assert response["status"] == "completed", response
    assert len(model_inputs.inputs) == 3
    assert bool(summary_inputs.inputs) == (mutation == "summarization")
    for batch, instructions in zip(
        model_inputs.inputs, ["root-only", "child-only", None], strict=True
    ):
        system_text = "\n".join(
            str(message.content)
            for message in batch
            if isinstance(message, SystemMessage)
        )
        assert "application-owned" in system_text
        for temporary in ("root-only", "child-only"):
            assert (temporary in system_text) == (temporary == instructions)
    for temporary in ("root-only", "child-only"):
        assert temporary not in str(summary_inputs.inputs)
        for checkpoint in saver.list(None):
            assert temporary not in str(checkpoint.checkpoint["channel_values"])
            assert temporary not in str(checkpoint.metadata)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("source_mode", ["messages", "context"])
def test_instruction_modes_can_continue_each_others_checkpoints(
    enabled: bool, source_mode: Literal["messages", "context"]
) -> None:
    captured = _ModelInputCapture()
    saver = InMemorySaver()
    graph = create_agent(
        FakeListChatModel(responses=["ok"], callbacks=[captured]),
        system_prompt="application-owned",
        middleware=[ResponsesInstructionsMiddleware()],
        checkpointer=saver,
    )
    provider = InMemoryResponseProvider()
    original_host = ResponsesHostServer(
        graph,
        store=provider,
        enable_response_branching=enabled,
        instructions_mode=source_mode,
    )
    with TestClient(
        original_host.app,
        headers={_INSTRUCTIONS_MODE_HEADER: "context"},
    ) as client:
        root = _post(
            client,
            [
                {"role": "system", "content": "explicit-system"},
                {"role": "developer", "content": "explicit-developer"},
                {"role": "user", "content": "A"},
            ],
            instructions="root-only",
        )
    assert root["status"] == "completed", root
    original_checkpoint = next(saver.list(None))
    target_host = ResponsesHostServer(
        graph,
        store=provider,
        enable_response_branching=enabled,
        instructions_mode="context" if source_mode == "messages" else "messages",
    )
    with TestClient(target_host.app) as client:
        child = _post(
            client, "B", previous_response_id=root["id"], instructions="child-only"
        )
        cleared = _post(
            client, "C", previous_response_id=child["id"], instructions=None
        )
        empty = _post(client, "D", previous_response_id=cleared["id"], instructions="")
    assert all(
        response["status"] == "completed" for response in (child, cleared, empty)
    )
    assert len(captured.inputs) == 4
    for batch, expected in zip(
        captured.inputs, ["root-only", "child-only", None, None], strict=True
    ):
        system_text = str(
            [message.content for message in batch if isinstance(message, SystemMessage)]
        )
        for permanent in ("application-owned", "explicit-system", "explicit-developer"):
            assert permanent in system_text
        for temporary in ("root-only", "child-only"):
            assert (temporary in system_text) == (temporary == expected)
    assert saver.get_tuple(original_checkpoint.config) == original_checkpoint


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("content_blocks", [False, True])
async def test_instruction_middleware_preserves_application_system_message(
    asynchronous: bool, content_blocks: bool
) -> None:
    captured = _ModelInputCapture()
    application = SystemMessage(
        content=(
            [{"type": "text", "text": "application-owned"}]
            if content_blocks
            else "application-owned"
        ),
        id="application-id",
        name="application-name",
        additional_kwargs={"application": "owned"},
    )
    original = application.model_copy(deep=True)
    graph = create_agent(
        FakeListChatModel(responses=["ok"], callbacks=[captured]),
        system_prompt=application,
        middleware=[ResponsesInstructionsMiddleware()],
    )
    for instructions in ("root-only", "child-only", None):
        config: RunnableConfig = {
            "configurable": {
                _INSTRUCTIONS_CONFIG_KEY: _ResponseInstructions(instructions)
            }
        }
        if asynchronous:
            await graph.ainvoke({"messages": [HumanMessage(content="hello")]}, config)
        else:
            graph.invoke({"messages": [HumanMessage(content="hello")]}, config)
        system_message = captured.inputs[-1][0]
        assert system_message.id == application.id
        assert system_message.name == application.name
        assert system_message.additional_kwargs == application.additional_kwargs
        assert "application-owned" in str(system_message.content)
        for temporary in ("root-only", "child-only"):
            assert (temporary in str(system_message.content)) == (
                temporary == instructions
            )
        assert application == original
    await graph.ainvoke({"messages": [HumanMessage(content="not hosted")]})
    assert captured.inputs[-1][0] == application
    assert get_response_instructions() is None
    assert get_response_instructions({"configurable": {}}) is None


@pytest.mark.parametrize("enabled", [False, True])
def test_context_instructions_are_isolated_between_concurrent_requests(
    enabled: bool,
) -> None:
    arrived = 0
    ready = asyncio.Event()

    async def record(state: _BranchState, config: RunnableConfig) -> dict[str, Any]:
        nonlocal arrived
        before = get_response_instructions()
        arrived += 1
        if arrived == 2:
            ready.set()
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert get_response_instructions(config) == before
        assert all(
            not isinstance(message, SystemMessage) for message in state["messages"]
        )
        return {"messages": [AIMessage(content=before or "missing")]}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
        instructions_mode="context",
    )
    with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_post, client, "hello", instructions=instructions)
            for instructions in ("first-only", "second-only")
        ]
        responses = [future.result(timeout=10) for future in futures]
    assert all(response["status"] == "completed" for response in responses)
    assert [_text(response) for response in responses] == ["first-only", "second-only"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("admitted_mode", [None, "messages", "context"])
async def test_instruction_recovery_retains_admitted_mode(
    enabled: bool, admitted_mode: Literal["messages", "context"] | None
) -> None:
    observed: list[tuple[str | None, list[str]]] = []

    async def record(state: _BranchState) -> dict[str, Any]:
        observed.append(
            (
                get_response_instructions(),
                [
                    str(message.content)
                    for message in state["messages"]
                    if isinstance(message, SystemMessage)
                ],
            )
        )
        return {"messages": [AIMessage(content="ok")]}

    builder = StateGraph(_BranchState)
    builder.add_node("record", record)
    builder.add_edge(START, "record")
    builder.add_edge("record", END)
    graph: CompiledStateGraph = builder.compile(checkpointer=InMemorySaver())
    original_host = ResponsesHostServer(
        graph,
        store=InMemoryResponseProvider(),
        instructions_mode=admitted_mode or "messages",
    )
    request = _request(
        previous_response_id="parent" if enabled else None,
        conversation=None if enabled else {"id": "root"},
        instructions="task-only",
    )
    context = _context(response_id="child", conversation_id=None if enabled else "root")
    graph_input = await original_host.build_input(request, context)
    graph_input["messages"].append(SystemMessage(content="application-owned"))
    config: RunnableConfig = {"configurable": {"thread_id": "root"}}
    prepared_input, config = await original_host._prepare_request_instructions(
        graph, config, graph_input, request, context
    )
    await graph.ainvoke(prepared_input, config, interrupt_before=["record"])
    saved = await graph.aget_state(config)
    ref = HostingRunnableConfig(saved.config).checkpoint_ref
    assert ref is not None
    recovering_host = ResponsesHostServer(
        graph,
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
        instructions_mode="messages" if admitted_mode == "context" else "context",
    )
    metadata: dict[str, Any] = {
        METADATA_LANGGRAPH_THREAD_ID: ref.thread_id,
        METADATA_LANGGRAPH_CHECKPOINT_ID: ref.checkpoint_id,
    }
    context.client_headers = {}
    if admitted_mode is not None:
        context.client_headers[_INSTRUCTIONS_MODE_HEADER] = admitted_mode
    if enabled:
        context.client_headers.update(
            {BRANCH_MODE_HEADER: BRANCH_MODE, BRANCH_OWNER_HEADER: "admitted-child"}
        )
        metadata[BRANCH_MODE_METADATA] = BRANCH_MODE
        await recovering_host._branch_executions.claim(
            "response",
            recovering_host._branch_executions.response_identity(
                "child", context.platform_context
            ),
            "admitted-child",
        )
        await recovering_host._conversation_chain_store.set(
            "child",
            BRANCH_ORIGIN_KEY,
            {
                **ResponseBranchStore._record(ref, paused=False),
                "mode": BRANCH_MODE,
                "parent_response_id": "parent",
            },
        )
    context.is_recovery = True
    context.persisted_response = _response_object("child", internal_metadata=metadata)
    events = [
        event
        async for event in recovering_host.handle_create(
            request, context, asyncio.Event()
        )
    ]
    assert events[-1]["response"]["status"] == "completed", events[-1]
    assert observed == (
        [("task-only", ["application-owned"])]
        if admitted_mode == "context"
        else [(None, ["task-only", "application-owned"])]
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "shared-text"])
def test_formatted_messages_reject_lost_instruction_provenance(
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

    assert root["status"] == "completed", root
    if instructions is None:
        assert child["status"] == "completed", child
        assert observed == [
            ["shared-text", "developer-context"],
            ["shared-text", "developer-context", "application-owned"],
        ]
    else:
        assert child["status"] == "failed", child
        assert child["error"]["code"] == "server_error"
        assert observed == [["shared-text", "shared-text", "developer-context"]]


@pytest.mark.parametrize("instructions_mode", ["messages", "context"])
async def test_ambiguous_legacy_instructions_fail_without_graph_execution(
    instructions_mode: Literal["messages", "context"],
) -> None:
    graph, executions = _branch_graph()
    await graph.ainvoke(
        {"messages": [SystemMessage(content="ambiguous"), HumanMessage(content="A")]},
        {"configurable": {"thread_id": "legacy"}},
    )
    executions.clear()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), instructions_mode=instructions_mode
    )
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


@pytest.mark.parametrize("lost_identity", ["summarized", "reformatted"])
async def test_context_mode_rejects_unverifiable_legacy_instruction_identity(
    lost_identity: str,
) -> None:
    graph, executions = _branch_graph()
    messages: list[AnyMessage] = [HumanMessage(content="Earlier summary")]
    if lost_identity == "reformatted":
        messages.insert(0, SystemMessage(content="old-request-only"))
    await graph.ainvoke(
        {"messages": messages},
        {
            "configurable": {"thread_id": "legacy"},
            "metadata": {
                "langchain_response_instructions_v1": "1",
                "langchain_response_instructions_source_v1": "parent",
            },
        },
    )
    executions.clear()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), instructions_mode="context"
    )
    events = [
        event
        async for event in server.handle_create(
            _request(conversation={"id": "legacy"}),
            _context(conversation_id="legacy"),
            asyncio.Event(),
        )
    ]
    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["code"] == "server_error"
    assert executions == []


async def test_recovery_rejects_unknown_instruction_mode_before_execution() -> None:
    graph, executions = _branch_graph()
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    context = _context()
    context.is_recovery = True
    context.client_headers = {_INSTRUCTIONS_MODE_HEADER: "unknown"}
    events = [
        event
        async for event in server.handle_create(_request(), context, asyncio.Event())
    ]
    assert events[-1]["response"]["status"] == "failed"
    assert events[-1]["response"]["error"]["code"] == "server_error"
    assert executions == []


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
    if boundary_published or progress in {"missing", "malformed"}:
        assert terminal["status"] == "failed"
        assert terminal["error"]["code"] == "server_error"
        assert executions == []
    else:
        assert terminal["status"] == "completed", terminal
        assert executions == (["B"] if progress == "none" else [])
    provider.get_response.assert_not_awaited()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    ("provenance", "instruction_source", "identity"),
    [
        ("1", None, "intact"),
        (None, None, "intact"),
        ("unknown", None, "intact"),
        ("1", "child", "intact"),
        ("1", "child", "tag-lost"),
        ("1", "child", "id-lost"),
        ("1", "other", "intact"),
        ("1", "", "intact"),
        ("1", None, "tag-lost"),
    ],
)
async def test_recovery_preserves_verified_instruction_provenance(
    enabled: bool,
    provenance: str | None,
    instruction_source: str | None,
    identity: str,
) -> None:
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
    if identity == "tag-lost":
        graph_input["messages"][0].additional_kwargs.clear()
    elif identity == "id-lost":
        graph_input["messages"][0].id = None
    checkpoint_metadata: dict[str, Any] = {}
    if provenance is not None:
        checkpoint_metadata["langchain_response_instructions_v1"] = provenance
    if instruction_source is not None:
        checkpoint_metadata["langchain_response_instructions_source_v1"] = (
            instruction_source
        )
    config: RunnableConfig = {
        "configurable": {"thread_id": "root"},
        "metadata": checkpoint_metadata,
    }
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
        context.client_headers = {
            BRANCH_MODE_HEADER: BRANCH_MODE,
            BRANCH_OWNER_HEADER: "admitted-child",
        }
        await server._branch_executions.claim(
            "response",
            server._branch_executions.response_identity(
                "child", context.platform_context
            ),
            "admitted-child",
        )
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
    if (
        provenance != "1"
        or instruction_source not in {None, "child"}
        or identity != "intact"
    ):
        assert events[-1]["response"]["status"] == "failed"
        assert events[-1]["response"]["error"]["code"] == "server_error"
        assert executions == []
        return

    assert events[-1]["response"]["status"] == "completed", events[-1]
    assert executions == ["B"]
    recovered = await graph.aget_state(config)
    assert recovered.metadata is not None
    assert recovered.metadata.get("langchain_response_instructions_v1") == "1"
    assert (
        recovered.metadata.get("langchain_response_instructions_source_v1") == "child"
    )
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
    ] == ["application-owned"]


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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "approval-only"])
def test_context_instructions_follow_the_current_approval_request(
    enabled: bool, stream: bool, instructions: str | None
) -> None:
    observed: list[str | None] = []

    async def approve(state: _BranchState) -> dict[str, Any]:
        interrupt("approval required")
        current = get_response_instructions()
        observed.append(current)
        assert all(
            not isinstance(message, SystemMessage) for message in state["messages"]
        )
        return {"messages": [AIMessage(content=current or "cleared")]}

    builder = StateGraph(_BranchState)
    builder.add_node("approve", approve)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", END)
    saver = InMemorySaver()
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver),
        store=InMemoryResponseProvider(),
        enable_response_branching=enabled,
        instructions_mode="context",
    )
    with TestClient(server.app) as client:
        paused = _post(client, "A", instructions="root-only", stream=stream)
        waiting = _post(
            client,
            "still waiting",
            previous_response_id=paused["id"],
            instructions="waiting-only",
            stream=stream,
        )
        assert observed == []
        pending = next(
            item for item in paused["output"] if item["type"] == "function_call"
        )
        approved = _post(
            client,
            [
                {
                    "type": "function_call_output",
                    "call_id": pending["call_id"],
                    "output": json.dumps({"resume": "approved"}),
                }
            ],
            previous_response_id=waiting["id"],
            instructions=instructions,
            stream=stream,
        )
    assert all(
        response["status"] == "completed" for response in (paused, waiting, approved)
    )
    assert observed == [instructions]
    assert _text(approved) == (instructions or "cleared")


@pytest.mark.parametrize("stream", [False, True])
def test_context_instructions_work_without_a_checkpointer(stream: bool) -> None:
    captured = _ModelInputCapture()
    graph = create_agent(
        FakeListChatModel(responses=["ok"], callbacks=[captured]),
        middleware=[ResponsesInstructionsMiddleware()],
    )
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), instructions_mode="context"
    )
    parent = None
    with TestClient(server.app) as client:
        for text, instructions in (
            ("A", "root-only"),
            ("B", "child-only"),
            ("C", None),
        ):
            response = _post(
                client,
                text,
                previous_response_id=parent,
                instructions=instructions,
                stream=stream,
            )
            assert response["status"] == "completed", response
            parent = response["id"]
            for temporary in ("root-only", "child-only"):
                assert (temporary in str(captured.inputs[-1])) == (
                    temporary == instructions
                )
    assert [
        message.content
        for message in captured.inputs[-1]
        if isinstance(message, HumanMessage)
    ] == ["A", "B", "C"]


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

    assert historical["status"] == "failed"
    assert executions == ["ask"]


@pytest.mark.parametrize("stream", [False, True])
def test_concurrent_approvals_claim_before_resume_execution(
    stream: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    executions: list[str] = []

    def ask(state: _BranchState) -> dict[str, Any]:
        executions.append("ask")
        return {"messages": [AIMessage(content=f"ok:{interrupt('name?')}")]}

    builder = StateGraph(_BranchState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=InMemorySaver()),
        store=InMemoryResponseProvider(),
        enable_response_branching=True,
    )
    original_check = server._branch_store.check_pause_owner
    barrier = asyncio.Barrier(2)

    async def synchronize_claim(
        response_key: str,
        ownership_store: ResponseExecutionStore,
        *,
        claim: bool = False,
    ) -> None:
        if claim:
            await asyncio.wait_for(barrier.wait(), timeout=5)
        await original_check(response_key, ownership_store, claim=claim)

    with TestClient(server.app) as client:
        paused = _post(client, "start", stream=stream)
        pending = next(
            item for item in paused["output"] if item["type"] == "function_call"
        )
        executions.clear()
        monkeypatch.setattr(
            server._branch_store, "check_pause_owner", synchronize_claim
        )

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

    assert sorted(response["status"] for response in responses) == [
        "completed",
        "failed",
    ]
    completed = next(
        response for response in responses if response["status"] == "completed"
    )
    assert _text(completed) in {"ok:Alice", "ok:Bob"}
    assert executions == ["ask"]


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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("initial_instructions", "resume_instructions"),
    [(None, None), ("initial", None), ("initial", "resumed")],
)
def test_parallel_approval_updates_preserve_messages_and_instructions(
    enabled: bool,
    stream: bool,
    initial_instructions: str | None,
    resume_instructions: str | None,
) -> None:
    graph = build_parallel_interrupt_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    with TestClient(server.app) as client:
        paused = _post(
            client, "start", instructions=initial_instructions, stream=stream
        )
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
            instructions=resume_instructions,
            stream=stream,
        )

    assert approved["status"] == "completed", approved
    assert "a=Alice" in _text(approved)
    assert "b=Paris" in _text(approved)
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    checkpoint = next(saver.list(None))
    system_messages = [
        message.content
        for message in checkpoint.checkpoint["channel_values"]["messages"]
        if isinstance(message, SystemMessage)
    ]
    expected = [resume_instructions] if resume_instructions else []
    assert system_messages == [*expected, "explicit:Alice", "explicit:Paris"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("resume_instructions", [None, "resumed"])
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
    resume_instructions: str | None,
    message_update: Any,
) -> None:
    graph = build_simple_interrupt_graph()
    server = ResponsesHostServer(
        graph, store=InMemoryResponseProvider(), enable_response_branching=enabled
    )
    with TestClient(server.app) as client:
        paused = _post(client, "start", instructions="initial", stream=stream)
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
            instructions=resume_instructions,
            stream=stream,
        )

    assert approved["status"] == "completed", approved
    assert _text(approved) == "ok:Alice"
    saver = graph.checkpointer
    assert isinstance(saver, BaseCheckpointSaver)
    checkpoint = next(saver.list(None))
    messages = checkpoint.checkpoint["channel_values"]["messages"]
    expected_instructions = [resume_instructions] if resume_instructions else []
    assert [message.content for message in messages] == [
        "start",
        *expected_instructions,
        "edited-input",
        "ok:Alice",
    ]


@pytest.mark.parametrize("failure", ["provider-timeout", "sdk-validation"])
def test_pre_admission_failure_allows_same_identity_retry(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph, executions = _branch_graph()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    lookup = provider.get_response
    request: dict[str, Any] = {"model": "test", "input": "A", "store": True}
    headers = {"x-agent-response-id": "caresp_" + "a" * 18 + "b" * 32}
    if failure == "provider-timeout":
        monkeypatch.setattr(
            provider, "get_response", AsyncMock(side_effect=TimeoutError("lookup"))
        )
    else:
        request.update(background=True, store=False)

    with TestClient(server.app) as client:
        rejected = client.post("/responses", json=request, headers=headers)
        assert rejected.status_code == (500 if failure == "provider-timeout" else 400)
        assert executions == []
        monkeypatch.setattr(provider, "get_response", lookup)
        retried = client.post(
            "/responses",
            json={"model": "test", "input": "A", "store": True},
            headers=headers,
        )

    assert retried.status_code == 200, retried.text
    assert retried.json()["status"] == "completed", retried.text
    assert executions == ["A"]


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


@pytest.mark.parametrize(
    ("outcome", "released"),
    [
        ("validation", True),
        ("missing-reference", True),
        ("handler-started", False),
        ("background-accepted", False),
        ("lookup-error", False),
        ("delete-error", False),
        ("server-error", False),
        ("unknown-error", False),
        ("incomplete-response", False),
        ("app-error", False),
    ],
)
async def test_admission_rollback_requires_confirmed_rejection(
    outcome: str, released: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azure.ai.agentserver.core import get_request_context
    from azure.ai.agentserver.responses import PlatformContext
    from starlette.responses import JSONResponse
    from starlette.types import Receive, Scope, Send

    from langchain_azure_ai.agents.hosting._responses import branching

    response_id = "caresp_" + "a" * 18 + "b" * 32
    provider = InMemoryResponseProvider()
    executions = ResponseExecutionStore()
    observed: list[tuple[str | None, str | None]] = []

    async def missing_response(*args: Any, **kwargs: Any) -> Any:
        observed.append((get_request_context().user_id, get_request_context().call_id))
        raise KeyError(response_id)

    monkeypatch.setattr(provider, "get_response", missing_response)

    async def start_handler() -> None:
        branching.mark_response_started()

    async def reject(scope: Scope, receive: Receive, send: Send) -> None:
        if outcome == "handler-started":
            await asyncio.create_task(start_handler())
        elif outcome == "background-accepted":
            monkeypatch.setattr(
                provider,
                "get_response",
                AsyncMock(return_value={"id": response_id, "status": "queued"}),
            )
        elif outcome == "lookup-error":
            monkeypatch.setattr(
                provider,
                "get_response",
                AsyncMock(side_effect=TimeoutError("private lookup details")),
            )
        elif outcome == "delete-error":
            monkeypatch.setattr(
                branching.FoundryStateStore,
                "delete_item",
                AsyncMock(side_effect=TimeoutError("private delete details")),
            )

        status = 404 if outcome == "missing-reference" else 400
        error_type = (
            "not_found_error"
            if outcome == "missing-reference"
            else "invalid_request_error"
        )
        if outcome == "server-error":
            status, error_type = 500, "server_error"
        elif outcome == "unknown-error":
            error_type = "unknown_error"
        payload = {"error": {"type": error_type, "message": "Rejected."}}
        if outcome == "incomplete-response":
            await send({"type": "http.response.start", "status": status, "headers": []})
            await send(
                {
                    "type": "http.response.body",
                    "body": json.dumps(payload).encode(),
                    "more_body": True,
                }
            )
        else:
            await JSONResponse(payload, status_code=status)(scope, receive, send)
        if outcome == "app-error":
            raise RuntimeError("private app details")

    middleware = branching.BranchingAdmissionMiddleware(
        reject, enabled=True, executions=executions, provider=provider
    )
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/responses",
        "headers": [
            (b"x-agent-response-id", response_id.encode()),
            (b"x-agent-user-id", b"user"),
            (b"x-agent-foundry-call-id", b"call"),
        ],
    }
    receive = AsyncMock(return_value={"type": "http.request", "body": b'{"input":"A"}'})
    send = AsyncMock()
    if outcome == "app-error":
        with pytest.raises(RuntimeError, match="private app details"):
            await middleware(scope, receive, send)
    else:
        await middleware(scope, receive, send)

    identity = executions.response_identity(
        response_id, PlatformContext(user_id_key="user")
    )
    assert (await executions.owner("response", identity) is None) == released
    assert await executions.claim("response", identity, "retry-owner") == released
    assert observed and all(context == ("user", "call") for context in observed)
    assert all(
        b"private" not in call.args[0].get("body", b"") for call in send.call_args_list
    )
    if outcome == "validation":
        assert len(observed) == 2


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stored", [False, True])
def test_admitted_failure_keeps_response_identity(stream: bool, stored: bool) -> None:
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
        retry = client.post("/responses", json=payload, headers=headers)

    assert failed.status_code == 200, failed.text
    if stream:
        assert any(kind == "response.failed" for kind, _ in _parse_sse(failed.text))
    else:
        assert failed.json()["status"] == "failed", failed.text
    assert retry.status_code == 409, retry.text
    assert executions == ["side-effect"]
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


@pytest.mark.parametrize("real_store", [False, True])
@pytest.mark.parametrize("replacement", [None, "other-owner", "original-owner"])
async def test_execution_claim_release_checks_owner_and_version(
    real_store: bool, replacement: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents.hosting._responses import branching

    if real_store:
        monkeypatch.setattr(branching, "FoundryStateStore", FoundryStateStore)
    executions = ResponseExecutionStore()
    assert await executions.claim("response", "identity", "original-owner")
    assert not await executions.release("response", "identity", "wrong-owner")
    assert await executions.owner("response", "identity") == "original-owner"

    get_item = branching.FoundryStateStore.get_item

    async def replace_after_read(store: Any, key: str, **kwargs: Any) -> Any:
        item = await get_item(store, key, **kwargs)
        if replacement is not None:
            await store.set_item(key, {"version": "1", "owner": replacement})
        return item

    with monkeypatch.context() as patch:
        patch.setattr(branching.FoundryStateStore, "get_item", replace_after_read)
        released = await executions.release("response", "identity", "original-owner")

    assert released == (replacement is None)
    assert await ResponseExecutionStore().owner("response", "identity") == replacement
    assert await executions.claim("response", "identity", "retry-owner") == released


@pytest.mark.parametrize("etag", [None, "", "*"])
async def test_execution_claim_release_rejects_missing_or_wildcard_version(
    etag: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from langchain_azure_ai.agents.hosting._responses import branching

    executions = ResponseExecutionStore()
    assert await executions.claim("response", "identity", "owner")
    get_item = branching.FoundryStateStore.get_item

    async def invalid_version(store: Any, key: str, **kwargs: Any) -> Any:
        item = await get_item(store, key, **kwargs)
        assert item is not None
        return SimpleNamespace(value=item.value, etag=etag)

    with monkeypatch.context() as patch:
        patch.setattr(branching.FoundryStateStore, "get_item", invalid_version)
        with pytest.raises(branching.BranchingError, match="version is invalid"):
            await executions.release("response", "identity", "owner")
    assert await executions.owner("response", "identity") == "owner"


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
