# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-scoped instructions through the public Responses endpoint."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncGenerator, Sequence
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from operator import add
from pathlib import Path
from typing import Annotated, Any, NamedTuple, cast

import pytest
from azure.ai.agentserver.responses import (
    InMemoryResponseProvider,
    ResponseContext,
    ResponseEventStream,
)
from azure.ai.agentserver.responses.hosting._orchestrator import ResponseModeFlags
from azure.ai.agentserver.responses.models import CreateResponse, ResponseObject
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler, CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.load import load
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    convert_to_openai_messages,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.runnables import RunnableConfig
from langgraph.cache.memory import InMemoryCache
from langgraph.channels import DeltaChannel, LastValue
from langgraph.checkpoint.base import CheckpointTuple, empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import TASKS
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, Messages, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import NodeBuilder, Pregel
from langgraph.types import CachePolicy, Command, Interrupt, Overwrite, Send, interrupt
from pydantic import BaseModel, PrivateAttr
from starlette.testclient import TestClient
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import (
    ResponsesHostServer,
    with_response_instructions,
)
from langchain_azure_ai.agents.hosting._checkpoint._checkpoint_reuse import (
    _RequestCheckpointSaver,
)
from langchain_azure_ai.agents.hosting._instructions import (
    _is_request_instruction,
    _ResponseInstructions,
    _without_request_instructions,
)
from langchain_azure_ai.agents.hosting._responses import TaskStorageManager


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))


class _CaptureMessages(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        self.calls.extend(deepcopy(messages))


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


class _AppendingMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add]


class _ReplacingMessagesState(TypedDict):
    messages: list[AnyMessage]


def _replace_messages(
    previous: list[AnyMessage], current: list[AnyMessage]
) -> list[AnyMessage]:
    return current


class _ReplacingReducerMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], _replace_messages]


def _replace_message_deltas(
    previous: list[AnyMessage], updates: Sequence[list[AnyMessage]]
) -> list[AnyMessage]:
    return updates[-1] if updates else previous


class _DeltaReplacingSnapshotState(TypedDict):
    messages: Annotated[
        list[AnyMessage], DeltaChannel(_replace_message_deltas, snapshot_frequency=1)
    ]


class _DeltaReplacingReplayState(TypedDict):
    messages: Annotated[list[AnyMessage], DeltaChannel(_replace_message_deltas)]


def _graph(
    capture: _CaptureMessages,
    saver: InMemorySaver | None,
    *,
    schema: type = MessagesState,
) -> CompiledStateGraph:
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder: StateGraph = StateGraph(schema)
    builder.add_node("answer", answer, input_schema=schema)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=saver)


def _create(
    client: TestClient,
    *,
    stream: bool,
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


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("schema", [MessagesState, _AppendingMessagesState])
def test_current_instructions_precede_complete_context_without_inheritance(
    checkpointed: bool, stream: bool, conversation: bool, schema: type
) -> None:
    capture = _CaptureMessages()
    saver = InMemorySaver() if checkpointed else None
    server = ResponsesHostServer(
        _graph(capture, saver, schema=schema), store=InMemoryResponseProvider()
    )
    linkage: dict[str, Any] = (
        {"conversation": {"id": "conv-instructions"}} if conversation else {}
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

    assert [message.content for message in capture.calls[0]] == ["turn-A", "first"]
    assert [message.content for message in capture.calls[1]] == [
        "turn-B",
        "first",
        "answer",
        "second",
    ]
    assert [message.content for message in capture.calls[2]] == [
        "first",
        "answer",
        "second",
        "answer",
        "third",
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize(
    "replacing_schema",
    [
        _ReplacingMessagesState,
        _ReplacingReducerMessagesState,
        _DeltaReplacingSnapshotState,
        _DeltaReplacingReplayState,
    ],
)
def test_replacing_message_updates_keep_their_current_instructions(
    stream: bool, overwrite: bool, replacing_schema: type
) -> None:
    class OverwritingHost(ResponsesHostServer):
        async def build_input(
            self,
            request: CreateResponse,
            context: ResponseContext,
            *,
            skip_call_ids: frozenset[str] | None = None,
        ) -> dict[str, Any]:
            payload = await super().build_input(
                request, context, skip_call_ids=skip_call_ids
            )
            return {**payload, "messages": Overwrite(payload["messages"])}

    capture = _CaptureMessages()
    host_type = OverwritingHost if overwrite else ResponsesHostServer
    schema = _AppendingMessagesState if overwrite else replacing_schema
    server = host_type(
        _graph(capture, InMemorySaver(), schema=schema),
        store=InMemoryResponseProvider(),
    )
    with TestClient(server.app) as client:
        first = _create(client, stream=stream, input="first", instructions="turn-A")
        second = _create(
            client,
            stream=stream,
            input="second",
            instructions="turn-B",
            previous_response_id=first["id"],
        )
        _create(
            client,
            stream=stream,
            input="third",
            previous_response_id=second["id"],
        )
    assert [[message.content for message in call] for call in capture.calls] == [
        ["turn-A", "first"],
        ["turn-B", "second"],
        ["third"],
    ]


@pytest.mark.parametrize("lookup_recovers", [False, True])
def test_instruction_placement_retries_failed_checkpoint_inspection(
    monkeypatch: pytest.MonkeyPatch, lookup_recovers: bool
) -> None:
    capture = _CaptureMessages()
    saver = InMemorySaver()
    server = ResponsesHostServer(
        _graph(capture, saver, schema=_AppendingMessagesState),
        store=InMemoryResponseProvider(),
    )
    with TestClient(server.app) as client:
        first = _create(
            client, stream=False, input="first", instructions="request-only"
        )
        get_tuple = saver.aget_tuple
        inspected = False

        async def inspect(config: RunnableConfig, **kwargs: Any) -> Any:
            nonlocal inspected
            if not inspected or not lookup_recovers:
                inspected = True
                raise RuntimeError("checkpoint inspection unavailable")
            return await get_tuple(config, **kwargs)

        monkeypatch.setattr(saver, "aget_tuple", inspect)
        result = client.post(
            "/responses",
            json={
                "input": "second",
                "instructions": "second-instruction",
                "previous_response_id": first["id"],
                "stream": False,
            },
        )
        assert result.status_code == 200, result.text
        response = result.json()
    assert inspected
    if lookup_recovers:
        assert response["status"] == "completed", response
        assert [message.content for message in capture.calls[1]] == [
            "second-instruction",
            "first",
            "answer",
            "second",
        ]
    else:
        assert response["status"] == "failed", response
        assert [[message.content for message in call] for call in capture.calls] == [
            ["request-only", "first"]
        ]


@pytest.mark.parametrize("schema", [MessagesState, _AppendingMessagesState])
def test_instructions_and_explicit_messages_preserve_developer_roles(
    schema: type,
) -> None:
    capture = _CaptureMessages()
    server = ResponsesHostServer(
        _graph(capture, InMemorySaver(), schema=schema),
        store=InMemoryResponseProvider(),
    )
    with TestClient(server.app) as client:
        response = _create(
            client,
            stream=False,
            instructions="request-only",
            input=[
                {"role": "system", "content": "explicit-system"},
                {"role": "developer", "content": "explicit-developer"},
                {"role": "user", "content": "first"},
            ],
        )
        items = client.get(
            f"/responses/{response['id']}/input_items", params={"order": "asc"}
        )
        assert items.status_code == 200, items.text
        _create(
            client,
            stream=False,
            input="second",
            instructions="request-only",
            previous_response_id=response["id"],
        )

    messages = convert_to_openai_messages(capture.calls[0])
    assert [message["role"] for message in messages] == [
        "developer",
        "system",
        "developer",
        "user",
    ]
    assert [message.content for message in capture.calls[0]] == [
        "request-only",
        "explicit-system",
        "explicit-developer",
        "first",
    ]
    assert response["instructions"] == "request-only"
    assert [item["role"] for item in items.json()["data"]] == [
        "system",
        "developer",
        "user",
    ]
    assert [
        (message["role"], message["content"])
        for message in convert_to_openai_messages(capture.calls[1])
    ] == [
        ("developer", "request-only"),
        ("system", "explicit-system"),
        ("developer", "explicit-developer"),
        ("user", "first"),
        ("assistant", "answer"),
        ("user", "second"),
    ]


def _contains_instruction(value: Any) -> bool:
    if _is_request_instruction(value):
        return True
    if isinstance(value, dict):
        return any(_contains_instruction(item) for item in value.values())
    if isinstance(value, (list, tuple, deque)):
        return any(_contains_instruction(item) for item in value)
    if isinstance(value, Send):
        return _contains_instruction(value.arg)
    if isinstance(value, Interrupt):
        return _contains_instruction(value.value)
    if isinstance(value, BaseModel):
        return any(
            _contains_instruction(getattr(value, name))
            for name in type(value).model_fields
        ) or _contains_instruction(value.model_extra)
    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_instruction(getattr(value, item.name)) for item in fields(value)
        )
    return False


def test_checkpoints_and_pending_writes_exclude_request_instructions() -> None:
    capture = _CaptureMessages()
    saver = InMemorySaver()
    server = ResponsesHostServer(
        _graph(capture, saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        _create(client, stream=False, input="first", instructions="request-only")

    checkpoints = list(saver.list(None))
    assert checkpoints
    for checkpoint in checkpoints:
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("instructions", [None, ""])
@pytest.mark.parametrize("conversation", [False, True])
def test_null_and_empty_instructions_do_not_restore_previous_instructions(
    instructions: str | None, conversation: bool
) -> None:
    capture = _CaptureMessages()
    server = ResponsesHostServer(
        _graph(capture, InMemorySaver()), store=InMemoryResponseProvider()
    )
    linkage: dict[str, Any] = (
        {"conversation": {"id": "conv-instructions"}} if conversation else {}
    )
    with TestClient(server.app) as client:
        first = _create(
            client,
            stream=False,
            input="first",
            instructions="request-only",
            **linkage,
        )
        if not conversation:
            linkage = {"previous_response_id": first["id"]}
        second = _create(
            client,
            stream=False,
            input="second",
            instructions=instructions,
            **linkage,
        )
    assert [message.content for message in capture.calls[1]] == [
        "first",
        "answer",
        "second",
    ]
    assert second.get("instructions") == instructions


def test_equal_explicit_messages_and_assistant_text_remain_in_history() -> None:
    capture = _CaptureMessages()
    saver = InMemorySaver()
    server = ResponsesHostServer(
        _graph(capture, saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(
            client,
            stream=False,
            instructions="answer",
            input=[
                {"role": "system", "content": "answer"},
                {"role": "developer", "content": "answer"},
                {"role": "user", "content": "first"},
            ],
        )
        _create(
            client,
            stream=False,
            input="second",
            previous_response_id=first["id"],
        )
    assert [
        (message["role"], message["content"])
        for message in convert_to_openai_messages(capture.calls[1])
    ] == [
        ("system", "answer"),
        ("developer", "answer"),
        ("user", "first"),
        ("assistant", "answer"),
        ("user", "second"),
    ]


def _checkpoint_config(response: ResponseObject) -> RunnableConfig:
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


@pytest.mark.parametrize("schema", [MessagesState, _AppendingMessagesState])
def test_branch_instructions_do_not_modify_the_parent_or_other_branches(
    schema: type,
) -> None:
    capture = _CaptureMessages()
    saver = InMemorySaver()
    provider = InMemoryResponseProvider()
    graph = _graph(capture, saver, schema=schema)
    server = ResponsesHostServer(graph, store=provider, enable_response_branching=True)
    with TestClient(server.app) as client:
        parent = _create(
            client, stream=False, input="parent", instructions="request-only"
        )
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, parent["id"])
        assert stored is not None
        config = _checkpoint_config(stored)
        original = deepcopy(graph.get_state(config).values)
        child = _create(
            client,
            stream=False,
            input="child",
            instructions="child-instruction",
            previous_response_id=parent["id"],
        )
        _create(
            client,
            stream=False,
            input="sibling",
            previous_response_id=parent["id"],
        )
        _create(
            client,
            stream=False,
            input="descendant",
            previous_response_id=child["id"],
        )
    assert [message.content for message in capture.calls[1]] == [
        "child-instruction",
        "parent",
        "answer",
        "child",
    ]
    assert [message.content for message in capture.calls[2]] == [
        "parent",
        "answer",
        "sibling",
    ]
    assert [message.content for message in capture.calls[3]] == [
        "parent",
        "answer",
        "child",
        "answer",
        "descendant",
    ]
    assert graph.get_state(config).values == original


@pytest.mark.parametrize("child_checkpointer", [None, True, "separate"])
def test_nested_graph_instructions_are_not_persisted(
    child_checkpointer: bool | str | None,
) -> None:
    capture = _CaptureMessages()
    child_saver = InMemorySaver()
    child = _graph(capture, child_saver)
    if child_checkpointer != "separate":
        child = child.copy({"checkpointer": child_checkpointer})
    saver = InMemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    graph = builder.compile(checkpointer=saver)
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        first = _create(
            client, stream=False, input="first", instructions="request-only"
        )
        _create(
            client,
            stream=False,
            input="second",
            previous_response_id=first["id"],
        )
    assert [message.content for message in capture.calls[0]] == [
        "request-only",
        "first",
    ]
    assert [message.content for message in capture.calls[1]] == [
        "first",
        "answer",
        "second",
    ]
    for backend in (saver, child_saver):
        for checkpoint in backend.list(None):
            assert not _contains_instruction(checkpoint.checkpoint)
            assert not _contains_instruction(checkpoint.pending_writes)


def test_send_task_payloads_do_not_persist_request_instructions() -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])

    def dispatch(state: MessagesState) -> Command:
        return Command(goto=Send("answer", state))

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    saver = InMemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node("dispatch", dispatch)
    builder.add_node("answer", answer)
    builder.add_edge(START, "dispatch")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        _create(client, stream=False, input="first", instructions="request-only")
    assert capture.calls[0][0].content == "request-only"
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@dataclass
class _DataclassTaskInput:
    messages: list[AnyMessage]


class _PydanticTaskInput(BaseModel):
    messages: list[AnyMessage]


class _PydanticTupleTaskInput(BaseModel, frozen=True):
    messages: tuple[AnyMessage, ...]


class _PydanticExtraTaskInput(BaseModel, extra="allow", frozen=True):
    label: str = "keep"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "", "resumed-instruction"])
@pytest.mark.parametrize("schema", [MessagesState, _AppendingMessagesState])
@pytest.mark.parametrize("forward_parent_messages", [False, True])
@pytest.mark.parametrize(
    "task_schema",
    [
        _ReplacingMessagesState,
        _DataclassTaskInput,
        _PydanticTaskInput,
        _PydanticTupleTaskInput,
        _PydanticExtraTaskInput,
    ],
    ids=["dict", "dataclass", "pydantic", "pydantic-tuple", "pydantic-extra"],
)
def test_saved_send_resume_uses_only_current_request_instructions(
    stream: bool,
    instructions: str | None,
    schema: type,
    task_schema: type,
    forward_parent_messages: bool,
) -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])
    dispatched: list[Any] = []

    def dispatch(state: MessagesState) -> Command:
        messages = (
            state["messages"]
            if forward_parent_messages
            else [HumanMessage(content="first")]
        )
        payload = task_schema(messages=messages)
        dispatched.append(payload)
        return Command(goto=Send("answer", payload))

    async def answer(
        state: (
            _ReplacingMessagesState
            | _DataclassTaskInput
            | _PydanticTaskInput
            | _PydanticTupleTaskInput
            | _PydanticExtraTaskInput
        ),
    ) -> dict[str, Any]:
        interrupt({"question": "continue?"})
        messages: list[AnyMessage] | tuple[AnyMessage, ...]
        if isinstance(state, dict):
            messages = state["messages"]
        elif isinstance(state, _PydanticExtraTaskInput):
            extra = state.model_extra
            assert extra is not None
            messages = extra["messages"]
        else:
            messages = state.messages
        return {"messages": [await model.ainvoke(messages)]}

    saver = InMemorySaver()
    builder: StateGraph = StateGraph(schema)
    builder.add_node("dispatch", dispatch, input_schema=schema)
    builder.add_node("answer", answer, input_schema=task_schema)
    builder.add_edge(START, "dispatch")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(
            client, stream=stream, input="first", instructions="request-only"
        )
        approval = next(
            item for item in first["output"] if item["type"] == "mcp_approval_request"
        )
        _create(
            client,
            stream=stream,
            previous_response_id=first["id"],
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
            ],
            **({"instructions": instructions} if instructions is not None else {}),
        )
    expected = [instructions, "first"] if instructions else ["first"]
    assert [message.content for message in capture.calls[0]] == expected
    if instructions:
        assert capture.calls[0][0].additional_kwargs == {"__openai_role__": "developer"}
    original_messages = (
        dispatched[0]["messages"]
        if isinstance(dispatched[0], dict)
        else dispatched[0].messages
    )
    assert [message.content for message in original_messages] == (
        ["request-only", "first"] if forward_parent_messages else ["first"]
    )
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


def test_send_overlay_preserves_frozen_payloads_and_explicit_messages() -> None:
    @dataclass(frozen=True, slots=True)
    class FrozenTaskInput:
        original: list[AnyMessage]
        messages: list[AnyMessage] = field(init=False)

        def __post_init__(self) -> None:
            object.__setattr__(self, "messages", self.original)

    class FrozenTaskModel(BaseModel, frozen=True):
        messages: list[AnyMessage]
        _resource: Any = PrivateAttr(default_factory=object)

    messages: list[AnyMessage] = [
        SystemMessage(content="request-only", id="explicit-system"),
        SystemMessage(
            content="explicit-developer",
            id="explicit-developer",
            additional_kwargs={"__openai_role__": "developer"},
        ),
        HumanMessage(content="saved-user", id="saved-user"),
    ]
    frozen = FrozenTaskInput(messages)
    model = FrozenTaskModel(messages=messages)
    packets = [Send("worker", frozen), Send("worker", model)]
    untouched = Send("worker", {"messages": "opaque-business-input"})
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"][TASKS] = [*packets, untouched]
    original = CheckpointTuple(
        config={"configurable": {"checkpoint_ns": "child:task"}},
        checkpoint=checkpoint,
        metadata={},
    )
    instructions = _ResponseInstructions.create("request-only", "response-test")
    first = instructions.overlay(original)
    assert first is not None
    for overlaid in (first, instructions.overlay(first)):
        assert overlaid is not None
        restored = overlaid.checkpoint["channel_values"][TASKS]
        for packet, source in zip(restored, packets):
            assert packet is not source
            assert packet.node == source.node
            assert type(packet.arg) is type(source.arg)
            assert packet.arg.messages[1:] == messages
            assert (
                sum(_is_request_instruction(item) for item in packet.arg.messages) == 1
            )
            assert packet.arg.messages[0].additional_kwargs == {
                "__openai_role__": "developer"
            }
            assert source.arg.messages == messages
        assert restored[0].arg.original is frozen.original
        assert restored[1].arg._resource is model._resource
        assert restored[-1] is untouched
        assert "messages" not in overlaid.checkpoint["channel_values"]
    assert checkpoint["channel_values"][TASKS] == [*packets, untouched]
    assert not _contains_instruction(original.checkpoint)
    assert _ResponseInstructions.create(None, "omitted").overlay(original) is original


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("instructions", [None, "", "request-only"])
def test_fresh_send_instructions_do_not_depend_on_checkpointing(
    checkpointed: bool, stream: bool, conditional: bool, instructions: str | None
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    messages = [
        SystemMessage(content="worker-system", id="worker-system"),
        HumanMessage(content="worker-input", id="worker-input"),
    ]
    payload = {"messages": messages}

    def route(state: MessagesState) -> list[Send]:
        return [Send("answer", payload)]

    def dispatch(state: MessagesState) -> Command:
        return Command(goto=route(state))

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    saver = InMemorySaver() if checkpointed else None
    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    if conditional:
        builder.add_conditional_edges(START, route)
    else:
        builder.add_node("dispatch", dispatch)
        builder.add_edge(START, "dispatch")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)
    original_tasks = graph.channels[TASKS]
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    assert graph.channels[TASKS] is original_tasks
    hosted_tasks = server.graph.channels[TASKS]
    prepared = with_response_instructions(server.graph)
    assert server.graph.channels[TASKS] is hosted_tasks
    with TestClient(server.app) as client:
        _create(
            client,
            stream=stream,
            input="parent-input",
            instructions=instructions,
        )
        assert client.portal is not None

        async def outside_host() -> None:
            await prepared.ainvoke(
                {"messages": [HumanMessage(content="outside-host")]},
                {"configurable": {"thread_id": "outside-host"}},
            )

        client.portal.call(outside_host)
    assert [[message.content for message in call] for call in model._calls] == [
        [*([instructions] if instructions else []), "worker-system", "worker-input"],
        ["worker-system", "worker-input"],
    ]
    assert payload["messages"] is messages
    assert [message.content for message in messages] == [
        "worker-system",
        "worker-input",
    ]
    if saver is not None:
        for checkpoint in saver.list(None):
            assert not _contains_instruction(checkpoint.checkpoint)
            assert not _contains_instruction(checkpoint.metadata)
            assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "task_schema",
    [
        _ReplacingMessagesState,
        _DataclassTaskInput,
        _PydanticTaskInput,
        _PydanticTupleTaskInput,
        _PydanticExtraTaskInput,
    ],
    ids=["dict", "dataclass", "pydantic", "pydantic-tuple", "pydantic-extra"],
)
def test_send_model_inputs_are_consistent_before_and_after_recovery(
    stream: bool, nested: bool, task_schema: type
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    failed = False
    messages = [
        SystemMessage(content="worker-system", id="worker-system"),
        HumanMessage(content="worker-input", id="worker-input"),
    ]
    payload = task_schema(messages=messages)

    def dispatch(state: MessagesState) -> Command:
        return Command(goto=Send("answer", payload))

    async def answer(state: Any) -> dict[str, Any]:
        nonlocal failed
        current = state["messages"] if isinstance(state, dict) else state.messages
        result = await model.ainvoke(current)
        if not failed:
            failed = True
            raise RuntimeError("retry this Send task")
        return {"messages": [result]}

    saver = InMemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node("dispatch", dispatch)
    builder.add_node("answer", answer, input_schema=task_schema)
    builder.add_edge(START, "dispatch")
    builder.add_edge("answer", END)
    provider = InMemoryResponseProvider()
    graph = builder.compile(checkpointer=True if nested else saver)
    if nested:
        parent = StateGraph(MessagesState)
        parent.add_node("child", graph)
        parent.add_edge(START, "child")
        parent.add_edge("child", END)
        graph = parent.compile(checkpointer=saver)
    server = ResponsesHostServer(graph, store=provider)
    request = cast(
        CreateResponse,
        {"input": "parent-input", "instructions": "request-only", "stream": stream},
    )
    with TestClient(server.app) as client:
        first = _create(
            client,
            stream=stream,
            expected_status="failed",
            input="parent-input",
            instructions="request-only",
        )
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, first["id"])
        assert stored is not None
        context = ResponseContext(
            response_id=stored["id"],
            mode_flags=ResponseModeFlags(stream=stream, store=True, background=False),
            request=request,
            provider=provider,
        )
        context.is_recovery = True
        context.persisted_response = stored

        async def recover() -> list[Any]:
            return [
                event
                async for event in server._handle_create_async_gen(
                    request, context, asyncio.Event()
                )
            ]

        events = client.portal.call(recover)
    assert events[-1]["type"] == "response.completed"
    assert [[message.content for message in call] for call in model._calls] == [
        ["request-only", "worker-system", "worker-input"],
        ["request-only", "worker-system", "worker-input"],
    ]
    original = payload["messages"] if isinstance(payload, dict) else payload.messages
    assert list(original) == messages
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.metadata)
        assert not _contains_instruction(checkpoint.pending_writes)


def test_existing_agent_middleware_and_system_prompt_remain_in_place() -> None:
    capture = _CaptureMessages()
    observed: list[list[BaseMessage]] = []

    class ExistingMiddleware(AgentMiddleware):
        def before_model(self, state: Any, runtime: Any) -> dict[str, Any]:
            observed.append(deepcopy(state["messages"]))
            return {"messages": list(state["messages"])}

    middleware = [ExistingMiddleware()]
    saver = InMemorySaver()
    graph = create_agent(
        model=FakeListChatModel(responses=["answer"], callbacks=[capture]),
        tools=[],
        middleware=middleware,
        system_prompt="agent-system",
        checkpointer=saver,
    )
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        first = _create(
            client, stream=False, input="first", instructions="request-only"
        )
        _create(
            client,
            stream=False,
            input="second",
            previous_response_id=first["id"],
        )
    assert [message.content for message in capture.calls[0]] == [
        "agent-system",
        "request-only",
        "first",
    ]
    assert [message.content for message in capture.calls[1]] == [
        "agent-system",
        "first",
        "answer",
        "second",
    ]
    assert observed[0][0].content == "request-only"
    assert observed[1][0].content == "first"
    assert graph.checkpointer is saver
    assert len(middleware) == 1
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("instructions", [None, "resumed-instruction"])
@pytest.mark.parametrize(
    "schema",
    [
        MessagesState,
        _AppendingMessagesState,
        _DeltaReplacingSnapshotState,
        _DeltaReplacingReplayState,
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_interrupt_resume_uses_only_the_new_response_instructions(
    instructions: str | None, schema: type, nested: bool
) -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])

    def pause(state: MessagesState) -> dict[str, Any]:
        interrupt({"question": "continue?", "state": state})
        return {}

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    saver = InMemorySaver()
    builder: StateGraph = StateGraph(schema)
    builder.add_node("pause", pause, input_schema=schema)
    builder.add_node("answer", answer, input_schema=schema)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=True if nested else saver)
    if nested:
        parent: StateGraph = StateGraph(schema)
        parent.add_node("child", graph, input_schema=schema)
        parent.add_edge(START, "child")
        parent.add_edge("child", END)
        graph = parent.compile(checkpointer=saver)
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        first = _create(
            client, stream=False, input="first", instructions="request-only"
        )
        approval = next(
            item for item in first["output"] if item["type"] == "mcp_approval_request"
        )
        _create(
            client,
            stream=False,
            instructions=instructions,
            previous_response_id=first["id"],
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
            ],
        )
    assert [message.content for message in capture.calls[0]] == (
        [instructions, "first"] if instructions else ["first"]
    )
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("schema", [MessagesState, _AppendingMessagesState])
def test_instructions_cover_every_model_call_in_a_tool_and_reasoning_turn(
    stream: bool, schema: type
) -> None:
    capture = _CaptureMessages()
    responses: list[BaseMessage] = []
    for index in range(2):
        responses.extend(
            [
                AIMessage(
                    content=[
                        {
                            "type": "reasoning",
                            "summary": [
                                {"type": "summary_text", "text": "ordinary reasoning"}
                            ],
                        }
                    ],
                    tool_calls=[{"name": "lookup", "args": {}, "id": f"call-{index}"}],
                ),
                AIMessage(content="answer"),
            ]
        )
    model = FakeMessagesListChatModel(responses=responses, callbacks=[capture])

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    def tool(state: MessagesState) -> dict[str, Any]:
        message = state["messages"][-1]
        assert isinstance(message, AIMessage)
        return {
            "messages": [
                ToolMessage(
                    content="tool result", tool_call_id=message.tool_calls[0]["id"]
                )
            ]
        }

    def route(state: MessagesState | _AppendingMessagesState) -> str:
        message = state["messages"][-1]
        return "tool" if isinstance(message, AIMessage) and message.tool_calls else END

    saver = InMemorySaver()
    builder: StateGraph = StateGraph(schema)
    builder.add_node("answer", answer, input_schema=schema)
    builder.add_node("tool", tool, input_schema=schema)
    builder.add_edge(START, "answer")
    builder.add_conditional_edges("answer", route)
    builder.add_edge("tool", "answer")
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(
            client, stream=stream, input="first", instructions="request-only"
        )
        _create(
            client,
            stream=stream,
            input="second",
            previous_response_id=first["id"],
        )
    assert all(call[0].content == "request-only" for call in capture.calls[:2])
    assert not any(
        message.content == "request-only"
        for call in capture.calls[2:]
        for message in call
    )
    assert any(isinstance(message, ToolMessage) for message in capture.calls[2])
    assert any(
        isinstance(message, AIMessage) and message.content == responses[0].content
        for message in capture.calls[2]
    )
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


def test_projection_preserves_explicit_messages_and_original_task_payloads() -> None:
    instruction = _ResponseInstructions.create("request-only", "response-test").message
    assert instruction is not None
    explicit = SystemMessage(content="request-only", id="explicit")
    messages = [instruction, explicit]
    packet = Send("model", {"messages": messages})
    payload = deque(
        [
            ("messages", messages),
            ("tasks", packet),
            ("update", Command(update={"messages": messages})),
            ("overwrite", Overwrite(messages)),
            ("interrupt", Interrupt({"messages": messages}, id="approval-test")),
        ]
    )
    result = _without_request_instructions(payload)
    assert result[0] == ("messages", [explicit])
    assert result[1][1].arg == {"messages": [explicit]}
    assert result[2][1].update == {"messages": [explicit]}
    assert result[3][1].value == [explicit]
    assert result[4][1].value == {"messages": [explicit]}
    assert result[4][1].id == "approval-test"
    assert messages == [instruction, explicit]
    assert packet.arg["messages"] is messages


def test_projection_cleans_nested_dataclasses_without_mutating_runtime_state() -> None:
    @dataclass(frozen=True, slots=True)
    class FrozenPayload:
        nested: _DataclassTaskInput
        messages: list[AnyMessage] = field(init=False)
        unchanged: object = field(default_factory=object, init=False)

        def __post_init__(self) -> None:
            object.__setattr__(self, "messages", self.nested.messages)

    instruction = _ResponseInstructions.create("request-only", "response-test").message
    assert instruction is not None
    explicit = SystemMessage(content="request-only", id="explicit")
    messages: list[AnyMessage] = [instruction, explicit]
    nested = _DataclassTaskInput(messages)
    payload = FrozenPayload(nested)
    packet = Send("model", payload)

    projected = _without_request_instructions(packet)

    assert projected is not packet
    assert type(projected.arg) is FrozenPayload
    assert projected.arg is not payload
    assert projected.arg.nested is not nested
    assert projected.arg.nested.messages == [explicit]
    assert projected.arg.messages == [explicit]
    assert projected.arg.unchanged is payload.unchanged
    assert packet.arg is payload
    assert payload.nested is nested
    assert payload.messages is messages
    assert nested.messages is messages
    assert messages == [instruction, explicit]

    clean = FrozenPayload(_DataclassTaskInput([explicit]))
    assert _without_request_instructions(clean) is clean
    assert _without_request_instructions(FrozenPayload) is FrozenPayload


def test_projection_cleans_pydantic_extras_without_mutating_runtime_state() -> None:
    class Payload(BaseModel, extra="allow", frozen=True):
        messages: list[AnyMessage]
        _resource: Any = PrivateAttr(default_factory=object)

    instruction = _ResponseInstructions.create("request-only", "response-extra").message
    assert instruction is not None
    explicit = HumanMessage(content="saved-user")
    messages: list[AnyMessage] = [instruction, explicit]
    original = Payload.model_validate(
        {
            "messages": messages,
            "nested": {"messages": messages},
            "label": "keep",
        }
    )
    projected = _without_request_instructions(original)
    assert type(projected) is Payload
    assert projected.messages == [explicit]
    assert projected.model_extra == {
        "nested": {"messages": [explicit]},
        "label": "keep",
    }
    assert projected._resource is original._resource
    assert original.messages == [instruction, explicit]
    assert original.model_extra == {
        "nested": {"messages": [instruction, explicit]},
        "label": "keep",
    }
    assert _without_request_instructions(projected) is projected


@pytest.mark.parametrize(
    "label",
    [
        {},
        {"type": "message_reference"},
        {"role": "audit_reference"},
        {"content": "keep"},
        {"type": "message_reference", "content": "keep"},
        {"role": "audit_reference", "content": "keep"},
    ],
)
def test_projection_preserves_business_references_to_instruction_ids(
    label: dict[str, str],
) -> None:
    instruction = _ResponseInstructions.create("request-only", "response-ref").message
    assert instruction is not None
    reference = {"id": instruction.id, **label, "label": "keep"}
    explicit = HumanMessage(content="saved-user")
    original = {
        "messages": [instruction, explicit],
        "reference": reference,
    }
    projected = _without_request_instructions(original)
    assert projected == {"messages": [explicit], "reference": reference}
    assert projected["reference"] is reference
    assert original["messages"] == [instruction, explicit]
    assert not _is_request_instruction(reference)


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("extra_data", [False, True])
@pytest.mark.parametrize(
    "message_type",
    [
        SystemMessage,
        SystemMessageChunk,
        HumanMessage,
        AIMessage,
        AIMessageChunk,
        RemoveMessage,
    ],
)
def test_projection_removes_owned_serialized_messages(
    serialized: bool, extra_data: bool, message_type: type[BaseMessage]
) -> None:
    instruction = _ResponseInstructions.create("request-only", "response-wire").message
    assert instruction is not None
    assert instruction.id is not None
    message = (
        RemoveMessage(id=instruction.id)
        if message_type is RemoveMessage
        else message_type(content="request-only", id=instruction.id)
    )
    if extra_data:
        message = message.model_copy(update={"data": {"label": "keep"}})
    owned = message_to_dict(message) if serialized else message.model_dump()
    explicit = SystemMessage(content="explicit-system", id="explicit")
    projected = _without_request_instructions(
        {"messages": [owned, explicit.model_dump()]}
    )
    assert projected == {"messages": [explicit.model_dump()]}
    assert _is_request_instruction(owned)


@pytest.mark.parametrize("role", ["system", "developer", "user", "assistant"])
@pytest.mark.parametrize("typed", [False, True])
def test_projection_removes_owned_openai_messages(role: str, typed: bool) -> None:
    instruction = _ResponseInstructions.create(
        "request-only", "response-openai"
    ).message
    assert instruction is not None
    message = {
        "role": role,
        "content": "request-only",
        "id": instruction.id,
        **({"type": "message"} if typed else {}),
    }
    assert _without_request_instructions({"messages": [message]}) == {"messages": []}


def test_projection_removes_developer_dumps_without_default_fields() -> None:
    instruction = _ResponseInstructions.create(
        "request-only", "response-minimal"
    ).message
    assert instruction is not None
    serialized = instruction.model_dump(exclude_defaults=True)
    assert "type" not in serialized
    assert _without_request_instructions({"messages": [serialized]}) == {"messages": []}


@pytest.mark.parametrize("serialized", [False, True])
def test_projection_preserves_explicit_messages_with_owned_nested_data(
    serialized: bool,
) -> None:
    instruction = _ResponseInstructions.create("request-only", "response-data").message
    assert instruction is not None
    explicit = SystemMessage(content="explicit-system", id="explicit").model_copy(
        update={"data": instruction.model_dump()}
    )
    original = message_to_dict(explicit) if serialized else explicit.model_dump()
    expected = deepcopy(original)
    if serialized:
        expected["data"]["data"] = []
    else:
        expected["data"] = []
    projected = _without_request_instructions({"messages": [original]})
    assert projected == {"messages": [expected]}
    assert not _is_request_instruction(original)
    assert explicit.model_extra == {"data": instruction.model_dump()}


@pytest.mark.parametrize("representation", ["role", "type", "serialized"])
def test_projection_removes_owned_removal_references(representation: str) -> None:
    instruction = _ResponseInstructions.create(
        "request-only", "response-remove"
    ).message
    assert instruction is not None
    assert instruction.id is not None
    if representation == "serialized":
        removal: dict[str, Any] = {
            "type": "remove",
            "data": {"id": instruction.id},
        }
        assert messages_from_dict([removal]) == [RemoveMessage(id=instruction.id)]
    else:
        removal = {representation: "remove", "id": instruction.id}
    explicit = HumanMessage(content="saved-user")
    assert _without_request_instructions({"messages": [removal, explicit]}) == {
        "messages": [explicit]
    }


def test_projection_distinguishes_message_tuples_from_write_tuples() -> None:
    class Write(NamedTuple):
        channel: str
        value: tuple[AnyMessage, ...]

    instruction = _ResponseInstructions.create("request-only", "response-tuple").message
    assert instruction is not None
    explicit = HumanMessage(content="saved-user")
    messages = (instruction, explicit)
    original = _PydanticTupleTaskInput(messages=messages)
    projected = _without_request_instructions(original)
    assert projected.messages == (explicit,)
    assert original.messages == messages
    assert _without_request_instructions(projected) is projected
    writes = [
        ("messages", messages),
        ("messages", instruction),
        ("reference", ("messages", instruction.id)),
        ("task", "messages", messages),
        ("update", Command(update=[("messages", messages)])),
        ("overwrite", Command(update={"messages": Overwrite(messages)})),
        ("interrupt", Interrupt({"messages": messages}, id="approval-tuple")),
    ]
    assert _without_request_instructions(writes) == [
        ("messages", (explicit,)),
        ("messages", []),
        ("reference", ("messages", instruction.id)),
        ("task", "messages", (explicit,)),
        ("update", Command(update=[("messages", (explicit,))])),
        ("overwrite", Command(update={"messages": Overwrite((explicit,))})),
        ("interrupt", Interrupt({"messages": (explicit,)}, id="approval-tuple")),
    ]
    projected_write = _without_request_instructions(Write("messages", messages))
    assert type(projected_write) is Write
    assert projected_write == Write("messages", (explicit,))


@pytest.mark.parametrize("synchronous", [False, True])
async def test_saved_tool_artifacts_exclude_instructions_without_losing_references(
    synchronous: bool,
) -> None:
    instruction = _ResponseInstructions.create(
        "request-only", "response-artifact"
    ).message
    assert instruction is not None
    explicit = SystemMessage(content="request-only", id="explicit")
    reference = {"id": instruction.id, "label": "keep"}
    artifact = {"messages": [instruction, explicit], "reference": reference}
    tool = ToolMessage(content="tool-result", tool_call_id="call-1", artifact=artifact)
    backend = InMemorySaver()
    saver = _RequestCheckpointSaver(backend)
    config: RunnableConfig = {
        "configurable": {"thread_id": "instruction-artifacts", "checkpoint_ns": ""}
    }
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"]["messages"] = [tool]
    checkpoint["channel_versions"]["messages"] = 1
    if synchronous:
        config = saver.put(
            config,
            checkpoint,
            {"source": "loop", "step": 0, "parents": {}},
            checkpoint["channel_versions"],
        )
        saver.put_writes(config, [("messages", [tool])], "tool-task")
    else:
        config = await saver.aput(
            config,
            checkpoint,
            {"source": "loop", "step": 0, "parents": {}},
            checkpoint["channel_versions"],
        )
        await saver.aput_writes(config, [("messages", [tool])], "tool-task")
    restored = await backend.aget_tuple(config)
    assert restored is not None
    saved = restored.checkpoint["channel_values"]["messages"][0]
    assert not _contains_instruction(saved)
    assert saved.artifact == {
        "messages": [explicit.model_dump()],
        "reference": reference,
    }
    assert saved.content == "tool-result"
    assert saved.tool_call_id == "call-1"
    assert restored.pending_writes is not None
    assert restored.pending_writes[0][2][0].artifact == saved.artifact
    assert tool.artifact is artifact
    assert artifact["messages"] == [instruction, explicit]


def _merge_message_deltas(state: Messages, updates: Sequence[Messages]) -> Messages:
    for update in updates:
        state = add_messages(state, update)
    return state


class _DeltaSnapshotState(TypedDict):
    messages: Annotated[
        list[AnyMessage], DeltaChannel(_merge_message_deltas, snapshot_frequency=1)
    ]


class _DeltaReplayState(TypedDict):
    messages: Annotated[list[AnyMessage], DeltaChannel(_merge_message_deltas)]


def _append_message_deltas(
    state: list[AnyMessage], updates: Sequence[list[AnyMessage]]
) -> list[AnyMessage]:
    return state + [message for update in updates for message in update]


class _DeltaAppendingSnapshotState(TypedDict):
    messages: Annotated[
        list[AnyMessage], DeltaChannel(_append_message_deltas, snapshot_frequency=1)
    ]


class _DeltaAppendingReplayState(TypedDict):
    messages: Annotated[list[AnyMessage], DeltaChannel(_append_message_deltas)]


@pytest.mark.parametrize("stream", [False, True])
def test_tuple_delta_updates_do_not_persist_current_instructions(stream: bool) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    forwarded: list[tuple[AnyMessage, ...]] = []

    def forward(state: MessagesState) -> dict[str, Any]:
        update = tuple(state["messages"])
        forwarded.append(update)
        return {"messages": update}

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    saver = InMemorySaver()
    builder = StateGraph(_DeltaAppendingReplayState)
    builder.add_node("forward", forward, input_schema=_DeltaAppendingReplayState)
    builder.add_node("answer", answer, input_schema=_DeltaAppendingReplayState)
    builder.add_edge(START, "forward")
    builder.add_edge("forward", "answer")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(
            client, stream=stream, input="first", instructions="request-only"
        )
        second = _create(
            client,
            stream=stream,
            input="second",
            previous_response_id=first["id"],
        )
    assert first["status"] == second["status"] == "completed"
    assert [message.content for message in model._calls[0]] == [
        "request-only",
        "first",
        "first",
    ]
    assert not any(_is_request_instruction(message) for message in model._calls[1])
    assert forwarded[0][0].content == "request-only"
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("schema", [_DeltaSnapshotState, _DeltaReplayState])
@pytest.mark.parametrize("reset", ["remove_all", "overwrite"])
def test_delta_history_resets_do_not_clear_current_instructions(
    stream: bool, schema: type, reset: str
) -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])

    async def answer(state: MessagesState) -> dict[str, Any]:
        message = await model.ainvoke(state["messages"])
        return {
            "messages": (
                [RemoveMessage(id=REMOVE_ALL_MESSAGES), message]
                if reset == "remove_all"
                else Overwrite([message])
            )
        }

    saver = InMemorySaver()
    builder: StateGraph = StateGraph(schema)
    builder.add_node("answer", answer, input_schema=schema)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(client, stream=stream, input="first", instructions="turn-A")
        second = _create(
            client,
            stream=stream,
            input="second",
            instructions="turn-B",
            previous_response_id=first["id"],
        )
        _create(
            client,
            stream=stream,
            input="third",
            previous_response_id=second["id"],
        )
    assert [[message.content for message in call] for call in capture.calls] == [
        ["turn-A", "first"],
        ["turn-B", "answer", "second"],
        ["answer", "third"],
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("branching", [False, True])
@pytest.mark.parametrize(
    "child_schema",
    [
        _AppendingMessagesState,
        _DeltaAppendingSnapshotState,
        _DeltaAppendingReplayState,
    ],
)
def test_persistent_appending_subgraphs_receive_current_instructions_once(
    stream: bool, branching: bool, child_schema: type
) -> None:
    capture = _CaptureMessages()
    child = _graph(capture, None, schema=child_schema).copy({"checkpointer": True})
    saver = InMemorySaver()
    builder = StateGraph(_ReplacingMessagesState)
    builder.add_node("child", child)
    builder.add_edge(START, "child")
    builder.add_edge("child", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver),
        store=InMemoryResponseProvider(),
        enable_response_branching=branching,
    )
    with TestClient(server.app) as client:
        first = _create(client, stream=stream, input="first", instructions="turn-A")
        second = _create(
            client,
            stream=stream,
            input="second",
            instructions="turn-B",
            previous_response_id=first["id"],
        )
        _create(
            client,
            stream=stream,
            input="third",
            previous_response_id=second["id"],
        )
        if branching:
            _create(
                client,
                stream=stream,
                input="sibling",
                instructions="turn-S",
                previous_response_id=first["id"],
            )
    assert [[message.content for message in call] for call in capture.calls[:3]] == [
        ["turn-A", "first"],
        ["turn-B", "first", "answer", "second"],
        ["first", "answer", "second", "answer", "third"],
    ]
    if branching:
        # Private subgraph history is not rewound; request instructions are isolated.
        assert [
            message.content
            for message in capture.calls[3]
            if _is_request_instruction(message)
        ] == ["turn-S"]
        assert capture.calls[3][-1].content == "sibling"
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("schema", [_DeltaSnapshotState, _DeltaReplayState])
async def test_delta_checkpoint_reads_overlay_after_history_without_mutation(
    synchronous: bool, schema: type
) -> None:
    explicit = SystemMessage(content="explicit-system", id="explicit")
    saved = HumanMessage(content="saved-user", id="saved")
    saver = InMemorySaver()
    builder: StateGraph = StateGraph(schema)
    builder.add_node(
        "reset",
        lambda state: {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), explicit, saved]
        },
        input_schema=schema,
    )
    builder.add_edge(START, "reset")
    builder.add_edge("reset", END)
    graph = builder.compile(checkpointer=saver)
    config: RunnableConfig = {"configurable": {"thread_id": "delta-instructions"}}
    await graph.ainvoke({"messages": [HumanMessage(content="initial")]}, config)
    config = (await graph.aget_state(config)).config
    original = deepcopy(
        saver.get_delta_channel_history(config=config, channels=["messages"])
    )
    wrapped = with_response_instructions(
        graph.copy({"checkpointer": _RequestCheckpointSaver(saver)})
    )
    for current in ("turn-A", "turn-B", None):
        instructions = _ResponseInstructions.create(current, f"response-{current}")
        selected = instructions.with_config(config)
        with instructions.bind():
            state = (
                wrapped.get_state(selected)
                if synchronous
                else await wrapped.aget_state(selected)
            )
        assert [message.content for message in state.values["messages"]] == [
            *([current] if current else []),
            "explicit-system",
            "saved-user",
        ]
    assert (
        saver.get_delta_channel_history(config=config, channels=["messages"])
        == original
    )
    assert (await graph.aget_state(config)).values["messages"] == [explicit, saved]


@pytest.mark.parametrize(
    "schema",
    [
        _DeltaSnapshotState,
        _DeltaReplayState,
        _DeltaAppendingSnapshotState,
        _DeltaAppendingReplayState,
    ],
)
def test_delta_messages_keep_history_without_persisting_instructions(
    schema: type,
) -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    saver = InMemorySaver()
    builder: StateGraph = StateGraph(schema)
    builder.add_node("answer", answer, input_schema=schema)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(
            client, stream=False, input="first", instructions="request-only"
        )
        second = _create(
            client,
            stream=False,
            input="second",
            instructions="second-instruction",
            previous_response_id=first["id"],
        )
        _create(
            client,
            stream=False,
            input="third",
            previous_response_id=second["id"],
        )
    assert [message.content for message in capture.calls[1]] == [
        "second-instruction",
        "first",
        "answer",
        "second",
    ]
    assert [message.content for message in capture.calls[2]] == [
        "first",
        "answer",
        "second",
        "answer",
        "third",
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize(
    "schema",
    [MessagesState, _AppendingMessagesState, _DeltaSnapshotState, _DeltaReplayState],
)
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_recovery_reapplies_instructions_without_duplicating_input(
    schema: type, depth: int
) -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])
    failed = False

    async def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [await model.ainvoke(state["messages"])]}

    def fail_once(state: MessagesState) -> dict[str, Any]:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("retry this node")
        return {}

    saver = InMemorySaver()
    builder: StateGraph = StateGraph(schema)
    builder.add_node("first", answer, input_schema=schema)
    builder.add_node("retry", fail_once, input_schema=schema)
    builder.add_node("second", answer, input_schema=schema)
    builder.add_edge(START, "first")
    builder.add_edge("first", "retry")
    builder.add_edge("retry", "second")
    builder.add_edge("second", END)
    provider = InMemoryResponseProvider()
    graph = builder.compile(checkpointer=True if depth else saver)
    for level in range(depth):
        parent: StateGraph = StateGraph(schema)
        name = f"child-{level}"
        parent.add_node(name, graph, input_schema=schema)
        parent.add_edge(START, name)
        parent.add_edge(name, END)
        graph = parent.compile(checkpointer=saver if level == depth - 1 else True)
    server = ResponsesHostServer(graph, store=provider)
    request = cast(
        CreateResponse,
        {"input": "first", "instructions": "request-only", "stream": False},
    )
    with TestClient(server.app) as client:
        first = client.post("/responses", json=dict(request))
        assert first.status_code == 200, first.text
        assert first.json()["status"] == "failed"
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, first.json()["id"])
        assert stored is not None
        context = ResponseContext(
            response_id=stored["id"],
            mode_flags=ResponseModeFlags(stream=False, store=True, background=False),
            request=request,
            provider=provider,
        )
        context.is_recovery = True
        context.persisted_response = stored

        async def recover() -> list[Any]:
            return [
                event
                async for event in server._handle_create_async_gen(
                    request, context, asyncio.Event()
                )
            ]

        events = client.portal.call(recover)
    assert events[-1]["type"] == "response.completed"
    assert [message.content for message in capture.calls[-1]] == [
        "request-only",
        "first",
        "answer",
    ]
    assert all(
        sum(_is_request_instruction(message) for message in call) == 1
        for call in capture.calls
    )
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "resumed-instruction"])
@pytest.mark.parametrize(
    "schema",
    [
        _DeltaSnapshotState,
        _DeltaReplayState,
        _DeltaReplacingSnapshotState,
        _DeltaReplacingReplayState,
        _DeltaAppendingSnapshotState,
        _DeltaAppendingReplayState,
    ],
)
def test_runtime_created_delta_subgraphs_resume_with_current_instructions(
    stream: bool, instructions: str | None, schema: type
) -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])

    def create_child() -> CompiledStateGraph:
        def pause(state: MessagesState) -> dict[str, Any]:
            interrupt({"question": "continue?"})
            return {}

        async def answer(state: MessagesState) -> dict[str, Any]:
            return {"messages": [await model.ainvoke(state["messages"])]}

        child: StateGraph = StateGraph(schema)
        child.add_node("pause", pause, input_schema=schema)
        child.add_node("answer", answer, input_schema=schema)
        child.add_edge(START, "pause")
        child.add_edge("pause", "answer")
        child.add_edge("answer", END)
        return with_response_instructions(child.compile(checkpointer=True))

    async def call_child(state: MessagesState) -> dict[str, Any]:
        return await create_child().ainvoke(state)

    saver = InMemorySaver()
    parent = StateGraph(MessagesState)
    parent.add_node("dynamic", call_child)
    parent.add_edge(START, "dynamic")
    parent.add_edge("dynamic", END)
    graph = parent.compile(checkpointer=saver)
    assert not list(graph.get_subgraphs(recurse=True))
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        first = _create(
            client, stream=stream, input="first", instructions="initial-instruction"
        )
        approval = next(
            item for item in first["output"] if item["type"] == "mcp_approval_request"
        )
        _create(
            client,
            stream=stream,
            previous_response_id=first["id"],
            instructions=instructions,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
            ],
        )
    assert [[message.content for message in call] for call in capture.calls] == [
        [*([instructions] if instructions else []), "first"]
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.metadata)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "resumed-instruction"])
@pytest.mark.parametrize(
    "child_schema",
    [
        _AppendingMessagesState,
        _DeltaAppendingSnapshotState,
        _DeltaAppendingReplayState,
    ],
)
def test_parent_approval_does_not_double_inject_into_fresh_persistent_children(
    stream: bool, instructions: str | None, child_schema: type
) -> None:
    capture = _CaptureMessages()
    child = _graph(capture, None, schema=child_schema).copy({"checkpointer": True})

    def pause(state: _ReplacingMessagesState) -> dict[str, Any]:
        if any(
            isinstance(message, HumanMessage) and message.content == "second"
            for message in state["messages"]
        ):
            interrupt({"question": "continue?"})
        return {}

    saver = InMemorySaver()
    parent = StateGraph(_ReplacingMessagesState)
    parent.add_node("pause", pause, input_schema=_ReplacingMessagesState)
    parent.add_node("child", child, input_schema=_ReplacingMessagesState)
    parent.add_edge(START, "pause")
    parent.add_edge("pause", "child")
    parent.add_edge("child", END)
    server = ResponsesHostServer(
        parent.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(client, stream=stream, input="first", instructions="turn-A")
        paused = _create(
            client,
            stream=stream,
            input="second",
            instructions="turn-B",
            previous_response_id=first["id"],
        )
        approval = next(
            item for item in paused["output"] if item["type"] == "mcp_approval_request"
        )
        _create(
            client,
            stream=stream,
            previous_response_id=paused["id"],
            instructions=instructions,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
            ],
        )
    assert [
        message.content
        for message in capture.calls[-1]
        if _is_request_instruction(message)
    ] == ([instructions] if instructions else [])
    assert [
        message.content
        for message in capture.calls[-1]
        if not _is_request_instruction(message)
    ] == ["first", "answer", "second"]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.metadata)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "request-only"])
def test_input_checkpoint_recovery_preserves_current_instructions(
    stream: bool, instructions: str | None
) -> None:
    failed = False

    def replace_once(
        previous: list[AnyMessage], current: list[AnyMessage]
    ) -> list[AnyMessage]:
        nonlocal failed
        if not failed and any(isinstance(message, HumanMessage) for message in current):
            failed = True
            raise RuntimeError("fail before the first graph step commits")
        return current

    InputRecoveryState = TypedDict(
        "InputRecoveryState",
        {"messages": Annotated[list[AnyMessage], replace_once]},
    )
    capture = _CaptureMessages()
    saver = InMemorySaver()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(
        _graph(capture, saver, schema=InputRecoveryState), store=provider
    )
    request = cast(
        CreateResponse,
        {
            "model": "test",
            "input": "first",
            "instructions": instructions,
            "stream": stream,
        },
    )
    with TestClient(server.app) as client:
        first = _create(
            client,
            stream=stream,
            expected_status="failed",
            input="first",
            instructions=instructions,
        )
        checkpoint = next(saver.list(None))
        assert checkpoint.metadata["source"] == "input"
        assert checkpoint.pending_writes is not None
        assert any(channel == "messages" for _, channel, _ in checkpoint.pending_writes)
        assert client.portal is not None
        stored = client.portal.call(provider.get_response, first["id"])
        assert stored is not None
        context = ResponseContext(
            response_id=stored["id"],
            mode_flags=ResponseModeFlags(stream=stream, store=True, background=False),
            request=request,
            provider=provider,
        )
        context.is_recovery = True
        context.persisted_response = stored

        async def recover() -> list[Any]:
            return [
                event
                async for event in server._handle_create_async_gen(
                    request, context, asyncio.Event()
                )
            ]

        events = client.portal.call(recover)
    assert events[-1]["type"] == "response.completed"
    assert [[message.content for message in call] for call in capture.calls] == [
        [*([instructions] if instructions else []), "first"]
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.metadata)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "schema",
    [
        MessagesState,
        _AppendingMessagesState,
        _ReplacingMessagesState,
        _ReplacingReducerMessagesState,
        _DeltaSnapshotState,
        _DeltaReplayState,
        _DeltaAppendingSnapshotState,
        _DeltaAppendingReplayState,
        _DeltaReplacingSnapshotState,
        _DeltaReplacingReplayState,
    ],
)
async def test_actual_model_inputs_are_isolated_across_concurrent_requests(
    stream: bool, schema: type
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    saver = InMemorySaver()
    gate = asyncio.Barrier(2)

    async def answer(state: MessagesState) -> dict[str, Any]:
        if state["messages"][-1].content in ("input-A", "input-B"):
            await asyncio.wait_for(gate.wait(), timeout=10)
        return {"messages": [await model.ainvoke(state["messages"])]}

    builder: StateGraph = StateGraph(schema)
    builder.add_node("answer", answer, input_schema=schema)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        with ThreadPoolExecutor(max_workers=2) as executor:
            requests = [
                executor.submit(
                    _create,
                    client,
                    stream=stream,
                    input=f"input-{name}",
                    instructions=f"instruction-{name}",
                )
                for name in ("A", "B")
            ]
            for request in requests:
                request.result(timeout=20)
        _create(client, stream=stream, input="plain-after")

    prepared = with_response_instructions(graph)
    await prepared.ainvoke(
        {
            "messages": [
                SystemMessage(content="explicit-system"),
                HumanMessage(content="outside-host"),
            ]
        },
        {"configurable": {"thread_id": "outside-host"}},
    )
    assert {
        call[-1].content: [message.content for message in call] for call in model._calls
    } == {
        "input-A": ["instruction-A", "input-A"],
        "input-B": ["instruction-B", "input-B"],
        "plain-after": ["plain-after"],
        "outside-host": ["explicit-system", "outside-host"],
    }
    assert prepared is not graph
    assert isinstance(prepared.channels["messages"], type(graph.channels["messages"]))
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.metadata)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("close_from_other_task", [False, True])
async def test_instruction_context_does_not_escape_a_suspended_response_stream(
    close_from_other_task: bool,
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)

    def answer(state: MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider)
    request = cast(
        CreateResponse,
        {
            "model": "test",
            "input": "inside-host",
            "instructions": "request-only",
            "stream": True,
        },
    )
    context = ResponseContext(
        response_id="response-context-isolation",
        mode_flags=ResponseModeFlags(stream=True, store=True, background=False),
        request=request,
        provider=provider,
    )
    iterator = server.handle_create(request, context, asyncio.Event())
    assert isinstance(iterator, AsyncGenerator)
    try:
        while not model._calls:
            await anext(iterator)
        await with_response_instructions(graph).ainvoke(
            {"messages": [HumanMessage(content="outside-host")]}
        )
    finally:
        if close_from_other_task:
            await asyncio.create_task(iterator.aclose())
        else:
            await iterator.aclose()
    assert [[message.content for message in call] for call in model._calls] == [
        ["request-only", "inside-host"],
        ["outside-host"],
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "resumed-instruction"])
@pytest.mark.parametrize("explicit_prepare", [False, True])
def test_statically_discovered_pregel_children_receive_resumed_instructions(
    stream: bool, instructions: str | None, explicit_prepare: bool
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)

    def answer(state: MessagesState) -> list[BaseMessage]:
        interrupt({"question": "continue?"})
        return [model.invoke(state["messages"])]

    child: Pregel = Pregel(
        nodes={
            "answer": (
                NodeBuilder()
                .subscribe_to("trigger", read=False)
                .read_from("messages")
                .do(answer)
                .write_to("messages")
            )
        },
        channels={"messages": LastValue(list), "trigger": LastValue(bool)},
        input_channels=["messages", "trigger"],
        output_channels=["messages"],
        checkpointer=True,
    )
    if explicit_prepare:
        child = with_response_instructions(child)

    async def call_child(state: MessagesState) -> dict[str, Any]:
        return await child.ainvoke({**state, "trigger": True})

    saver = InMemorySaver()
    parent = StateGraph(MessagesState)
    parent.add_node("low_level", call_child)
    parent.add_edge(START, "low_level")
    parent.add_edge("low_level", END)
    graph = parent.compile(checkpointer=saver)
    assert dict(graph.get_subgraphs(recurse=True))["low_level"] is child
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        first = _create(
            client, stream=stream, input="first", instructions="initial-instruction"
        )
        approval = next(
            item for item in first["output"] if item["type"] == "mcp_approval_request"
        )
        _create(
            client,
            stream=stream,
            previous_response_id=first["id"],
            instructions=instructions,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": approval["id"],
                    "approve": True,
                }
            ],
        )
    assert [[message.content for message in call] for call in model._calls] == [
        [*([instructions] if instructions else []), "first"]
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.metadata)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize(
    "message_type",
    [SystemMessage, SystemMessageChunk, HumanMessage, AIMessage, AIMessageChunk],
)
def test_projection_removes_owned_constructor_messages(
    message_type: type[BaseMessage],
) -> None:
    instruction = _ResponseInstructions.create("request-only", "constructor").message
    assert instruction is not None
    owned = message_type(content="request-only", id=instruction.id).to_json()
    explicit = SystemMessage(content="explicit-system", id="explicit").to_json()
    reference = {
        "lc": 1,
        "type": "constructor",
        "id": ["application", "AuditRecord"],
        "kwargs": {"id": instruction.id, "label": "keep"},
    }

    assert _is_request_instruction(owned)
    projected = _without_request_instructions([owned, explicit, reference])
    assert projected == [explicit, reference]
    restored = load(projected[0], allowed_objects="messages")
    assert isinstance(restored, SystemMessage)
    assert restored.content == "explicit-system"
    assert not _is_request_instruction(reference)


class _SerializedArchiveState(MessagesState):
    archive: list[dict[str, Any]]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "request-only"])
def test_constructor_message_archives_round_trip_across_responses(
    stream: bool, instructions: str | None
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    saver = InMemorySaver()

    async def answer(state: _SerializedArchiveState) -> dict[str, Any]:
        for value in state.get("archive", []):
            load(value, allowed_objects="messages")
        return {
            "messages": [await model.ainvoke(state["messages"])],
            "archive": [message.to_json() for message in state["messages"]],
        }

    builder = StateGraph(_SerializedArchiveState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
    with TestClient(server.app) as client:
        first = _create(
            client,
            stream=stream,
            input=[
                {"role": "developer", "content": "explicit-developer"},
                {"role": "user", "content": "first"},
            ],
            instructions=instructions,
        )
        _create(
            client,
            stream=stream,
            input="second",
            previous_response_id=first["id"],
        )
    assert [[message.content for message in call] for call in model._calls] == [
        [*([instructions] if instructions else []), "explicit-developer", "first"],
        ["explicit-developer", "first", "answer", "second"],
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("instructions", [None, "", "current-only"])
@pytest.mark.parametrize("resume_kind", ["approval", "recovery"])
@pytest.mark.parametrize("graph_kind", ["root", "subgraph", "functional"])
@pytest.mark.parametrize("async_task", [False, True])
def test_cached_task_result_rebinds_current_instructions(
    stream: bool,
    instructions: str | None,
    resume_kind: str,
    graph_kind: str,
    async_task: bool,
) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    task_inputs: list[list[AnyMessage]] = []
    failed = False

    @task
    def retain(messages: list[AnyMessage]) -> list[AnyMessage]:
        task_inputs.append(deepcopy(messages))
        return messages

    @task
    async def aretain(messages: list[AnyMessage]) -> list[AnyMessage]:
        task_inputs.append(deepcopy(messages))
        return messages

    async def answer(state: MessagesState) -> dict[str, Any]:
        nonlocal failed
        messages = (
            await aretain(state["messages"])
            if async_task
            else await retain(state["messages"])
        )
        if resume_kind == "approval":
            interrupt({"question": "continue?"})
        elif not failed:
            failed = True
            raise RuntimeError("retry after caching the task result")
        return {"messages": [await model.ainvoke(messages)]}

    saver = InMemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)
    if graph_kind != "root":
        child: Pregel = (
            entrypoint(checkpointer=saver)(answer)
            if graph_kind == "functional"
            else builder.compile(checkpointer=True)
        )
        child = with_response_instructions(child)
        parent = StateGraph(MessagesState)
        parent.add_node("child", child)
        parent.add_edge(START, "child")
        parent.add_edge("child", END)
        graph = parent.compile(checkpointer=saver)
    provider = InMemoryResponseProvider()
    server = ResponsesHostServer(graph, store=provider)
    initial = "initial-only" if resume_kind == "approval" else instructions
    request = cast(
        CreateResponse,
        {
            "model": "test",
            "input": [
                {"role": "developer", "content": "explicit-developer"},
                {"role": "user", "content": "first"},
            ],
            "instructions": initial,
            "stream": stream,
        },
    )
    with TestClient(server.app) as client:
        first = _create(
            client,
            stream=stream,
            input=request["input"],
            instructions=initial,
            expected_status="failed" if resume_kind == "recovery" else "completed",
        )
        if resume_kind == "approval":
            approval = next(
                item
                for item in first["output"]
                if item["type"] == "mcp_approval_request"
            )
            _create(
                client,
                stream=stream,
                instructions=instructions,
                previous_response_id=first["id"],
                input=[
                    {
                        "type": "mcp_approval_response",
                        "approval_request_id": approval["id"],
                        "approve": True,
                    }
                ],
            )
        else:
            assert client.portal is not None
            stored = client.portal.call(provider.get_response, first["id"])
            assert stored is not None
            context = ResponseContext(
                response_id=stored["id"],
                mode_flags=ResponseModeFlags(
                    stream=stream, store=True, background=False
                ),
                request=request,
                provider=provider,
            )
            context.is_recovery = True
            context.persisted_response = stored

            async def recover() -> list[Any]:
                return [
                    event
                    async for event in server._handle_create_async_gen(
                        request, context, asyncio.Event()
                    )
                ]

            events = client.portal.call(recover)
            assert events[-1]["type"] == "response.completed"
    assert len(task_inputs) == 1
    assert [[message.content for message in call] for call in model._calls] == [
        [*([instructions] if instructions else []), "explicit-developer", "first"]
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)
        assert "initial-only" not in repr(checkpoint)
        assert "current-only" not in repr(checkpoint)


@pytest.mark.parametrize("stream", [False, True])
def test_task_cache_rebinds_only_owned_instruction_slots(stream: bool) -> None:
    model = _ActualInputModel(responses=["answer"], disable_streaming=True)
    cache: InMemoryCache[Any] = InMemoryCache()
    calls: list[str] = []

    def prompt_key(messages: list[AnyMessage]) -> str | bytes:
        return "shared"

    def reply_key() -> str | bytes:
        return "shared"

    @task(cache_policy=CachePolicy(key_func=prompt_key))
    def retain(messages: list[AnyMessage]) -> list[AnyMessage]:
        calls.append("prompt")
        return messages

    @task(cache_policy=CachePolicy(key_func=reply_key))
    def reply() -> list[AIMessage]:
        calls.append("reply")
        return [AIMessage(content="ordinary-reply")]

    async def answer(state: MessagesState) -> dict[str, Any]:
        messages = await retain(state["messages"])
        ordinary = await reply()
        assert [message.content for message in ordinary] == ["ordinary-reply"]
        return {"messages": [await model.ainvoke(messages)]}

    builder = StateGraph(MessagesState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(cache=cache)
    server = ResponsesHostServer(graph, store=InMemoryResponseProvider())
    with TestClient(server.app) as client:
        for instructions in ("initial-only", "current-only", None):
            _create(
                client,
                stream=stream,
                input="first",
                instructions=instructions,
            )
        assert client.portal is not None

        async def outside_host() -> None:
            await server.graph.ainvoke({"messages": [HumanMessage(content="first")]})

        client.portal.call(outside_host)
    assert graph.cache is cache
    assert calls == ["prompt", "reply"]
    assert [[message.content for message in call] for call in model._calls] == [
        ["initial-only", "first"],
        ["current-only", "first"],
        ["first"],
        ["first"],
    ]
