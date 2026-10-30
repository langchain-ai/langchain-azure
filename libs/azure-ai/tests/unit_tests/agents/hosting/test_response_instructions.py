# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-scoped instructions through the public Responses endpoint."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from operator import add
from pathlib import Path
from typing import Annotated, Any, cast

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
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.channels import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import Messages, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, Overwrite, Send, interrupt
from starlette.testclient import TestClient
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.agents.hosting._instructions import (
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


class _AppendingMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add]


class _ReplacingMessagesState(TypedDict):
    messages: list[AnyMessage]


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


def _create(client: TestClient, *, stream: bool, **body: Any) -> ResponseObject:
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
    assert response["status"] == "completed", response
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
    linkage = {"conversation": {"id": "conv-instructions"}} if conversation else {}
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
def test_replacing_message_updates_keep_their_current_instructions(
    stream: bool, overwrite: bool
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
    schema = _AppendingMessagesState if overwrite else _ReplacingMessagesState
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
    server = ResponsesHostServer(
        _graph(capture, InMemorySaver(), schema=_AppendingMessagesState),
        store=InMemoryResponseProvider(),
    )
    with TestClient(server.app) as client:
        first = _create(
            client, stream=False, input="first", instructions="request-only"
        )
        get_state = server.graph.aget_state
        inspected = False

        async def inspect(config: RunnableConfig, **kwargs: Any) -> Any:
            nonlocal inspected
            if not inspected or not lookup_recovers:
                inspected = True
                raise RuntimeError("checkpoint inspection unavailable")
            return await get_state(config, **kwargs)

        monkeypatch.setattr(server.graph, "aget_state", inspect)
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
    if isinstance(value, BaseMessage):
        return value.content == "request-only"
    if isinstance(value, dict):
        return any(_contains_instruction(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_instruction(item) for item in value)
    if isinstance(value, Send):
        return _contains_instruction(value.arg)
    if isinstance(value, Interrupt):
        return _contains_instruction(value.value)
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
    linkage = {"conversation": {"id": "conv-instructions"}} if conversation else {}
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


@pytest.mark.parametrize("stream", [False, True])
def test_dataclass_send_resume_does_not_inherit_request_instructions(
    stream: bool,
) -> None:
    capture = _CaptureMessages()
    model = FakeListChatModel(responses=["answer"], callbacks=[capture])
    dispatched: list[_DataclassTaskInput] = []

    def dispatch(state: MessagesState) -> Command:
        payload = _DataclassTaskInput(state["messages"])
        dispatched.append(payload)
        return Command(goto=Send("answer", payload))

    async def answer(state: _DataclassTaskInput) -> dict[str, Any]:
        interrupt({"question": "continue?"})
        return {"messages": [await model.ainvoke(state.messages)]}

    saver = InMemorySaver()
    builder = StateGraph(MessagesState)
    builder.add_node("dispatch", dispatch)
    builder.add_node("answer", answer, input_schema=_DataclassTaskInput)
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
        )
    assert [message.content for message in capture.calls[0]] == ["first"]
    assert [message.content for message in dispatched[0].messages] == [
        "request-only",
        "first",
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
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
@pytest.mark.parametrize("schema", [MessagesState, _AppendingMessagesState])
def test_interrupt_resume_uses_only_the_new_response_instructions(
    instructions: str | None, schema: type
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
    server = ResponsesHostServer(
        builder.compile(checkpointer=saver), store=InMemoryResponseProvider()
    )
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


@pytest.mark.parametrize("schema", [MessagesState, _AppendingMessagesState])
def test_recovery_reapplies_instructions_without_replaying_input(schema: type) -> None:
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
    server = ResponsesHostServer(builder.compile(checkpointer=saver), store=provider)
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
    assert [message.content for message in capture.calls[1]] == [
        "request-only",
        "first",
        "answer",
    ]
    for checkpoint in saver.list(None):
        assert not _contains_instruction(checkpoint.checkpoint)
        assert not _contains_instruction(checkpoint.pending_writes)
