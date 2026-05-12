# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Responses-API human-in-the-loop converter and host wiring."""

from __future__ import annotations

import json
from typing import Annotated, Any, ClassVar

import pytest
from azure.ai.agentserver.responses.models import FunctionCallOutputItemParam
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Interrupt, interrupt
from pydantic import BaseModel
from starlette.testclient import TestClient
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import AzureAIResponsesAgentHost
from langchain_azure_ai.agents.hosting._converters import (
    HITL_FUNCTION_NAME,
    interrupt_arguments_json,
    parse_resume_command,
)


# ---------------------------------------------------------------------------
# _hitl.parse_resume_command
# ---------------------------------------------------------------------------


def _pending(*, id: str = "int-1", value: Any = "Q?") -> Interrupt:
    return Interrupt(value=value, id=id)


def test_parse_resume_command_returns_none_when_no_pending() -> None:
    items = [FunctionCallOutputItemParam(call_id="int-1", output='{"resume": "x"}')]
    command, consumed = parse_resume_command(items, ())
    assert command is None
    assert consumed == frozenset()


def test_parse_resume_command_returns_none_when_no_matching_item() -> None:
    items = [FunctionCallOutputItemParam(call_id="other", output='{"resume": "x"}')]
    command, consumed = parse_resume_command(items, (_pending(id="int-1"),))
    assert command is None
    assert consumed == frozenset()


def test_parse_resume_command_decodes_json_envelope() -> None:
    items = [
        FunctionCallOutputItemParam(
            call_id="int-1",
            output='{"resume": "Seattle"}',
        )
    ]
    command, consumed = parse_resume_command(items, (_pending(id="int-1"),))
    assert isinstance(command, Command)
    assert command.resume == "Seattle"
    assert consumed == frozenset({"int-1"})


def test_parse_resume_command_supports_update_and_goto() -> None:
    body = json.dumps({"update": {"k": 1}, "goto": "next"})
    items = [FunctionCallOutputItemParam(call_id="int-1", output=body)]
    command, _ = parse_resume_command(items, (_pending(id="int-1"),))
    assert command is not None
    assert command.update == {"k": 1}
    assert command.goto == "next"


def test_parse_resume_command_treats_plain_string_as_resume() -> None:
    items = [FunctionCallOutputItemParam(call_id="int-1", output="Seattle")]
    command, _ = parse_resume_command(items, (_pending(id="int-1"),))
    assert command is not None
    assert command.resume == "Seattle"


def test_parse_resume_command_treats_unrelated_json_as_resume() -> None:
    items = [FunctionCallOutputItemParam(call_id="int-1", output='{"x": 1}')]
    command, _ = parse_resume_command(items, (_pending(id="int-1"),))
    assert command is not None
    # No resume/update/goto keys → keep raw string as resume.
    assert command.resume == '{"x": 1}'


def test_parse_resume_command_ignores_blank_output() -> None:
    items = [FunctionCallOutputItemParam(call_id="int-1", output="   ")]
    command, consumed = parse_resume_command(items, (_pending(id="int-1"),))
    assert command is None
    assert consumed == frozenset()


# ---------------------------------------------------------------------------
# _hitl.interrupt_arguments_json
# ---------------------------------------------------------------------------


def test_interrupt_arguments_json_passes_strings_through() -> None:
    assert interrupt_arguments_json(_pending(value="Where?")) == "Where?"


def test_interrupt_arguments_json_serializes_objects() -> None:
    out = interrupt_arguments_json(_pending(value={"question": "Where?"}))
    assert json.loads(out) == {"question": "Where?"}


def test_interrupt_arguments_json_falls_back_for_non_serializable() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    out = interrupt_arguments_json(_pending(value=Opaque()))
    assert json.loads(out) == "opaque-value"


# ---------------------------------------------------------------------------
# End-to-end pause + resume through the responses host
# ---------------------------------------------------------------------------


class _MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _AskHuman(BaseModel):
    question: str


@tool
def _get_weather(location: str) -> str:
    """Fake weather tool."""
    return f"It's sunny in {location}."


class _ScriptedModel:
    """Tiny chat model that yields preset assistant messages on each call.

    The graph calls ``model.invoke(state["messages"])`` once per ``agent``
    node visit. We hand back successive scripted ``AIMessage`` payloads so
    the test fully controls when the graph decides to ``AskHuman`` and
    when it produces the final answer.
    """

    _scripted: ClassVar[dict[str, list[AIMessage]]] = {}

    def __init__(self, key: str) -> None:
        self._key = key

    def invoke(self, _messages: list[BaseMessage]) -> AIMessage:
        queue = self._scripted[self._key]
        if not queue:
            raise AssertionError("scripted model exhausted")
        return queue.pop(0)


def _build_hitl_graph(key: str) -> Any:
    model = _ScriptedModel(key)
    tools = [_get_weather]
    tool_node = ToolNode(tools)

    def call_model(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    def ask_human(state: _MessagesState) -> dict[str, Any]:
        last = state["messages"][-1]
        tool_call = last.tool_calls[0]  # type: ignore[attr-defined]
        question = _AskHuman.model_validate(tool_call["args"]).question
        answer = interrupt(question)
        return {
            "messages": [
                ToolMessage(content=str(answer), tool_call_id=tool_call["id"])
            ]
        }

    def should_continue(state: _MessagesState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None)
        if not calls:
            return END
        if calls[0]["name"] == "AskHuman":
            return "ask_human"
        return "action"

    builder = StateGraph(_MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("action", tool_node)
    builder.add_node("ask_human", ask_human)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", should_continue, path_map=["ask_human", "action", END]
    )
    builder.add_edge("action", "agent")
    builder.add_edge("ask_human", "agent")
    return builder.compile(checkpointer=InMemorySaver())


def _client(host: AzureAIResponsesAgentHost) -> TestClient:
    return TestClient(host.app)


def test_responses_host_emits_interrupt_function_call_and_resumes() -> None:
    key = "hitl-test"
    _ScriptedModel._scripted[key] = [
        # Turn 1: model decides to ask the user for their location.
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_ask_1",
                    "name": "AskHuman",
                    "args": {"question": "Where are you?"},
                }
            ],
        ),
        # Turn 2 (after resume): model produces the final answer.
        AIMessage(content="It's sunny in Seattle."),
    ]
    try:
        graph = _build_hitl_graph(key)
        host = AzureAIResponsesAgentHost(graph)
        conversation_id = "conv-hitl-1"

        with _client(host) as client:
            # 1. Initial turn — graph should pause and the response should
            #    expose a __hosted_agent_adapter_interrupt__ function_call item.
            first = client.post(
                "/responses",
                json={
                    "input": "Look up the weather where I am.",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            first_payload = first.json()
            assert first_payload["status"] == "completed"
            interrupts = [
                item
                for item in first_payload["output"]
                if item.get("type") == "function_call"
                and item.get("name") == HITL_FUNCTION_NAME
            ]
            assert len(interrupts) == 1, first_payload
            interrupt_item = interrupts[0]
            assert interrupt_item["arguments"] == "Where are you?"
            call_id = interrupt_item["call_id"]
            assert call_id  # LangGraph interrupt id

            # 2. Resume turn — submit a function_call_output keyed by the
            #    interrupt id. The host should resume the graph and return
            #    the assistant's final message.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"resume": "Seattle"}),
                        }
                    ],
                },
            )
            assert second.status_code == 200, second.text
            second_payload = second.json()
            assert second_payload["status"] == "completed"
            # No new pending interrupt this time.
            assert not [
                it
                for it in second_payload["output"]
                if it.get("type") == "function_call"
                and it.get("name") == HITL_FUNCTION_NAME
            ]
            # And we should see the final assistant message text.
            text = "".join(
                part.get("text", "")
                for item in second_payload["output"]
                if item.get("type") == "message"
                for part in item.get("content", [])
            )
            assert "Seattle" in text
    finally:
        _ScriptedModel._scripted.pop(key, None)


def test_responses_host_falls_back_when_resume_call_id_mismatches() -> None:
    """A function_call_output with an unknown call_id should be treated as
    a normal input (not a resume) and not crash the host."""
    key = "hitl-fallback"
    _ScriptedModel._scripted[key] = [
        AIMessage(content="ack"),
    ]
    try:
        host = AzureAIResponsesAgentHost(_build_hitl_graph(key))
        with _client(host) as client:
            resp = client.post(
                "/responses",
                json={
                    "conversation": {"id": "conv-fallback"},
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "no-such-interrupt",
                            "output": '{"resume": "x"}',
                        }
                    ],
                },
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "completed"
    finally:
        _ScriptedModel._scripted.pop(key, None)


@pytest.mark.parametrize("stream", [False, True])
def test_responses_host_interrupt_works_in_both_modes(stream: bool) -> None:
    key = f"hitl-mode-{stream}"
    _ScriptedModel._scripted[key] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_ask_2",
                    "name": "AskHuman",
                    "args": {"question": "Which city?"},
                }
            ],
        ),
    ]
    try:
        host = AzureAIResponsesAgentHost(_build_hitl_graph(key))
        with _client(host) as client:
            resp = client.post(
                "/responses",
                json={
                    "input": "ask me a city",
                    "conversation": {"id": f"conv-mode-{stream}"},
                    "stream": stream,
                },
            )
        assert resp.status_code == 200, resp.text
        body = resp.text
        # In both modes the interrupt name must appear somewhere in the
        # response payload (output item name for non-streaming, or as part
        # of an SSE event payload for streaming).
        assert HITL_FUNCTION_NAME in body
    finally:
        _ScriptedModel._scripted.pop(key, None)
