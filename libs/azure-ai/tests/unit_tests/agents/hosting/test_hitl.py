# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Responses-API human-in-the-loop converter and host wiring."""

from __future__ import annotations

import json
import sys
from typing import Annotated, Any, ClassVar, Optional
from unittest.mock import MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.responses import ResponseEventStream  # noqa: E402
from azure.ai.agentserver.responses.models import (  # noqa: E402
    FunctionCallOutputItemParam,
    MCPApprovalResponse,
)
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import Command, Interrupt, interrupt  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from typing_extensions import TypedDict  # noqa: E402

from langchain_azure_ai.agents.hosting import ResponsesHostServer  # noqa: E402
from langchain_azure_ai.agents.hosting._converters import (  # noqa: E402
    HITL_FUNCTION_NAME,
    HITL_MCP_SERVER_LABEL,
    detect_approval_rejection,
    emit_interrupts,
    interrupt_arguments_json,
    parse_resume_command,
)

_REAL_INTERRUPT_ASYNC_XFAIL = pytest.mark.xfail(
    sys.version_info < (3, 11),
    reason=(
        "LangGraph interrupt() loses runnable config in async graph execution "
        "on Python < 3.11."
    ),
    strict=True,
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


def test_parse_resume_command_preserves_false_resume() -> None:
    # ``{"resume": false}`` is the reject half of the approve/reject
    # pattern. A falsy-but-present resume must survive decoding, or the
    # node would never take its "cancel" branch.
    items = [FunctionCallOutputItemParam(call_id="int-1", output='{"resume": false}')]
    command, consumed = parse_resume_command(items, (_pending(id="int-1"),))
    assert command is not None
    assert command.resume is False
    assert consumed == frozenset({"int-1"})


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


def test_interrupt_arguments_json_emits_envelope_for_strings() -> None:
    out = interrupt_arguments_json(_pending(id="int-1", value="Where?"))
    assert json.loads(out) == {"interrupt_id": "int-1", "value": "Where?"}


def test_interrupt_arguments_json_serializes_objects() -> None:
    out = interrupt_arguments_json(_pending(id="int-1", value={"question": "Where?"}))
    assert json.loads(out) == {
        "interrupt_id": "int-1",
        "value": {"question": "Where?"},
    }


def test_interrupt_arguments_json_falls_back_for_non_serializable() -> None:
    class Opaque:
        def __str__(self) -> str:
            return "opaque-value"

    out = interrupt_arguments_json(_pending(id="int-1", value=Opaque()))
    assert json.loads(out) == {"interrupt_id": "int-1", "value": "opaque-value"}


# ---------------------------------------------------------------------------
# _hitl.parse_resume_command — mcp_approval_response paths
# ---------------------------------------------------------------------------


def test_parse_resume_command_approve_true_echoes_interrupt_value() -> None:
    pending = _pending(id="int-1", value={"question": "Where?"})
    items = [MCPApprovalResponse(approval_request_id="int-1", approve=True)]
    command, consumed = parse_resume_command(items, (pending,))
    assert command is not None
    # approve=True echoes the original interrupt value back as the
    # resume payload (matches Agent Framework's behavior).
    assert command.resume == {"question": "Where?"}
    assert consumed == frozenset({"int-1"})


def test_parse_resume_command_approve_false_yields_no_command() -> None:
    # Rejection is surfaced via ``detect_approval_rejection``, not here.
    pending = _pending(id="int-1")
    items = [MCPApprovalResponse(approval_request_id="int-1", approve=False)]
    command, consumed = parse_resume_command(items, (pending,))
    assert command is None
    assert consumed == frozenset()


def test_parse_resume_command_function_call_output_wins_over_approval() -> None:
    pending = _pending(id="int-1", value="original")
    items = [
        FunctionCallOutputItemParam(call_id="int-1", output="Seattle"),
        MCPApprovalResponse(approval_request_id="int-1", approve=True),
    ]
    command, consumed = parse_resume_command(items, (pending,))
    assert command is not None
    # function_call_output (richer payload) wins over the approval echo.
    assert command.resume == "Seattle"
    assert consumed == frozenset({"int-1"})


def test_parse_resume_command_approval_for_unknown_id_is_ignored() -> None:
    pending = _pending(id="int-1")
    items = [MCPApprovalResponse(approval_request_id="other", approve=True)]
    command, consumed = parse_resume_command(items, (pending,))
    assert command is None
    assert consumed == frozenset()


# ---------------------------------------------------------------------------
# _hitl.parse_resume_command — parallel interrupts
#
# LangGraph refuses a bare resume value when more than one interrupt is
# pending: "When there are multiple pending interrupts, you must specify
# the interrupt id when resuming." So the converter must fold every
# matched item into an id-keyed resume map.
# https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts
# ---------------------------------------------------------------------------


def test_parse_resume_command_builds_resume_map_for_parallel_interrupts() -> None:
    pending = (_pending(id="int-a", value="a?"), _pending(id="int-b", value="b?"))
    items = [
        FunctionCallOutputItemParam(call_id="int-a", output='{"resume": "A"}'),
        FunctionCallOutputItemParam(call_id="int-b", output='{"resume": "B"}'),
    ]
    command, consumed = parse_resume_command(items, pending)
    assert command is not None
    assert command.resume == {"int-a": "A", "int-b": "B"}
    assert consumed == frozenset({"int-a", "int-b"})


def test_parse_resume_command_map_allows_partial_answers() -> None:
    # Answering only one of two parallel pauses is legal; LangGraph keeps
    # the unanswered branch suspended.
    pending = (_pending(id="int-a"), _pending(id="int-b"))
    items = [FunctionCallOutputItemParam(call_id="int-a", output="A")]
    command, consumed = parse_resume_command(items, pending)
    assert command is not None
    assert command.resume == {"int-a": "A"}
    assert consumed == frozenset({"int-a"})


def test_parse_resume_command_map_mixes_both_resume_channels() -> None:
    pending = (_pending(id="int-a"), _pending(id="int-b", value="echo-me"))
    items = [
        FunctionCallOutputItemParam(call_id="int-a", output="A"),
        MCPApprovalResponse(approval_request_id="int-b", approve=True),
    ]
    command, consumed = parse_resume_command(items, pending)
    assert command is not None
    assert command.resume == {"int-a": "A", "int-b": "echo-me"}
    assert consumed == frozenset({"int-a", "int-b"})


def test_parse_resume_command_map_skips_rejected_approvals() -> None:
    pending = (_pending(id="int-a"), _pending(id="int-b"))
    items = [
        FunctionCallOutputItemParam(call_id="int-a", output="A"),
        MCPApprovalResponse(approval_request_id="int-b", approve=False),
    ]
    command, _ = parse_resume_command(items, pending)
    assert command is not None
    assert command.resume == {"int-a": "A"}


# ---------------------------------------------------------------------------
# _hitl — mcp_approval_request id encoding round-trip
#
# ``emit_interrupts`` cannot reuse the raw LangGraph interrupt id as the
# ``mcp_approval_request.id`` (storage requires an ``mcpr_*`` shape), so it
# encodes the interrupt id into the generated id. The inbound path must be
# able to recover it.
# ---------------------------------------------------------------------------


async def _emitted_items(interrupts: Any) -> list[Any]:
    stream = ResponseEventStream(response_id="resp-emit", request=MagicMock())
    # The stream's state machine requires the lifecycle prologue before any
    # output item may be emitted.
    stream.emit_created()
    stream.emit_in_progress()
    events = [event async for event in emit_interrupts(interrupts, stream)]
    assert events  # emission must not be silent
    return list(stream.response.output or [])


async def test_emit_interrupts_emits_paired_channels_per_interrupt() -> None:
    items = await _emitted_items(
        (_pending(id="int-a", value="a?"), _pending(id="int-b", value="b?"))
    )
    function_calls = [it for it in items if it.type == "function_call"]
    approvals = [it for it in items if it.type == "mcp_approval_request"]

    assert [it.call_id for it in function_calls] == ["int-a", "int-b"]
    assert all(it.name == HITL_FUNCTION_NAME for it in function_calls)
    assert len(approvals) == 2
    assert all(it.id.startswith("mcpr_") for it in approvals)
    assert all(it.server_label == HITL_MCP_SERVER_LABEL for it in approvals)
    # The two channels carry the same envelope so a client may pick either.
    assert [it.arguments for it in approvals] == [it.arguments for it in function_calls]
    # And each approval id must be distinct so partial answers stay routable.
    assert approvals[0].id != approvals[1].id


async def test_parse_resume_command_accepts_emitted_approval_id() -> None:
    pending = _pending(id="int-1", value="echo-me")
    items = await _emitted_items((pending,))
    approval_id = next(it.id for it in items if it.type == "mcp_approval_request")
    assert approval_id != "int-1"  # encoded, not the raw interrupt id

    command, consumed = parse_resume_command(
        [MCPApprovalResponse(approval_request_id=approval_id, approve=True)],
        (pending,),
    )
    assert command is not None
    assert command.resume == "echo-me"
    assert consumed == frozenset({approval_id})


async def test_detect_approval_rejection_accepts_emitted_approval_id() -> None:
    pending = _pending(id="int-1")
    items = await _emitted_items((pending,))
    approval_id = next(it.id for it in items if it.type == "mcp_approval_request")

    message = detect_approval_rejection(
        [MCPApprovalResponse(approval_request_id=approval_id, approve=False)],
        (pending,),
    )
    assert message is not None
    assert approval_id in message


# ---------------------------------------------------------------------------
# _hitl.detect_approval_rejection
# ---------------------------------------------------------------------------


def test_detect_approval_rejection_returns_message_when_approve_false() -> None:
    pending = _pending(id="int-1")
    items = [
        MCPApprovalResponse(
            approval_request_id="int-1", approve=False, reason="too risky"
        )
    ]
    msg = detect_approval_rejection(items, (pending,))
    assert msg is not None
    assert "int-1" in msg
    assert "too risky" in msg


def test_detect_approval_rejection_returns_none_when_approve_true() -> None:
    pending = _pending(id="int-1")
    items = [MCPApprovalResponse(approval_request_id="int-1", approve=True)]
    assert detect_approval_rejection(items, (pending,)) is None


def test_detect_approval_rejection_returns_none_when_id_mismatches() -> None:
    pending = _pending(id="int-1")
    items = [MCPApprovalResponse(approval_request_id="other", approve=False)]
    assert detect_approval_rejection(items, (pending,)) is None


def test_detect_approval_rejection_returns_none_when_no_pending() -> None:
    items = [MCPApprovalResponse(approval_request_id="int-1", approve=False)]
    assert detect_approval_rejection(items, ()) is None


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
            "messages": [ToolMessage(content=str(answer), tool_call_id=tool_call["id"])]
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


def _client(host: ResponsesHostServer) -> TestClient:
    return TestClient(host.app)


@_REAL_INTERRUPT_ASYNC_XFAIL
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
        host = ResponsesHostServer(graph)
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
            envelope = json.loads(interrupt_item["arguments"])
            assert envelope["value"] == "Where are you?"
            call_id = interrupt_item["call_id"]
            assert call_id  # LangGraph interrupt id
            assert envelope["interrupt_id"] == call_id

            # The host should ALSO have emitted a paired mcp_approval_request
            # item with a storage-compatible id and the same arguments envelope.
            approvals = [
                item
                for item in first_payload["output"]
                if item.get("type") == "mcp_approval_request"
                and item.get("name") == HITL_FUNCTION_NAME
            ]
            assert len(approvals) == 1, first_payload
            assert approvals[0]["id"].startswith("mcpr_")
            assert approvals[0]["server_label"] == HITL_MCP_SERVER_LABEL
            assert approvals[0]["arguments"] == interrupt_item["arguments"]
            assert json.loads(approvals[0]["arguments"])["interrupt_id"] == call_id

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
        host = ResponsesHostServer(_build_hitl_graph(key))
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


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_reemits_interrupt_when_resume_call_id_mismatches() -> None:
    """Pending interrupt + wrong-call_id resume → host re-emits the
    sentinel instead of driving the graph with a malformed message list.

    This is the recovery path for a client that echoed the wrong
    function_call's call_id on resume (e.g. the LLM's ``AskHuman`` id
    instead of the interrupt sentinel id).
    """
    key = "hitl-bad-resume"
    _ScriptedModel._scripted[key] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_ask_bad",
                    "name": "AskHuman",
                    "args": {"question": "Where?"},
                }
            ],
        ),
        # Second AIMessage should NEVER be consumed — the host must not
        # drive the graph on the bad-resume turn.
        AIMessage(content="should not be reached"),
    ]
    try:
        host = ResponsesHostServer(_build_hitl_graph(key))
        conversation_id = "conv-bad-resume"
        with _client(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "ask me",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            interrupt_items = [
                it
                for it in first.json()["output"]
                if it.get("type") == "function_call"
                and it.get("name") == HITL_FUNCTION_NAME
            ]
            assert len(interrupt_items) == 1
            sentinel_call_id = interrupt_items[0]["call_id"]

            # Client mistakenly echoes the AskHuman call_id instead of
            # the sentinel.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_ask_bad",
                            "output": '{"resume": "Seattle"}',
                        }
                    ],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed"
            # Host re-emits the SAME pending sentinel (both channels) so
            # the client can retry with the correct call_id.
            sentinels = [
                it
                for it in payload["output"]
                if it.get("type") == "function_call"
                and it.get("name") == HITL_FUNCTION_NAME
            ]
            assert len(sentinels) == 1
            assert sentinels[0]["call_id"] == sentinel_call_id
            approvals = [
                it
                for it in payload["output"]
                if it.get("type") == "mcp_approval_request"
                and it.get("name") == HITL_FUNCTION_NAME
            ]
            assert len(approvals) == 1
            assert approvals[0]["id"].startswith("mcpr_")
            assert (
                json.loads(approvals[0]["arguments"])["interrupt_id"]
                == sentinel_call_id
            )
            # And no spurious assistant message from a second LLM call.
            assert not [it for it in payload["output"] if it.get("type") == "message"]
        # The second scripted AIMessage must remain un-consumed because
        # the graph was not driven on the bad-resume turn.
        assert len(_ScriptedModel._scripted[key]) == 1
    finally:
        _ScriptedModel._scripted.pop(key, None)


@pytest.mark.parametrize("stream", [False, True])
@_REAL_INTERRUPT_ASYNC_XFAIL
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
        host = ResponsesHostServer(_build_hitl_graph(key))
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


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_resumes_via_mcp_approval_response_approve() -> None:
    """Client resumes a paused graph via ``mcp_approval_response{approve:true}``;
    the host should drive the graph with ``Command(resume=interrupt.value)``
    (echoing the original interrupt value back, per design)."""
    key = "hitl-approve"
    _ScriptedModel._scripted[key] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_ask_approve",
                    "name": "AskHuman",
                    "args": {"question": "Confirm: run weather lookup?"},
                }
            ],
        ),
        AIMessage(content="OK, lookup completed."),
    ]
    try:
        host = ResponsesHostServer(_build_hitl_graph(key))
        conversation_id = "conv-approve"
        with _client(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "do the thing",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            approvals = [
                it
                for it in first.json()["output"]
                if it.get("type") == "mcp_approval_request"
                and it.get("name") == HITL_FUNCTION_NAME
            ]
            assert len(approvals) == 1
            approval_id = approvals[0]["id"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approval_id,
                            "approve": True,
                        }
                    ],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed"
            # No new pending interrupt this time.
            assert not [
                it
                for it in payload["output"]
                if it.get("type") in ("function_call", "mcp_approval_request")
                and it.get("name") == HITL_FUNCTION_NAME
            ]
            text = "".join(
                part.get("text", "")
                for item in payload["output"]
                if item.get("type") == "message"
                for part in item.get("content", [])
            )
            assert "lookup completed" in text
    finally:
        _ScriptedModel._scripted.pop(key, None)


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_rejects_via_mcp_approval_response() -> None:
    """``mcp_approval_response{approve:false}`` short-circuits the turn into
    ``response.failed(code='interrupt_rejected', …)``; the graph is NOT
    driven on the rejection turn."""
    key = "hitl-reject"
    _ScriptedModel._scripted[key] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_ask_reject",
                    "name": "AskHuman",
                    "args": {"question": "Confirm: irreversible action?"},
                }
            ],
        ),
        # MUST NOT be consumed — the host should not drive the graph on
        # the rejection turn.
        AIMessage(content="should not be reached"),
    ]
    try:
        host = ResponsesHostServer(_build_hitl_graph(key))
        conversation_id = "conv-reject"
        with _client(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "do something risky",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            approvals = [
                it
                for it in first.json()["output"]
                if it.get("type") == "mcp_approval_request"
            ]
            assert len(approvals) == 1
            approval_id = approvals[0]["id"]

            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        {
                            "type": "mcp_approval_response",
                            "approval_request_id": approval_id,
                            "approve": False,
                            "reason": "user said no",
                        }
                    ],
                },
            )
            # The agentserver Responses lifecycle still returns 200 with
            # a ``failed`` status payload (mirrors how other failures are
            # surfaced).
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "failed", payload
            err = payload.get("error") or {}
            assert err.get("code") == "interrupt_rejected", payload
            assert approval_id in (err.get("message") or "")
            assert "user said no" in (err.get("message") or "")
        # The second scripted AIMessage must remain un-consumed because
        # the graph was not driven on the rejection turn.
        assert len(_ScriptedModel._scripted[key]) == 1
    finally:
        _ScriptedModel._scripted.pop(key, None)


# ---------------------------------------------------------------------------
# Interrupt rules and patterns from
# https://docs.langchain.com/oss/python/langgraph/interrupts
#
# The graphs below are the user-authored side of the contract: whatever a
# customer writes with ``langgraph.types.interrupt``, the host must surface
# and resume faithfully. Each test names the rule it pins.
# ---------------------------------------------------------------------------


def _sentinels(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``function_call`` HITL sentinels in a response payload."""
    return [
        item
        for item in payload["output"]
        if item.get("type") == "function_call"
        and item.get("name") == HITL_FUNCTION_NAME
    ]


def _approval_requests(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``mcp_approval_request`` HITL sentinels in a payload."""
    return [
        item
        for item in payload["output"]
        if item.get("type") == "mcp_approval_request"
        and item.get("name") == HITL_FUNCTION_NAME
    ]


def _assistant_text(payload: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in payload["output"]
        if item.get("type") == "message"
        for part in item.get("content", [])
    )


def _interrupt_value(item: dict[str, Any]) -> Any:
    return json.loads(item["arguments"])["value"]


def _resume(call_id: str, value: Any) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps({"resume": value}),
    }


# --- Handling multiple interrupts ------------------------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts


def _build_parallel_interrupt_graph() -> Any:
    """Fan out to two nodes that each pause in the same superstep."""

    def ask_a(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"a={interrupt('question_a')}")]}

    def ask_b(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"b={interrupt('question_b')}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("a", ask_a)
    builder.add_node("b", ask_b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_emits_and_resumes_parallel_interrupts() -> None:
    """Two branches pause at once → one sentinel pair each, resumable together.

    LangGraph rejects a bare resume value while several interrupts are
    pending ("you must specify the interrupt id when resuming"), so the
    host must fold both answers into an id-keyed resume map.
    """
    host = ResponsesHostServer(_build_parallel_interrupt_graph())
    conversation_id = "conv-parallel"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "go", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        first_payload = first.json()
        sentinels = _sentinels(first_payload)
        assert len(sentinels) == 2, first_payload
        call_ids = {_interrupt_value(it): it["call_id"] for it in sentinels}
        assert set(call_ids) == {"question_a", "question_b"}
        # Distinct LangGraph ids — otherwise answers could not be routed.
        assert len(set(call_ids.values())) == 2

        approvals = _approval_requests(first_payload)
        assert len(approvals) == 2, first_payload
        assert len({it["id"] for it in approvals}) == 2

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [
                    _resume(call_ids["question_a"], "A"),
                    _resume(call_ids["question_b"], "B"),
                ],
            },
        )
        assert second.status_code == 200, second.text
        payload = second.json()
        assert payload["status"] == "completed", payload
        assert not _sentinels(payload), payload
        text = _assistant_text(payload)
        assert "a=A" in text and "b=B" in text, payload


# --- Interrupts in tools ----------------------------------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#interrupts-in-tools


@tool
def _send_email(to: str, subject: str) -> str:
    """Send an email to a recipient."""
    response = interrupt({"action": "send_email", "to": to, "subject": subject})
    if isinstance(response, dict) and response.get("action") == "approve":
        return f"Email sent to {response.get('to', to)}"
    return "Email cancelled by user"


def _build_tool_interrupt_graph(key: str) -> Any:
    """Agent whose tool pauses for approval from inside ``ToolNode``."""
    model = _ScriptedModel(key)

    def call_model(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [model.invoke(state["messages"])]}

    def should_continue(state: _MessagesState) -> str:
        return "action" if getattr(state["messages"][-1], "tool_calls", None) else END

    builder = StateGraph(_MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("action", ToolNode([_send_email]))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, path_map=["action", END])
    builder.add_edge("action", "agent")
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_surfaces_interrupt_raised_inside_a_tool() -> None:
    """``interrupt()`` inside a ``@tool`` must reach the client, and the
    resume payload must be handed back to the tool so it can act on the
    (possibly edited) arguments."""
    key = "hitl-tool"
    _ScriptedModel._scripted[key] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "call_send_1",
                    "name": "_send_email",
                    "args": {"to": "alice@example.com", "subject": "Meeting"},
                }
            ],
        ),
        AIMessage(content="All done."),
    ]
    try:
        host = ResponsesHostServer(_build_tool_interrupt_graph(key))
        conversation_id = "conv-tool-interrupt"
        with _client(host) as client:
            first = client.post(
                "/responses",
                json={
                    "input": "email alice about the meeting",
                    "conversation": {"id": conversation_id},
                },
            )
            assert first.status_code == 200, first.text
            sentinels = _sentinels(first.json())
            assert len(sentinels) == 1, first.json()
            # Structured payloads survive the envelope round-trip.
            assert _interrupt_value(sentinels[0]) == {
                "action": "send_email",
                "to": "alice@example.com",
                "subject": "Meeting",
            }

            # Approve, editing the recipient on the way through.
            second = client.post(
                "/responses",
                json={
                    "conversation": {"id": conversation_id},
                    "input": [
                        _resume(
                            sentinels[0]["call_id"],
                            {"action": "approve", "to": "ops@example.com"},
                        )
                    ],
                },
            )
            assert second.status_code == 200, second.text
            payload = second.json()
            assert payload["status"] == "completed", payload
            assert not _sentinels(payload), payload
            tool_outputs = [
                item["output"]
                for item in payload["output"]
                if item.get("type") == "function_call_output"
            ]
            assert any(
                "Email sent to ops@example.com" in str(out) for out in tool_outputs
            ), payload
    finally:
        _ScriptedModel._scripted.pop(key, None)


# --- Validating human input (re-prompt loop) --------------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#validating-human-input


class _FormState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    age: Optional[int]
    pending_question: Optional[str]


def _build_reprompt_graph() -> Any:
    """Collect an age, re-prompting through a conditional edge until valid.

    ``interrupt()`` is called exactly once per node invocation — the
    documented alternative to a ``while True`` loop inside the node.
    """

    def collect_age(state: _FormState) -> dict[str, Any]:
        question = state.get("pending_question") or "What is your age?"
        answer = interrupt(question)
        if isinstance(answer, int) and answer > 0:
            return {
                "age": answer,
                "pending_question": None,
                "messages": [AIMessage(content=f"age={answer}")],
            }
        return {"pending_question": f"'{answer}' is not a valid age."}

    def route(state: _FormState) -> str:
        return END if state.get("age") is not None else "collect_age"

    builder = StateGraph(_FormState)
    builder.add_node("collect_age", collect_age)
    builder.add_edge(START, "collect_age")
    builder.add_conditional_edges("collect_age", route, path_map=["collect_age", END])
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_reemits_a_new_sentinel_when_the_graph_pauses_again() -> None:
    """A resume turn that itself pauses must emit a *fresh* sentinel.

    Without this the client would have no id to answer the re-prompt with,
    and the conversation would dead-end after one invalid answer.
    """
    host = ResponsesHostServer(_build_reprompt_graph())
    conversation_id = "conv-reprompt"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "start", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        first_sentinels = _sentinels(first.json())
        assert len(first_sentinels) == 1, first.json()
        assert _interrupt_value(first_sentinels[0]) == "What is your age?"

        # Invalid answer → the node loops back and pauses again.
        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(first_sentinels[0]["call_id"], "thirty")],
            },
        )
        assert second.status_code == 200, second.text
        second_payload = second.json()
        assert second_payload["status"] == "completed", second_payload
        second_sentinels = _sentinels(second_payload)
        assert len(second_sentinels) == 1, second_payload
        assert "not a valid age" in _interrupt_value(second_sentinels[0])
        # A new pause is a new LangGraph interrupt, so a new call_id.
        assert second_sentinels[0]["call_id"] != first_sentinels[0]["call_id"], (
            second_payload
        )

        # Valid answer → the graph finishes.
        third = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(second_sentinels[0]["call_id"], 30)],
            },
        )
        assert third.status_code == 200, third.text
        third_payload = third.json()
        assert third_payload["status"] == "completed", third_payload
        assert not _sentinels(third_payload), third_payload
        assert "age=30" in _assistant_text(third_payload), third_payload


# --- Do not wrap interrupt calls in try/except ------------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-wrap-interrupt-calls-in-try%2Fexcept


class _ToolFailure(Exception):
    """Domain error a node might legitimately want to catch."""


def _build_swallowed_interrupt_graph() -> Any:
    """The documented anti-pattern: a bare ``except`` eats ``GraphInterrupt``."""

    def ask(state: _MessagesState) -> dict[str, Any]:
        try:
            answer = interrupt("What's your name?")
        except Exception as exc:  # noqa: BLE001 - deliberately wrong
            return {"messages": [AIMessage(content=f"swallowed:{type(exc).__name__}")]}
        return {"messages": [AIMessage(content=f"ok:{answer}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


def _build_specific_except_graph() -> Any:
    """The documented fix: catch a specific type so the interrupt bubbles up."""

    def ask(state: _MessagesState) -> dict[str, Any]:
        try:
            answer = interrupt("What's your name?")
        except _ToolFailure:
            answer = "fallback"
        return {"messages": [AIMessage(content=f"ok:{answer}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_responses_host_emits_nothing_when_a_node_swallows_the_interrupt() -> None:
    """Regression guard for the most common HITL support question.

    A bare ``except Exception`` around ``interrupt()`` catches the
    ``GraphInterrupt`` LangGraph uses to suspend, so nothing is ever
    checkpointed and the host has no pause to surface. The turn completes
    normally with no sentinel — pinning this makes the failure mode
    diagnosable instead of looking like a host bug.
    """
    host = ResponsesHostServer(_build_swallowed_interrupt_graph())
    with _client(host) as client:
        resp = client.post(
            "/responses",
            json={"input": "hi", "conversation": {"id": "conv-swallow"}},
        )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "completed", payload
    assert not _sentinels(payload), payload
    assert not _approval_requests(payload), payload
    assert "swallowed:GraphInterrupt" in _assistant_text(payload), payload


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_surfaces_interrupt_past_a_narrow_except_clause() -> None:
    """The ✅ counterpart: a narrow ``except`` leaves the pause intact."""
    host = ResponsesHostServer(_build_specific_except_graph())
    conversation_id = "conv-narrow-except"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "hi", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        sentinels = _sentinels(first.json())
        assert len(sentinels) == 1, first.json()

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(sentinels[0]["call_id"], "Ada")],
            },
        )
        assert second.status_code == 200, second.text
        assert "ok:Ada" in _assistant_text(second.json()), second.text


# --- Side effects before interrupt must be idempotent -----------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#side-effects-called-before-interrupt-must-be-idempotent


def _build_side_effect_graph(trace: list[str]) -> Any:
    """Record what runs before and after the pause, to expose node replay."""

    def ask(state: _MessagesState) -> dict[str, Any]:
        trace.append("before")
        answer = interrupt("confirm?")
        trace.append("after")
        return {"messages": [AIMessage(content=f"done:{answer}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_resume_replays_the_node_from_its_start() -> None:
    """Resuming re-runs the whole node, not just the line after ``interrupt``.

    This is why the docs require pre-interrupt side effects to be
    idempotent. Hosting must not paper over it — a customer debugging
    duplicated writes needs the replay to be observable.
    """
    trace: list[str] = []
    host = ResponsesHostServer(_build_side_effect_graph(trace))
    conversation_id = "conv-replay"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "go", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        sentinels = _sentinels(first.json())
        assert len(sentinels) == 1, first.json()
        # Paused before the post-interrupt work ever ran.
        assert trace == ["before"]

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(sentinels[0]["call_id"], "yes")],
            },
        )
        assert second.status_code == 200, second.text
        assert "done:yes" in _assistant_text(second.json()), second.text
    assert trace == ["before", "before", "after"]


# --- Interrupts inside subgraphs called as functions ------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#using-with-subgraphs-called-as-functions


def _build_subgraph_interrupt_graph() -> Any:
    """Parent node invokes a subgraph that pauses."""

    def sub_node(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"sub:{interrupt('sub-question')}")]}

    sub_builder = StateGraph(_MessagesState)
    sub_builder.add_node("inner", sub_node)
    sub_builder.add_edge(START, "inner")
    sub_builder.add_edge("inner", END)
    subgraph = sub_builder.compile()

    def parent_node(state: _MessagesState) -> dict[str, Any]:
        result = subgraph.invoke({"messages": state["messages"]})
        return {"messages": result["messages"][-1:]}

    builder = StateGraph(_MessagesState)
    builder.add_node("parent", parent_node)
    builder.add_edge(START, "parent")
    builder.add_edge("parent", END)
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_surfaces_interrupt_raised_inside_a_subgraph() -> None:
    """An interrupt raised in a nested subgraph bubbles to the parent
    checkpoint, so the host surfaces and resumes it like any other pause."""
    host = ResponsesHostServer(_build_subgraph_interrupt_graph())
    conversation_id = "conv-subgraph"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "go", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        sentinels = _sentinels(first.json())
        assert len(sentinels) == 1, first.json()
        assert _interrupt_value(sentinels[0]) == "sub-question"

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(sentinels[0]["call_id"], "Ada")],
            },
        )
        assert second.status_code == 200, second.text
        payload = second.json()
        assert payload["status"] == "completed", payload
        assert not _sentinels(payload), payload
        assert "sub:Ada" in _assistant_text(payload), payload


# --- A checkpointer is required ---------------------------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#pause-using-interrupt


def test_responses_host_cannot_surface_an_interrupt_without_a_checkpointer() -> None:
    """No checkpointer → no persisted pause → nothing to surface or resume.

    Documents the first prerequisite in the interrupt docs: a HITL graph
    compiled without a checkpointer silently loses its pause, so the turn
    just completes.
    """

    def ask(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"ok:{interrupt('name?')}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)

    host = ResponsesHostServer(builder.compile())
    with _client(host) as client:
        resp = client.post(
            "/responses",
            json={"input": "hi", "conversation": {"id": "conv-no-checkpointer"}},
        )
    assert resp.status_code == 200, resp.text
    assert HITL_FUNCTION_NAME not in resp.text


def _build_simple_interrupt_graph() -> Any:
    """Minimal single-pause graph, reused by the transport-level tests."""

    def ask(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content=f"ok:{interrupt('name?')}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


# --- Do not reorder interrupt calls within a node ---------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-reorder-interrupt-calls-within-a-node


def _build_sequential_interrupt_graph() -> Any:
    """One node, two interrupts, always issued in the same order (✅)."""

    def ask(state: _MessagesState) -> dict[str, Any]:
        name = interrupt("name?")
        city = interrupt("city?")
        return {"messages": [AIMessage(content=f"{name}@{city}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_walks_sequential_interrupts_in_one_node() -> None:
    """Several ``interrupt()`` calls in one node surface one pause at a time.

    LangGraph stores resume values per *task* and matches them to
    ``interrupt()`` calls strictly by index, so both pauses share the
    same ``call_id``. That is a real constraint on clients: within a
    node, the id does not identify *which question* is being asked — only
    the ``value`` envelope does. Pinning it here stops anyone from
    "fixing" the host to mint a fresh id per pause, which would break
    resume matching.
    """
    host = ResponsesHostServer(_build_sequential_interrupt_graph())
    conversation_id = "conv-sequential"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "go", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        first_sentinels = _sentinels(first.json())
        assert len(first_sentinels) == 1, first.json()
        assert _interrupt_value(first_sentinels[0]) == "name?"
        call_id = first_sentinels[0]["call_id"]

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(call_id, "Ada")],
            },
        )
        assert second.status_code == 200, second.text
        second_payload = second.json()
        second_sentinels = _sentinels(second_payload)
        assert len(second_sentinels) == 1, second_payload
        assert _interrupt_value(second_sentinels[0]) == "city?"
        # Same task → same interrupt id, even though it is a new question.
        assert second_sentinels[0]["call_id"] == call_id, second_payload

        third = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(call_id, "Paris")],
            },
        )
        assert third.status_code == 200, third.text
        third_payload = third.json()
        assert third_payload["status"] == "completed", third_payload
        assert not _sentinels(third_payload), third_payload
        # Both stored answers replayed into the right slots.
        assert "Ada@Paris" in _assistant_text(third_payload), third_payload


def _build_skipping_interrupt_graph(flags: dict[str, bool]) -> Any:
    """The 🔴 pattern: a conditional ``interrupt()`` shifts the call order."""

    def ask(state: _MessagesState) -> dict[str, Any]:
        name = interrupt("name?")
        if flags["ask_age"]:
            interrupt("age?")
        city = interrupt("city?")
        return {"messages": [AIMessage(content=f"{name}@{city}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_misroutes_answers_when_a_node_skips_an_interrupt() -> None:
    """Skipping an ``interrupt()`` on replay silently misbinds answers.

    Resume values are matched by index, so dropping the second question
    shifts everything after it up one slot: the answer given for "age?"
    is handed back as the answer to "city?", and "city?" is never asked.
    The host has no way to detect or repair this — it faithfully relays
    whatever the graph does — so the test exists to make the corruption
    reproducible and attributable to the graph, not to hosting.
    """
    flags = {"ask_age": True}
    host = ResponsesHostServer(_build_skipping_interrupt_graph(flags))
    conversation_id = "conv-skip"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "go", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        sentinels = _sentinels(first.json())
        assert len(sentinels) == 1, first.json()
        assert _interrupt_value(sentinels[0]) == "name?"
        call_id = sentinels[0]["call_id"]

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(call_id, "Ada")],
            },
        )
        assert second.status_code == 200, second.text
        second_sentinels = _sentinels(second.json())
        assert len(second_sentinels) == 1, second.json()
        assert _interrupt_value(second_sentinels[0]) == "age?"

        # The branch condition flips before the node replays.
        flags["ask_age"] = False
        third = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(second_sentinels[0]["call_id"], "30")],
            },
        )
        assert third.status_code == 200, third.text
        third_payload = third.json()
        assert third_payload["status"] == "completed", third_payload
        # "city?" was never asked...
        assert not _sentinels(third_payload), third_payload
        # ...and the age answer landed in the city slot.
        assert "Ada@30" in _assistant_text(third_payload), third_payload


# --- Do not return complex values in interrupt calls ------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#do-not-return-complex-values-in-interrupt-calls


def _build_unserializable_value_graph() -> Any:
    """Pause on a payload the checkpointer accepts but ``json`` rejects.

    A ``set`` survives LangGraph's msgpack serializer, so the pause is
    checkpointed normally and only the *wire* encoding has a problem.
    """

    def ask(state: _MessagesState) -> dict[str, Any]:
        answer = interrupt({"q": "pick one", "choices": {"a", "b"}})
        return {"messages": [AIMessage(content=f"picked:{answer}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_degrades_gracefully_for_non_json_interrupt_values() -> None:
    """A non-JSON-serializable payload must not take the turn down.

    The docs tell graph authors to keep interrupt payloads simple, but
    hosting cannot enforce that. The envelope falls back to ``str()`` so
    the client still gets a routable sentinel and resume keeps working —
    degraded rendering instead of a failed response.
    """
    host = ResponsesHostServer(_build_unserializable_value_graph())
    conversation_id = "conv-unserializable"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "go", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        first_payload = first.json()
        sentinels = _sentinels(first_payload)
        assert len(sentinels) == 1, first_payload
        # Arguments stay valid JSON; the value degrades to its repr.
        value = _interrupt_value(sentinels[0])
        assert isinstance(value, str), first_payload
        assert "pick one" in value, first_payload
        # Both channels still agree so either can be used to resume.
        approvals = _approval_requests(first_payload)
        assert len(approvals) == 1, first_payload
        assert approvals[0]["arguments"] == sentinels[0]["arguments"]

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(sentinels[0]["call_id"], "a")],
            },
        )
        assert second.status_code == 200, second.text
        payload = second.json()
        assert payload["status"] == "completed", payload
        assert "picked:a" in _assistant_text(payload), payload


# --- Review and edit state --------------------------------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#review-and-edit-state


def _build_review_graph() -> Any:
    """Draft, then pause so a human can rewrite the draft."""

    def draft(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="Initial draft")]}

    def review(state: _MessagesState) -> dict[str, Any]:
        edited = interrupt(
            {
                "instruction": "Review and edit this content",
                "content": state["messages"][-1].content,
            }
        )
        return {"messages": [AIMessage(content=str(edited))]}

    builder = StateGraph(_MessagesState)
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", END)
    return builder.compile(checkpointer=InMemorySaver())


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_supports_review_and_edit_state() -> None:
    """The pause must carry the current state out, and the edit back in."""
    host = ResponsesHostServer(_build_review_graph())
    conversation_id = "conv-review"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "write something", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        sentinels = _sentinels(first.json())
        assert len(sentinels) == 1, first.json()
        assert _interrupt_value(sentinels[0]) == {
            "instruction": "Review and edit this content",
            "content": "Initial draft",
        }

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(sentinels[0]["call_id"], "Improved draft")],
            },
        )
        assert second.status_code == 200, second.text
        payload = second.json()
        assert payload["status"] == "completed", payload
        assert not _sentinels(payload), payload
        assert "Improved draft" in _assistant_text(payload), payload


# --- Approve or reject ------------------------------------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#approve-or-reject


def _build_approval_routing_graph() -> Any:
    """Route to a different node depending on the resume value."""

    def approval(state: _MessagesState) -> Command:
        decision = interrupt({"question": "Do you want to proceed?"})
        return Command(goto="proceed" if decision else "cancel")

    def proceed(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="status:approved")]}

    def cancel(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="status:rejected")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("approval", approval)
    builder.add_node("proceed", proceed)
    builder.add_node("cancel", cancel)
    builder.add_edge(START, "approval")
    builder.add_edge("proceed", END)
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.mark.parametrize(
    ("decision", "expected"),
    [(True, "status:approved"), (False, "status:rejected")],
)
@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_routes_approve_or_reject_through_command_goto(
    decision: bool, expected: str
) -> None:
    """``resume=false`` is a real answer, not a missing one.

    Distinct from the ``mcp_approval_response{approve:false}`` path,
    which fails the turn: here the *graph* owns the rejection semantics,
    so a falsy resume value has to reach the node intact and route it to
    the cancel branch.
    """
    host = ResponsesHostServer(_build_approval_routing_graph())
    conversation_id = f"conv-approve-{decision}"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "do the thing", "conversation": {"id": conversation_id}},
        )
        assert first.status_code == 200, first.text
        sentinels = _sentinels(first.json())
        assert len(sentinels) == 1, first.json()

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(sentinels[0]["call_id"], decision)],
            },
        )
        assert second.status_code == 200, second.text
        payload = second.json()
        assert payload["status"] == "completed", payload
        assert not _sentinels(payload), payload
        assert expected in _assistant_text(payload), payload


# --- Static interrupts are not human-in-the-loop interrupts -----------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#debugging-with-interrupts


def test_responses_host_does_not_surface_static_interrupt_breakpoints() -> None:
    """``interrupt_before`` pauses the graph without producing an ``Interrupt``.

    The docs steer people away from static breakpoints for HITL; this
    pins *why* it matters for hosting: the snapshot's ``next`` is set but
    ``interrupts`` is empty, so there is nothing for the host to emit and
    the client is left with a silently truncated turn.
    """

    def node(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="node-ran")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("n", node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["n"])

    host = ResponsesHostServer(graph)
    with _client(host) as client:
        resp = client.post(
            "/responses",
            json={"input": "hi", "conversation": {"id": "conv-static"}},
        )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert not _sentinels(payload), payload
    assert not _approval_requests(payload), payload
    # The node never ran, so there is no output to show either.
    assert "node-ran" not in _assistant_text(payload), payload


# --- Resume is scoped to the thread that paused -----------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#resuming-interrupts


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_does_not_resume_a_pause_from_another_conversation() -> None:
    """An interrupt id is only meaningful on the thread that produced it.

    Conversation id maps to LangGraph's ``thread_id``, so replaying a
    resume item against a different conversation must not reach into the
    original thread. The second conversation starts its own run (and its
    own pause), and the first stays resumable.
    """
    host = ResponsesHostServer(_build_simple_interrupt_graph())
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={"input": "hi", "conversation": {"id": "conv-thread-a"}},
        )
        assert first.status_code == 200, first.text
        sentinels = _sentinels(first.json())
        assert len(sentinels) == 1, first.json()
        call_id = sentinels[0]["call_id"]

        # Same resume item, wrong thread: nothing is pending there, so it
        # is treated as ordinary input and that thread pauses on its own.
        other = client.post(
            "/responses",
            json={
                "conversation": {"id": "conv-thread-b"},
                "input": [_resume(call_id, "Ada")],
            },
        )
        assert other.status_code == 200, other.text
        other_payload = other.json()
        other_sentinels = _sentinels(other_payload)
        assert len(other_sentinels) == 1, other_payload
        assert other_sentinels[0]["call_id"] != call_id, other_payload
        assert "ok:Ada" not in _assistant_text(other_payload), other_payload

        # The original thread is untouched and still resumable.
        resumed = client.post(
            "/responses",
            json={
                "conversation": {"id": "conv-thread-a"},
                "input": [_resume(call_id, "Ada")],
            },
        )
        assert resumed.status_code == 200, resumed.text
        payload = resumed.json()
        assert payload["status"] == "completed", payload
        assert not _sentinels(payload), payload
        assert "ok:Ada" in _assistant_text(payload), payload


# --- Streaming with human-in-the-loop interrupts ----------------------------
# https://docs.langchain.com/oss/python/langgraph/interrupts#stream-with-human-in-the-loop-hitl-interrupts


def _sse_payloads(body: str) -> list[Any]:
    """Decode the JSON payloads carried by an SSE response body."""
    payloads: list[Any] = []
    for line in body.splitlines():
        if not line.startswith("data:"):
            continue
        chunk = line[len("data:") :].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            payloads.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return payloads


def _hitl_items_in(payloads: Any) -> list[dict[str, Any]]:
    """Collect every HITL item nested anywhere in the decoded SSE events.

    Event *names* are the SDK's business and change between previews, so
    we scan structurally for the item shapes we are contractually
    required to emit instead of hard-coding event types.
    """
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("name") == HITL_FUNCTION_NAME and "type" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payloads)
    return found


@_REAL_INTERRUPT_ASYNC_XFAIL
def test_responses_host_streams_both_interrupt_channels_and_resumes() -> None:
    """Streaming clients must see the same two channels, and be able to
    resume over the streaming endpoint too.

    The existing parity test only checks that the sentinel name appears
    somewhere in the streamed body; this one pins that *both* item types
    really reach the wire and that a streamed resume completes the run.
    """
    host = ResponsesHostServer(_build_simple_interrupt_graph())
    conversation_id = "conv-stream-hitl"
    with _client(host) as client:
        first = client.post(
            "/responses",
            json={
                "input": "hi",
                "conversation": {"id": conversation_id},
                "stream": True,
            },
        )
        assert first.status_code == 200, first.text
        items = _hitl_items_in(_sse_payloads(first.text))
        assert items, first.text
        assert {"function_call", "mcp_approval_request"} <= {
            item["type"] for item in items
        }, first.text
        call_ids = {
            item["call_id"] for item in items if item["type"] == "function_call"
        }
        assert len(call_ids) == 1, first.text
        call_id = call_ids.pop()

        second = client.post(
            "/responses",
            json={
                "conversation": {"id": conversation_id},
                "input": [_resume(call_id, "Ada")],
                "stream": True,
            },
        )
        assert second.status_code == 200, second.text
        assert not _hitl_items_in(_sse_payloads(second.text)), second.text
        assert "ok:Ada" in second.text, second.text
