# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Responses-API human-in-the-loop converter and host wiring."""

from __future__ import annotations

import json
import sys
from typing import Annotated, Any, ClassVar

import pytest

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

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
