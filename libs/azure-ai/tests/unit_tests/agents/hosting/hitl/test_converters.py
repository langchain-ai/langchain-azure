# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the human-in-the-loop converter functions in ``_hitl.py``.

These exercise the pure translation layer between LangGraph ``Interrupt``
objects and Responses-API items — no graph and no host involved.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses.models import (
    FunctionCallOutputItemParam,
    ItemFunctionToolCall,
    MCPApprovalResponse,
)
from langchain_core.runnables import RunnableLambda
from langgraph.types import Command

from langchain_azure_ai.agents.hosting._converters import (
    HITL_FUNCTION_NAME,
    HITL_MCP_SERVER_LABEL,
    build_messages_input,
    detect_approval_rejection,
    detect_pending_interrupts,
    hitl_call_ids,
    interrupt_arguments_json,
    parse_resume_command,
    track_pending_interrupts,
)

from .conftest import emitted_items, pending_interrupt


async def test_detect_pending_interrupts_returns_empty_for_stateless_runnable() -> None:
    graph = RunnableLambda(lambda value: value)

    pending = await detect_pending_interrupts(graph, {})

    assert pending == ()


async def test_detect_pending_interrupts_skips_completed_empty_result() -> None:
    answered = pending_interrupt(id="int-a", value="answered")
    outstanding = pending_interrupt(id="int-b", value="outstanding")
    graph = MagicMock()
    graph.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            tasks=(
                SimpleNamespace(result={}, interrupts=(answered,)),
                SimpleNamespace(result=None, interrupts=(outstanding,)),
            )
        )
    )

    pending = await detect_pending_interrupts(graph, {})

    assert pending == (outstanding,)


async def test_detect_pending_interrupts_keeps_sequential_partial_result() -> None:
    outstanding = pending_interrupt(id="int-a", value="next question")
    graph = MagicMock()
    graph.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            tasks=(SimpleNamespace(result={}, interrupts=(outstanding,)),)
        )
    )

    pending = await detect_pending_interrupts(graph, {})

    assert pending == (outstanding,)


async def test_track_pending_interrupts_passes_through_and_accumulates() -> None:
    first = pending_interrupt(id="int-a", value="first")
    second = pending_interrupt(id="int-b", value="second")
    refreshed_second = pending_interrupt(id="int-b", value="second-updated")
    chunks = [
        ("updates", {"__interrupt__": (first,)}),
        ("updates", {"__interrupt__": (second,)}),
        ("updates", {"__interrupt__": (refreshed_second,)}),
        ("updates", {"node": {"messages": []}}),
    ]

    async def graph_stream() -> AsyncIterator[Any]:
        for chunk in chunks:
            yield chunk

    pending: list = []
    passed_through = [
        chunk async for chunk in track_pending_interrupts(graph_stream(), pending)
    ]

    assert passed_through == chunks
    assert pending == [first, refreshed_second]


def _sentinel_call(call_id: str, value: str = "Where are you?") -> ItemFunctionToolCall:
    """The ``function_call`` item the host emits for a pending interrupt."""
    return ItemFunctionToolCall(
        type="function_call",
        call_id=call_id,
        name=HITL_FUNCTION_NAME,
        arguments=json.dumps({"interrupt_id": call_id, "value": value}),
    )


def _tool_call(call_id: str, name: str) -> ItemFunctionToolCall:
    """An ordinary model-issued tool call."""
    return ItemFunctionToolCall(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments="{}",
    )


def _tool_output(call_id: str, output: str) -> FunctionCallOutputItemParam:
    """The client's answer to either kind of call."""
    return FunctionCallOutputItemParam(
        type="function_call_output",
        call_id=call_id,
        output=output,
    )


def _approval_response(
    approval_request_id: str,
    approve: bool,
    reason: str | None = None,
) -> MCPApprovalResponse:
    return MCPApprovalResponse(
        type="mcp_approval_response",
        approval_request_id=approval_request_id,
        approve=approve,
        reason=reason,
    )


def _tool_call_names(messages: list) -> list[str]:
    """Every tool-call name across a message list."""
    return [
        call["name"]
        for message in messages
        for call in getattr(message, "tool_calls", None) or ()
    ]


class TestParseResumeCommand:
    """The ``function_call_output`` resume channel."""

    def test_returns_none_when_no_pending(self) -> None:
        items = [_tool_output("int-1", '{"resume": "x"}')]
        command, consumed = parse_resume_command(items, ())
        assert command is None
        assert consumed == frozenset()

    def test_returns_none_when_no_matching_item(self) -> None:
        items = [_tool_output("other", '{"resume": "x"}')]
        pending = (pending_interrupt(id="int-1"),)
        command, consumed = parse_resume_command(items, pending)
        assert command is None
        assert consumed == frozenset()

    def test_decodes_json_envelope(self) -> None:
        items = [_tool_output("int-1", '{"resume": "Seattle"}')]
        pending = (pending_interrupt(id="int-1"),)
        command, consumed = parse_resume_command(items, pending)
        assert isinstance(command, Command)
        assert command.resume == "Seattle"
        assert consumed == frozenset({"int-1"})

    def test_supports_update_and_goto(self) -> None:
        body = json.dumps({"update": {"k": 1}, "goto": "next"})
        items = [_tool_output("int-1", body)]
        command, _ = parse_resume_command(items, (pending_interrupt(id="int-1"),))
        assert command is not None
        assert command.update == {"k": 1}
        assert command.goto == "next"

    def test_preserves_false_resume(self) -> None:
        # ``{"resume": false}`` is the reject half of the approve/reject
        # pattern. A falsy-but-present resume must survive decoding, or the
        # node would never take its "cancel" branch.
        items = [_tool_output("int-1", '{"resume": false}')]
        pending = (pending_interrupt(id="int-1"),)
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume is False
        assert consumed == frozenset({"int-1"})

    def test_preserves_explicit_null_resume(self) -> None:
        items = [_tool_output("int-1", '{"resume": null}')]
        pending = (pending_interrupt(id="int-1"),)
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-1": None}
        assert consumed == frozenset({"int-1"})

    def test_treats_plain_string_as_resume(self) -> None:
        items = [_tool_output("int-1", "Seattle")]
        command, _ = parse_resume_command(items, (pending_interrupt(id="int-1"),))
        assert command is not None
        assert command.resume == "Seattle"

    def test_treats_unrelated_json_as_resume(self) -> None:
        items = [_tool_output("int-1", '{"x": 1}')]
        command, _ = parse_resume_command(items, (pending_interrupt(id="int-1"),))
        assert command is not None
        # No resume/update/goto keys → keep raw string as resume.
        assert command.resume == '{"x": 1}'

    def test_ignores_blank_output(self) -> None:
        items = [_tool_output("int-1", "   ")]
        pending = (pending_interrupt(id="int-1"),)
        command, consumed = parse_resume_command(items, pending)
        assert command is None
        assert consumed == frozenset()


class TestInterruptArgumentsJson:
    """The outbound ``{"interrupt_id", "value"}`` envelope."""

    def test_emits_envelope_for_strings(self) -> None:
        out = interrupt_arguments_json(pending_interrupt(id="int-1", value="Where?"))
        assert json.loads(out) == {"interrupt_id": "int-1", "value": "Where?"}

    def test_falls_back_for_non_serializable(self) -> None:
        class Opaque:
            def __str__(self) -> str:
                return "opaque-value"

        out = interrupt_arguments_json(pending_interrupt(id="int-1", value=Opaque()))
        assert json.loads(out) == {"interrupt_id": "int-1", "value": "opaque-value"}


class TestApprovalResumeChannel:
    """The ``mcp_approval_response`` resume channel."""

    def test_approve_true_echoes_interrupt_value(self) -> None:
        pending = pending_interrupt(id="int-1", value={"question": "Where?"})
        items = [_approval_response("int-1", True)]
        command, consumed = parse_resume_command(items, (pending,))
        assert command is not None
        # approve=True echoes the original interrupt value back as the
        # resume payload (matches Agent Framework's behavior).
        assert command.resume == {"question": "Where?"}
        assert consumed == frozenset({"int-1"})

    def test_approve_false_yields_no_command(self) -> None:
        # Rejection is surfaced via ``detect_approval_rejection``, not here.
        pending = pending_interrupt(id="int-1")
        items = [_approval_response("int-1", False)]
        command, consumed = parse_resume_command(items, (pending,))
        assert command is None
        assert consumed == frozenset()

    def test_function_call_output_wins_over_approval(self) -> None:
        pending = pending_interrupt(id="int-1", value="original")
        items = [
            _tool_output("int-1", "Seattle"),
            _approval_response("int-1", True),
        ]
        command, consumed = parse_resume_command(items, (pending,))
        assert command is not None
        # function_call_output (richer payload) wins over the approval echo.
        assert command.resume == "Seattle"
        assert consumed == frozenset({"int-1"})

    def test_approval_for_unknown_id_is_ignored(self) -> None:
        pending = pending_interrupt(id="int-1")
        items = [_approval_response("other", True)]
        command, consumed = parse_resume_command(items, (pending,))
        assert command is None
        assert consumed == frozenset()


class TestParallelInterruptResumeMap:
    """Several pauses outstanding at once.

    LangGraph refuses a bare resume value when more than one interrupt is
    pending: "When there are multiple pending interrupts, you must specify
    the interrupt id when resuming." So the converter must fold every
    matched item into an id-keyed resume map.
    https://docs.langchain.com/oss/python/langgraph/interrupts#handling-multiple-interrupts
    """

    def test_builds_resume_map_for_parallel_interrupts(self) -> None:
        pending = (
            pending_interrupt(id="int-a", value="a?"),
            pending_interrupt(id="int-b", value="b?"),
        )
        items = [
            _tool_output("int-a", '{"resume": "A"}'),
            _tool_output("int-b", '{"resume": "B"}'),
        ]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": "A", "int-b": "B"}
        assert consumed == frozenset({"int-a", "int-b"})

    def test_map_allows_partial_answers(self) -> None:
        # Answering only one of two parallel pauses is legal; LangGraph keeps
        # the unanswered branch suspended.
        pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
        items = [_tool_output("int-a", "A")]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": "A"}
        assert consumed == frozenset({"int-a"})

    def test_update_only_does_not_implicitly_resume_with_none(self) -> None:
        pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
        items = [_tool_output("int-a", '{"update": {"marker": "set"}}')]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume is None
        assert command.update == {"marker": "set"}
        assert consumed == frozenset({"int-a"})

    def test_explicit_null_still_resumes_with_none(self) -> None:
        pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
        items = [_tool_output("int-a", '{"resume": null}')]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": None}
        assert consumed == frozenset({"int-a"})

    def test_preserves_updates_and_gotos_from_every_answer(self) -> None:
        pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
        items = [
            _tool_output(
                "int-a",
                json.dumps({"resume": "A", "update": {"a": 1}, "goto": "route-a"}),
            ),
            _tool_output(
                "int-b",
                json.dumps({"resume": "B", "update": {"b": 2}, "goto": "route-b"}),
            ),
        ]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": "A", "int-b": "B"}
        assert command.update == [("a", 1), ("b", 2)]
        assert command.goto == ("route-a", "route-b")
        assert consumed == frozenset({"int-a", "int-b"})

    def test_map_mixes_both_resume_channels(self) -> None:
        pending = (
            pending_interrupt(id="int-a"),
            pending_interrupt(id="int-b", value="echo-me"),
        )
        items = [
            _tool_output("int-a", "A"),
            _approval_response("int-b", True),
        ]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": "A", "int-b": "echo-me"}
        assert consumed == frozenset({"int-a", "int-b"})

    def test_map_skips_rejected_approvals(self) -> None:
        pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
        items = [
            _tool_output("int-a", "A"),
            _approval_response("int-b", False),
        ]
        command, _ = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": "A"}

    def test_routes_by_id_not_by_position(self) -> None:
        # Clients are under no obligation to answer in emission order, and
        # nothing in the Responses API preserves it. Pairing items to pending
        # interrupts by index would silently swap the answers here.
        pending = (
            pending_interrupt(id="int-a", value="a?"),
            pending_interrupt(id="int-b", value="b?"),
        )
        items = [
            _tool_output("int-b", '{"resume": "B"}'),
            _tool_output("int-a", '{"resume": "A"}'),
        ]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": "A", "int-b": "B"}
        assert consumed == frozenset({"int-a", "int-b"})

    def test_first_answer_wins_when_one_id_is_answered_twice(self) -> None:
        # Two answers for the same pause in a single request: the first is
        # kept and the duplicate dropped, rather than the last write winning.
        # The other pause must be unaffected either way.
        pending = (pending_interrupt(id="int-a"), pending_interrupt(id="int-b"))
        items = [
            _tool_output("int-a", '{"resume": "first"}'),
            _tool_output("int-a", '{"resume": "second"}'),
            _tool_output("int-b", '{"resume": "B"}'),
        ]
        command, consumed = parse_resume_command(items, pending)
        assert command is not None
        assert command.resume == {"int-a": "first", "int-b": "B"}
        assert consumed == frozenset({"int-a", "int-b"})


class TestApprovalIdRoundTrip:
    """``mcp_approval_request`` id encoding must survive the round trip.

    ``emit_interrupts`` cannot reuse the raw LangGraph interrupt id as the
    ``mcp_approval_request.id`` (storage requires an ``mcpr_*`` shape), so it
    encodes the interrupt id into the generated id. The inbound path must be
    able to recover it.
    """

    async def test_emit_interrupts_emits_paired_channels_per_interrupt(self) -> None:
        items = await emitted_items(
            (
                pending_interrupt(id="int-a", value="a?"),
                pending_interrupt(id="int-b", value="b?"),
            )
        )
        function_calls = [it for it in items if it["type"] == "function_call"]
        approvals = [it for it in items if it["type"] == "mcp_approval_request"]

        assert [it["call_id"] for it in function_calls] == ["int-a", "int-b"]
        assert all(it["name"] == HITL_FUNCTION_NAME for it in function_calls)
        assert len(approvals) == 2
        assert all(it["id"].startswith("mcpr_") for it in approvals)
        assert all(it["server_label"] == HITL_MCP_SERVER_LABEL for it in approvals)
        # The two channels carry the same envelope so a client may pick either.
        assert [it["arguments"] for it in approvals] == [
            it["arguments"] for it in function_calls
        ]
        # And each approval id must be distinct so partial answers stay routable.
        assert approvals[0]["id"] != approvals[1]["id"]

    async def test_parse_resume_command_accepts_emitted_approval_id(self) -> None:
        pending = pending_interrupt(id="int-1", value="echo-me")
        items = await emitted_items((pending,))
        approval_id = next(
            it["id"] for it in items if it["type"] == "mcp_approval_request"
        )
        assert approval_id != "int-1"  # encoded, not the raw interrupt id

        command, consumed = parse_resume_command(
            [_approval_response(approval_id, True)],
            (pending,),
        )
        assert command is not None
        assert command.resume == "echo-me"
        assert consumed == frozenset({approval_id})

    async def test_detect_approval_rejection_accepts_emitted_approval_id(self) -> None:
        pending = pending_interrupt(id="int-1")
        items = await emitted_items((pending,))
        approval_id = next(
            it["id"] for it in items if it["type"] == "mcp_approval_request"
        )

        message = detect_approval_rejection(
            [_approval_response(approval_id, False)],
            (pending,),
        )
        assert message is not None
        assert approval_id in message


class TestDetectApprovalRejection:
    """Turning ``approve=false`` into a failure message."""

    def test_returns_message_when_approve_false(self) -> None:
        pending = pending_interrupt(id="int-1")
        items = [_approval_response("int-1", False, reason="too risky")]
        msg = detect_approval_rejection(items, (pending,))
        assert msg is not None
        assert "int-1" in msg
        assert "too risky" in msg

    def test_returns_none_when_approve_true(self) -> None:
        pending = pending_interrupt(id="int-1")
        items = [_approval_response("int-1", True)]
        assert detect_approval_rejection(items, (pending,)) is None

    def test_returns_none_when_id_mismatches(self) -> None:
        pending = pending_interrupt(id="int-1")
        items = [_approval_response("other", False)]
        assert detect_approval_rejection(items, (pending,)) is None

    def test_returns_none_when_no_pending(self) -> None:
        items = [_approval_response("int-1", False)]
        assert detect_approval_rejection(items, ()) is None

    def test_function_call_output_wins_over_rejection(self) -> None:
        pending = pending_interrupt(id="int-1")
        items = [
            _approval_response("int-1", False),
            _tool_output("int-1", '{"resume": true}'),
        ]
        assert detect_approval_rejection(items, (pending,)) is None

    def test_blank_function_output_does_not_override_rejection(self) -> None:
        pending = pending_interrupt(id="int-1")
        items = [
            _tool_output("int-1", "  "),
            _approval_response("int-1", False),
        ]
        assert detect_approval_rejection(items, (pending,)) is not None


class TestHitlSentinelFiltering:
    """Wire plumbing must never reach the model, pending or not.

    On the turn that consumes an interrupt the host strips the sentinel
    pair through the resume path's consumed-id set. But stateless clients
    echo the previous turn's output items back with every request, so the
    sentinel keeps arriving long after the pause closed — when nothing is
    pending and no consumed-id set exists. Filtering therefore keys off
    the reserved function name instead.
    """

    def test_reserves_only_the_hitl_function_name(self) -> None:
        items = [_sentinel_call("int-a"), _tool_call("call_1", "get_weather")]
        assert hitl_call_ids(items) == frozenset({"int-a"})

    def test_drops_an_echoed_sentinel_pair(self) -> None:
        # No skip_call_ids: this is a turn *after* the pause was consumed,
        # so the host has no consumed-id set left to filter with.
        items = [_sentinel_call("int-a"), _tool_output("int-a", '{"resume": "Paris"}')]
        assert build_messages_input(items)["messages"] == []

    def test_keeps_ordinary_tool_round_trips(self) -> None:
        items = [_tool_call("call_1", "get_weather"), _tool_output("call_1", "sunny")]
        messages = build_messages_input(items)["messages"]
        assert _tool_call_names(messages) == ["get_weather"]
        assert messages[-1].content == "sunny"

    def test_drops_only_the_sentinel_when_both_are_present(self) -> None:
        # Adjacent function_call items are folded into one AIMessage, so
        # the sentinel has to be dropped without taking the real call with
        # it or leaving its answer behind as an orphan.
        items = [
            _sentinel_call("int-a"),
            _tool_call("call_1", "get_weather"),
            _tool_output("int-a", '{"resume": "Paris"}'),
            _tool_output("call_1", "sunny"),
        ]
        messages = build_messages_input(items)["messages"]
        assert _tool_call_names(messages) == ["get_weather"]
        assert [m.content for m in messages] == ["", "sunny"]
