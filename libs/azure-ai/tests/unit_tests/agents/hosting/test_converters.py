# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for converters."""

from __future__ import annotations

from azure.ai.agentserver.responses.models import (
    FunctionCallOutputItemParam,
    ItemFunctionToolCall,
    ItemMessage,
    MessageContentInputTextContent,
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting._converters import (
    build_messages_input,
    build_messages_input_from_text,
    extract_text,
    is_messages_state_schema,
    items_to_messages,
    last_ai_message_text,
)


class _WithMessages(TypedDict):
    messages: list


class _NoMessages(TypedDict):
    name: str


def test_is_messages_state_schema_detects_messages_field() -> None:
    assert is_messages_state_schema(_WithMessages) is True
    assert is_messages_state_schema(_NoMessages) is False


def test_extract_text_handles_string_list_and_none() -> None:
    assert extract_text("hello") == "hello"
    assert extract_text(None) == ""
    assert extract_text(["a", "b"]) == "ab"
    assert extract_text([{"text": "hi"}, {"text": " there"}]) == "hi there"


def test_last_ai_message_text_picks_final_aimessage() -> None:
    messages = [
        HumanMessage(content="hello"),
        AIMessage(content="first"),
        AIMessage(content="second"),
    ]
    assert last_ai_message_text(messages) == "second"
    assert last_ai_message_text([]) == ""


def test_items_to_messages_handles_message_function_call_and_output() -> None:
    items = [
        ItemMessage(
            role="user",
            content=[
                MessageContentInputTextContent(
                    {"type": "input_text", "text": "hi"}
                )
            ],
        ),
        ItemFunctionToolCall(
            call_id="call_1",
            name="get_weather",
            arguments='{"city": "Seattle"}',
        ),
        FunctionCallOutputItemParam(call_id="call_1", output="sunny"),
    ]

    messages = items_to_messages(items)
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "hi"
    ai = messages[1]
    assert isinstance(ai, AIMessage)
    assert ai.tool_calls and ai.tool_calls[0]["name"] == "get_weather"
    tool = messages[2]
    assert isinstance(tool, ToolMessage)
    assert tool.tool_call_id == "call_1"
    assert tool.content == "sunny"


def test_build_messages_input_prepends_instructions() -> None:
    result = build_messages_input([], instructions="be concise")
    assert isinstance(result["messages"][0], SystemMessage)


def test_build_messages_input_from_text_creates_human_message() -> None:
    result = build_messages_input_from_text("hello", instructions="be brief")
    assert isinstance(result["messages"][0], SystemMessage)
    assert isinstance(result["messages"][1], HumanMessage)
    assert result["messages"][1].content == "hello"


def test_build_messages_input_from_text_skips_empty_text() -> None:
    result = build_messages_input_from_text("")
    assert result["messages"] == []


def test_build_messages_input_drops_orphan_tool_message() -> None:
    """A bare ``function_call_output`` (no matching prior ``function_call``)
    must be filtered out so the chat model isn't sent an invalid
    ``role: tool`` message without a preceding ``tool_calls``.

    This guards against a poisoned conversation store where a failed
    earlier request persisted a ``function_call_output`` that never had
    a paired tool call.
    """
    items = [
        FunctionCallOutputItemParam(call_id="orphan", output="stale"),
        ItemMessage(
            role="user",
            content=[
                MessageContentInputTextContent(
                    {"type": "input_text", "text": "do something new"}
                )
            ],
        ),
    ]

    result = build_messages_input(items)
    # Only the user message survives; the orphan ToolMessage is dropped.
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)


def test_build_messages_input_drops_unanswered_tool_call() -> None:
    """An ``AIMessage`` whose ``tool_calls`` are not closed by a paired
    ``ToolMessage`` is dropped so the chat model isn't asked to reason
    about a dangling tool call."""
    items = [
        ItemMessage(
            role="user",
            content=[
                MessageContentInputTextContent(
                    {"type": "input_text", "text": "hi"}
                )
            ],
        ),
        ItemFunctionToolCall(
            call_id="call_pending",
            name="get_weather",
            arguments='{"city": "Seattle"}',
        ),
        # No matching function_call_output — incomplete sequence.
    ]
    result = build_messages_input(items)
    # Only the user message survives.
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)


def test_build_messages_input_keeps_balanced_tool_call_pair() -> None:
    """Sanity check: a properly paired call + output is preserved."""
    items = [
        ItemMessage(
            role="user",
            content=[
                MessageContentInputTextContent(
                    {"type": "input_text", "text": "weather?"}
                )
            ],
        ),
        ItemFunctionToolCall(
            call_id="call_ok",
            name="get_weather",
            arguments="{}",
        ),
        FunctionCallOutputItemParam(call_id="call_ok", output="sunny"),
    ]
    result = build_messages_input(items)
    assert len(result["messages"]) == 3
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    assert isinstance(result["messages"][2], ToolMessage)
