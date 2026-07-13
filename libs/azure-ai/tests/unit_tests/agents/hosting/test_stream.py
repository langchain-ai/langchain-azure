# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the streaming converter (``_stream``)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("azure.ai.agentserver.responses")

from azure.ai.agentserver.responses import ResponseEventStream  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AIMessageChunk,
    ToolMessage,
)

from langchain_azure_ai.agents.hosting._converters._stream import (  # noqa: E402
    _extract_checkpoint_id,
    stream_graph_to_events,
)
from langchain_azure_ai.agents.hosting.utils import (  # noqa: E402
    METADATA_LANGGRAPH_CHECKPOINT_ID,
)


def _reasoning_chunk(text: str) -> AIMessageChunk:
    """Return an ``AIMessageChunk`` carrying a single reasoning summary fragment.

    Mirrors the content shape produced by a chat model configured with
    ``reasoning={"summary": "auto"}`` on the Responses path.
    """
    return AIMessageChunk(
        content=[
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": text}],
            }
        ]
    )


async def _agen(items: list[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


async def _drive(items: list[Any]) -> list[Any]:
    """Run *items* through the converter and return the emitted events.

    The host is responsible for the surrounding ``response.created`` /
    ``response.in_progress`` (and terminal ``response.completed``)
    lifecycle events, so emit those here too — the SDK's event-stream
    validator rejects any output item that is not bracketed by them.
    """
    stream = ResponseEventStream(response_id="resp-test")
    stream.emit_created()
    stream.emit_in_progress()
    events: list[Any] = []
    async for event in stream_graph_to_events(
        _agen(items),
        stream,
        cancellation_signal=asyncio.Event(),
    ):
        events.append(event)
    stream.emit_completed()
    return events


def _types(events: list[Any]) -> list[str]:
    return [event.type for event in events]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, None),
        ({}, None),
        ({"config": None}, None),
        ({"config": {"configurable": None}}, None),
        ({"config": {"configurable": {"checkpoint_id": ""}}}, None),
        ({"config": {"configurable": {"checkpoint_id": 1}}}, None),
        (
            {"config": {"configurable": {"checkpoint_id": "checkpoint-1"}}},
            "checkpoint-1",
        ),
    ],
)
def test_extract_checkpoint_id(payload: Any, expected: str | None) -> None:
    assert _extract_checkpoint_id(payload) == expected


async def test_reasoning_chunk_emits_summary_text_delta() -> None:
    """A reasoning content block is surfaced as a ``reasoning`` output item
    with streaming ``reasoning_summary_text.delta`` events."""
    events = await _drive(
        [
            ("messages", (_reasoning_chunk("Let me think"), {})),
            ("messages", (_reasoning_chunk(" about it."), {})),
        ]
    )

    types = _types(events)
    assert "response.output_item.added" in types
    assert "response.reasoning_summary_part.added" in types
    assert "response.reasoning_summary_text.delta" in types
    assert "response.reasoning_summary_text.done" in types
    assert "response.reasoning_summary_part.done" in types
    assert "response.output_item.done" in types

    deltas = [
        event.delta
        for event in events
        if event.type == "response.reasoning_summary_text.delta"
    ]
    assert deltas == ["Let me think", " about it."]


async def test_reasoning_item_closes_before_assistant_text() -> None:
    """The reasoning item is finalized before any assistant message text,
    so the two output items never interleave."""
    events = await _drive(
        [
            ("messages", (_reasoning_chunk("thinking"), {})),
            ("messages", (AIMessageChunk(content="The answer is 42."), {})),
        ]
    )

    types = _types(events)
    reasoning_done = types.index("response.reasoning_summary_part.done")
    first_text_delta = types.index("response.output_text.delta")
    assert reasoning_done < first_text_delta

    text_deltas = [
        event.delta for event in events if event.type == "response.output_text.delta"
    ]
    assert "".join(text_deltas) == "The answer is 42."


async def test_empty_reasoning_fragment_separates_sections() -> None:
    """An empty summary fragment marks a new section and is rendered as a
    newline delta (only once content already exists)."""
    events = await _drive(
        [
            ("messages", (_reasoning_chunk("first"), {})),
            ("messages", (_reasoning_chunk(""), {})),
            ("messages", (_reasoning_chunk("second"), {})),
        ]
    )

    deltas = [
        event.delta
        for event in events
        if event.type == "response.reasoning_summary_text.delta"
    ]
    assert deltas == ["first", "\n", "second"]


async def test_leading_empty_reasoning_fragment_is_dropped() -> None:
    """A leading empty fragment does not emit a spurious newline, and does not
    open a reasoning output item or summary part before real content arrives."""
    events = await _drive(
        [
            ("messages", (_reasoning_chunk(""), {})),
            ("messages", (_reasoning_chunk("body"), {})),
        ]
    )

    deltas = [
        event.delta
        for event in events
        if event.type == "response.reasoning_summary_text.delta"
    ]
    assert deltas == ["body"]

    reasoning_items_added = [
        event
        for event in events
        if event.type == "response.output_item.added" and event.item.type == "reasoning"
    ]
    assert len(reasoning_items_added) == 1


async def test_lone_leading_empty_reasoning_fragment_opens_no_item() -> None:
    """A lone leading empty fragment opens no reasoning item or part, so the
    stream emits no reasoning output at all and ``flush`` cannot close a
    spurious empty reasoning item."""
    events = await _drive([("messages", (_reasoning_chunk(""), {}))])

    types = _types(events)
    assert not any(typename.startswith("response.reasoning") for typename in types)
    assert not any(
        event.type == "response.output_item.added" and event.item.type == "reasoning"
        for event in events
    )


async def test_reasoning_item_closes_before_tool_call() -> None:
    """The reasoning item is finalized before a function call opens."""
    update = {
        "agent": {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_weather", "id": "call_1", "args": {"city": "X"}}
                    ],
                )
            ]
        }
    }
    events = await _drive(
        [
            ("messages", (_reasoning_chunk("deciding to call a tool"), {})),
            ("updates", update),
        ]
    )

    types = _types(events)
    reasoning_done = types.index("response.reasoning_summary_part.done")
    function_call = types.index("response.function_call_arguments.delta")
    assert reasoning_done < function_call


async def test_reasoning_item_closes_before_tool_output() -> None:
    """The reasoning item is finalized before a function-call output opens."""
    update = {
        "tools": {"messages": [ToolMessage(content="sunny", tool_call_id="call_1")]}
    }
    events = await _drive(
        [
            ("messages", (_reasoning_chunk("inspecting the tool result"), {})),
            ("updates", update),
        ]
    )

    types = _types(events)
    assert "response.reasoning_summary_part.done" in types

    reasoning_item_done = next(
        index
        for index, event in enumerate(events)
        if event.type == "response.output_item.done" and event.item.type == "reasoning"
    )
    tool_output_item_added = next(
        index
        for index, event in enumerate(events)
        if event.type == "response.output_item.added"
        and event.item.type == "function_call_output"
    )
    assert reasoning_item_done < tool_output_item_added


async def test_chunk_without_reasoning_emits_no_reasoning_events() -> None:
    """Plain text chunks must not produce any reasoning events."""
    events = await _drive([("messages", (AIMessageChunk(content="hello world"), {}))])

    types = _types(events)
    assert not any(typename.startswith("response.reasoning") for typename in types)
    assert "response.output_text.delta" in types


async def test_checkpoint_event_captures_config_and_commits_response() -> None:
    stream = ResponseEventStream(response_id="resp-test")
    stream.emit_created()
    stream.emit_in_progress()
    events = [
        event
        async for event in stream_graph_to_events(
            _agen(
                [
                    (
                        "updates",
                        {"agent": {"messages": [AIMessage(content="committed")]}},
                    ),
                    (
                        "checkpoints",
                        {
                            "config": {
                                "configurable": {
                                    "thread_id": "thread-1",
                                    "checkpoint_id": "checkpoint-1",
                                }
                            }
                        },
                    )
                ]
            ),
            stream,
            cancellation_signal=asyncio.Event(),
        )
    ]
    stream.emit_completed()

    assert events[-2].type == "response.output_item.done"
    assert type(events[-1]).__name__ == "ResponseCheckpointEvent"
    assert (
        stream.internal_metadata[METADATA_LANGGRAPH_CHECKPOINT_ID]
        == "checkpoint-1"
    )


async def test_checkpoint_event_is_committed_after_cancellation() -> None:
    stream = ResponseEventStream(response_id="resp-test")
    stream.emit_created()
    stream.emit_in_progress()
    cancellation_signal = asyncio.Event()

    async def graph_stream() -> AsyncIterator[Any]:
        yield ("updates", {})
        cancellation_signal.set()
        yield (
            "checkpoints",
            {
                "config": {
                    "configurable": {
                        "thread_id": "thread-1",
                        "checkpoint_id": "checkpoint-1",
                    }
                }
            },
        )

    events = [
        event
        async for event in stream_graph_to_events(
            graph_stream(),
            stream,
            cancellation_signal=cancellation_signal,
        )
    ]
    stream.emit_completed()

    assert len(events) == 1
    assert type(events[0]).__name__ == "ResponseCheckpointEvent"
