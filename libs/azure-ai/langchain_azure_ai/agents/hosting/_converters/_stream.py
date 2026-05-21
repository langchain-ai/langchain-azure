# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Translate LangGraph streaming output into Responses API events.

Drives :meth:`CompiledStateGraph.astream` with
``stream_mode=["updates", "messages"]`` so the converter receives both
per-token text chunks and per-node state updates. This lets us surface
intermediate tool calls and tool-message results to the client in real
time, not only the final assistant message.

Lifecycle per turn (a "turn" is everything appended after the last
:class:`HumanMessage`):

1. Token text from any LLM node arrives as :class:`AIMessageChunk`
   payloads under the ``messages`` channel. Consecutive non-empty
   chunks are streamed through one ``message`` output item with
   ``output_text.delta`` events.
2. When a node finishes, an ``updates`` payload arrives. We finalize
   any open message item, then walk the messages produced by that node:

   - :class:`AIMessage.tool_calls` → ``function_call`` output items
     (with the full JSON arguments emitted as a single
     ``function_call_arguments.delta`` followed by ``done``).
   - :class:`ToolMessage` → ``function_call_output`` output items.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from azure.ai.agentserver.responses import ResponseEventStream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)

from ._utils import extract_text


async def stream_graph_to_events(
    graph_stream: AsyncIterator[Any],
    stream: ResponseEventStream,
    *,
    cancellation_signal: asyncio.Event,
) -> AsyncIterator[Any]:
    """Iterate the graph stream and yield Responses API events.

    The caller is responsible for emitting ``response.created`` /
    ``response.in_progress`` before invoking this generator and
    ``response.completed`` (or ``response.failed`` /
    ``response.cancelled``) after it returns.

    Args:
        graph_stream: The ``CompiledStateGraph.astream`` iterator,
            opened with ``stream_mode=["updates", "messages"]``.
        stream: The :class:`ResponseEventStream` to emit events through.
        cancellation_signal: Set by the responses host when the request
            is cancelled or the server is draining; iteration stops on set.

    Yields:
        Responses API event payload dicts.
    """
    state = _StreamState(stream)

    async for chunk in graph_stream:
        if cancellation_signal.is_set():
            break
        mode, payload = _split_chunk(chunk)
        if mode == "messages":
            async for event in state.handle_message_chunk(payload):
                yield event
        elif mode == "updates":
            async for event in state.handle_update(payload):
                yield event

    async for event in state.flush():
        yield event


class _StreamState:
    """Track in-flight builders and tool-call IDs already emitted."""

    def __init__(self, stream: ResponseEventStream) -> None:
        self._stream = stream
        self._message_builder: Any = None
        self._text_builder: Any = None
        self._text_buffer: list[str] = []
        self._emitted_tool_call_ids: set[str] = set()
        self._emitted_tool_output_call_ids: set[str] = set()

    async def handle_message_chunk(self, payload: Any) -> AsyncIterator[Any]:
        """Handle a payload from ``stream_mode="messages"``."""
        message_chunk = _extract_message_chunk(payload)
        if message_chunk is None:
            return

        text = extract_text(message_chunk.content)
        if not text:
            return

        if self._message_builder is None:
            self._message_builder = self._stream.add_output_item_message()
            yield self._message_builder.emit_added()
        if self._text_builder is None:
            self._text_builder = self._message_builder.add_text_content()
            yield self._text_builder.emit_added()
        self._text_buffer.append(text)
        yield self._text_builder.emit_delta(text)

    async def handle_update(self, payload: Any) -> AsyncIterator[Any]:
        """Handle a payload from ``stream_mode="updates"``.

        ``payload`` is ``{node_name: state_update}``; ``state_update`` is
        the partial state returned by the node, which for
        ``MessagesState`` graphs contains a ``messages`` channel with the
        messages that node appended.
        """
        for messages in _extract_node_messages(payload):
            # Close any in-flight assistant message before emitting the
            # tool calls / tool outputs that just arrived from this node.
            async for event in self._close_open_message():
                yield event

            for message in messages:
                if isinstance(message, AIMessage):
                    for call in message.tool_calls or []:
                        async for event in self._emit_tool_call(call):
                            yield event
                elif isinstance(message, ToolMessage):
                    async for event in self._emit_tool_output(message):
                        yield event

    async def flush(self) -> AsyncIterator[Any]:
        """Close any in-flight builders. Called after the graph stream ends."""
        async for event in self._close_open_message():
            yield event

    async def _close_open_message(self) -> AsyncIterator[Any]:
        if self._text_builder is not None:
            yield self._text_builder.emit_text_done("".join(self._text_buffer))
            yield self._text_builder.emit_done()
            self._text_builder = None
            self._text_buffer = []
        if self._message_builder is not None:
            yield self._message_builder.emit_done()
            self._message_builder = None

    async def _emit_tool_call(self, call: Any) -> AsyncIterator[Any]:
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or call.get("call_id") or "")
        if not name or not call_id or call_id in self._emitted_tool_call_ids:
            return
        self._emitted_tool_call_ids.add(call_id)

        args = call.get("args")
        arguments_json = args if isinstance(args, str) else json.dumps(args or {})

        fn = self._stream.add_output_item_function_call(name, call_id)
        yield fn.emit_added()
        if arguments_json:
            yield fn.emit_arguments_delta(arguments_json)
        yield fn.emit_arguments_done(arguments_json)
        yield fn.emit_done()

    async def _emit_tool_output(self, message: ToolMessage) -> AsyncIterator[Any]:
        call_id = str(getattr(message, "tool_call_id", "") or "")
        if not call_id or call_id in self._emitted_tool_output_call_ids:
            return
        self._emitted_tool_output_call_ids.add(call_id)
        output_text = extract_text(message.content)
        fn_out = self._stream.add_output_item_function_call_output(call_id)
        yield fn_out.emit_added(output_text)
        yield fn_out.emit_done(output_text)


def _split_chunk(chunk: Any) -> tuple[str | None, Any]:
    """Decode a multi-mode ``astream`` payload.

    With ``stream_mode=["updates", "messages"]`` LangGraph yields
    ``(mode_name, payload)`` tuples. When a single mode is configured,
    the iterator yields raw payloads, in which case we treat them as
    ``"messages"`` for backwards compatibility.

    Args:
        chunk: One value yielded by ``graph.astream``.

    Returns:
        A ``(mode, payload)`` pair, with ``mode`` set to ``None`` when
        the value cannot be classified.
    """
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        return chunk[0], chunk[1]
    return "messages", chunk


def _extract_message_chunk(payload: Any) -> AIMessageChunk | None:
    """Pull an ``AIMessageChunk`` out of a ``messages`` payload."""
    if isinstance(payload, AIMessageChunk):
        return payload
    if isinstance(payload, tuple) and payload:
        candidate = payload[0]
        if isinstance(candidate, AIMessageChunk):
            return candidate
    return None


def _extract_node_messages(payload: Any) -> list[list[BaseMessage]]:
    """Extract message lists from each node update inside an ``updates`` payload.

    LangGraph 1.x emits ``{node_name: {"messages": [...]}}`` per node.
    Older releases occasionally surface the per-node update directly
    (``{"messages": [...]}``); we accept both shapes.

    Args:
        payload: The ``updates`` payload from ``graph.astream``.

    Returns:
        A list of message lists, one per node update found.
    """
    result: list[list[BaseMessage]] = []
    if not isinstance(payload, dict):
        return result
    # Per-node form: {node_name: {"messages": [...]}}
    saw_node_form = False
    for value in payload.values():
        if isinstance(value, dict) and "messages" in value:
            saw_node_form = True
            messages = value.get("messages") or []
            if isinstance(messages, list):
                result.append(messages)
    if saw_node_form:
        return result
    # Direct form: {"messages": [...]}
    messages = payload.get("messages") or []
    if isinstance(messages, list):
        result.append(messages)
    return result
