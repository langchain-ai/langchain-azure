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
2. Reasoning summaries (emitted when the chat model is configured with
   ``reasoning={"summary": "auto"}``) arrive in the same
   :class:`AIMessageChunk` payloads as ``reasoning`` content blocks.
   They are streamed through a ``reasoning`` output item with
   ``reasoning_summary_text.delta`` events. At most one reasoning item
   is open at a time; it is closed before any assistant text, tool call,
   or tool output is emitted so output items stay correctly ordered.
3. When a node finishes, an ``updates`` payload arrives. We finalize
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

from ._utils import extract_reasoning_summary_fragments, extract_text


async def stream_graph_to_events(
    graph_stream: AsyncIterator[Any],
    stream: ResponseEventStream,
    *,
    cancellation_signal: asyncio.Event,
    checkpoint_each_phase: bool = False,
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
        checkpoint_each_phase: When ``True``, treat each LangGraph ``updates``
            payload as a resilient phase: after the payload's output items are
            closed, ``yield stream.checkpoint()`` so the response snapshot is
            persisted at the LangGraph superstep boundary. This is
            what lets a crash-recovered attempt seed itself from
            ``context.persisted_response`` and resume past the output already
            committed. ``stream.checkpoint()`` is a no-op unless the deployment
            has ``resilient_background=True`` and the request is
            ``background=true``, so it is safe to enable unconditionally.
    Yields:
        Responses API event payload dicts.
    """
    state = _StreamState(stream, checkpoint_each_phase=checkpoint_each_phase)

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

    def __init__(
        self, stream: ResponseEventStream, *, checkpoint_each_phase: bool = False
    ) -> None:
        self._stream = stream
        self._checkpoint_each_phase = checkpoint_each_phase
        self._message_builder: Any = None
        self._text_builder: Any = None
        self._text_buffer: list[str] = []
        self._reasoning_builder: Any = None
        self._reasoning_part_builder: Any = None
        self._reasoning_buffer: list[str] = []
        self._emitted_tool_call_ids: set[str] = set()
        self._emitted_tool_output_call_ids: set[str] = set()
        # AIMessage ids whose text was streamed token-by-token via
        # ``stream_mode="messages"`` (LLM nodes). Used to avoid re-emitting the
        # same assistant text as a whole ``message`` item when the node's
        # ``updates`` payload arrives carrying the finished AIMessage.
        self._streamed_message_ids: set[str] = set()
        # AIMessage ids already surfaced as a whole ``message`` item from an
        # ``updates`` payload (non-LLM nodes that return assistant text
        # directly, e.g. a deterministic ``finalize`` node), so we emit each
        # such message exactly once.
        self._emitted_message_ids: set[str] = set()

    async def handle_message_chunk(self, payload: Any) -> AsyncIterator[Any]:
        """Handle a payload from ``stream_mode="messages"``."""
        message_chunk = _extract_message_chunk(payload)
        if message_chunk is None:
            return

        chunk_id = getattr(message_chunk, "id", None)
        if isinstance(chunk_id, str) and chunk_id:
            self._streamed_message_ids.add(chunk_id)

        for fragment in extract_reasoning_summary_fragments(message_chunk.content):
            async for event in self._emit_reasoning_fragment(fragment):
                yield event

        text = extract_text(message_chunk.content)
        if not text:
            return

        # Assistant text closes any in-flight reasoning item so output
        # items stay ordered: reasoning is emitted before the answer.
        async for event in self._close_open_reasoning():
            yield event

        if self._message_builder is None:
            self._message_builder = self._stream.add_output_item_message()
            yield self._message_builder.emit_added()
        if self._text_builder is None:
            self._text_builder = self._message_builder.add_text_content()
            yield self._text_builder.emit_added()
        self._text_buffer.append(text)
        yield self._text_builder.emit_delta(text)

    async def _emit_reasoning_fragment(self, fragment: str) -> AsyncIterator[Any]:
        """Stream one reasoning summary text fragment.

        Opens a reasoning output item and summary part on first use. An
        empty fragment marks the start of a new summary section; once a
        section has already received text, it is rendered as a newline
        delta so consecutive sections stay visually separated within the
        single open summary part. A leading empty fragment (before any
        content is buffered) is ignored before any item or part opens, so
        it never produces a spurious empty reasoning output item.
        """
        if not fragment and not self._reasoning_buffer:
            return
        if self._reasoning_builder is None:
            self._reasoning_builder = self._stream.add_output_item_reasoning_item()
            yield self._reasoning_builder.emit_added()
        if self._reasoning_part_builder is None:
            self._reasoning_part_builder = self._reasoning_builder.add_summary_part()
            yield self._reasoning_part_builder.emit_added()
        delta = fragment or "\n"
        self._reasoning_buffer.append(delta)
        yield self._reasoning_part_builder.emit_text_delta(delta)

    async def handle_update(self, payload: Any) -> AsyncIterator[Any]:
        """Handle a payload from ``stream_mode="updates"``.

        ``payload`` is ``{node_name: state_update}``; ``state_update`` is
        the partial state returned by the node, which for
        ``MessagesState`` graphs contains a ``messages`` channel with the
        messages that node appended.

        When ``checkpoint_each_phase`` is enabled, the whole completed
        ``updates`` payload is treated as a resilient phase: after all node
        outputs in the payload are closed, ``yield stream.checkpoint()``
        persists the response snapshot at the LangGraph superstep boundary.
        """
        for node_name, messages in _extract_node_updates(payload):
            # Close any in-flight reasoning item and assistant message
            # before emitting the tool calls / tool outputs that just
            # arrived from this node, so output items stay ordered.
            async for event in self._close_open_reasoning():
                yield event
            async for event in self._close_open_message():
                yield event

            for message in messages:
                if isinstance(message, AIMessage):
                    async for event in self._emit_ai_message_text(message):
                        yield event
                    for call in message.tool_calls or []:
                        async for event in self._emit_tool_call(call):
                            yield event
                elif isinstance(message, ToolMessage):
                    async for event in self._emit_tool_output(message):
                        yield event

        if self._checkpoint_each_phase:
            async for event in self._close_open_message():
                yield event
            yield self._stream.checkpoint()

    async def flush(self) -> AsyncIterator[Any]:
        """Close any in-flight builders. Called after the graph stream ends."""
        async for event in self._close_open_reasoning():
            yield event
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

    async def _close_open_reasoning(self) -> AsyncIterator[Any]:
        if self._reasoning_part_builder is not None:
            yield self._reasoning_part_builder.emit_text_done(
                "".join(self._reasoning_buffer)
            )
            yield self._reasoning_part_builder.emit_done()
            self._reasoning_part_builder = None
            self._reasoning_buffer = []
        if self._reasoning_builder is not None:
            yield self._reasoning_builder.emit_done()
            self._reasoning_builder = None

    async def _emit_ai_message_text(self, message: AIMessage) -> AsyncIterator[Any]:
        """Emit a whole ``message`` item for assistant text from a node update.

        LangGraph's ``stream_mode="messages"`` only streams tokens from chat
        model invocations. A node that returns an :class:`AIMessage` with text
        content *without* calling an LLM (e.g. a deterministic ``finalize``
        node) therefore never produces ``messages``-mode chunks, so its text
        would otherwise be dropped from the stream. Here we surface it as a
        complete ``message`` output item.

        The emission is deduplicated two ways so LLM-streamed text is not
        doubled up: messages whose id was already streamed token-by-token
        (tracked in ``_streamed_message_ids``) are skipped, as are messages
        already emitted here (``_emitted_message_ids``).
        """
        text = extract_text(message.content)
        if not text:
            return
        message_id = message.id
        if isinstance(message_id, str) and message_id:
            if message_id in self._streamed_message_ids:
                return
            if message_id in self._emitted_message_ids:
                return
            self._emitted_message_ids.add(message_id)
        elif self._text_buffer:
            # No id to dedup on, but text was already streamed this segment.
            return

        async for event in self._close_open_reasoning():
            yield event
        async for event in self._close_open_message():
            yield event

        message_builder = self._stream.add_output_item_message()
        yield message_builder.emit_added()
        text_builder = message_builder.add_text_content()
        yield text_builder.emit_added()
        yield text_builder.emit_delta(text)
        yield text_builder.emit_text_done(text)
        yield text_builder.emit_done()
        yield message_builder.emit_done()

    async def _emit_tool_call(self, call: Any) -> AsyncIterator[Any]:
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or call.get("call_id") or "")
        if not name or not call_id or call_id in self._emitted_tool_call_ids:
            return
        async for event in self._close_open_reasoning():
            yield event
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
        async for event in self._close_open_reasoning():
            yield event
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


def _extract_node_updates(payload: Any) -> list[tuple[str, list[BaseMessage]]]:
    """Extract ``(node_name, messages)`` pairs from an ``updates`` payload.

    LangGraph 1.x emits ``{node_name: {"messages": [...]}}`` per node.
    Older releases occasionally surface the per-node update directly
    (``{"messages": [...]}``); we accept both shapes and label the direct
    form with an empty node name.

    Args:
        payload: The ``updates`` payload from ``graph.astream``.

    Returns:
        A list of ``(node_name, messages)`` pairs, one per node update found.
    """
    result: list[tuple[str, list[BaseMessage]]] = []
    if not isinstance(payload, dict):
        return result
    # Per-node form: {node_name: {"messages": [...]}}
    saw_node_form = False
    for node_name, value in payload.items():
        if isinstance(value, dict) and "messages" in value:
            saw_node_form = True
            messages = value.get("messages") or []
            if isinstance(messages, list):
                result.append((str(node_name), messages))
    if saw_node_form:
        return result
    # Direct form: {"messages": [...]}
    messages = payload.get("messages") or []
    if isinstance(messages, list):
        result.append(("", messages))
    return result
