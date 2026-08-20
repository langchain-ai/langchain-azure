# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Translate LangGraph streaming output into Responses API events.

Drives :meth:`CompiledStateGraph.astream` with
``stream_mode=["updates", "messages", "checkpoints"]`` so the converter
receives per-token text chunks, per-node state updates, and the exact persisted
checkpoint config for the invocation. Checkpoint events stay internal while
tool calls and tool-message results are surfaced to the client in real time.

Lifecycle per turn (a "turn" is everything appended after the last
:class:`HumanMessage`):

1. Assistant text arrives under the ``messages`` channel in one of two
   shapes. A streaming chat model produces :class:`AIMessageChunk`
   payloads that share one message id. A non-streaming chat model call,
   or a node that returns an :class:`AIMessage` directly (e.g. a
   deterministic ``finalize`` node), produces a single whole
   :class:`AIMessage`. LangGraph emits each message on this channel
   exactly once — its ``StreamMessagesHandler`` deduplicates by message
   id across token chunks, LLM completions and node returns — so the
   converter treats both shapes the same way: consecutive non-empty
   payloads sharing a message id are streamed through one ``message``
   output item with ``output_text.delta`` events.
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
from typing import Any, cast

from azure.ai.agentserver.responses import ResponseEventStream
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from .._responses import CheckpointRef, HostingRunnableConfig, TaskStorageManager
from ._utils import extract_reasoning_summary_fragments, extract_text


async def stream_graph_to_events(
    graph_stream: AsyncIterator[Any],
    stream: ResponseEventStream,
    *,
    cancellation_signal: asyncio.Event,
    shutdown_signal: asyncio.Event | None = None,
) -> AsyncIterator[Any]:
    """Iterate the graph stream and yield Responses API events.

    Each invocation handles one Responses turn by consuming the complete stream
    from one LangGraph execution. A graph run may contain multiple supersteps
    and checkpoint events, all of which are processed by this invocation.

    The caller is responsible for emitting ``response.created`` /
    ``response.in_progress`` before invoking this generator and
    ``response.completed`` (or ``response.failed`` /
    ``response.cancelled``) after it returns.

    Args:
        graph_stream: The ``CompiledStateGraph.astream`` iterator,
            opened with ``stream_mode=["updates", "messages", "checkpoints"]``.
        stream: The :class:`ResponseEventStream` to emit events through.
        cancellation_signal: Set by the responses host when the request
            is cancelled; iteration stops on set.
        shutdown_signal: Set when the host is draining. Iteration stops on set
            so the caller can defer resilient work to the next lifetime.

    Yields:
        Responses API event payload dicts.
    """
    converter = StreamConverter(stream)
    task_storage = TaskStorageManager.from_stream(stream)

    # Common timeline:
    #   ...
    #   -> execute superstep A
    #   -> "messages" chunks while nodes run
    #   -> "updates" chunks as nodes finish
    #   -> commit LangGraph checkpoint
    #   -> "checkpoints" chunk for the resulting state
    #   -> commit responses store  <--- THE recovery boundary
    #       A crash before it resumes from superstep A.
    #       A crash after it resumes from superstep B.
    #   -> execute superstep B
    #   -> ...
    # At each checkpoint boundary, close partial Responses output and persist
    # the Responses layer before requesting another LangGraph event.
    def stop_requested() -> bool:
        return cancellation_signal.is_set() or bool(
            shutdown_signal is not None and shutdown_signal.is_set()
        )

    async for chunk in graph_stream:
        mode, payload = _split_chunk(chunk)
        if stop_requested() and mode != "checkpoints":
            break
        if mode == "messages":
            async for event in converter.handle_message_chunk(payload):
                yield event
                if stop_requested():
                    break
        elif mode == "updates":
            async for event in converter.handle_update(payload):
                yield event
                if stop_requested():
                    break
        elif mode == "checkpoints":
            checkpoint_ref = _extract_checkpoint_ref(payload)
            if checkpoint_ref is not None:
                task_storage.store_checkpoint_ref(checkpoint_ref)
            async for event in converter.checkpoint():
                yield event
                if stop_requested():
                    break
        if stop_requested():
            break

    async for event in converter.flush():
        yield event


class StreamConverter:
    """Convert one LangGraph invocation stream into Responses events.

    One converter is created per Responses call. It caches transient conversion
    state, such as a partially built message or IDs used for deduplication.
    """

    def __init__(self, stream: ResponseEventStream) -> None:
        self._stream = stream
        self._message_builder: Any = None
        self._text_builder: Any = None
        self._text_buffer: list[str] = []
        self._reasoning_builder: Any = None
        self._reasoning_part_builder: Any = None
        self._reasoning_buffer: list[str] = []
        self._emitted_tool_call_ids: set[str] = set()
        self._emitted_tool_output_call_ids: set[str] = set()
        # Id of the AI message currently streamed into open output items.
        self._current_message_id: str | None = None

    async def checkpoint(self) -> AsyncIterator[Any]:
        """Close partial output and emit an Agent Server checkpoint event."""
        async for event in self.flush():
            yield event
        yield self._stream.checkpoint()

    async def handle_message_chunk(self, payload: Any) -> AsyncIterator[Any]:
        """Handle a payload from ``stream_mode="messages"``.

        The payload carries either one token chunk of a streaming chat model
        response or a whole :class:`AIMessage` (non-streaming LLM call, or a
        node that built the message itself). Payloads sharing a message id
        accumulate into a single ``message`` output item; a different id
        closes the open item and starts a new one.
        """
        message = _extract_ai_message(payload)
        if message is None:
            return

        message_id = message.id if isinstance(message.id, str) and message.id else None
        if (
            message_id is not None
            and self._current_message_id is not None
            and message_id != self._current_message_id
        ):
            async for event in self._close_open_reasoning():
                yield event
            async for event in self._close_open_message():
                yield event
        if message_id is not None:
            self._current_message_id = message_id

        for fragment in extract_reasoning_summary_fragments(message.content):
            async for event in self._emit_reasoning_fragment(fragment):
                yield event

        text = extract_text(message.content)
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
                    # Assistant text is not emitted here: LangGraph already
                    # published this message on the ``messages`` channel.
                    for call in message.tool_calls or []:
                        async for event in self._emit_tool_call(call):
                            yield event
                elif isinstance(message, ToolMessage):
                    async for event in self._emit_tool_output(message):
                        yield event

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
        self._current_message_id = None

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


def _extract_checkpoint_ref(payload: Any) -> CheckpointRef | None:
    """Extract the runnable config from a LangGraph checkpoint event."""
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    return HostingRunnableConfig(cast(RunnableConfig, config)).checkpoint_ref


def _extract_ai_message(payload: Any) -> AIMessage | None:
    """Pull an ``AIMessage`` out of a ``messages`` payload.

    Accepts :class:`AIMessage` and its :class:`AIMessageChunk` subclass so
    both token chunks and whole messages are surfaced. Other message types
    (notably :class:`ToolMessage`, which LangGraph also publishes on this
    channel) are ignored here and handled from the ``updates`` channel.
    """
    if isinstance(payload, AIMessage):
        return payload
    if isinstance(payload, tuple) and payload:
        candidate = payload[0]
        if isinstance(candidate, AIMessage):
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
