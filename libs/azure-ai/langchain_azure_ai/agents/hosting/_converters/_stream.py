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
   converter routes payloads by message id, including interleaved parallel
   streams. A shared queue publishes all output item types in arrival order.
   Only its head streams immediately; later items wait for completion so
   clients never need to revisit an earlier output index.
2. Reasoning summaries (emitted when the chat model is configured with
   ``reasoning={"summary": "auto"}``) arrive in the same
   :class:`AIMessageChunk` payloads as ``reasoning`` content blocks.
   They are streamed through a ``reasoning`` output item with
   ``reasoning_summary_text.delta`` events. At most one reasoning item
   is published at a time, while each message owns its pending reasoning.
   A message's text or completion ends only that message's reasoning item.
3. A final message chunk or an ``updates`` payload closes only matching
   message outputs. Checkpoints and stream termination drain the same queue.
   Node updates also surface:

   - :class:`AIMessage.tool_calls` → ``function_call`` output items
     (with the full JSON arguments emitted as a single
     ``function_call_arguments.delta`` followed by ``done``).
   - :class:`ToolMessage` → ``function_call_output`` output items.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

from azure.ai.agentserver.responses import ResponseEventStream
from azure.ai.agentserver.responses.models import ResponseUsage
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig

from .._responses import CheckpointRef, HostingRunnableConfig, TaskStorageManager
from ._text import _TextMessageEmitter, text_deltas
from ._utils import extract_reasoning_summary_fragments, tool_output


async def stream_graph_to_events(
    graph_stream: AsyncIterator[Any],
    stream: ResponseEventStream,
    *,
    cancellation_signal: asyncio.Event,
    shutdown_signal: asyncio.Event | None = None,
    usage: UsageAccumulator | None = None,
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
        usage: Optional accumulator for LangChain AI message usage metadata.

    Yields:
        Responses API event payload dicts.
    """
    converter = StreamConverter(stream, usage=usage)
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


class _InputTokensDetails(TypedDict):
    """Preserve cache-write usage across supported Responses SDK schemas."""

    cached_tokens: int
    cache_write_tokens: int


class UsageAccumulator:
    """Aggregate LangChain usage metadata into Responses API usage."""

    def __init__(self) -> None:
        self._has_usage = False
        self._input_tokens = 0
        self._output_tokens = 0
        self._total_tokens = 0
        self._cached_tokens = 0
        self._cache_write_tokens = 0
        self._reasoning_tokens = 0

    def add(self, message: AIMessage) -> None:
        """Add usage reported by one AI message or message chunk."""
        metadata = _coerce_mapping(
            getattr(message, "usage_metadata", None),
            (
                "input_tokens",
                "prompt_tokens",
                "output_tokens",
                "completion_tokens",
                "total_tokens",
                "input_token_details",
                "input_tokens_details",
                "output_token_details",
                "output_tokens_details",
            ),
        )
        if metadata is None:
            return

        input_tokens = _first_int(metadata, ("input_tokens", "prompt_tokens"))
        output_tokens = _first_int(metadata, ("output_tokens", "completion_tokens"))
        total_tokens = _first_int(metadata, ("total_tokens",))
        input_details = _coerce_mapping(
            metadata.get("input_token_details") or metadata.get("input_tokens_details"),
            (
                "cache_read",
                "cached_tokens",
                "cache_creation",
                "cache_write_tokens",
            ),
        )
        output_details = _coerce_mapping(
            metadata.get("output_token_details")
            or metadata.get("output_tokens_details"),
            ("reasoning", "reasoning_tokens"),
        )
        cached_tokens = _first_int(input_details, ("cache_read", "cached_tokens"))
        cache_write_tokens = _first_int(
            input_details, ("cache_creation", "cache_write_tokens")
        )
        reasoning_tokens = _first_int(output_details, ("reasoning", "reasoning_tokens"))
        if all(
            value is None
            for value in (
                input_tokens,
                output_tokens,
                total_tokens,
                cached_tokens,
                cache_write_tokens,
                reasoning_tokens,
            )
        ):
            return

        self._has_usage = True
        self._input_tokens += input_tokens or 0
        self._output_tokens += output_tokens or 0
        self._total_tokens += (
            total_tokens
            if total_tokens is not None
            else (input_tokens or 0) + (output_tokens or 0)
        )
        self._cached_tokens += cached_tokens or 0
        self._cache_write_tokens += cache_write_tokens or 0
        self._reasoning_tokens += reasoning_tokens or 0

    @property
    def response_usage(self) -> ResponseUsage | None:
        """Return accumulated usage in the standard Responses API shape."""
        if not self._has_usage:
            return None
        input_details: _InputTokensDetails = {
            "cached_tokens": self._cached_tokens,
            "cache_write_tokens": self._cache_write_tokens,
        }
        usage: ResponseUsage = {
            "input_tokens": self._input_tokens,
            "input_tokens_details": input_details,
            "output_tokens": self._output_tokens,
            "output_tokens_details": {"reasoning_tokens": self._reasoning_tokens},
            "total_tokens": self._total_tokens,
        }
        return usage


def _coerce_mapping(
    value: Any,
    attribute_names: Sequence[str],
) -> Mapping[str, Any] | None:
    """Return mapping or model data without assuming one concrete type."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            result = method(exclude_none=True)
        except TypeError:
            result = method()
        if isinstance(result, Mapping):
            return result
    extracted = {
        name: attribute
        for name in attribute_names
        if (attribute := getattr(value, name, None)) is not None
    }
    return extracted or None


def _first_int(
    values: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> int | None:
    """Return the first integer-like value under the requested keys."""
    if values is None:
        return None
    for key in keys:
        value = values.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass
class _PendingOutput:
    operations: deque[Iterator[Any]] = field(default_factory=deque)
    finished: bool = False


class _OutputQueue:
    """Publish complete item lifecycles in arrival order for every output type.

    Builder operations are lazy: only the head allocates SDK IDs/indexes and
    emits events. Producers keep running while later output waits in memory.
    """

    def __init__(self) -> None:
        self._items: dict[tuple[str, str | None], _PendingOutput] = {}

    def add(self, key: tuple[str, str | None], events: Iterator[Any]) -> None:
        """Schedule builder operations for an open output item."""
        self._items.setdefault(key, _PendingOutput()).operations.append(events)

    def finish(self, key: tuple[str, str | None]) -> None:
        """Mark an item's scheduled operations as its complete lifecycle."""
        self._items[key].finished = True

    def drain(self) -> Iterator[Any]:
        """Publish the head's available operations, advancing only when done."""
        while self._items:
            key = next(iter(self._items))
            item = self._items[key]
            while item.operations:
                # Retain the iterator until exhausted, including across yields.
                for event in item.operations[0]:
                    yield event
                item.operations.popleft()
            if not item.finished:
                break
            del self._items[key]


class _ReasoningEmitter:
    """Build one message's reasoning summary through public SDK builders."""

    def __init__(self, stream: ResponseEventStream) -> None:
        self._stream = stream
        self._builder: Any = None
        self._part: Any = None
        self._fragments: list[str] = []

    def add(self, fragments: list[str]) -> Iterator[Any]:
        """Keep summary sections separated, omitting leading empty sections."""
        for fragment in fragments:
            if not fragment and not self._fragments:
                continue
            if self._builder is None:
                self._builder = self._stream.add_output_item_reasoning_item()
                yield self._builder.emit_added()
                self._part = self._builder.add_summary_part()
                yield self._part.emit_added()
            delta = fragment or "\n"
            self._fragments.append(delta)
            yield self._part.emit_text_delta(delta)

    def close(self) -> Iterator[Any]:
        """Finalize only this message's reasoning output."""
        if self._builder is not None:
            yield self._part.emit_text_done("".join(self._fragments))
            yield self._part.emit_done()
            yield self._builder.emit_done()


class StreamConverter:
    """Route graph messages to one ordered output lifecycle queue.

    Message identity owns text/reasoning state; the queue alone decides when
    any output type may publish. Tool calls/results are complete queue items.
    """

    def __init__(
        self,
        stream: ResponseEventStream,
        *,
        usage: UsageAccumulator | None = None,
    ) -> None:
        self._stream = stream
        self._usage = usage or UsageAccumulator()
        self._outputs = _OutputQueue()
        self._emitters: dict[
            tuple[str, str | None], _TextMessageEmitter | _ReasoningEmitter
        ] = {}
        self._emitted_tool_call_ids: set[str] = set()
        self._emitted_tool_output_call_ids: set[str] = set()

    def _finish(self, key: tuple[str, str | None]) -> None:
        emitter = self._emitters.pop(key, None)
        if emitter is not None:
            self._outputs.add(key, emitter.close())
            self._outputs.finish(key)

    def _finish_message(self, message_id: str | None) -> None:
        self._finish(("reasoning", message_id))
        self._finish(("text", message_id))

    async def checkpoint(self) -> AsyncIterator[Any]:
        """Drain all item lifecycles before persisting the SDK checkpoint."""
        async for event in self.flush():
            yield event
        yield self._stream.checkpoint()

    async def handle_message_chunk(self, payload: Any) -> AsyncIterator[Any]:
        """Accumulate per-message output; publish only the queue head."""
        message = _extract_ai_message(payload)
        if message is None:
            return
        self._usage.add(message)
        message_id = message.id or None
        reasoning_key = ("reasoning", message_id)
        fragments = extract_reasoning_summary_fragments(message.content)
        if any(fragments) or (fragments and reasoning_key in self._emitters):
            if reasoning_key not in self._emitters:
                self._emitters[reasoning_key] = _ReasoningEmitter(self._stream)
            reasoning = cast(_ReasoningEmitter, self._emitters[reasoning_key])
            self._outputs.add(reasoning_key, reasoning.add(fragments))
        if deltas := list(text_deltas(message.content)):
            # Text ends this message's reasoning, never another producer's.
            self._finish(reasoning_key)
            text_key = ("text", message_id)
            if text_key not in self._emitters:
                self._emitters[text_key] = _TextMessageEmitter(self._stream)
            text = cast(_TextMessageEmitter, self._emitters[text_key])
            self._outputs.add(text_key, text.add(deltas))
        if not isinstance(message, AIMessageChunk) or message.chunk_position == "last":
            self._finish_message(message_id)
        for event in self._outputs.drain():
            yield event

    async def handle_update(self, payload: Any) -> AsyncIterator[Any]:
        """Complete matching messages and enqueue their tool calls/results."""
        for _, messages in _extract_node_updates(payload):
            # ID-less legacy streams have only the node update as a boundary.
            self._finish_message(None)
            for message in messages:
                if isinstance(message, AIMessage):
                    self._finish_message(message.id or None)
                    for call in message.tool_calls or []:
                        self._queue_tool_call(call)
                elif isinstance(message, ToolMessage):
                    call_id = message.tool_call_id
                    if call_id and call_id not in self._emitted_tool_output_call_ids:
                        self._emitted_tool_output_call_ids.add(call_id)
                        key = ("function_call_output", call_id)
                        self._outputs.add(
                            key,
                            self._tool_output_events(
                                call_id, tool_output(message.content)
                            ),
                        )
                        self._outputs.finish(key)
            for event in self._outputs.drain():
                yield event

    async def flush(self) -> AsyncIterator[Any]:
        """Complete every pending output in order; repeated calls are no-ops."""
        for key in list(self._emitters):
            self._finish(key)
        for event in self._outputs.drain():
            yield event

    def _queue_tool_call(self, call: Any) -> None:
        name = str(call.get("name") or "")
        call_id = str(call.get("id") or call.get("call_id") or "")
        if not name or not call_id or call_id in self._emitted_tool_call_ids:
            return
        self._emitted_tool_call_ids.add(call_id)
        args = call.get("args")
        arguments = args if isinstance(args, str) else json.dumps(args or {})
        key = ("function_call", call_id)
        self._outputs.add(key, self._tool_call_events(name, call_id, arguments))
        self._outputs.finish(key)

    def _tool_call_events(
        self, name: str, call_id: str, arguments: str
    ) -> Iterator[Any]:
        builder = self._stream.add_output_item_function_call(name, call_id)
        yield builder.emit_added()
        if arguments:
            yield builder.emit_arguments_delta(arguments)
        yield builder.emit_arguments_done(arguments)
        yield builder.emit_done()

    def _tool_output_events(self, call_id: str, output: Any) -> Iterator[Any]:
        builder = self._stream.add_output_item_function_call_output(call_id)
        yield builder.emit_added(output)
        yield builder.emit_done(output)


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
