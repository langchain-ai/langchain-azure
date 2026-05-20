# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Translate a non-streaming LangGraph result into Responses API events.

A LangGraph ``ainvoke`` returns the final state. For ``MessagesState``
graphs we walk every message produced *during this turn* (i.e. messages
appended after the last :class:`HumanMessage`) and surface them to the
Responses API client as a faithful trace:

- :class:`AIMessage` text content → ``message`` output item with text deltas.
- :class:`AIMessage.tool_calls` entries → ``function_call`` output items.
- :class:`ToolMessage` → ``function_call_output`` output items.

This preserves the call/result trace for ReAct-style agents instead of
returning only the final assistant message.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from azure.ai.agentserver.responses import ResponseEventStream
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from ._utils import extract_text


async def state_to_events(
    state: dict[str, Any] | Any,
    stream: ResponseEventStream,
) -> AsyncIterator[Any]:
    """Yield the Responses API events that summarise a final graph state.

    Walks every message appended after the last :class:`HumanMessage` so
    intermediate tool calls and tool results are surfaced to the client
    (not just the final assistant text).

    Args:
        state: The value returned by ``CompiledStateGraph.ainvoke``.
            Expected shape is a mapping with a ``"messages"`` channel.
        stream: The :class:`ResponseEventStream` to emit events through.

    Yields:
        Responses API event payload dicts.
    """
    for message in _messages_for_this_turn(state):
        if isinstance(message, AIMessage):
            text = extract_text(message.content)
            if text:
                async for event in _emit_message(stream, text):
                    yield event
            for call in message.tool_calls or []:
                async for event in _emit_function_call(stream, call):
                    yield event
        elif isinstance(message, ToolMessage):
            async for event in _emit_function_call_output(stream, message):
                yield event


def _messages_for_this_turn(state: Any) -> list[BaseMessage]:
    """Return only messages produced during the current turn.

    Includes everything after (and excluding) the last
    :class:`HumanMessage` in the channel. When no human message is
    present, returns the entire channel.

    Args:
        state: The graph result.

    Returns:
        The list of messages to emit.
    """
    if isinstance(state, dict):
        messages = list(state.get("messages") or [])
    else:
        messages = list(getattr(state, "messages", None) or [])
    if not messages:
        return []
    last_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            last_human_index = index
    return messages[last_human_index + 1 :]


async def _emit_message(stream: ResponseEventStream, text: str) -> AsyncIterator[Any]:
    message_builder = stream.add_output_item_message()
    yield message_builder.emit_added()
    text_builder = message_builder.add_text_content()
    yield text_builder.emit_added()
    yield text_builder.emit_delta(text)
    yield text_builder.emit_text_done(text)
    yield text_builder.emit_done()
    yield message_builder.emit_done()


async def _emit_function_call(
    stream: ResponseEventStream, call: Any
) -> AsyncIterator[Any]:
    name = str(call.get("name") or "")
    call_id = str(call.get("id") or call.get("call_id") or "")
    args = call.get("args")
    arguments_json = args if isinstance(args, str) else json.dumps(args or {})

    if not name or not call_id:
        return

    fn = stream.add_output_item_function_call(name, call_id)
    yield fn.emit_added()
    if arguments_json:
        yield fn.emit_arguments_delta(arguments_json)
    yield fn.emit_arguments_done(arguments_json)
    yield fn.emit_done()


async def _emit_function_call_output(
    stream: ResponseEventStream, message: ToolMessage
) -> AsyncIterator[Any]:
    call_id = str(getattr(message, "tool_call_id", "") or "")
    if not call_id:
        return
    output_text = extract_text(message.content)
    fn_out = stream.add_output_item_function_call_output(call_id)
    yield fn_out.emit_added(output_text)
    yield fn_out.emit_done(output_text)
