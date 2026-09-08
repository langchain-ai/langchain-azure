# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Convert Responses API requests into LangGraph ``MessagesState`` input."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence, TypeGuard

from azure.ai.agentserver.responses.models import (
    FunctionCallOutputItemParam,
    ItemFunctionToolCall,
    ItemMessage,
    MessageContentInputTextContent,
    MessageContentOutputTextContent,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall

from ._hitl import hitl_call_ids

logger = logging.getLogger(__name__)

_ROLE_TO_MESSAGE_CLS: dict[
    str, type[HumanMessage] | type[SystemMessage] | type[AIMessage]
] = {
    "user": HumanMessage,
    "system": SystemMessage,
    "developer": SystemMessage,
    "assistant": AIMessage,
}


def _is_function_call(item: Any) -> TypeGuard[ItemFunctionToolCall]:
    return isinstance(item, dict) and item.get("type") == "function_call"


def _is_function_call_output(item: Any) -> TypeGuard[FunctionCallOutputItemParam]:
    return isinstance(item, dict) and item.get("type") == "function_call_output"


def _is_message(item: Any) -> TypeGuard[ItemMessage]:
    return isinstance(item, dict) and item.get("type") == "message"


def _is_text_content(
    part: Any,
) -> TypeGuard[MessageContentInputTextContent | MessageContentOutputTextContent]:
    return isinstance(part, dict) and part.get("type") in {
        "input_text",
        "output_text",
    }


def items_to_messages(
    items: Sequence[Any],
    *,
    skip_call_ids: Iterable[str] = (),
) -> list[AnyMessage]:
    """Translate resolved Responses API items into LangChain messages.

    Accepts the typed ``Item`` subtypes returned by
    :meth:`ResponseContext.get_input_items`. Items that do not map cleanly
    to a LangChain message type are skipped.

    HITL wire-protocol items are *always* dropped, whether or not the
    caller lists them in *skip_call_ids*: the ``function_call`` sentinel
    this host emits for a pending interrupt and the
    ``function_call_output`` answering it are transport plumbing, and
    replaying them as a tool-call round trip would put a reserved internal
    function name into the model's context. See :func:`.hitl_call_ids`.

    Args:
        items: Resolved input items from the request.
        skip_call_ids: ``function_call`` / ``function_call_output`` items
            whose ``call_id`` is in this set are skipped. Used by the
            HITL resume path to keep the resume ``function_call_output``
            out of the regular message channel.

    Returns:
        A list of LangChain messages suitable for a ``MessagesState`` graph.
    """
    reserved = hitl_call_ids(items)
    if reserved:
        logger.debug(
            "Dropping %d HITL sentinel call id(s) from graph input.", len(reserved)
        )
    skip = frozenset(skip_call_ids) | reserved
    messages: list[AnyMessage] = []
    index = 0
    while index < len(items):
        item = items[index]
        if _is_function_call(item):
            tool_calls: list[ToolCall] = []
            while index < len(items):
                function_call = items[index]
                if not _is_function_call(function_call):
                    break
                if not skip or function_call["call_id"] not in skip:
                    tool_calls.append(_function_call_to_tool_call(function_call))
                index += 1
            if tool_calls:
                messages.append(AIMessage(content="", tool_calls=tool_calls))
            continue

        if skip and _item_call_id(item) in skip:
            index += 1
            continue
        message = _item_to_message(item)
        if message is not None:
            messages.append(message)
        index += 1
    return messages


def _item_call_id(item: Any) -> str | None:
    if _is_function_call(item):
        return item["call_id"]
    if _is_function_call_output(item):
        return item["call_id"]
    return None


def _item_to_message(item: Any) -> AnyMessage | None:
    if _is_message(item):
        text = _content_to_text(item["content"])
        role = item["role"]
        role_value = getattr(role, "value", role)
        cls = _ROLE_TO_MESSAGE_CLS.get(str(role_value))
        if cls is None:
            return None
        return cls(content=text)

    if _is_function_call(item):
        return AIMessage(
            content="",
            tool_calls=[_function_call_to_tool_call(item)],
        )

    if _is_function_call_output(item):
        output = item["output"]
        if isinstance(output, list):
            output = _content_to_text(output)
        return ToolMessage(content=output or "", tool_call_id=item["call_id"])

    return None


def _function_call_to_tool_call(item: ItemFunctionToolCall) -> ToolCall:
    arguments = item["arguments"]
    try:
        args = json.loads(arguments) if arguments else {}
    except json.JSONDecodeError:
        args = {}
    return ToolCall(id=item["call_id"], name=item["name"], args=args)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if _is_text_content(part):
                text = part["text"]
                if text:
                    parts.append(text)
            elif isinstance(part, dict):
                dict_text = part.get("text")
                if isinstance(dict_text, str):
                    parts.append(dict_text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content) if content is not None else ""


def build_messages_input(
    items: Sequence[Any],
    *,
    instructions: str | None = None,
    skip_call_ids: Iterable[str] = (),
) -> dict[str, list[AnyMessage]]:
    """Build a ``{"messages": [...]}`` LangGraph input from resolved items.

    Prepends a ``SystemMessage`` when *instructions* is non-empty. Empty
    item lists yield an empty messages list — callers should typically have
    at least one user message before invoking the graph.

    Incomplete tool-call sequences are filtered out: any ``AIMessage`` with
    unanswered ``tool_calls`` and any orphan ``ToolMessage`` (whose
    ``tool_call_id`` has no matching preceding ``AIMessage.tool_calls``
    entry) are dropped. This prevents a poisoned conversation store —
    e.g. one containing a ``function_call_output`` from a prior failed
    request — from producing a message list the chat model would reject
    ("messages with role 'tool' must be a response to a preceding message
    with 'tool_calls'").

    Args:
        items: Resolved input items from the request.
        instructions: Optional system instructions from the request.
        skip_call_ids: Forwarded to :func:`items_to_messages`. Used by the
            HITL resume path to filter out the resume ``function_call_output``.

    Returns:
        The ``{"messages": [...]}`` payload accepted by ``MessagesState`` graphs.
    """
    messages: list[AnyMessage] = []
    if instructions:
        messages.append(SystemMessage(content=instructions))
    messages.extend(
        _filter_incomplete_tool_calls(
            items_to_messages(items, skip_call_ids=skip_call_ids)
        )
    )
    return {"messages": messages}


def _filter_incomplete_tool_calls(messages: Sequence[AnyMessage]) -> list[AnyMessage]:
    """Drop incomplete tool-call sequences from a message list.

    Walks the messages once and keeps only those that participate in a
    matched ``AIMessage.tool_calls`` ↔ ``ToolMessage`` pair (or aren't
    tool-call related at all). Specifically:

    - An ``AIMessage`` whose ``tool_calls`` are not all answered by a
      subsequent ``ToolMessage`` is dropped.
    - A ``ToolMessage`` whose ``tool_call_id`` has no matching prior
      ``AIMessage.tool_calls`` entry is dropped.

    This mirrors the defensive filter used by
    ``azure-ai-agentserver-langgraph``.
    """
    tool_responses: set[str] = set()
    for msg in messages:
        if isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None)
            if isinstance(call_id, str) and call_id:
                tool_responses.add(call_id)

    valid_tool_calls: set[str] = set()
    result: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            call_ids = [
                _tool_call_id(tc)
                for tc in msg.tool_calls
                if _tool_call_id(tc) is not None
            ]
            if not call_ids or not all(cid in tool_responses for cid in call_ids):
                continue
            valid_tool_calls.update(call_ids)  # type: ignore[arg-type]
            result.append(msg)
        elif isinstance(msg, ToolMessage):
            call_id = getattr(msg, "tool_call_id", None)
            if not isinstance(call_id, str) or call_id not in valid_tool_calls:
                continue
            result.append(msg)
        else:
            result.append(msg)
    return result


def _tool_call_id(tc: Any) -> str | None:
    if isinstance(tc, dict):
        value = tc.get("id")
    else:
        value = getattr(tc, "id", None)
    return value if isinstance(value, str) and value else None


def build_messages_input_from_text(
    text: str,
    *,
    instructions: str | None = None,
) -> dict[str, list[AnyMessage]]:
    """Build a ``{"messages": [...]}`` payload from a single user text string.

    Convenience wrapper used by the Invocations converter when the body is
    already a plain string.

    Args:
        text: The user's message text.
        instructions: Optional system instructions.

    Returns:
        The ``{"messages": [...]}`` payload.
    """
    messages: list[AnyMessage] = []
    if instructions:
        messages.append(SystemMessage(content=instructions))
    if text:
        messages.append(HumanMessage(content=text))
    return {"messages": messages}
