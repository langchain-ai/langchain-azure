"""Convert LangChain messages to Foundry agent evaluation format.

Foundry agent evaluators (TaskCompletion, TaskAdherence, ToolCallAccuracy,
etc.) expect conversations in OpenAI message schema with ``query`` and
``response`` arrays. This module bridges LangChain's message types to that
format.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def messages_to_foundry_format(
    messages: Sequence[BaseMessage],
    *,
    tool_definitions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert a LangChain message sequence to Foundry evaluator input.

    Splits the conversation into ``query`` (system + user messages up to
    the last user message) and ``response`` (everything after, typically
    assistant and tool messages).

    Args:
        messages: Ordered list of LangChain BaseMessage objects.
        tool_definitions: Optional tool schemas to include in the
            evaluator input (used by ToolCallAccuracy, ToolSelection,
            etc.).

    Returns:
        Dict with ``query``, ``response``, and optionally
        ``tool_definitions`` keys, ready for Foundry evaluators.
    """
    if not messages:
        return {"query": "", "response": "", "tool_definitions": None}

    # Find the split point: last user message
    split_idx = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, (HumanMessage, SystemMessage)):
            split_idx = i + 1

    query_messages = list(messages[:split_idx])
    response_messages = list(messages[split_idx:])

    # If there are no response messages but we have AI messages in query,
    # move all AI/Tool messages after the last Human to response
    if not response_messages:
        for i in range(len(query_messages) - 1, -1, -1):
            if isinstance(query_messages[i], HumanMessage):
                response_messages = query_messages[i + 1 :]
                query_messages = query_messages[: i + 1]
                break

    query = _convert_to_foundry_array(query_messages)
    response = _convert_to_foundry_array(response_messages)

    result: dict[str, Any] = {
        "query": query if len(query) > 1 else (query[0]["content"] if query else ""),
        "response": response
        if len(response) > 1
        else (response[0]["content"] if response else ""),
        "tool_definitions": tool_definitions,
    }
    return result


def _convert_to_foundry_array(
    messages: Sequence[BaseMessage],
) -> list[dict[str, Any]]:
    """Convert a list of LangChain messages to Foundry message array format."""
    result = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for msg in messages:
        converted = _convert_single_message(msg, timestamp=now)
        if converted is not None:
            result.append(converted)

    return result


def _convert_single_message(
    msg: BaseMessage,
    *,
    timestamp: str,
) -> dict[str, Any] | None:
    """Convert a single LangChain message to Foundry format."""
    if isinstance(msg, SystemMessage):
        return {
            "role": "system",
            "content": _text_content(msg),
        }

    if isinstance(msg, HumanMessage):
        return {
            "createdAt": timestamp,
            "role": "user",
            "content": [{"type": "text", "text": _text_content(msg)}],
        }

    if isinstance(msg, AIMessage):
        content_parts: list[dict[str, Any]] = []

        # Add tool calls if present
        if msg.tool_calls:
            for tc in msg.tool_calls:
                part: dict[str, Any] = {
                    "type": "tool_call",
                    "name": tc["name"],
                    "arguments": tc["args"],
                }
                if "id" in tc:
                    part["tool_call_id"] = tc["id"]
                content_parts.append(part)

        # Add text content if present (and not empty)
        text = _text_content(msg)
        if text:
            content_parts.append({"type": "text", "text": text})

        # If no parts at all, just add the text content
        if not content_parts:
            content_parts.append({"type": "text", "text": ""})

        return {
            "createdAt": timestamp,
            "role": "assistant",
            "content": content_parts,
        }

    if isinstance(msg, ToolMessage):
        tool_result: Any
        try:
            tool_result = (
                json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            )
        except (json.JSONDecodeError, TypeError):
            tool_result = msg.content

        result: dict[str, Any] = {
            "createdAt": timestamp,
            "role": "tool",
            "content": [{"type": "tool_result", "tool_result": tool_result}],
        }
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        return result

    # Unknown message type — skip
    return None


def _text_content(msg: BaseMessage) -> str:
    """Extract text content from a message."""
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        parts = []
        for item in msg.content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return str(msg.content)


def tool_schemas_to_foundry_format(
    tools: Sequence[Any],
) -> list[dict[str, Any]]:
    """Convert LangChain tool objects to Foundry tool_definitions format.

    Args:
        tools: Sequence of LangChain tool objects (must have ``name``,
            ``description``, and ``args_schema`` or ``args`` attributes).

    Returns:
        List of tool definition dicts for Foundry evaluators.
    """
    definitions = []
    for t in tools:
        defn: dict[str, Any] = {
            "name": getattr(t, "name", str(t)),
            "description": getattr(t, "description", ""),
        }
        # Try to get the JSON schema from the tool
        if hasattr(t, "args_schema") and t.args_schema is not None:
            try:
                defn["parameters"] = t.args_schema.model_json_schema()
            except Exception:
                pass
        elif hasattr(t, "args"):
            defn["parameters"] = t.args
        definitions.append(defn)
    return definitions
