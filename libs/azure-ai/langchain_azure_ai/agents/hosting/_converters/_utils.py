# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Internal helpers: state-schema validation and message text extraction."""

from __future__ import annotations

from typing import Any, Iterable, get_type_hints

from langchain_core.messages import AIMessage, BaseMessage


def is_messages_state_schema(state_schema: Any) -> bool:
    """Return ``True`` when *state_schema* exposes a ``messages`` field.

    LangGraph compiles graphs against a TypedDict (or dataclass-like) state
    schema. The default request/response converters only know how to build
    ``{"messages": [...]}`` payloads, so we fast-fail when the schema does
    not expose that key.

    Args:
        state_schema: The state schema class associated with a compiled graph.

    Returns:
        ``True`` when the schema declares a ``messages`` field.
    """
    fields = _get_typeddict_fields(state_schema)
    return "messages" in fields


def _get_typeddict_fields(schema_class: Any) -> dict[str, Any]:
    try:
        return get_type_hints(schema_class)
    except (TypeError, AttributeError):
        if hasattr(schema_class, "__annotations__"):
            return dict(schema_class.__annotations__)
    return {}


def extract_text(content: Any) -> str:
    """Return the plain-text representation of a LangChain message ``content``.

    LangChain message ``content`` is either a string or a list of content
    parts. String parts are concatenated; non-string parts (images, files,
    tool-use blocks, etc.) are ignored for text extraction.

    Args:
        content: The raw ``content`` field of a LangChain message.

    Returns:
        The flattened text content.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def extract_reasoning_summary_fragments(content: Any) -> list[str]:
    """Return the reasoning summary text fragments in a message ``content``.

    When a chat model is configured to stream reasoning summaries
    (e.g. ``AzureChatOpenAI(reasoning={"summary": "auto"})``), each
    :class:`~langchain_core.messages.AIMessageChunk` carries a list
    ``content`` that may include reasoning blocks shaped
    ``{"type": "reasoning", "summary": [{"type": "summary_text",
    "text": "<delta>"}]}``. The summary text arrives incrementally across
    chunks; an empty fragment (``""``) marks the start of a new summary
    section.

    :func:`extract_text` deliberately ignores these blocks (they carry no
    top-level ``text`` key), so reasoning needs its own extractor.

    Args:
        content: The raw ``content`` field of a LangChain message.

    Returns:
        The ordered list of summary text fragments. Fragments are kept as
        emitted by the model, including empty strings that signal a new
        summary section. Returns an empty list when *content* carries no
        reasoning blocks.
    """
    if not isinstance(content, list):
        return []
    fragments: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "reasoning":
            continue
        summary = part.get("summary")
        if not isinstance(summary, list):
            continue
        for summary_part in summary:
            if not isinstance(summary_part, dict):
                continue
            text = summary_part.get("text")
            if isinstance(text, str):
                fragments.append(text)
    return fragments


def last_ai_message_text(messages: Iterable[Any]) -> str:
    """Return the text content of the last ``AIMessage`` in *messages*.

    Used by the default Invocations and non-streaming Responses converters
    to surface the assistant's final answer after a graph run. When no
    ``AIMessage`` is present the function returns an empty string.

    Args:
        messages: An iterable of LangChain messages, typically the
            ``messages`` channel from a graph result.

    Returns:
        The text content of the last ``AIMessage`` or ``""``.
    """
    last: BaseMessage | None = None
    for message in messages:
        if isinstance(message, AIMessage):
            last = message
    if last is None:
        return ""
    return extract_text(last.content)
