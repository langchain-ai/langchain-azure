# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test fixtures: tiny in-process LangGraph builders used to drive the host classes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class _MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class _NoMessagesState(TypedDict):
    name: str


def _last_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
    return ""


def make_echo_graph():
    """Return a compiled graph that echoes the user's message as ``Echo: ...``."""

    async def echo(state: _MessagesState) -> dict[str, Any]:
        text = _last_user_text(state["messages"])
        return {"messages": [AIMessage(content=f"Echo: {text}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile()


def make_streaming_graph():
    """Return a compiled graph whose node yields chunked tokens via writer events.

    LangGraph propagates ``AIMessageChunk`` instances yielded from
    ``stream_mode="messages"``-aware nodes. The simplest deterministic
    way to emit chunks is to stream them through a custom node returning
    multiple updates — here we just return a final aggregated AI message
    plus a set of per-token chunks delivered via the ``messages`` channel
    using ``AIMessageChunk``.
    """
    tokens = ["Hello", ", ", "world", "!"]

    async def streamer(state: _MessagesState) -> AsyncIterator[dict[str, Any]]:
        accumulated = ""
        for token in tokens:
            accumulated += token
            yield {"messages": [AIMessageChunk(content=token)]}
        # Also produce a final coherent AIMessage so non-streaming paths see it.
        yield {"messages": [AIMessage(content=accumulated)]}

    async def collect(state: _MessagesState) -> dict[str, Any]:
        return {}

    builder = StateGraph(_MessagesState)
    builder.add_node("collect", collect)
    builder.add_edge(START, "collect")
    builder.add_edge("collect", END)
    return builder.compile()


def make_custom_state_graph():
    """Return a graph with a state schema that lacks a ``messages`` field."""

    async def noop(state: _NoMessagesState) -> dict[str, Any]:
        return {}

    builder = StateGraph(_NoMessagesState)
    builder.add_node("noop", noop)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile()
