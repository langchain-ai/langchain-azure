# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test fixtures: tiny in-process LangGraph builders used to drive the host classes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Annotated, Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
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


def make_echo_graph() -> CompiledStateGraph:
    """Return a compiled graph that echoes the user's message as ``Echo: ...``."""

    async def echo(state: _MessagesState) -> dict[str, Any]:
        text = _last_user_text(state["messages"])
        return {"messages": [AIMessage(content=f"Echo: {text}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile()


def make_checkpointed_echo_graph() -> CompiledStateGraph:
    """Return an echo graph compiled with a LangGraph checkpointer."""

    async def echo(state: _MessagesState) -> dict[str, Any]:
        text = _last_user_text(state["messages"])
        return {"messages": [AIMessage(content=f"Echo: {text}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=InMemorySaver())


def make_streaming_graph() -> CompiledStateGraph:
    """Return a graph-shaped fixture that emits chunked tokens."""
    tokens = ["Hello", ", ", "world", "!"]

    class _StreamingGraph:
        builder = SimpleNamespace(state_schema=_MessagesState)

        async def astream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[AIMessageChunk]:
            del args, kwargs
            for token in tokens:
                yield AIMessageChunk(content=token)

        async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return {"messages": [AIMessage(content="".join(tokens))]}

    return cast(CompiledStateGraph, _StreamingGraph())


def make_custom_state_graph() -> CompiledStateGraph:
    """Return a graph with a state schema that lacks a ``messages`` field."""

    async def noop(state: _NoMessagesState) -> dict[str, Any]:
        return {}

    builder = StateGraph(_NoMessagesState)
    builder.add_node("noop", noop)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile()
