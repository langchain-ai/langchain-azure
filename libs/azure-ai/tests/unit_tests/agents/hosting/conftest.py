# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test fixtures: tiny in-process LangGraph builders used to drive the host classes."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from copy import deepcopy
from types import SimpleNamespace
from typing import Annotated, Any, cast

import pytest
from azure.ai.agentserver.core import get_request_context
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
from langgraph.types import Interrupt
from typing_extensions import TypedDict


@pytest.fixture(autouse=True)
def foundry_state_stores(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Replace FoundryStateStore with a process-local store keyed by name."""
    stores: dict[str, dict[str, Any]] = {}

    class FakeFoundryStateStore:
        def __init__(self, name: str) -> None:
            self.name = name

        @classmethod
        async def get_or_create(cls, name: str, **_: Any) -> "FakeFoundryStateStore":
            stores.setdefault(name, {})
            return cls(name)

        async def __aenter__(self) -> "FakeFoundryStateStore":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

        async def get_item(self, key: str) -> SimpleNamespace | None:
            value = stores[self.name].get(key)
            return SimpleNamespace(value=deepcopy(value)) if value is not None else None

        async def set_item(
            self,
            key: str,
            value: dict[str, Any],
            **_: Any,
        ) -> SimpleNamespace:
            stores[self.name][key] = deepcopy(value)
            return SimpleNamespace(etag='"test"')

    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._responses."
        "conversation_chain_storage_manager.FoundryStateStore",
        FakeFoundryStateStore,
    )
    return stores


@pytest.fixture(autouse=True)
def invocation_state_store(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[tuple[str | None, str], dict[str, Any]]:
    """Replace invocation state with process-local, user-partitioned storage."""
    records: dict[tuple[str | None, str], dict[str, Any]] = {}

    class FakeInvocationStateStore:
        async def get(self, invocation_id: str) -> dict[str, Any] | None:
            key = (get_request_context().user_id, invocation_id)
            value = records.get(key)
            return deepcopy(value) if value is not None else None

        async def set(self, envelope: dict[str, Any]) -> None:
            invocation_id = str(envelope["id"])
            key = (get_request_context().user_id, invocation_id)
            current = records.get(key)
            current_sequence = (
                current.get("sequence_number", -1) if current is not None else -1
            )
            if current_sequence >= envelope.get("sequence_number", -1):
                return
            records[key] = deepcopy(envelope)

    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._invoke_host.create_invocation_state_store",
        lambda **_: FakeInvocationStateStore(),
    )
    return records


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


def make_checkpointed_two_node_graph() -> CompiledStateGraph:
    """Return a checkpointed graph with distinct plan and research updates."""

    async def plan(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="plan")]}

    async def research(state: _MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage(content="research")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("plan", plan)
    builder.add_node("research", research)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", END)
    return builder.compile(checkpointer=InMemorySaver())


def make_checkpointed_steering_graph(
    first_turn_started: threading.Event,
    cancellation_started: threading.Event | None = None,
    release_cancellation: threading.Event | None = None,
) -> CompiledStateGraph:
    """Return a graph whose first turn blocks until hosting cancels it."""

    async def echo(state: _MessagesState) -> dict[str, Any]:
        text = _last_user_text(state["messages"])
        if text == "first":
            first_turn_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if cancellation_started is not None:
                    cancellation_started.set()
                if release_cancellation is not None:
                    await asyncio.to_thread(release_cancellation.wait, 5)
                raise
        return {"messages": [AIMessage(content=f"Echo: {text}")]}

    builder = StateGraph(_MessagesState)
    builder.add_node("echo", echo)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=InMemorySaver())


def make_streaming_graph(
    captured_config: dict[str, Any] | None = None,
) -> CompiledStateGraph:
    """Return a graph-shaped fixture that emits chunked tokens."""
    tokens = ["Hello", ", ", "world", "!"]

    class _StreamingGraph:
        builder = SimpleNamespace(state_schema=_MessagesState)

        async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            del args
            if captured_config is not None:
                captured_config.update(kwargs["config"])
            if isinstance(kwargs.get("stream_mode"), list):
                for token in tokens:
                    yield "messages", (AIMessageChunk(content=token), {})
                yield "values", {"messages": [AIMessage(content="".join(tokens))]}
                return
            for token in tokens:
                yield AIMessageChunk(content=token)

        async def ainvoke(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return {"messages": [AIMessage(content="".join(tokens))]}

    return cast(CompiledStateGraph, _StreamingGraph())


def make_recovery_probe_graph(
    captured: dict[str, Any],
    pending_interrupt: Interrupt | None = None,
) -> CompiledStateGraph:
    """Return a graph-shaped fixture that records recovered invocation input."""

    class _RecoveryGraph:
        builder = SimpleNamespace(state_schema=_MessagesState)
        checkpointer = object()

        async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[Any]:
            captured["input"] = graph_input
            captured["config"] = kwargs["config"]
            yield "values", {"messages": [AIMessage(content="Recovered")]}
            if pending_interrupt is not None:
                yield "updates", {"__interrupt__": (pending_interrupt,)}

        async def aget_state(self, config: dict[str, Any]) -> Any:
            captured["state_config"] = config
            if pending_interrupt is None:
                return SimpleNamespace(tasks=())
            task = SimpleNamespace(result=None, interrupts=(pending_interrupt,))
            return SimpleNamespace(tasks=(task,))

    return cast(CompiledStateGraph, _RecoveryGraph())


def make_shutdown_checkpoint_graph() -> CompiledStateGraph:
    """Return a graph that publishes a checkpoint as shutdown is signalled."""

    class _ShutdownCheckpointGraph:
        builder = SimpleNamespace(state_schema=_MessagesState)
        checkpointer = object()

        async def astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            del args
            context = kwargs["config"]["configurable"]["invocation_context"]
            context.shutdown.set()
            yield (
                "checkpoints",
                {
                    "config": {
                        "configurable": {
                            "thread_id": "shutdown-thread",
                            "checkpoint_id": "checkpoint-ready",
                        }
                    }
                },
            )

    return cast(CompiledStateGraph, _ShutdownCheckpointGraph())


def make_custom_state_graph() -> CompiledStateGraph:
    """Return a graph with a state schema that lacks a ``messages`` field."""

    async def noop(state: _NoMessagesState) -> dict[str, Any]:
        return {}

    builder = StateGraph(_NoMessagesState)
    builder.add_node("noop", noop)
    builder.add_edge(START, "noop")
    builder.add_edge("noop", END)
    return builder.compile()
