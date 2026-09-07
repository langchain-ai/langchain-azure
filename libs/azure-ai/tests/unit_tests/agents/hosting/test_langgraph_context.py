"""Regression tests for LangGraph async context propagation in hosting."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import textwrap
from contextvars import Context, ContextVar, copy_context
from typing import Any, Literal
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tracers.langchain import LangChainTracer
from langgraph._internal import _runnable
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langsmith import get_current_run_tree
from typing_extensions import TypedDict

from langchain_azure_ai.agents.hosting import (
    _install_langgraph_async_context_patch,
)


class _State(TypedDict):
    value: str


class _RunRecorder(BaseCallbackHandler):
    run_inline = True

    def __init__(self) -> None:
        self.runs: dict[str, tuple[UUID, UUID | None]] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.runs[kwargs["name"]] = (run_id, parent_run_id)


@pytest.mark.parametrize("stream", [False, True])
async def test_conditional_route_preserves_nested_callback_parentage(
    stream: bool,
) -> None:
    _install_langgraph_async_context_patch()
    recorder = _RunRecorder()
    tracer = LangChainTracer(client=MagicMock())
    trace_runs: dict[str, UUID | None] = {}
    original_run = get_current_run_tree()

    def record_trace(name: str) -> None:
        current_run = get_current_run_tree()
        trace_runs[name] = current_run.id if current_run is not None else None

    async def classifier(state: _State) -> str:
        record_trace("classification")
        return END

    classification = RunnableLambda(classifier, name="classification")

    async def node(state: _State) -> _State:
        record_trace("node")
        return state

    async def route(state: _State) -> str:
        record_trace("route")
        return await classification.ainvoke(state)

    builder = StateGraph(_State)
    builder.add_node("node", node)
    builder.add_edge(START, "node")
    builder.add_conditional_edges("node", route, {END: END})
    graph = builder.compile()
    if stream:
        results = [
            result
            async for result in graph.astream(
                {"value": "hello"},
                {"callbacks": [recorder, tracer]},
                stream_mode="values",
            )
        ]
        assert results[-1]["value"] == "hello"
    else:
        result = await graph.ainvoke(
            {"value": "hello"}, {"callbacks": [recorder, tracer]}
        )
        assert result["value"] == "hello"

    assert recorder.runs["route"][1] == recorder.runs["node"][0]
    assert recorder.runs["classification"][1] == recorder.runs["route"][0]
    for name in ("node", "route", "classification"):
        assert trace_runs[name] == recorder.runs[name][0]
    assert get_current_run_tree() is original_run


@pytest.mark.parametrize("mode", ["callable", "sequence", "stream"])
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_runnable_context_is_isolated_and_restored(
    mode: Literal["callable", "sequence", "stream"],
    outcome: Literal["success", "error", "cancel"],
) -> None:
    marker = ContextVar("node_context_marker", default="caller")
    started = asyncio.Event()
    finished = asyncio.Event()
    outer_config: RunnableConfig = {"configurable": {"request_id": "caller"}}

    async def node(value: str) -> str:
        try:
            assert get_config()["configurable"]["request_id"] == "caller"
            marker.set("node")
            started.set()
            if outcome == "error":
                raise ValueError("expected node failure")
            if outcome == "cancel":
                await asyncio.Event().wait()
            return value
        finally:
            finished.set()

    context_node = _runnable.RunnableCallable(None, node, name="context_node")
    runnable: Runnable = context_node
    if mode != "callable":
        runnable = _runnable.RunnableSeq(
            context_node, RunnableLambda(lambda value: value)
        )

    async def invoke() -> None:
        if mode == "stream":
            assert [chunk async for chunk in runnable.astream("ok")] == [None]
        else:
            assert await runnable.ainvoke("ok") == "ok"

    token = var_child_runnable_config.set(outer_config)
    try:
        if outcome == "cancel":
            pending = asyncio.create_task(invoke())
            try:
                await asyncio.wait_for(started.wait(), 5)
            finally:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
        elif outcome == "error":
            with pytest.raises(ValueError, match="expected node failure"):
                await invoke()
        else:
            await invoke()

        assert finished.is_set()
        assert marker.get() == "caller"
        assert var_child_runnable_config.get() is outer_config
    finally:
        var_child_runnable_config.reset(token)


async def test_concurrent_calls_keep_their_own_context() -> None:
    marker = ContextVar("parallel_context_marker", default="caller")
    both_started = asyncio.Event()
    arrivals = 0
    original_config = var_child_runnable_config.get()

    async def node(value: str) -> str:
        nonlocal arrivals
        assert get_config()["configurable"]["request_id"] == value
        marker.set(value)
        arrivals += 1
        if arrivals == 2:
            both_started.set()
        await both_started.wait()
        assert marker.get() == value
        assert get_config()["configurable"]["request_id"] == value
        return value

    runnable = _runnable.RunnableCallable(None, node)
    results = await asyncio.wait_for(
        asyncio.gather(
            runnable.ainvoke("left", {"configurable": {"request_id": "left"}}),
            runnable.ainvoke("right", {"configurable": {"request_id": "right"}}),
        ),
        5,
    )
    assert results == ["left", "right"]
    assert marker.get() == "caller"
    assert var_child_runnable_config.get() is original_config


@pytest.mark.parametrize("context_kind", ["implicit", "none", "copy", "empty"])
async def test_task_adapter_inherits_context_and_preserves_name(
    context_kind: str,
) -> None:
    marker = ContextVar("task_context_marker", default="unset")
    token = marker.set("caller")

    async def read_marker() -> tuple[str, str]:
        current_task = asyncio.current_task()
        assert current_task is not None
        return marker.get(), current_task.get_name()

    try:
        kwargs: dict[str, Any] = {"name": "context-probe"}
        expected = "caller"
        if context_kind == "none":
            kwargs["context"] = None
        elif context_kind == "copy":
            context = copy_context()
            context.run(marker.set, "copied")
            kwargs["context"] = context
            expected = "copied"
        elif context_kind == "empty":
            kwargs["context"] = Context()
            expected = "unset"

        task = _runnable.asyncio.create_task(read_marker(), **kwargs)
        assert isinstance(task, asyncio.Task)
        assert await task == (expected, "context-probe")
        assert marker.get() == "caller"
    finally:
        marker.reset(token)


def test_installation_is_idempotent_and_scoped() -> None:
    program = textwrap.dedent(
        """
        import asyncio
        import sys
        from langgraph._internal import _runnable

        original_asyncio = _runnable.asyncio
        original_create_task = asyncio.create_task
        original_methods = (
            _runnable.RunnableCallable.ainvoke,
            _runnable.RunnableSeq.ainvoke,
            _runnable.RunnableSeq.astream,
        )

        from langchain_azure_ai.agents.hosting import (
            _install_langgraph_async_context_patch,
        )

        installed_asyncio = _runnable.asyncio
        assert asyncio.create_task is original_create_task
        assert original_methods == (
            _runnable.RunnableCallable.ainvoke,
            _runnable.RunnableSeq.ainvoke,
            _runnable.RunnableSeq.astream,
        )
        if sys.version_info >= (3, 11):
            assert installed_asyncio is original_asyncio
        else:
            assert installed_asyncio is not original_asyncio
            assert _runnable.ASYNCIO_ACCEPTS_CONTEXT is True
            assert installed_asyncio.get_running_loop is asyncio.get_running_loop

        _install_langgraph_async_context_patch()
        _install_langgraph_async_context_patch()
        assert _runnable.asyncio is installed_asyncio
        assert asyncio.create_task is original_create_task
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
