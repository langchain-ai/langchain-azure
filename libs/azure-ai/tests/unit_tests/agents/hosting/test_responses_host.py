# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end tests for ``ResponsesHostServer`` via Starlette TestClient."""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import threading
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from anyio import create_task_group

pytest.importorskip("azure.ai.agentserver.responses")
pytest.importorskip("starlette")

from azure.ai.agentserver.core.streaming import streams
from azure.ai.agentserver.responses.models import (
    CreateResponse,
    ItemMessage,
    MessageContentInputTextContent,
    ResponseObject,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Interrupt
from starlette.testclient import TestClient
from starlette.types import Message

from langchain_azure_ai.agents.hosting import (
    HostingFeature,
    ResponsesHostServer,
    ResponsesServerOptions,
)
from langchain_azure_ai.agents.hosting._responses import (
    CONVERSATION_CHAIN_STORE_PREFIX,
    CONVERSATION_CHECKPOINT_KEY,
    METADATA_LANGGRAPH_CHECKPOINT_ID,
    METADATA_LANGGRAPH_ROOT_RESPONSE_ID,
    METADATA_LANGGRAPH_THREAD_ID,
    CheckpointRef,
    FoundryConversationChainStore,
    HostingRunnableConfig,
)
from langchain_azure_ai.agents.hosting._responses_host import (
    METADATA_STEERABLE_CONVERSATION,
)

from .conftest import (  # noqa: E402
    make_checkpointed_echo_graph,
    make_checkpointed_two_node_graph,
    make_custom_state_graph,
    make_echo_graph,
    make_recovery_probe_graph,
    make_streaming_graph,
)


def _client(server: ResponsesHostServer) -> TestClient:
    return TestClient(server.app)


@pytest.mark.parametrize("ending", ["success", "failure", "cancel", "close"])
async def test_handler_owns_clients_through_generator_exit(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents.hosting._storage import get_state_store

    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    server = ResponsesHostServer(make_echo_graph())
    settled = []

    async def handle(*args: Any) -> AsyncGenerator[Any, None]:
        try:
            async with get_state_store(server, FoundryStateStore, "chain"):
                pass
            yield "admitted"
            async with get_state_store(server, FoundryStateStore, "chain"):
                store.aclose.assert_not_awaited()
                if ending == "failure":
                    raise RuntimeError("failed")
                if ending == "cancel":
                    raise asyncio.CancelledError()
        finally:
            settled.append(True)

    monkeypatch.setattr(server, "handle_create", handle)
    events = server._handle_create_async_gen(_request(), _context(), asyncio.Event())
    assert await anext(events) == "admitted"
    store.aclose.assert_not_awaited()
    if ending == "close":
        await cast(AsyncGenerator[Any, None], events).aclose()
    else:
        expected = {
            "success": StopAsyncIteration,
            "failure": RuntimeError,
            "cancel": asyncio.CancelledError,
        }[ending]
        with pytest.raises(expected):
            await anext(events)
    assert settled == [True]
    assert acquire.await_count == 1
    store.aclose.assert_awaited_once()


@pytest.mark.parametrize("boundary", ["graph", "handler"])
@pytest.mark.parametrize("primary_kind", ["error", "recovery", "generator_exit"])
@pytest.mark.parametrize("cancelled_close", [False, True])
async def test_iterator_cleanup_preserves_primary_error(
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    primary_kind: str,
    cancelled_close: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from azure.ai.agentserver.responses import ResponseExitForRecovery

    from langchain_azure_ai.agents.hosting._responses_host import _closing_stream

    primary = {
        "error": ValueError("execution failed"),
        "recovery": ResponseExitForRecovery(),
        "generator_exit": GeneratorExit(),
    }[primary_kind]

    async def close() -> None:
        if cancelled_close:
            worker = asyncio.create_task(asyncio.sleep(0))
            worker.cancel("close worker cancelled")
            await worker
        raise RuntimeError("close failed")

    class Iterator:
        aclose = AsyncMock(side_effect=close)

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            raise primary

    iterator = Iterator()
    with pytest.raises(type(primary)) as caught:
        if boundary == "graph":
            async with _closing_stream(iterator):
                raise primary
        else:
            server = ResponsesHostServer(make_echo_graph())
            monkeypatch.setattr(server, "handle_create", lambda *args: iterator)
            await anext(
                server._handle_create_async_gen(_request(), _context(), asyncio.Event())
            )
    assert caught.value is primary
    task = asyncio.current_task()
    assert task is not None and task.cancelling() == 0
    assert "Hosting cleanup failed during handler exit" in caplog.text
    iterator.aclose.assert_awaited_once()


@pytest.mark.parametrize("custom_handler", [False, True])
@pytest.mark.parametrize("file_provider", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
async def test_background_recovery_survives_client_close_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    custom_handler: bool,
    file_provider: bool,
    close_fails: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore
    from azure.ai.agentserver.responses import (
        FileResponseStore,
        InMemoryResponseProvider,
        PlatformContext,
        ResponseContext,
        ResponseExitForRecovery,
    )
    from azure.ai.agentserver.responses.hosting._orchestrator import (
        ResponseExecution,
        ResponseModeFlags,
        _run_background_non_stream,
    )

    from langchain_azure_ai.agents.hosting._storage import get_state_store

    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    provider = (
        FileResponseStore(storage_dir=str(tmp_path))
        if file_provider
        else InMemoryResponseProvider()
    )
    request = _request(background=True, store=True, stream=False)
    modes = ResponseModeFlags(stream=False, store=True, background=True)
    context = ResponseContext(
        response_id="resp-recovery",
        mode_flags=modes,
        request=request,
        provider=provider,
        input_items=[],
        platform_context=PlatformContext(),
    )
    context._task_context = cast(Any, SimpleNamespace())
    recovery = AsyncMock(wraps=context.exit_for_recovery)
    monkeypatch.setattr(context, "exit_for_recovery", recovery)
    options = ResponsesServerOptions(resilient_background=True)
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        store=provider,
        options=ResponsesServerOptions(resilient_background=file_provider),
    )
    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    if close_fails:
        store.aclose.side_effect = RuntimeError("client close failed")
    monkeypatch.setattr(
        FoundryStateStore, "get_or_create", AsyncMock(return_value=store)
    )

    async def graph_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async with get_state_store(server, FoundryStateStore, "checkpoint"):
            pass
        yield "messages", (AIMessage(content="partial"), {})
        context.shutdown.set()
        yield "updates", {}

    async def handler(*args: Any) -> AsyncGenerator[Any, None]:
        stream = server._new_stream(request, context)
        yield stream.emit_created()
        yield stream.emit_in_progress()
        async with get_state_store(server, FoundryStateStore, "checkpoint"):
            pass
        yield stream.checkpoint()
        context.shutdown.set()
        await context.exit_for_recovery()

    if custom_handler:
        monkeypatch.setattr(server, "handle_create", handler)
    else:
        monkeypatch.setattr(server, "build_runnable_config", AsyncMock(return_value={}))
        monkeypatch.setattr(
            server, "build_input", AsyncMock(return_value={"messages": []})
        )
        monkeypatch.setattr(server.graph, "astream", graph_stream)
    signal = asyncio.Event()
    record = ResponseExecution(
        response_id=context.response_id,
        mode_flags=modes,
        response_context=context,
        cancel_signal=signal,
    )
    with pytest.raises(ResponseExitForRecovery):
        await _run_background_non_stream(
            create_fn=server._handle_create_async_gen,
            parsed=request,
            context=context,
            cancellation_signal=signal,
            record=record,
            response_id=context.response_id,
            agent_reference={"type": "agent_reference", "name": "test"},
            model="test",
            provider=provider,
            runtime_options=options,
        )
    recovery.assert_awaited_once_with()
    store.aclose.assert_awaited_once()
    assert record.status == "in_progress"
    if file_provider:
        provider = FileResponseStore(storage_dir=str(tmp_path))
    assert (await provider.get_response(context.response_id))["status"] == "in_progress"
    if close_fails:
        assert "client close failed" in caplog.text


@pytest.mark.parametrize("close_fails", [False, True])
async def test_anyio_cancelled_handler_waits_for_cleanup_without_spinning(
    monkeypatch: pytest.MonkeyPatch,
    close_fails: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents.hosting._storage import get_state_store

    started = asyncio.Event()
    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    server = ResponsesHostServer(make_echo_graph())
    cancellations = []
    wait_calls = 0
    wait = asyncio.wait

    def counted_wait(*args: Any, **kwargs: Any) -> Any:
        nonlocal wait_calls
        wait_calls += 1
        return wait(*args, **kwargs)

    async def close() -> None:
        await asyncio.sleep(0.02)
        if close_fails:
            raise RuntimeError("close failed")

    async def handle(*args: Any) -> AsyncGenerator[Any, None]:
        async with get_state_store(server, FoundryStateStore, "store"):
            pass
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            cancellations.append(exc)
            raise
        yield None

    async def consume() -> None:
        try:
            async for _ in server._handle_create_async_gen(
                _request(), _context(), asyncio.Event()
            ):
                pytest.fail("Cancelled handler must not yield output")
        except asyncio.CancelledError as exc:
            assert exc is cancellations[0]
            raise

    store.aclose.side_effect = close
    monkeypatch.setattr(server, "handle_create", handle)
    monkeypatch.setattr(asyncio, "wait", counted_wait)
    async with create_task_group() as group:
        group.start_soon(consume)
        await started.wait()
        group.cancel_scope.cancel()
    assert len(cancellations) == 1
    assert wait_calls < 10, "Cleanup must wait instead of retrying level cancellation"
    acquire.assert_awaited_once()
    store.aclose.assert_awaited_once()
    if close_fails:
        assert "close failed" in caplog.text


@pytest.mark.parametrize("ending", ["complete", "close", "cancel"])
async def test_handler_context_is_bound_per_iteration_across_tasks(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents import hosting
    from langchain_azure_ai.agents.hosting._checkpoint._checkpoint_reuse import (
        _checkpoint_reads,
    )
    from langchain_azure_ai.agents.hosting._storage import get_state_store
    from langchain_azure_ai.agents.hosting._storage.state_store_factory import (
        _current_factory,
    )

    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    server = ResponsesHostServer(make_echo_graph())
    monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "test")
    monkeypatch.setattr(hosting, "_process_hosting_features", HostingFeature.NONE)
    factories = []
    checkpoint_contexts = []
    expected_features = HostingFeature.RESPONSES | HostingFeature.HITL

    async def handle(*args: Any) -> AsyncGenerator[Any, None]:
        factory = _current_factory.get()
        assert factory is not None
        factories.append(factory)
        checkpoints = _checkpoint_reads.get()
        assert checkpoints is not None
        checkpoint_contexts.append(checkpoints)
        try:
            async with get_state_store(server, FoundryStateStore, "store"):
                pass
            hosting._add_request_hosting_features(HostingFeature.HITL)
            yield "first"
            assert _current_factory.get() is factory
            assert _checkpoint_reads.get() is checkpoints
            assert hosting.get_hosting_features() == expected_features
            if ending == "cancel":
                raise asyncio.CancelledError("stop")
            yield "second"
        finally:
            assert _current_factory.get() is factory
            assert _checkpoint_reads.get() is checkpoints
            assert checkpoints.active
            assert hosting.get_hosting_features() == expected_features
            async with get_state_store(server, FoundryStateStore, "store"):
                store.aclose.assert_not_awaited()

    monkeypatch.setattr(server, "handle_create", handle)
    iterator = server._handle_create_async_gen(_request(), _context(), asyncio.Event())

    async def advance() -> Any:
        return await anext(iterator)

    try:
        assert await asyncio.create_task(advance()) == "first"
        assert _current_factory.get() is None
        assert _checkpoint_reads.get() is None
        if ending == "close":
            await cast(AsyncGenerator[Any, None], iterator).aclose()
        elif ending == "cancel":
            with pytest.raises(asyncio.CancelledError, match="stop"):
                await asyncio.create_task(advance())
        else:
            assert await asyncio.create_task(advance()) == "second"
            with pytest.raises(StopAsyncIteration):
                await asyncio.create_task(advance())
        assert _current_factory.get() is None
        assert hosting._request_hosting_features.get() == HostingFeature.NONE
        acquire.assert_awaited_once()
        store.aclose.assert_awaited_once()
        assert factories[0].session is not None and factories[0].session.closed
        assert not checkpoint_contexts[0].active
        assert checkpoint_contexts[0].cache is None
    finally:
        await factories[0].stack.aclose()


@pytest.mark.parametrize("other_borrower", [False, True])
async def test_cross_task_close_does_not_cancel_handler_tasks(
    monkeypatch: pytest.MonkeyPatch, other_borrower: bool
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents.hosting._storage import (
        get_state_store,
        state_store_factory,
    )

    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    server = ResponsesHostServer(make_echo_graph())
    monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "test")
    held = []
    factories = []
    children: list[asyncio.Task[None]] = []
    started, advanced, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def values() -> AsyncGenerator[str, None]:
        async with get_state_store(server, FoundryStateStore, "store"):
            yield "item"

    async def handle(*args: Any) -> AsyncGenerator[Any, None]:
        factory = state_store_factory._current_factory.get()
        assert factory is not None
        factories.append(factory)
        for _ in range(2):
            iterator = values()
            held.append(iterator)
            yield await anext(iterator)

    async def child() -> None:
        async with get_state_store(server, FoundryStateStore, "store"):
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(server, "handle_create", handle)

    async def handler() -> None:
        events = cast(
            AsyncGenerator[Any, None],
            server._handle_create_async_gen(_request(), _context(), asyncio.Event()),
        )

        async def advance() -> None:
            assert await anext(events) == "item"
            advanced.set()
            await release.wait()

        advancer = None
        try:
            assert await anext(events) == "item"
            advancer = asyncio.create_task(advance())
            await advanced.wait()
            if other_borrower:
                with factories[0].bind():
                    children.append(asyncio.create_task(child()))
                await started.wait()
            await asyncio.create_task(events.aclose())
            assert advancer.cancelling() == 0
            store.aclose.assert_awaited_once()
            session = factories[0].session
            assert session is not None and session.closed
        finally:
            release.set()
            for iterator in held:
                await iterator.aclose()
            await asyncio.gather(
                *children, *([advancer] if advancer else []), return_exceptions=True
            )
            await events.aclose()

    task = asyncio.create_task(handler())
    await asyncio.wait_for(task, 1)
    assert task.cancelling() == 0
    assert all(child.cancelled() for child in children)


@pytest.mark.parametrize("ignore_cancellation", [False, True])
async def test_iterator_shutdown_timeout_still_closes_factory(
    monkeypatch: pytest.MonkeyPatch,
    ignore_cancellation: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents.hosting._storage import (
        get_state_store,
        state_store_factory,
    )

    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    server = ResponsesHostServer(make_echo_graph())
    monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "test")
    closing, release = asyncio.Event(), asyncio.Event()

    async def graph_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async with get_state_store(server, FoundryStateStore, "graph"):
            pass
        try:
            yield "messages", (AIMessage(content="ready"), {})
        finally:
            closing.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                if not ignore_cancellation:
                    raise
                await release.wait()

    monkeypatch.setattr(server.graph, "astream", graph_stream)

    async def handler() -> None:
        iterator = server._handle_create_async_gen(
            _request(), _context(), asyncio.Event()
        )
        async for event in iterator:
            if (
                isinstance(event, dict)
                and event.get("type") == "response.output_text.delta"
            ):
                break
        await cast(AsyncGenerator[Any, None], iterator).aclose()

    task = asyncio.create_task(handler())
    try:
        await asyncio.wait_for(closing.wait(), 1)
        done, _ = await asyncio.wait({task}, timeout=1)
        assert task in done, "Iterator shutdown must not block factory cleanup"
        await task
        store.aclose.assert_awaited_once()
        assert acquire.call_args.kwargs["transport"].session.closed
        resources = {
            record.args[0]
            for record in caplog.records
            if "timed out" in record.getMessage() and isinstance(record.args, tuple)
        }
        assert "handler_iterator" in resources
        assert resources <= {"handler_iterator", "graph_iterator"}
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("shutdown", [False, True])
async def test_graph_close_timeout_preserves_generator_exit(
    monkeypatch: pytest.MonkeyPatch, shutdown: bool
) -> None:
    from contextlib import suppress

    from azure.ai.agentserver.responses import ResponseExitForRecovery

    from langchain_azure_ai.agents.hosting._storage import state_store_factory

    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    server = ResponsesHostServer(make_echo_graph())
    context = _context()
    context.exit_for_recovery.side_effect = ResponseExitForRecovery
    release, closed = asyncio.Event(), asyncio.Event()

    async def graph_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        try:
            yield "messages", (AIMessage(content="partial"), {})
        finally:
            try:
                await release.wait()
            finally:
                closed.set()

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    iterator = server.handle_create(_request(), context, asyncio.Event())
    try:
        async for event in iterator:
            if (
                isinstance(event, dict)
                and event.get("type") == "response.output_text.delta"
            ):
                break
        if shutdown:
            context.shutdown.set()
        await asyncio.wait_for(cast(AsyncGenerator[Any, None], iterator).aclose(), 1)
        context.exit_for_recovery.assert_not_awaited()
    finally:
        release.set()
        with suppress(BaseException):
            await cast(AsyncGenerator[Any, None], iterator).aclose()
        await asyncio.wait_for(closed.wait(), 1)


async def test_agentserver_cancel_winddown_preserves_handler_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore
    from azure.ai.agentserver.responses.hosting._orchestrator import _iter_with_winddown

    from langchain_azure_ai.agents.hosting import get_hosting_features
    from langchain_azure_ai.agents.hosting._storage import get_state_store

    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    server = ResponsesHostServer(make_echo_graph())
    cancellation_signal = asyncio.Event()
    settled = []

    async def handle(*args: Any) -> AsyncGenerator[Any, None]:
        try:
            async with get_state_store(server, FoundryStateStore, "store") as first:
                pass
            yield "ready"
        finally:
            assert get_hosting_features() & HostingFeature.RESPONSES
            async with get_state_store(server, FoundryStateStore, "store") as reused:
                assert reused is first
                store.aclose.assert_not_awaited()
            settled.append(True)

    monkeypatch.setattr(server, "handle_create", handle)
    events = _iter_with_winddown(
        server._handle_create_async_gen(_request(), _context(), cancellation_signal),
        cancellation_signal,
    )
    assert await anext(events) == "ready"
    cancellation_signal.set()
    with pytest.raises(StopAsyncIteration):
        await anext(events)
    assert settled == [True]
    acquire.assert_awaited_once()
    store.aclose.assert_awaited_once()


async def test_early_handler_close_settles_graph_before_closing_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents.hosting._storage import get_state_store

    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    monkeypatch.setattr(
        FoundryStateStore, "get_or_create", AsyncMock(return_value=store)
    )
    server = ResponsesHostServer(make_echo_graph())
    settled = []

    async def graph_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async with get_state_store(server, FoundryStateStore, "graph"):
            try:
                yield "messages", (AIMessage(content="ready"), {})
                await asyncio.Event().wait()
            finally:
                store.aclose.assert_not_awaited()
                settled.append(True)

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    events = server._handle_create_async_gen(_request(), _context(), asyncio.Event())
    async with asyncio.timeout(5):
        async for event in events:
            if (
                isinstance(event, dict)
                and event.get("type") == "response.output_text.delta"
            ):
                break
    await cast(AsyncGenerator[Any, None], events).aclose()
    assert settled == [True]
    store.aclose.assert_awaited_once()


@pytest.mark.parametrize("streaming", [False, True])
def test_requests_reuse_two_clients_and_restore_checkpoint_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, streaming: bool
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore
    from langgraph.graph import END, START, MessagesState, StateGraph

    from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver
    from langchain_azure_ai.agents.hosting._responses import conversation_chain_store

    monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    monkeypatch.setattr(
        conversation_chain_store, "FoundryStateStore", FoundryStateStore
    )
    created: list[FoundryStateStore] = []
    closed: list[FoundryStateStore] = []
    get_or_create = FoundryStateStore.get_or_create
    aclose = FoundryStateStore.aclose
    get_item = FoundryStateStore.get_item
    list_keys = FoundryStateStore.list_keys
    checkpoint_reads: list[str] = []
    checkpoint_lists: list[dict[str, Any]] = []
    reference_reads: list[str] = []

    async def read(store: FoundryStateStore, key: str, **kwargs: Any) -> Any:
        if key == CONVERSATION_CHECKPOINT_KEY:
            reference_reads.append(key)
        if "FoundryCheckpointSaver/" in store.name and "/writes/" not in key:
            checkpoint_reads.append(key)
        return await get_item(store, key, **kwargs)

    async def keys(store: FoundryStateStore, **kwargs: Any) -> Any:
        if (kwargs.get("tags") or {}).get("kind") == "checkpoint":
            checkpoint_lists.append(kwargs)
        return await list_keys(store, **kwargs)

    async def acquire(*args: Any, **kwargs: Any) -> FoundryStateStore:
        store = await get_or_create(*args, **kwargs)
        created.append(store)
        return store

    async def close(store: FoundryStateStore) -> None:
        await aclose(store)
        closed.append(store)

    async def reply(state: MessagesState) -> dict[str, Any]:
        assert all(store not in closed for store in created[-2:])
        return {"messages": [AIMessage(content=f"messages={len(state['messages'])}")]}

    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    monkeypatch.setattr(FoundryStateStore, "aclose", close)
    monkeypatch.setattr(FoundryStateStore, "get_item", read)
    monkeypatch.setattr(FoundryStateStore, "list_keys", keys)
    builder = StateGraph(MessagesState)
    builder.add_node("reply", reply)
    builder.add_edge(START, "reply")
    builder.add_edge("reply", END)
    graph = builder.compile(checkpointer=FoundryCheckpointSaver(credential=AsyncMock()))
    server = ResponsesHostServer(graph)
    with _client(server) as client:
        for turn in (1, 2):
            checkpoint_reads.clear()
            checkpoint_lists.clear()
            reference_reads.clear()
            response = client.post(
                "/responses",
                json={"input": "hello", "conversation": "reuse", "stream": streaming},
            )
            assert response.status_code == 200, response.text
            if streaming:
                payload = next(
                    event["response"]
                    for name, event in _parse_sse(response.text)
                    if name == "response.completed"
                )
            else:
                payload = response.json()
            assert payload["status"] == "completed", payload
            assert f"messages={2 * turn - 1}" in str(payload["output"])
            assert len(created) == len(closed) == 2 * turn
            assert len({id(store) for store in created}) == 2 * turn
            assert len({store.name for store in created}) == 2
            assert len(reference_reads) == 1
            if turn == 2:
                assert len(checkpoint_reads) == 2, checkpoint_reads
                assert len(checkpoint_lists) == 1


@pytest.mark.parametrize("recovering", [False, True])
async def test_request_pins_reference_after_recovery_selection(
    monkeypatch: pytest.MonkeyPatch, recovering: bool
) -> None:
    from langchain_azure_ai.agents.hosting import FoundryCheckpointSaver
    from langchain_azure_ai.agents.hosting._checkpoint._checkpoint_reuse import (
        _checkpoint_reads,
    )

    graph = make_checkpointed_echo_graph()
    graph.checkpointer = FoundryCheckpointSaver(credential=AsyncMock())
    server = ResponsesHostServer(graph)
    reference = CheckpointRef("input-thread", "input-checkpoint")
    chain_get = AsyncMock(return_value=reference.to_dict())
    monkeypatch.setattr(server._conversation_chain_store, "get", chain_get)
    context = _context()
    context.is_recovery = recovering
    if recovering:
        context.persisted_response = _response_object(
            context.response_id,
            internal_metadata={
                METADATA_LANGGRAPH_THREAD_ID: "recovery-thread",
                METADATA_LANGGRAPH_CHECKPOINT_ID: "recovery-checkpoint",
            },
        )
        reference = CheckpointRef("recovery-thread", "recovery-checkpoint")

    async def run(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        reads = _checkpoint_reads.get()
        assert reads is not None and reads.key is not None
        assert reads.key[2:] == (reference.thread_id, "")
        assert reads.input_checkpoint_id == reference.checkpoint_id
        yield "messages", (AIMessage(content="done"), {})

    monkeypatch.setattr(server.graph, "astream", run)
    events = [
        event
        async for event in server._handle_create_async_gen(
            _request(), context, asyncio.Event()
        )
    ]
    assert events[-1]["type"] == "response.completed"
    chain_get.assert_awaited_once()


@pytest.mark.parametrize("streaming", [False, True])
def test_background_handler_keeps_clients_after_http_request_ends(
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    from azure.ai.agentserver.core.storage import FoundryStateStore

    from langchain_azure_ai.agents.hosting._storage import get_state_store

    started, closed = threading.Event(), threading.Event()
    release = asyncio.Event()
    store = SimpleNamespace(get=AsyncMock(), aclose=AsyncMock())
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    server = ResponsesHostServer(make_echo_graph())
    resumed = []

    async def close() -> None:
        closed.set()

    store.aclose.side_effect = close

    async def graph_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        async with get_state_store(server, FoundryStateStore, "background") as first:
            started.set()
        # No active borrower while the HTTP request finishes. The handler owns it.
        await release.wait()
        async with get_state_store(server, FoundryStateStore, "background") as reused:
            assert reused is first
            store.aclose.assert_not_awaited()
            resumed.append(True)
        yield "messages", (AIMessage(content="completed"), {})

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    with _client(server) as client:
        try:
            if streaming:

                async def disconnect_stream() -> None:
                    received = False
                    sent: list[Message] = []

                    async def receive() -> Message:
                        nonlocal received
                        if not received:
                            received = True
                            return {
                                "type": "http.request",
                                "body": json.dumps(
                                    {
                                        "input": "hello",
                                        "background": True,
                                        "stream": True,
                                    }
                                ).encode(),
                                "more_body": False,
                            }
                        assert await asyncio.to_thread(started.wait, 5)
                        return {"type": "http.disconnect"}

                    async def send(message: Message) -> None:
                        sent.append(message)

                    await asyncio.wait_for(
                        server.app(
                            {
                                "type": "http",
                                "asgi": {"version": "3.0", "spec_version": "2.0"},
                                "http_version": "1.1",
                                "method": "POST",
                                "scheme": "http",
                                "path": "/responses",
                                "raw_path": b"/responses",
                                "query_string": b"",
                                "root_path": "",
                                "headers": [(b"content-type", b"application/json")],
                                "client": ("127.0.0.1", 1234),
                                "server": ("testserver", 80),
                                "state": {},
                            },
                            receive,
                            send,
                        ),
                        5,
                    )
                    assert sent[0]["status"] == 200

                assert client.portal is not None
                client.portal.call(disconnect_stream)
            else:
                response = client.post(
                    "/responses",
                    json={"input": "hello", "background": True, "stream": False},
                )
                assert response.status_code == 200, response.text
                assert response.json()["status"] == "in_progress"
            assert started.wait(5)
            acquire.assert_awaited_once()
            store.aclose.assert_not_awaited()
            assert not resumed
        finally:
            assert client.portal is not None
            client.portal.call(release.set)
        assert closed.wait(5)
    assert resumed == [True]
    acquire.assert_awaited_once()
    store.aclose.assert_awaited_once()


def test_constructor_registers_responses_features() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._responses_host."
        "_add_process_hosting_features"
    ) as add_process_features:
        server = ResponsesHostServer(make_checkpointed_echo_graph())

    assert server._hosting_features == HostingFeature.RESPONSES
    add_process_features.assert_called_once_with(HostingFeature.RESPONSES)


def test_constructor_registers_resilient_background_feature() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._responses_host."
        "_add_process_hosting_features"
    ) as add_process_features:
        server = ResponsesHostServer(
            make_checkpointed_echo_graph(),
            options=ResponsesServerOptions(resilient_background=True),
        )

    expected = HostingFeature.RESPONSES | HostingFeature.RESILIENT_BACKGROUND
    assert server._hosting_features == expected
    add_process_features.assert_called_once_with(expected)


def test_constructor_registers_steerable_conversations_feature() -> None:
    with patch(
        "langchain_azure_ai.agents.hosting._responses_host."
        "_add_process_hosting_features"
    ) as add_process_features:
        server = ResponsesHostServer(
            make_checkpointed_echo_graph(),
            options=ResponsesServerOptions(steerable_conversations=True),
        )

    expected = HostingFeature.RESPONSES | HostingFeature.STEERABLE_CONVERSATIONS
    assert server._hosting_features == expected
    add_process_features.assert_called_once_with(expected)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_type = ""
    for line in body.splitlines():
        if line.startswith("event:"):
            current_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if not data:
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                payload = {"raw": data}
            events.append((current_type, payload))
    return events


def _message_item(text: str) -> ItemMessage:
    return ItemMessage(
        type="message",
        role="user",
        content=[MessageContentInputTextContent({"type": "input_text", "text": text})],
    )


def _request(**kwargs: object) -> CreateResponse:
    return cast(
        CreateResponse,
        {key: value for key, value in kwargs.items() if value is not None},
    )


def _response_object(
    response_id: str,
    *,
    previous_response_id: str | None = None,
    conversation_id: str | None = None,
    internal_metadata: dict[str, object] | str | None = None,
) -> ResponseObject:
    response: dict[str, object] = {"id": response_id, "output": []}
    if previous_response_id is not None:
        response["previous_response_id"] = previous_response_id
    if conversation_id is not None:
        response["conversation"] = {"id": conversation_id}
    if internal_metadata is not None:
        response["metadata"] = {"_internal_metadata": internal_metadata}
    return cast(ResponseObject, response)


async def _capture_graph_call(
    server: ResponsesHostServer,
    request: CreateResponse,
    context: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, RunnableConfig]:
    captured: dict[str, Any] = {}

    async def graph_stream(
        graph_input: Any,
        *,
        config: RunnableConfig,
        **_: Any,
    ) -> AsyncGenerator[Any, None]:
        captured["input"] = graph_input
        captured["config"] = config
        if False:
            yield None

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    _ = [
        event
        async for event in server.handle_create(
            request,
            context,
            asyncio.Event(),
        )
    ]
    return captured["input"], cast(RunnableConfig, captured["config"])


def _state_store_name(conversation_chain_id: str) -> str:
    return f"{CONVERSATION_CHAIN_STORE_PREFIX}/{conversation_chain_id}"


def _seed_conversation_checkpoint(
    foundry_state_stores: dict[str, dict[str, Any]],
    conversation_chain_id: str,
    *,
    thread_id: str,
    checkpoint_id: str,
) -> None:
    foundry_state_stores[_state_store_name(conversation_chain_id)] = {
        CONVERSATION_CHECKPOINT_KEY: {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }


def _context(
    *,
    response_id: str = "resp-current",
    conversation_id: str | None = "conv-test",
    conversation_chain_id: str | None = None,
    current_text: str = "hello",
    history: list[object] | None = None,
    provider: object | None = None,
    user_id_key: str | None = None,
) -> MagicMock:
    context = MagicMock()
    context.response_id = response_id
    context.conversation_id = conversation_id
    context.conversation_chain_id = (
        conversation_chain_id or conversation_id or f"resp-{response_id}"
    )
    context._provider = provider
    context.isolation = None
    context.is_recovery = False
    context.persisted_response = None
    context.client_cancelled = False
    context._cancellation_signal = asyncio.Event()
    context.shutdown = asyncio.Event()
    context.exit_for_recovery = AsyncMock()
    context.platform_context = SimpleNamespace(user_id_key=user_id_key)
    context.get_input_items = AsyncMock(return_value=[_message_item(current_text)])
    context.get_history = AsyncMock(return_value=history or [])
    return context


def test_non_streaming_request_returns_completed_response() -> None:
    server = ResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.post("/responses", json={"input": "hello", "model": "test"})
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["status"] == "completed"
    output = payload["output"]
    assert any(item.get("type") == "message" for item in output)
    text = "".join(
        part.get("text", "")
        for item in output
        if item.get("type") == "message"
        for part in item.get("content", [])
    )
    assert "Echo: hello" in text


def test_streaming_request_emits_sse_lifecycle_events() -> None:
    server = ResponsesHostServer(make_streaming_graph())
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={"input": "ignored", "stream": True, "model": "test"},
        )
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    types = [t for t, _ in events]
    assert "response.created" in types
    assert "response.in_progress" in types
    assert "response.output_text.delta" in types
    assert "response.completed" in types
    deltas = [p["delta"] for t, p in events if t == "response.output_text.delta"]
    assert "".join(deltas) == "Hello, world!"


async def test_completed_response_includes_langchain_usage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ResponsesHostServer(make_echo_graph())
    messages = [
        AIMessage(
            content="first",
            id="msg-1",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "input_token_details": {"cache_read": 3},
                "output_token_details": {"reasoning": 2},
            },
        ),
        AIMessage(
            content="second",
            id="msg-2",
            usage_metadata={
                "input_tokens": 6,
                "output_tokens": 5,
                "total_tokens": 11,
            },
        ),
    ]

    async def graph_stream(*_: Any, **__: Any) -> AsyncGenerator[Any, None]:
        for message in messages:
            yield "messages", (message, {})

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    events = [
        event
        async for event in server.handle_create(
            _request(),
            _context(),
            asyncio.Event(),
        )
    ]

    completed = next(
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "response.completed"
    )
    assert completed["response"]["usage"] == {
        "input_tokens": 16,
        "input_tokens_details": {
            "cached_tokens": 3,
            "cache_write_tokens": 0,
        },
        "output_tokens": 9,
        "output_tokens_details": {"reasoning_tokens": 2},
        "total_tokens": 25,
    }


def test_steerable_capability_metadata_is_true_when_enabled() -> None:
    server = ResponsesHostServer(
        make_streaming_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={
                "input": "hello",
                "metadata": {
                    "client.key": "kept",
                },
            },
        )

    metadata = resp.json()["metadata"]
    assert metadata[METADATA_STEERABLE_CONVERSATION] == "true"
    assert metadata["client.key"] == "kept"


def test_steerable_capability_metadata_is_false_when_disabled() -> None:
    server = ResponsesHostServer(make_streaming_graph())
    with _client(server) as client:
        resp = client.post(
            "/responses",
            json={
                "input": "hello",
                "metadata": {
                    "client.key": "kept",
                },
            },
        )

    metadata = resp.json()["metadata"]
    assert metadata[METADATA_STEERABLE_CONVERSATION] == "false"
    assert metadata["client.key"] == "kept"


async def test_steering_pressure_completes_superseded_response() -> None:
    server = ResponsesHostServer(make_streaming_graph())
    context = _context(current_text="original turn")
    context.client_cancelled = False
    cancellation_signal = asyncio.Event()
    cancellation_signal.set()

    events = [
        event
        async for event in server.handle_create(
            _request(),
            context,
            cancellation_signal,
        )
    ]

    event_types = [event["type"] for event in events if "type" in event]
    assert event_types[-1] == "response.completed"
    assert "response.failed" not in event_types


async def test_shutdown_before_graph_execution_defers_for_recovery() -> None:
    class _ExitForRecovery(BaseException):
        pass

    server = ResponsesHostServer(make_streaming_graph())
    context = _context()
    context.shutdown.set()
    context.exit_for_recovery.side_effect = _ExitForRecovery

    events = server.handle_create(_request(), context, asyncio.Event())

    assert (await anext(events))["type"] == "response.created"
    with pytest.raises(_ExitForRecovery):
        await anext(events)
    context.exit_for_recovery.assert_awaited_once_with()


async def test_shutdown_after_stream_event_defers_for_recovery() -> None:
    class _ExitForRecovery(BaseException):
        pass

    server = ResponsesHostServer(make_streaming_graph())
    context = _context()
    context.exit_for_recovery.side_effect = _ExitForRecovery
    events = server.handle_create(_request(), context, asyncio.Event())

    while True:
        event = await anext(events)
        if (
            isinstance(event, dict)
            and event.get("type") == "response.output_text.delta"
        ):
            context.shutdown.set()
            break

    with pytest.raises(_ExitForRecovery):
        while True:
            await anext(events)
    context.exit_for_recovery.assert_awaited_once_with()


@pytest.mark.parametrize("signal", ["shutdown", "steering", "client_cancel"])
@pytest.mark.parametrize("slow_close", [False, True])
async def test_graph_close_budget_preserves_signal_finalization(
    monkeypatch: pytest.MonkeyPatch,
    signal: str,
    slow_close: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from langchain_azure_ai.agents.hosting._responses import TaskStorageManager
    from langchain_azure_ai.agents.hosting._storage import state_store_factory

    class ExitForRecovery(BaseException):
        pass

    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.02)
    server = ResponsesHostServer(make_echo_graph())
    context = _context()
    context.exit_for_recovery.side_effect = ExitForRecovery
    cancellation_signal = asyncio.Event()
    chain_set = AsyncMock()
    monkeypatch.setattr(server._conversation_chain_store, "set", chain_set)
    checkpoint_ref = CheckpointRef("thread-test", "checkpoint-test")
    new_stream = server._new_stream

    def stream_with_checkpoint(*args: Any) -> Any:
        stream = new_stream(*args)
        TaskStorageManager.from_stream(stream).store_checkpoint_ref(checkpoint_ref)
        return stream

    monkeypatch.setattr(server, "_new_stream", stream_with_checkpoint)
    release_write = asyncio.Event()
    if not slow_close:
        release_write.set()
    pending_write = asyncio.create_task(release_write.wait())

    async def graph_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        try:
            yield "messages", (AIMessage(content="partial"), {})
        finally:
            await asyncio.shield(pending_write)

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    events = []

    async def consume() -> None:
        async for event in server._handle_create_async_gen(
            _request(), context, cancellation_signal
        ):
            events.append(event)
            if (
                isinstance(event, dict)
                and event.get("type") == "response.output_text.delta"
            ):
                if signal == "shutdown":
                    context.shutdown.set()
                else:
                    context.client_cancelled = signal == "client_cancel"
                    cancellation_signal.set()

    try:
        if signal == "shutdown":
            with pytest.raises(ExitForRecovery):
                await asyncio.wait_for(consume(), 1)
            context.exit_for_recovery.assert_awaited_once_with()
            assert not any(
                event.get("type") in {"response.failed", "response.completed"}
                for event in events
                if isinstance(event, dict)
            )
        else:
            await asyncio.wait_for(consume(), 1)
            terminal = events[-1]
            if signal == "client_cancel":
                assert terminal["response"]["error"]["code"] == "cancelled"
            elif slow_close:
                from openai.types.responses import ResponseError

                error = ResponseError.model_validate(terminal["response"]["error"])
                assert error.code == "server_error"
                assert error.message == "An internal error occurred."
                assert "graph_iterator" not in json.dumps(terminal)
                assert "cleanup timed out: resource=graph_iterator" in caplog.text
            else:
                assert terminal["type"] == "response.completed"
            context.exit_for_recovery.assert_not_awaited()
        if slow_close:
            assert not pending_write.done()
            chain_set.assert_not_awaited()
        else:
            chain_set.assert_awaited_once_with(
                context.conversation_chain_id,
                CONVERSATION_CHECKPOINT_KEY,
                checkpoint_ref.to_dict(),
            )
    finally:
        release_write.set()
        await pending_write


@pytest.mark.parametrize("signal", ["shutdown", "steering", "client_cancel"])
@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("graph execution timed out"),
        RuntimeError("private credential detail"),
    ],
)
async def test_graph_execution_errors_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    signal: str,
    error: Exception,
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context()
    cancellation_signal = asyncio.Event()

    async def graph_stream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        if signal == "shutdown":
            context.shutdown.set()
        else:
            context.client_cancelled = signal == "client_cancel"
            cancellation_signal.set()
        raise error
        yield  # pragma: no cover

    monkeypatch.setattr(server.graph, "astream", graph_stream)
    events = [
        event
        async for event in server._handle_create_async_gen(
            _request(), context, cancellation_signal
        )
    ]
    assert events[-1]["response"]["error"] == {
        "code": "server_error",
        "message": "An internal error occurred.",
    }
    assert all(
        str(error) not in json.dumps(event)
        for event in events
        if isinstance(event, dict)
    )
    assert str(error) in caplog.text
    context.exit_for_recovery.assert_not_awaited()


async def test_graph_close_timeout_preserves_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_azure_ai.agents.hosting._responses_host import _closing_stream
    from langchain_azure_ai.agents.hosting._storage import state_store_factory

    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)

    async def graph_stream() -> AsyncGenerator[Any, None]:
        try:
            yield "ready"
        finally:
            await asyncio.Event().wait()

    stream = graph_stream()
    assert await anext(stream) == "ready"
    original = TimeoutError("graph execution timed out")
    with pytest.raises(TimeoutError) as caught:
        async with _closing_stream(stream):
            raise original
    assert caught.value is original


@pytest.mark.asyncio
async def test_recovered_response_reclaims_stale_replay_stream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(resilient_background=True),
    )
    response_id = f"stale-lock-{uuid.uuid4()}"
    stream_path = tmp_path / "streams" / f"{response_id}.jsonl"
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    stream_path.touch()
    lock_path = stream_path.with_suffix(".jsonl.lock")
    lock_path.touch()
    attempts = 0

    class BodyObserved(Exception):
        pass

    async def fail_then_continue(_candidate_id: str) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            cause = FileExistsError(errno.EEXIST, "File exists", str(lock_path))
            raise RuntimeError("replay stream lock contention") from cause
        if attempts == 2:
            return object()
        raise BodyObserved

    monkeypatch.setattr(streams, "get_or_create", fail_then_continue)
    context = _context(response_id=response_id)
    context.is_recovery = True
    context.platform_context = SimpleNamespace(user_id_key=None, call_id=None)
    record = SimpleNamespace(input_items=[], previous_response_id=None)
    orchestrator = server.app._orchestrator
    assert orchestrator is not None

    with pytest.raises(BodyObserved):
        await orchestrator._run_resilient_stream_body(
            parsed=_request(stream=True),
            context=context,
            cancellation_signal=asyncio.Event(),
            record=record,
            response_id=response_id,
            agent_reference={"type": "agent_reference", "name": "test"},
            model="test",
            store=True,
            agent_session_id=None,
            conversation_id="conv-test",
            background=True,
        )

    assert attempts == 3
    assert not lock_path.exists()


@pytest.mark.asyncio
async def test_fresh_response_does_not_reclaim_replay_stream_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(resilient_background=True),
    )
    response_id = f"live-lock-{uuid.uuid4()}"
    stream_path = tmp_path / "streams" / f"{response_id}.jsonl"
    stream_path.touch()
    lock_path = stream_path.with_suffix(".jsonl.lock")
    lock_path.touch()
    attempts = 0

    async def locked(_candidate_id: str) -> object:
        nonlocal attempts
        attempts += 1
        cause = FileExistsError(errno.EEXIST, "File exists", str(lock_path))
        raise RuntimeError("replay stream lock contention") from cause

    monkeypatch.setattr(streams, "get_or_create", locked)
    context = _context(response_id=response_id)
    context.platform_context = SimpleNamespace(user_id_key=None, call_id=None)
    record = SimpleNamespace(input_items=[], previous_response_id=None)
    orchestrator = server.app._orchestrator
    assert orchestrator is not None

    with pytest.raises(RuntimeError, match="lock contention"):
        await orchestrator._run_resilient_stream_body(
            parsed=_request(stream=True),
            context=context,
            cancellation_signal=asyncio.Event(),
            record=record,
            response_id=response_id,
            agent_reference={"type": "agent_reference", "name": "test"},
            model="test",
            store=True,
            agent_session_id=None,
            conversation_id="conv-test",
            background=True,
        )

    assert attempts == 1
    assert lock_path.exists()


async def test_handle_create_passes_cancellation_signal_to_graph() -> None:
    captured_config: dict[str, object] = {}
    server = ResponsesHostServer(make_streaming_graph(captured_config))
    context = _context()
    cancellation_signal = asyncio.Event()

    _ = [
        event
        async for event in server.handle_create(
            _request(),
            context,
            cancellation_signal,
        )
    ]

    assert (
        captured_config["configurable"]["response_cancellation_signal"]  # type: ignore[index]
        is cancellation_signal
    )


async def test_handle_create_checkpoints_admission_before_config_resolution() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None)
    events = server.handle_create(
        _request(previous_response_id="resp-parent"), context, asyncio.Event()
    )

    assert (await anext(events))["type"] == "response.created"
    assert (await anext(events))["type"] == "response.in_progress"
    admission_checkpoint = await anext(events)

    assert type(admission_checkpoint).__name__ == "ResponseCheckpointEvent"
    assert "_internal_metadata" not in admission_checkpoint.response["metadata"]
    await cast(AsyncGenerator[object, None], events).aclose()


async def test_handle_create_persists_conversation_checkpoint(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())

    _ = [
        event
        async for event in server.handle_create(
            _request(),
            _context(conversation_id=None, conversation_chain_id="chain-1"),
            asyncio.Event(),
        )
    ]

    checkpoint = foundry_state_stores[_state_store_name("chain-1")][
        CONVERSATION_CHECKPOINT_KEY
    ]
    assert checkpoint["thread_id"] == "resp-current"
    assert isinstance(
        checkpoint["checkpoint_id"],
        str,
    )


@pytest.mark.parametrize("checkpointed", [False, True])
async def test_handle_create_persists_root_response_id(checkpointed: bool) -> None:
    responses: dict[str, ResponseObject] = {}
    provider = SimpleNamespace(
        get_response=AsyncMock(
            side_effect=lambda response_id, **_: responses[response_id]
        )
    )
    graph = make_checkpointed_echo_graph() if checkpointed else make_echo_graph()
    server = ResponsesHostServer(graph)
    for response_id, previous_id in [
        ("resp-root", None),
        ("resp-child", "resp-root"),
        ("resp-grandchild", "resp-child"),
    ]:
        context = _context(
            response_id=response_id,
            conversation_id=None,
            provider=provider,
            user_id_key="user-a",
        )
        request = _request(previous_response_id=previous_id)
        stream = server._new_stream(request, context)
        with patch.object(server, "_new_stream", return_value=stream):
            events = [
                event
                async for event in server.handle_create(
                    request, context, asyncio.Event()
                )
            ]
        assert events[-1]["type"] == "response.completed"
        assert (
            stream.internal_metadata[METADATA_LANGGRAPH_ROOT_RESPONSE_ID] == "resp-root"
        )
        responses[response_id] = cast(ResponseObject, dict(stream.response))
    assert provider.get_response.await_count == 2


@pytest.mark.parametrize("failure", ["provider", "cycle"])
async def test_unresolved_response_chain_does_not_cache_root(failure: str) -> None:
    parent = _response_object("resp-parent", previous_response_id="resp-parent")
    if failure == "provider":
        parent = _response_object("resp-parent", previous_response_id="resp-missing")
    provider = SimpleNamespace(
        get_response=AsyncMock(side_effect=[parent, RuntimeError("unavailable")])
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)
    request = _request(previous_response_id="resp-parent")
    stream = server._new_stream(request, context)
    with patch.object(server, "_new_stream", return_value=stream):
        events = [
            event
            async for event in server.handle_create(request, context, asyncio.Event())
        ]
    assert events[-1]["type"] == "response.completed"
    assert METADATA_LANGGRAPH_ROOT_RESPONSE_ID not in stream.internal_metadata


async def test_custom_conversation_chain_store() -> None:
    class CustomStore:
        def __init__(self) -> None:
            self.values: dict[tuple[str, str], dict[str, str]] = {}

        async def get(
            self,
            conversation_chain_id: str,
            key: str,
        ) -> dict[str, str] | None:
            return self.values.get((conversation_chain_id, key))

        async def set(
            self,
            conversation_chain_id: str,
            key: str,
            data: dict[str, str],
        ) -> None:
            self.values[(conversation_chain_id, key)] = data

    store = CustomStore()
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        conversation_chain_store=store,
    )

    _ = [
        event
        async for event in server.handle_create(
            _request(),
            _context(conversation_id=None, conversation_chain_id="custom-chain"),
            asyncio.Event(),
        )
    ]

    checkpoint_ref = CheckpointRef.from_dict(
        store.values[("custom-chain", CONVERSATION_CHECKPOINT_KEY)]
    )
    assert checkpoint_ref is not None
    assert checkpoint_ref.thread_id == "resp-current"
    config = await server.build_runnable_config(
        _request(previous_response_id="resp-current"),
        _context(conversation_id=None, conversation_chain_id="custom-chain"),
    )
    assert config["configurable"]["checkpoint_id"] == checkpoint_ref.checkpoint_id


async def test_recovery_replays_input_without_current_response_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "chain-1",
        thread_id="resp-root",
        checkpoint_id="checkpoint-parent",
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        conversation_chain_id="chain-1",
        current_text="replay me",
    )
    context.is_recovery = True
    context.persisted_response = _response_object("resp-current")
    request = _request(previous_response_id="resp-parent")
    graph_input, config = await _capture_graph_call(
        server,
        request,
        context,
        monkeypatch,
    )

    assert graph_input is not None
    assert config["configurable"]["checkpoint_id"] == "checkpoint-parent"
    assert [message.content for message in graph_input["messages"]] == ["replay me"]


async def test_recovery_replays_hitl_approval_without_current_response_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "chain-1",
        thread_id="resp-root",
        checkpoint_id="checkpoint-parent",
    )
    captured: dict[str, Any] = {}
    pending = Interrupt(value="Approve recovered action?", id="interrupt-1")
    server = ResponsesHostServer(make_recovery_probe_graph(captured, pending))
    context = _context(conversation_id=None, conversation_chain_id="chain-1")
    context.is_recovery = True
    context.persisted_response = _response_object("resp-current")
    context.get_input_items.return_value = [
        {
            "type": "mcp_approval_response",
            "approval_request_id": pending.id,
            "approve": True,
        }
    ]

    graph_input, config = await _capture_graph_call(
        server,
        _request(),
        context,
        monkeypatch,
    )

    assert isinstance(graph_input, Command)
    assert graph_input.resume == pending.value
    assert captured["state_config"]["configurable"]["checkpoint_id"] == (
        "checkpoint-parent"
    )
    assert config["configurable"]["checkpoint_id"] == "checkpoint-parent"


async def test_recovery_replays_hitl_rejection_without_current_response_checkpoint(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "chain-1",
        thread_id="resp-root",
        checkpoint_id="checkpoint-parent",
    )
    captured: dict[str, Any] = {}
    pending = Interrupt(value="Approve recovered action?", id="interrupt-1")
    server = ResponsesHostServer(make_recovery_probe_graph(captured, pending))
    context = _context(conversation_id=None, conversation_chain_id="chain-1")
    context.is_recovery = True
    context.persisted_response = _response_object("resp-current")
    context.get_input_items.return_value = [
        {
            "type": "mcp_approval_response",
            "approval_request_id": pending.id,
            "approve": False,
        }
    ]

    events = [
        event
        async for event in server.handle_create(
            _request(),
            context,
            asyncio.Event(),
        )
    ]

    assert "input" not in captured
    assert captured["state_config"]["configurable"]["checkpoint_id"] == (
        "checkpoint-parent"
    )
    failed = next(event for event in events if event.get("type") == "response.failed")
    assert failed["response"]["error"]["code"] == "interrupt_rejected"


async def test_detect_rejection_override_remains_the_extension_point() -> None:
    captured: dict[str, Any] = {}
    pending = Interrupt(value="Approve action?", id="interrupt-1")

    class RejectingHost(ResponsesHostServer):
        calls = 0

        async def detect_rejection(
            self, request: Any, context: Any, pending: Any
        ) -> str | None:
            del request, context, pending
            self.calls += 1
            return "blocked by custom rejection policy"

    server = RejectingHost(make_recovery_probe_graph(captured, pending))
    context = _context()
    context.get_input_items.return_value = [
        {
            "type": "mcp_approval_response",
            "approval_request_id": pending.id,
            "approve": True,
        }
    ]

    events = [
        event
        async for event in server.handle_create(_request(), context, asyncio.Event())
    ]

    assert server.calls == 1
    assert "input" not in captured
    failed = next(
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == "response.failed"
    )
    assert failed["response"]["error"] == {
        "code": "interrupt_rejected",
        "message": "blocked by custom rejection policy",
    }


@pytest.mark.parametrize(
    "input_items",
    [
        [_message_item("do not replay")],
        [
            {
                "type": "mcp_approval_response",
                "approval_request_id": "interrupt-1",
                "approve": True,
            }
        ],
        [
            {
                "type": "mcp_approval_response",
                "approval_request_id": "interrupt-1",
                "approve": False,
            }
        ],
    ],
)
async def test_recovery_resumes_from_current_response_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    input_items: list[object],
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, current_text="do not replay")
    context.is_recovery = True
    context.persisted_response = _response_object(
        "resp-current",
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-current",
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        },
    )
    context.get_input_items.return_value = input_items
    request = _request()
    graph_input, config = await _capture_graph_call(
        server,
        request,
        context,
        monkeypatch,
    )

    assert graph_input is None
    assert config["configurable"]["thread_id"] == "resp-root"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-current"
    context.get_input_items.assert_not_awaited()


def test_readiness_endpoint_is_available() -> None:
    server = ResponsesHostServer(make_echo_graph())
    with _client(server) as client:
        resp = client.get("/readiness")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_constructor_accepts_reexported_response_options() -> None:
    options = ResponsesServerOptions(default_model="test")
    server = ResponsesHostServer(make_echo_graph(), options=options)

    with _client(server) as client:
        resp = client.post("/responses", json={"input": "hello"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"


def test_constructor_rejects_resilient_background_without_checkpointer() -> None:
    options = ResponsesServerOptions(resilient_background=True)

    with pytest.raises(
        ValueError,
        match="requires a LangGraph checkpointer when resilient_background=True",
    ):
        ResponsesHostServer(make_echo_graph(), options=options)


def test_constructor_accepts_resilient_background_with_checkpointer() -> None:
    options = ResponsesServerOptions(resilient_background=True)

    ResponsesHostServer(make_checkpointed_echo_graph(), options=options)


def test_constructor_rejects_non_messages_state_schema() -> None:
    with pytest.raises(ValueError, match="messages"):
        ResponsesHostServer(make_custom_state_graph())


async def test_checkpointed_graph_uses_current_input_only() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context()

    result = await server.build_input(_request(), context)

    context.get_history.assert_not_called()
    assert len(result["messages"]) == 1
    assert result["messages"][0].content == "hello"


async def test_checkpointed_previous_response_id_does_not_duplicate_history() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        history=[_message_item("from responses transcript")],
        current_text="current turn",
    )

    result = await server.build_input(
        _request(previous_response_id="resp-previous"),
        context,
    )

    context.get_history.assert_not_called()
    assert [message.content for message in result["messages"]] == ["current turn"]


async def test_non_checkpointed_graph_uses_responses_history() -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(history=[_message_item("from history")])

    result = await server.build_input(_request(), context)

    context.get_history.assert_awaited_once()
    assert [message.content for message in result["messages"]] == [
        "from history",
        "hello",
    ]


async def test_non_checkpointed_previous_response_id_includes_history_once() -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(
        conversation_id=None,
        history=[_message_item("turn one"), _message_item("turn two")],
        current_text="turn three",
    )

    result = await server.build_input(
        _request(previous_response_id="resp-2"),
        context,
    )

    context.get_history.assert_awaited_once()
    assert [message.content for message in result["messages"]] == [
        "turn one",
        "turn two",
        "turn three",
    ]


async def test_root_response_id_is_thread_id() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(response_id="resp-1", conversation_id=None)

    with patch.object(
        FoundryConversationChainStore,
        "get",
        new_callable=AsyncMock,
    ) as get_conversation_value:
        config = await server.build_runnable_config(_request(), context)

    get_conversation_value.assert_not_awaited()
    assert config["configurable"]["thread_id"] == "resp-1"
    assert config["configurable"]["response_context"] is context


async def test_explicit_conversation_id_is_thread_id() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(response_id="resp-1", conversation_id="conv-context")

    config = await server.build_runnable_config(
        _request(conversation="conv-request"),
        context,
    )

    assert config["configurable"]["thread_id"] == "conv-request"
    assert config["configurable"]["response_context"] is context


async def test_checkpointed_conversation_is_isolated_by_user() -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())

    with _client(server) as client:
        first = client.post(
            "/responses",
            headers={"x-agent-user-id": "user-a"},
            json={"input": "secret-a", "conversation": {"id": "shared"}},
        )
        second = client.post(
            "/responses",
            headers={"x-agent-user-id": "user-b"},
            json={"input": "hello-b", "conversation": {"id": "shared"}},
        )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    first_config = await server.build_runnable_config(
        _request(),
        _context(conversation_id="shared", user_id_key="user-a"),
    )
    second_config = await server.build_runnable_config(
        _request(),
        _context(conversation_id="shared", user_id_key="user-b"),
    )
    assert (
        first_config["configurable"]["thread_id"]
        != second_config["configurable"]["thread_id"]
    )
    repeated_first_config = await server.build_runnable_config(
        _request(),
        _context(conversation_id="shared", user_id_key="user-a"),
    )
    assert (
        first_config["configurable"]["thread_id"]
        == repeated_first_config["configurable"]["thread_id"]
    )

    first_state = await server.graph.aget_state(first_config)
    second_state = await server.graph.aget_state(second_config)
    assert [message.content for message in first_state.values["messages"]] == [
        "secret-a",
        "Echo: secret-a",
    ]
    assert [message.content for message in second_state.values["messages"]] == [
        "hello-b",
        "Echo: hello-b",
    ]


async def test_checkpoint_reference_is_isolated_by_user(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    first_context = _context(conversation_id="shared", user_id_key="user-a")
    second_context = _context(conversation_id="shared", user_id_key="user-b")

    first_initial = await server.build_runnable_config(_request(), first_context)
    second_initial = await server.build_runnable_config(_request(), second_context)
    first_thread_id = first_initial["configurable"]["thread_id"]
    second_thread_id = second_initial["configurable"]["thread_id"]
    _seed_conversation_checkpoint(
        foundry_state_stores,
        first_thread_id,
        thread_id=first_thread_id,
        checkpoint_id="checkpoint-user-a",
    )
    _seed_conversation_checkpoint(
        foundry_state_stores,
        second_thread_id,
        thread_id=second_thread_id,
        checkpoint_id="checkpoint-user-b",
    )

    first_config = await server.build_runnable_config(_request(), first_context)
    second_config = await server.build_runnable_config(_request(), second_context)

    assert first_config["configurable"]["checkpoint_id"] == "checkpoint-user-a"
    assert second_config["configurable"]["checkpoint_id"] == "checkpoint-user-b"


async def test_previous_response_id_chain_resolves_root_thread_id() -> None:
    class _Provider:
        async def get_response(
            self,
            response_id: str,
            *,
            context: object = None,
        ) -> dict[str, str | None]:
            del context
            responses: dict[str, dict[str, str | None]] = {
                "resp-2": {"previous_response_id": "resp-1"},
                "resp-1": {"previous_response_id": None},
            }
            return responses[response_id]

    server = ResponsesHostServer(make_checkpointed_echo_graph())
    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        _context(conversation_id=None, provider=_Provider()),
    )

    assert config["configurable"]["thread_id"] == "resp-1"


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("user_id_key", [None, "user-a"])
@pytest.mark.parametrize("include_checkpoint", [False, True])
async def test_previous_response_root_metadata_avoids_ancestor_reads(
    as_json: bool, user_id_key: str | None, include_checkpoint: bool
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    root_config = await server.build_runnable_config(
        _request(),
        _context(
            response_id="resp-root",
            conversation_id=None,
            user_id_key=user_id_key,
        ),
    )
    thread_id = root_config["configurable"]["thread_id"]
    metadata: dict[str, object] = {
        METADATA_LANGGRAPH_ROOT_RESPONSE_ID: "resp-root",
        METADATA_LANGGRAPH_THREAD_ID: "opaque-thread-for-another-user",
    }
    if include_checkpoint:
        metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = "checkpoint-parent"
    parent = _response_object(
        "resp-parent",
        previous_response_id="resp-root",
        internal_metadata=json.dumps(metadata) if as_json else metadata,
    )
    provider = SimpleNamespace(get_response=AsyncMock(return_value=parent))
    context = _context(conversation_id=None, provider=provider, user_id_key=user_id_key)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"), context
    )

    assert config["configurable"]["thread_id"] == thread_id
    assert "checkpoint_id" not in config["configurable"]
    provider.get_response.assert_awaited_once_with(
        "resp-parent", context=context.platform_context
    )


@pytest.mark.parametrize("as_model", [False, True])
async def test_cached_root_lookup_does_not_copy_response_payload(
    as_model: bool,
) -> None:
    class UncopyablePayload:
        def __deepcopy__(self, memo: dict[int, Any]) -> Any:
            raise AssertionError("Root lookup must not copy the response payload")

    parent = _response_object(
        "resp-parent",
        internal_metadata={METADATA_LANGGRAPH_ROOT_RESPONSE_ID: "resp-root"},
    )
    parent["output"] = cast(Any, [UncopyablePayload()])
    response = SimpleNamespace(**parent) if as_model else parent
    provider = SimpleNamespace(get_response=AsyncMock(return_value=response))
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"), context
    )

    assert config["configurable"]["thread_id"] == "resp-root"
    provider.get_response.assert_awaited_once_with(
        "resp-parent", context=context.platform_context
    )


@pytest.mark.parametrize(
    ("owner_key", "caller_key"),
    [("user-a", "user-b"), ("user-a", None), (None, "user-b")],
)
async def test_cached_root_scopes_thread_for_current_user(
    owner_key: str | None, caller_key: str | None
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    owner_config = await server.build_runnable_config(
        _request(),
        _context(response_id="resp-root", conversation_id=None, user_id_key=owner_key),
    )
    expected_config = await server.build_runnable_config(
        _request(),
        _context(response_id="resp-root", conversation_id=None, user_id_key=caller_key),
    )
    provider = SimpleNamespace(
        get_response=AsyncMock(
            side_effect=[
                _response_object(
                    "resp-parent",
                    previous_response_id="resp-root",
                    internal_metadata={
                        METADATA_LANGGRAPH_ROOT_RESPONSE_ID: "resp-root",
                        METADATA_LANGGRAPH_THREAD_ID: owner_config["configurable"][
                            "thread_id"
                        ],
                    },
                ),
                _response_object("resp-root"),
            ]
        )
    )
    context = _context(conversation_id=None, provider=provider, user_id_key=caller_key)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"), context
    )

    assert (
        config["configurable"]["thread_id"]
        == expected_config["configurable"]["thread_id"]
    )
    assert (
        config["configurable"]["thread_id"] != owner_config["configurable"]["thread_id"]
    )
    provider.get_response.assert_awaited_once_with(
        "resp-parent", context=context.platform_context
    )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "invalid-json",
        "[]",
        {
            METADATA_LANGGRAPH_ROOT_RESPONSE_ID: 7,
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-parent",
        },
        {
            METADATA_LANGGRAPH_ROOT_RESPONSE_ID: "",
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-parent",
        },
    ],
)
async def test_previous_response_root_metadata_falls_back_to_root(
    metadata: dict[str, object] | str | None,
) -> None:
    provider = SimpleNamespace(
        get_response=AsyncMock(
            side_effect=[
                _response_object(
                    "resp-parent",
                    previous_response_id="resp-root",
                    internal_metadata=metadata,
                ),
                _response_object("resp-root"),
            ]
        )
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)

    request = _request(previous_response_id="resp-parent")
    stream = server._new_stream(request, context)
    with patch.object(server, "_new_stream", return_value=stream):
        events = [
            event
            async for event in server.handle_create(request, context, asyncio.Event())
        ]
    assert events[-1]["type"] == "response.completed"
    assert stream.internal_metadata[METADATA_LANGGRAPH_ROOT_RESPONSE_ID] == "resp-root"
    assert provider.get_response.await_args_list == [
        call("resp-parent", context=context.platform_context),
        call("resp-root", context=context.platform_context),
    ]


async def test_previous_response_conversation_takes_priority_over_metadata() -> None:
    provider = SimpleNamespace(
        get_response=AsyncMock(
            return_value=_response_object(
                "resp-parent",
                conversation_id="conv-existing",
                internal_metadata={
                    METADATA_LANGGRAPH_ROOT_RESPONSE_ID: "resp-root",
                    METADATA_LANGGRAPH_THREAD_ID: "thread-parent",
                    METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-parent",
                },
            )
        )
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id=None, provider=provider)

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-parent"), context
    )

    assert config["configurable"]["thread_id"] == "conv-existing"
    assert HostingRunnableConfig(config).root_response_id is None


async def test_previous_response_uses_conversation_state_store(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "chain-1",
        thread_id="resp-1",
        checkpoint_id="checkpoint-2",
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        conversation_id=None,
        conversation_chain_id="chain-1",
        provider=SimpleNamespace(
            get_response=AsyncMock(
                return_value=_response_object(
                    "resp-2",
                    internal_metadata={
                        METADATA_LANGGRAPH_ROOT_RESPONSE_ID: "resp-root",
                        METADATA_LANGGRAPH_THREAD_ID: "thread-parent",
                        METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-parent",
                    },
                )
            )
        ),
    )

    config = await server.build_runnable_config(
        _request(previous_response_id="resp-2"),
        context,
    )

    assert config["configurable"]["thread_id"] == "resp-1"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"
    assert config["configurable"]["checkpoint_ns"] == ""
    assert HostingRunnableConfig(config).root_response_id == "resp-root"


async def test_explicit_conversation_uses_conversation_state_store(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    _seed_conversation_checkpoint(
        foundry_state_stores,
        "conv-api",
        thread_id="conv-api",
        checkpoint_id="checkpoint-2",
    )
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(conversation_id="conv-api")

    config = await server.build_runnable_config(
        _request(conversation="conv-api"),
        context,
    )

    assert config["configurable"]["thread_id"] == "conv-api"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-2"


async def test_recovery_uses_persisted_thread_and_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ResponsesHostServer(make_checkpointed_echo_graph())
    context = _context(
        response_id="resp-current",
        conversation_id=None,
        provider=SimpleNamespace(
            get_response=AsyncMock(
                return_value=_response_object(
                    "resp-parent",
                    internal_metadata={
                        METADATA_LANGGRAPH_ROOT_RESPONSE_ID: "resp-root",
                        METADATA_LANGGRAPH_THREAD_ID: "thread-parent",
                        METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-parent",
                    },
                )
            )
        ),
    )
    context.is_recovery = True
    context.persisted_response = _response_object(
        "resp-current",
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: "checkpoint-committed",
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        },
    )

    request = _request(previous_response_id="resp-parent")
    _, config = await _capture_graph_call(
        server,
        request,
        context,
        monkeypatch,
    )

    assert config["configurable"]["thread_id"] == "resp-root"
    assert config["configurable"]["checkpoint_id"] == "checkpoint-committed"
    assert config["configurable"]["checkpoint_ns"] == ""


async def test_recovery_ignores_checkpoint_newer_than_response_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = ResponsesHostServer(make_checkpointed_two_node_graph())
    request = _request()
    context = _context(
        response_id="resp-root",
        conversation_id=None,
        current_text="turn one",
    )
    config = await server.build_runnable_config(request, context)
    graph_input = await server.build_input(request, context)
    committed_checkpoint_id: str | None = None
    graph_stream = server.graph.astream(
        graph_input,
        config,
        stream_mode=["updates", "checkpoints"],
        durability="sync",
    )
    saw_plan = False
    async for mode, payload in graph_stream:
        if mode == "updates" and "plan" in payload:
            saw_plan = True
        elif mode == "checkpoints" and saw_plan and committed_checkpoint_id is None:
            checkpoint = cast(dict[str, Any], payload)
            committed_checkpoint_id = checkpoint["config"]["configurable"][
                "checkpoint_id"
            ]

    assert committed_checkpoint_id is not None
    latest_snapshot = await server.graph.aget_state(config)
    assert (
        latest_snapshot.config["configurable"]["checkpoint_id"]
        != committed_checkpoint_id
    )

    recovery_context = _context(
        response_id="resp-root",
        conversation_id=None,
        current_text="turn one",
    )
    recovery_context.is_recovery = True
    recovery_context.persisted_response = _response_object(
        "resp-root",
        internal_metadata={
            METADATA_LANGGRAPH_CHECKPOINT_ID: committed_checkpoint_id,
            METADATA_LANGGRAPH_THREAD_ID: "resp-root",
        },
    )
    with monkeypatch.context() as recovery_patch:
        _, recovery_config = await _capture_graph_call(
            server,
            request,
            recovery_context,
            recovery_patch,
        )
    assert recovery_config["configurable"]["checkpoint_id"] == committed_checkpoint_id
    recovered_nodes = [
        node
        async for update in server.graph.astream(
            None,
            recovery_config,
            stream_mode="updates",
            durability="sync",
        )
        for node in update
    ]

    assert recovered_nodes == ["research"]


async def test_conversation_state_store_supports_linear_response_chain(
    foundry_state_stores: dict[str, dict[str, Any]],
) -> None:
    server = ResponsesHostServer(
        make_checkpointed_echo_graph(),
        options=ResponsesServerOptions(steerable_conversations=True),
    )
    conversation_chain_id = "chain-1"

    async def invoke(
        response_id: str,
        text: str,
        previous_response_id: str | None = None,
    ) -> tuple[dict[str, Any], RunnableConfig]:
        request = _request(previous_response_id=previous_response_id)
        context = _context(
            response_id=response_id,
            conversation_id=None,
            conversation_chain_id=conversation_chain_id,
            current_text=text,
        )
        config = await server.build_runnable_config(request, context)
        graph_input = await server.build_input(request, context)
        state: dict[str, Any] = {}
        graph_stream = server.graph.astream(
            graph_input,
            config=config,
            stream_mode=["values", "checkpoints"],
            durability="sync",
        )
        async for mode, payload in graph_stream:
            if mode == "values":
                state = cast(dict[str, Any], payload)
            elif mode == "checkpoints":
                checkpoint = cast(dict[str, Any], payload)
                checkpoint_ref = HostingRunnableConfig(
                    cast(RunnableConfig, checkpoint["config"])
                ).checkpoint_ref
                assert checkpoint_ref is not None
                await FoundryConversationChainStore().set(
                    conversation_chain_id,
                    CONVERSATION_CHECKPOINT_KEY,
                    checkpoint_ref.to_dict(),
                )
        return state, config

    root_state, root_config = await invoke("resp-root", "root")
    child_state, child_config = await invoke("resp-child", "child", "resp-root")
    grandchild_state, grandchild_config = await invoke(
        "resp-grandchild",
        "grandchild",
        "resp-child",
    )

    root_configurable = root_config.get("configurable")
    child_configurable = child_config.get("configurable")
    grandchild_configurable = grandchild_config.get("configurable")
    assert root_configurable is not None
    assert child_configurable is not None
    assert grandchild_configurable is not None
    root_checkpoint_id = child_configurable["checkpoint_id"]
    assert child_configurable["checkpoint_id"] == root_checkpoint_id
    assert grandchild_configurable["checkpoint_id"] != root_checkpoint_id
    assert "checkpoint_id" not in root_configurable
    assert [message.content for message in root_state["messages"]] == [
        "root",
        "Echo: root",
    ]
    assert [message.content for message in child_state["messages"]] == [
        "root",
        "Echo: root",
        "child",
        "Echo: child",
    ]
    assert [message.content for message in grandchild_state["messages"]] == [
        "root",
        "Echo: root",
        "child",
        "Echo: child",
        "grandchild",
        "Echo: grandchild",
    ]
    assert (
        CONVERSATION_CHECKPOINT_KEY
        in foundry_state_stores[_state_store_name(conversation_chain_id)]
    )


async def test_previous_response_id_thread_is_scoped_by_user() -> None:
    class _Provider:
        async def get_response(
            self,
            response_id: str,
            *,
            context: object = None,
        ) -> dict[str, str | None]:
            del context
            responses: dict[str, dict[str, str | None]] = {
                "resp-2": {"previous_response_id": "resp-1"},
                "resp-1": {"previous_response_id": None},
            }
            return responses[response_id]

    server = ResponsesHostServer(make_checkpointed_echo_graph())
    request = _request(previous_response_id="resp-2")
    first_config = await server.build_runnable_config(
        request,
        _context(conversation_id=None, provider=_Provider(), user_id_key="user-a"),
    )
    second_config = await server.build_runnable_config(
        request,
        _context(conversation_id=None, provider=_Provider(), user_id_key="user-b"),
    )

    assert (
        first_config["configurable"]["thread_id"]
        != second_config["configurable"]["thread_id"]
    )


async def test_conversation_management_debug_log_has_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = ResponsesHostServer(make_echo_graph())
    context = _context(history=[_message_item("from history")])

    caplog.set_level(
        logging.DEBUG,
        logger="langchain_azure_ai.agents.hosting._responses_host",
    )
    await server.build_input(_request(), context)

    assert "mode=responses_history" in caplog.text
    assert "history_items=1" in caplog.text
    assert "history_messages=1" in caplog.text
    assert "current_items=1" in caplog.text
    assert "current_messages=1" in caplog.text
