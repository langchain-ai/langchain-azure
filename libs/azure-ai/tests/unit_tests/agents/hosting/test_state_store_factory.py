# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Client ownership, identity isolation and real HTTP connection reuse."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)
from azure.ai.agentserver.core.storage import FoundryStateStore
from azure.core.pipeline.transport import AioHttpTransport
from langchain_core.runnables import RunnableConfig

from langchain_azure_ai.agents.hosting import (
    FoundryCheckpointSaver,
    FoundryConversationChainStore,
)
from langchain_azure_ai.agents.hosting._storage import (
    get_state_store,
    state_store_client_scope,
    state_store_factory,
)


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "test")
    created: list[Any] = []

    async def acquire(name: str, *args: Any, **options: Any) -> Any:
        await asyncio.sleep(0)
        store = SimpleNamespace(
            name=name,
            args=args,
            options=options,
            get=AsyncMock(),
            aclose=AsyncMock(),
        )
        created.append(store)
        return store

    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    return created


async def test_concurrent_borrowers_resolve_and_close_once(clients: list[Any]) -> None:
    owner = object()

    async def borrow() -> Any:
        async with get_state_store(owner, FoundryStateStore, "store") as store:
            await asyncio.sleep(0)
            cast(Any, store).aclose.assert_not_awaited()
            return store

    async with state_store_client_scope():
        borrowed = await asyncio.gather(*(borrow() for _ in range(8)))
        assert all(store is clients[0] for store in borrowed)
        assert len(clients) == 1
        clients[0].get.assert_not_awaited()
    clients[0].aclose.assert_awaited_once()


@pytest.mark.parametrize(
    "actual_isolation", [False, True], ids=["whole-store", "user-items"]
)
async def test_saver_deletion_resolves_store_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, actual_isolation: bool
) -> None:
    monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    saver = FoundryCheckpointSaver(
        credential=AsyncMock(), user_isolation=actual_isolation
    )
    async with state_store_client_scope():
        async with saver._get_or_create_store("thread") as first:
            await first.set_item("old", {"value": "old"})
        await saver.adelete_thread("thread")
        async with saver._get_or_create_store("thread") as replacement:
            assert replacement is not first
            assert await replacement.get_item("old") is None
            await replacement.set_item("new", {"value": "new"})
        async with saver._get_or_create_store("thread") as reused:
            assert reused is replacement
            item = await reused.get_item("new")
            assert item is not None and item.value == {"value": "new"}


async def test_deletion_waits_for_acquisition_and_blocks_replacement(
    monkeypatch: pytest.MonkeyPatch, clients: list[Any]
) -> None:
    owner = object()
    acquire_started, delete_started = asyncio.Event(), asyncio.Event()
    allow_acquire, allow_delete = asyncio.Event(), asyncio.Event()
    get_or_create = FoundryStateStore.get_or_create

    async def acquire(*args: Any, **kwargs: Any) -> Any:
        store = await get_or_create(*args, **kwargs)
        acquire_started.set()
        await allow_acquire.wait()
        return store

    async def borrow() -> Any:
        async with get_state_store(owner, FoundryStateStore, "store") as store:
            return store

    async def delete() -> None:
        async with state_store_factory._state_store_deletion_scope(owner, "store"):
            delete_started.set()
            await allow_delete.wait()

    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    async with state_store_client_scope():
        first = asyncio.create_task(borrow())
        tasks = [first]
        try:
            await asyncio.wait_for(acquire_started.wait(), 2)
            deletion = asyncio.create_task(delete())
            tasks.append(deletion)
            await asyncio.sleep(0)
            assert not delete_started.is_set()
            allow_acquire.set()
            old = await asyncio.wait_for(asyncio.shield(first), 2)
            await asyncio.wait_for(delete_started.wait(), 2)
            borrowers = [asyncio.create_task(borrow()) for _ in range(4)]
            tasks.extend(borrowers)
            await asyncio.sleep(0)
            assert all(not task.done() for task in borrowers)
            old.aclose.assert_not_awaited()
            allow_delete.set()
            await asyncio.wait_for(asyncio.shield(deletion), 2)
            borrowed = await asyncio.gather(*borrowers)
            assert all(store is clients[1] for store in borrowed)
            assert len(clients) == 2
        finally:
            allow_acquire.set()
            allow_delete.set()
            await asyncio.gather(*tasks, return_exceptions=True)
    for client in clients:
        client.get.assert_not_awaited()
        client.aclose.assert_awaited_once()


async def test_store_consumer_and_user_bindings_are_separate(
    clients: list[Any],
) -> None:
    first, second = object(), object()
    async with state_store_client_scope():
        for owner, name, user in [
            (first, "checkpoint", "a"),
            (first, "chain", "a"),
            (second, "checkpoint", "a"),
            (first, "checkpoint", "b"),
        ]:
            token = set_request_context(FoundryAgentRequestContext(user_id=user))
            try:
                async with get_state_store(owner, FoundryStateStore, name):
                    pass
            finally:
                reset_request_context(token)
        assert len(clients) == 4
        assert len({id(c.options["transport"].session) for c in clients}) == 1
    for client in clients:
        client.aclose.assert_awaited_once()


@pytest.mark.parametrize(
    "error", [None, RuntimeError("delete failed"), asyncio.CancelledError()]
)
async def test_deletion_reacquires_without_closing_other_borrowers(
    clients: list[Any], error: BaseException | None
) -> None:
    owner = object()
    async with state_store_client_scope():
        async with get_state_store(owner, FoundryStateStore, "store") as old:
            expectation = (
                pytest.raises(type(error)) if error is not None else nullcontext()
            )
            with expectation as raised:
                async with state_store_factory._state_store_deletion_scope(
                    owner, "store"
                ):
                    if error is not None:
                        raise error
            if error is not None:
                assert raised is not None and raised.value is error
            async with get_state_store(owner, FoundryStateStore, "store") as new:
                assert old is not new
                cast(Any, old).aclose.assert_not_awaited()
            async with get_state_store(owner, FoundryStateStore, "store") as reused:
                assert reused is new
    for client in clients:
        client.aclose.assert_awaited_once()


@pytest.mark.parametrize("error", [RuntimeError("backend"), asyncio.CancelledError()])
async def test_failed_acquisition_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, clients: list[Any], error: BaseException
) -> None:
    owner = object()
    get_or_create = FoundryStateStore.get_or_create
    monkeypatch.setattr(
        FoundryStateStore, "get_or_create", AsyncMock(side_effect=error)
    )
    async with state_store_client_scope() as factory:
        with pytest.raises(type(error)) as raised:
            async with get_state_store(owner, FoundryStateStore, "store") as store:
                pytest.fail("Failed acquisition must not lend a client")
        assert raised.value is error
        assert factory.clients == {}
        monkeypatch.setattr(FoundryStateStore, "get_or_create", get_or_create)
        async with get_state_store(owner, FoundryStateStore, "store") as store:
            assert store is clients[0]
        async with get_state_store(owner, FoundryStateStore, "store") as reused:
            assert reused is store
        assert len(clients) == 1
        cast(Any, clients[0]).get.assert_not_awaited()
    cast(Any, clients[0]).aclose.assert_awaited_once()


async def test_cancelled_owner_waits_for_active_borrowers(clients: list[Any]) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    owner = object()

    async def child() -> None:
        async with get_state_store(owner, FoundryStateStore, "store") as store:
            started.set()
            await release.wait()
            cast(Any, store).aclose.assert_not_awaited()

    async def handler() -> None:
        async with state_store_client_scope():
            task = asyncio.create_task(child())
            await started.wait()
            try:
                await asyncio.Event().wait()
            finally:
                # The scope waits even if the owner leaves before this child.
                assert not task.done()

    task = asyncio.create_task(handler())
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    clients[0].aclose.assert_not_awaited()
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    clients[0].aclose.assert_awaited_once()


async def test_late_child_cannot_reopen_closed_scope(clients: list[Any]) -> None:
    release = asyncio.Event()

    async def child() -> None:
        await release.wait()
        async with get_state_store(object(), FoundryStateStore, "store"):
            pytest.fail("Closed scope must reject new work")

    async with state_store_client_scope():
        task = asyncio.create_task(child())
    release.set()
    with pytest.raises(RuntimeError, match="already ended"):
        await task
    assert not clients


async def test_scopes_never_share_clients(clients: list[Any]) -> None:
    owner = object()

    async def handler() -> Any:
        async with state_store_client_scope():
            async with get_state_store(owner, FoundryStateStore, "store") as store:
                await asyncio.sleep(0)
                return store

    a, b = await asyncio.gather(handler(), handler())
    assert a is not b
    assert a.options["transport"].session is not b.options["transport"].session
    assert all(c.options["transport"].session.closed for c in clients)


async def test_scope_reset_failure_still_closes_resources(clients: list[Any]) -> None:
    async def events() -> Any:
        async with state_store_client_scope():
            async with get_state_store(object(), FoundryStateStore, "store"):
                pass
            yield "ready"

    iterator = events()
    assert await asyncio.create_task(anext(iterator)) == "ready"
    try:
        with pytest.raises(ValueError, match="different Context"):
            await iterator.aclose()
        clients[0].aclose.assert_awaited_once()
        assert clients[0].options["transport"].session.closed
    finally:
        if not clients[0].options["transport"].session.closed:
            await clients[0].options["transport"].session.close()


async def test_factory_rejects_another_event_loop_before_borrowing(
    clients: list[Any],
) -> None:
    owner = object()

    async def borrow_from_another_loop() -> None:
        async with get_state_store(owner, FoundryStateStore, "store"):
            pass

    async with state_store_client_scope():
        async with get_state_store(owner, FoundryStateStore, "store") as first:
            pass
        with pytest.raises(RuntimeError, match="original event loop"):
            await asyncio.to_thread(lambda: asyncio.run(borrow_from_another_loop()))
        assert len(clients) == 1
        clients[0].get.assert_not_awaited()
        clients[0].aclose.assert_not_awaited()
        async with get_state_store(owner, FoundryStateStore, "store") as reused:
            assert reused is first
    clients[0].aclose.assert_awaited_once()


async def test_scoped_saver_reuses_store_without_existence_checks(
    clients: list[Any],
) -> None:
    saver = FoundryCheckpointSaver(credential=AsyncMock())
    config: RunnableConfig = {"configurable": {"thread_id": "thread"}}
    async with state_store_client_scope():
        async with get_state_store(
            saver, FoundryStateStore, saver._store_name("thread")
        ) as store:
            cast(Any, store).get.return_value = SimpleNamespace(
                user_isolation=False, item_ttl_seconds=600
            )
            cast(Any, store).list_keys = AsyncMock(
                return_value=SimpleNamespace(keys=[], has_more=False)
            )
        assert await saver.aget_tuple(config) is None
        assert await saver.aget_tuple(config) is None
        assert len(clients) == 1
        cast(Any, store).get.assert_not_awaited()
        assert cast(Any, store).list_keys.await_count == 2
    cast(Any, store).aclose.assert_awaited_once()


@pytest.mark.parametrize("on_mismatch", ["fail", "ignore"])
async def test_scoped_chain_store_reuses_existing_settings(
    monkeypatch: pytest.MonkeyPatch, clients: list[Any], on_mismatch: str
) -> None:
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._responses."
        "conversation_chain_store.FoundryStateStore",
        FoundryStateStore,
    )
    chain = FoundryConversationChainStore(
        user_isolation=True, on_mismatch=cast(Any, on_mismatch)
    )
    data = {"value": "stored"}
    async with state_store_client_scope():
        async with get_state_store(
            chain, FoundryStateStore, chain._store_name("chain")
        ) as store:
            fake = cast(Any, store)
            fake.get.return_value = SimpleNamespace(
                user_isolation=False, item_ttl_seconds=600
            )
            fake.set_item = AsyncMock()
            fake.get_item = AsyncMock(return_value=SimpleNamespace(value=data))
        await chain.set("chain", "key", data)
        assert await chain.get("chain", "key") == data
        assert len(clients) == 1
        fake.get.assert_not_awaited()
        fake.set_item.assert_awaited_once_with("key", data)
    fake.aclose.assert_awaited_once()


async def test_failed_acquisition_closes_owned_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "test")
    transports = []

    async def fail(*args: Any, **options: Any) -> None:
        transports.append(options["transport"])
        raise RuntimeError("acquisition failed")

    monkeypatch.setattr(FoundryStateStore, "get_or_create", fail)
    with pytest.raises(RuntimeError, match="acquisition failed"):
        async with state_store_client_scope():
            async with get_state_store(object(), FoundryStateStore, "store"):
                pass
    assert transports[0].session.closed


@pytest.mark.parametrize(
    "close_error",
    [RuntimeError("close failed"), asyncio.CancelledError("close cancelled")],
)
async def test_close_failure_still_closes_remaining_resources(
    clients: list[Any], close_error: BaseException
) -> None:
    with pytest.raises(type(close_error)):
        async with state_store_client_scope():
            for name in ("a", "b"):
                async with get_state_store(object(), FoundryStateStore, name):
                    pass
            clients[-1].aclose.side_effect = close_error
    assert all(c.options["transport"].session.closed for c in clients)
    for client in clients:
        client.aclose.assert_awaited_once()


@pytest.mark.parametrize("primary_kind", ["error", "recovery", "generator_exit"])
@pytest.mark.parametrize("cancel_during_close", [False, True])
@pytest.mark.parametrize("cancelled_close", [False, True])
async def test_cleanup_preserves_primary_error_and_new_cancellation(
    clients: list[Any],
    primary_kind: str,
    cancel_during_close: bool,
    cancelled_close: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from azure.ai.agentserver.responses import ResponseExitForRecovery

    primary = {
        "error": ValueError("execution failed"),
        "recovery": ResponseExitForRecovery(),
        "generator_exit": GeneratorExit(),
    }[primary_kind]
    closing, release = asyncio.Event(), asyncio.Event()

    async def close() -> None:
        closing.set()
        await release.wait()
        if cancelled_close:
            worker = asyncio.create_task(asyncio.sleep(0))
            worker.cancel("close worker cancelled")
            await worker
        raise RuntimeError("close failed")

    async def handler() -> None:
        async with state_store_client_scope():
            async with get_state_store(object(), FoundryStateStore, "store"):
                pass
            clients[0].aclose.side_effect = close
            raise primary

    task = asyncio.create_task(handler())
    try:
        await asyncio.wait_for(closing.wait(), 1)
        if cancel_during_close:
            task.cancel("cancel during cleanup")
        release.set()
        expected = asyncio.CancelledError if cancel_during_close else type(primary)
        with pytest.raises(expected) as caught:
            await task
        if cancel_during_close:
            assert caught.value.args == ("cancel during cleanup",)
        else:
            assert caught.value is primary
        assert task.cancelling() == int(cancel_during_close)
        assert "Hosting cleanup failed during handler exit" in caplog.text
        clients[0].aclose.assert_awaited_once()
        assert clients[0].options["transport"].session.closed
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize(
    "option", ["transport", "pipeline", "session", "use_env_settings"]
)
async def test_custom_pipeline_options_keep_operation_ownership(
    monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    store = AsyncMock()
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    value = object()
    async with state_store_client_scope():
        async with get_state_store(
            object(), FoundryStateStore, "store", **{option: value}
        ):
            pass
        store.__aexit__.assert_awaited_once()
    assert acquire.call_args.kwargs == {option: value}


async def test_standalone_operations_keep_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AsyncMock()
    acquire = AsyncMock(return_value=store)
    monkeypatch.setattr(FoundryStateStore, "get_or_create", acquire)
    owner = object()
    for _ in range(2):
        async with get_state_store(owner, FoundryStateStore, "store"):
            pass
    assert acquire.await_count == store.__aexit__.await_count == 2
    assert acquire.call_args.kwargs == {}


async def test_local_store_reuse_does_not_create_http_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.delenv("FOUNDRY_HOSTING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("AGENTSERVER_STATE_ROOT", str(tmp_path))
    session = AsyncMock(side_effect=AssertionError("Local storage needs no session"))
    monkeypatch.setattr(
        "langchain_azure_ai.agents.hosting._storage.state_store_factory."
        "aiohttp.ClientSession",
        session,
    )
    owner = object()
    async with state_store_client_scope():
        async with get_state_store(owner, FoundryStateStore, "local") as first:
            await first.set_item("key", {"value": "stored"})
        async with get_state_store(owner, FoundryStateStore, "local") as second:
            assert first is second
            item = await second.get_item("key")
            assert item is not None and item.value == {"value": "stored"}
    session.assert_not_called()


async def test_parallel_stores_share_one_session(clients: list[Any]) -> None:
    async def acquire(name: str) -> Any:
        async with get_state_store(object(), FoundryStateStore, name) as store:
            return store

    async with state_store_client_scope():
        await asyncio.gather(*(acquire(str(n)) for n in range(8)))
        sessions = [c.options["transport"].session for c in clients]
        assert len({id(session) for session in sessions}) == 1
        assert not sessions[0].closed
    assert sessions[0].closed


async def test_transport_receives_only_connection_options(
    monkeypatch: pytest.MonkeyPatch, clients: list[Any]
) -> None:
    transport_options = []

    def transport(**kwargs: Any) -> AioHttpTransport:
        transport_options.append(kwargs)
        return AioHttpTransport(**kwargs)

    monkeypatch.setattr(state_store_factory, "AioHttpTransport", transport)
    connection_options = {
        "connection_timeout": 12,
        "read_timeout": 34,
        "connection_verify": "test-ca.pem",
        "connection_cert": "test-client.pem",
        "connection_data_block_size": 8192,
    }
    async with state_store_client_scope():
        async with get_state_store(
            object(),
            FoundryStateStore,
            "store",
            user_isolation=True,
            item_ttl_seconds=600,
            description="test",
            tags={"test": "yes"},
            user_id="user",
            api_version="v1",
            get_server_version=lambda: "test",
            retry_total=3,
            **connection_options,
        ):
            pass
    options = transport_options[0]
    assert options.pop("session_owner") is False
    assert options.pop("session").closed
    assert options == connection_options
    assert clients[0].options["retry_total"] == 3
    assert clients[0].options["user_isolation"] is True


async def test_cleanup_cancels_stuck_borrower_before_closing(
    monkeypatch: pytest.MonkeyPatch,
    clients: list[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    started, settled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def borrower() -> None:
        try:
            async with get_state_store(object(), FoundryStateStore, "store"):
                started.set()
                await release.wait()
        finally:
            clients[0].aclose.assert_not_awaited()
            settled.set()

    async def handler() -> None:
        async with state_store_client_scope():
            children.append(asyncio.create_task(borrower()))
            await started.wait()

    children: list[asyncio.Task[None]] = []
    owner = asyncio.create_task(handler())
    try:
        done, _ = await asyncio.wait({owner}, timeout=1)
        assert owner in done, "Scope cleanup must not wait forever"
        await owner
        assert settled.is_set()
        assert children[0].cancelled()
        clients[0].aclose.assert_awaited_once()
        assert clients[0].options["transport"].session.closed
        assert "timed out" in caplog.text
        record = next(r for r in caplog.records if "borrower_drain" in r.getMessage())
        elapsed, borrowers = cast(tuple[float, int], record.args)
        assert elapsed >= 0
        assert borrowers == 1
    finally:
        release.set()
        await asyncio.gather(owner, *children, return_exceptions=True)


@pytest.mark.parametrize("other_borrower", [False, True])
async def test_cleanup_does_not_cancel_finalizing_task_with_suspended_borrow(
    monkeypatch: pytest.MonkeyPatch, clients: list[Any], other_borrower: bool
) -> None:
    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    started = asyncio.Event()
    children: list[asyncio.Task[None]] = []

    async def values() -> AsyncGenerator[str, None]:
        async with get_state_store(object(), FoundryStateStore, "owner"):
            yield "item"

    async def child() -> None:
        async with get_state_store(object(), FoundryStateStore, "child"):
            started.set()
            await asyncio.Event().wait()

    async def handler() -> None:
        iterator = values()
        try:
            async with state_store_client_scope():
                assert await anext(iterator) == "item"
                if other_borrower:
                    children.append(asyncio.create_task(child()))
                    await started.wait()
            for client in clients:
                client.aclose.assert_awaited_once()
                assert client.options["transport"].session.closed
        finally:
            await iterator.aclose()
            await asyncio.gather(*children, return_exceptions=True)

    task = asyncio.create_task(handler())
    await asyncio.wait_for(task, 1)
    assert task.cancelling() == 0
    assert all(child.cancelled() for child in children)


@pytest.mark.parametrize("close_fails", [False, True])
async def test_cleanup_preserves_original_cancellation(
    clients: list[Any], close_fails: bool, caplog: pytest.LogCaptureFixture
) -> None:
    closing, release = asyncio.Event(), asyncio.Event()
    cancellations = []

    async def close() -> None:
        closing.set()
        await release.wait()
        if close_fails:
            raise RuntimeError("close failed")

    async def handler() -> None:
        async with state_store_client_scope():
            async with get_state_store(object(), FoundryStateStore, "store"):
                pass
            clients[0].aclose.side_effect = close
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                cancellations.append(exc)
                raise

    owner = asyncio.create_task(handler())
    while not clients:
        await asyncio.sleep(0)
    owner.cancel("original cancellation")
    try:
        await asyncio.wait_for(closing.wait(), 1)
        owner.cancel("second cancellation")
        await asyncio.sleep(0)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await owner
    assert raised.value is cancellations[0]
    assert raised.value.args == ("original cancellation",)
    assert owner.cancelled()
    assert clients[0].options["transport"].session.closed
    if close_fails:
        assert "close failed" in caplog.text


@pytest.mark.parametrize("ignore_cancellation", [False, True])
async def test_cleanup_bounds_hanging_client_close(
    monkeypatch: pytest.MonkeyPatch,
    clients: list[Any],
    ignore_cancellation: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    closing, release = asyncio.Event(), asyncio.Event()

    async def close() -> None:
        closing.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            if not ignore_cancellation:
                raise
            await release.wait()

    async def handler() -> None:
        async with state_store_client_scope():
            async with get_state_store(
                object(), FoundryStateStore, "private-store-name"
            ):
                pass
            clients[0].aclose.side_effect = close

    owner = asyncio.create_task(handler())
    try:
        await asyncio.wait_for(closing.wait(), 1)
        done, _ = await asyncio.wait({owner}, timeout=1)
        assert owner in done, "Closing a client must not wait forever"
        with pytest.raises(TimeoutError):
            await owner
        assert clients[0].options["transport"].session.closed
        record = next(r for r in caplog.records if "timed out" in r.getMessage())
        resource, elapsed, borrowers = cast(tuple[str, float, int], record.args)
        assert resource == "store_client"
        assert elapsed >= 0
        assert borrowers == 0
        assert "private-store-name" not in caplog.text
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)


async def test_session_timeout_log_has_safe_resource_label(
    monkeypatch: pytest.MonkeyPatch,
    clients: list[Any],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    session = SimpleNamespace(close=AsyncMock())

    async def close() -> None:
        await asyncio.sleep(1)

    session.close.side_effect = close
    monkeypatch.setattr(
        state_store_factory.aiohttp, "ClientSession", lambda **kw: session
    )
    with pytest.raises(TimeoutError):
        async with state_store_client_scope():
            async with get_state_store(object(), FoundryStateStore, "private-store"):
                pass
    record = next(r for r in caplog.records if "timed out" in r.getMessage())
    resource, elapsed, borrowers = cast(tuple[str, float, int], record.args)
    assert resource == "shared_session"
    assert elapsed >= 0
    assert borrowers == 0
    assert "private-store" not in caplog.text


async def test_cleanup_preserves_asyncio_timeout(clients: list[Any]) -> None:
    closing, release = asyncio.Event(), asyncio.Event()

    async def close() -> None:
        closing.set()
        await release.wait()

    async def unblock() -> None:
        await closing.wait()
        await asyncio.sleep(0.02)
        release.set()

    async def handler() -> None:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                async with state_store_client_scope():
                    async with get_state_store(object(), FoundryStateStore, "store"):
                        pass
                    clients[0].aclose.side_effect = close
                    await asyncio.Event().wait()
        task = asyncio.current_task()
        assert task is not None and task.cancelling() == 0

    await asyncio.gather(handler(), unblock())
    assert clients[0].options["transport"].session.closed


async def test_late_acquisition_after_cleanup_closes_client_without_lending(
    monkeypatch: pytest.MonkeyPatch, clients: list[Any]
) -> None:
    monkeypatch.setattr(state_store_factory, "_CLEANUP_TIMEOUT_SECONDS", 0.01)
    started, release = asyncio.Event(), asyncio.Event()
    acquire = FoundryStateStore.get_or_create
    sessions = []

    async def slow_acquire(*args: Any, **options: Any) -> Any:
        sessions.append(options["transport"].session)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return await acquire(*args, **options)

    monkeypatch.setattr(FoundryStateStore, "get_or_create", slow_acquire)

    async def borrower() -> None:
        async with get_state_store(object(), FoundryStateStore, "store"):
            pytest.fail("Closed factory must not lend a late client")

    async def handler() -> None:
        async with state_store_client_scope():
            children.append(asyncio.create_task(borrower()))
            await started.wait()

    children: list[asyncio.Task[None]] = []
    owner = asyncio.create_task(handler())
    try:
        done, _ = await asyncio.wait({owner}, timeout=1)
        assert owner in done
        await owner
        assert sessions[0].closed
        release.set()
        with pytest.raises(RuntimeError, match="already ended"):
            await asyncio.wait_for(children[0], 1)
        assert len(clients) == 1
        clients[0].aclose.assert_awaited_once()
    finally:
        release.set()
        await asyncio.gather(owner, *children, return_exceptions=True)
