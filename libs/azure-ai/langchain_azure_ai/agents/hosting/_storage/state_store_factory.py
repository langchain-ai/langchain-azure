# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Handler-owned State Store clients; standalone operations keep local ownership."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import Any

import aiohttp
from azure.ai.agentserver.core import get_request_context
from azure.ai.agentserver.core.storage import (
    FoundryStateStore,
    FoundryStorageNotFoundError,
)
from azure.core.pipeline.transport import AioHttpTransport

logger = logging.getLogger(__name__)
_CLEANUP_TIMEOUT_SECONDS = 5.0
_CONNECTION_OPTIONS = (
    "connection_timeout",
    "read_timeout",
    "connection_verify",
    "connection_cert",
    "connection_data_block_size",
)

_current_factory: ContextVar[StateStoreClientFactory | None] = ContextVar(
    "langchain_azure_state_store_client_factory", default=None
)


async def _close_resource(close: Callable[[], Awaitable[None]]) -> None:
    task = asyncio.ensure_future(close())
    done, _ = await asyncio.wait({task}, timeout=_CLEANUP_TIMEOUT_SECONDS)
    if not done:
        task.cancel()
        # Retrieve eventual errors without waiting for cancellation-resistant code.
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        logger.warning("State Store resource cleanup timed out")
        raise TimeoutError("State Store resource cleanup timed out")
    await task


class StateStoreClientFactory:
    """Create, lend and close compatible clients within one handler.

    A consumer instance supplies fixed credentials, endpoint and client options.
    Distinct consumers, store names and platform users never share a client.
    Only the connection pool is shared; descriptors are refreshed on every use.
    All borrowers must run on the event loop that created the factory.
    """

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.stack = AsyncExitStack()
        self.clients: dict[tuple[object, str, str | None], FoundryStateStore] = {}
        self.locks: dict[tuple[object, str, str | None], asyncio.Lock] = {}
        self.idle = asyncio.Event()
        self.idle.set()
        self.borrowers: dict[asyncio.Task[Any], int] = {}
        self.closing = False
        self.closed = False
        self.session: aiohttp.ClientSession | None = None

    async def acquire(
        self,
        owner: object,
        store_type: type[FoundryStateStore],
        name: str,
        *args: Any,
        **options: Any,
    ) -> FoundryStateStore:
        """Resolve a client while preserving get_or_create's metadata read."""
        # Each consumer instance fixes endpoint, credential and pipeline options.
        key = (owner, name, get_request_context().user_id)
        while True:
            if self.closed:
                raise RuntimeError("State Store client factory has already ended")
            store = self.clients.get(key)
            if store is not None:
                try:
                    # Replace get_or_create's metadata read, never cache descriptors.
                    await store.get()
                    return store
                except FoundryStorageNotFoundError:
                    pass
            async with self.locks.setdefault(key, asyncio.Lock()):
                if self.closed:
                    raise RuntimeError("State Store client factory has already ended")
                # A concurrent acquisition may have replaced the missing client.
                if self.clients.get(key) is not store:
                    continue
                if store is not None:
                    # Existing borrowers retain the old client until scope exit.
                    del self.clients[key]
                if os.environ.get("FOUNDRY_HOSTING_ENVIRONMENT"):
                    if self.session is None:
                        self.session = aiohttp.ClientSession(
                            cookie_jar=aiohttp.DummyCookieJar(),
                            trust_env=True,
                            auto_decompress=False,
                        )
                        self.stack.push_async_callback(
                            _close_resource, self.session.close
                        )
                    options["transport"] = AioHttpTransport(
                        session=self.session,
                        session_owner=False,
                        **{
                            key: options[key]
                            for key in _CONNECTION_OPTIONS
                            if key in options
                        },
                    )
                store = await store_type.get_or_create(name, *args, **options)
                if self.closed:
                    await _close_resource(store.aclose)
                    raise RuntimeError("State Store client factory has already ended")
                self.stack.push_async_callback(_close_resource, store.aclose)
                self.clients[key] = store
                return store


@asynccontextmanager
async def state_store_client_scope() -> AsyncIterator[None]:
    """Own reusable clients and bound cleanup after the handler exits."""
    factory = StateStoreClientFactory()
    token = _current_factory.set(factory)
    cancellation: asyncio.CancelledError | None = None
    try:
        yield
    except asyncio.CancelledError as exc:
        cancellation = exc
        raise
    finally:
        factory.closing = True
        _current_factory.reset(token)

        async def close() -> None:
            try:
                await asyncio.wait_for(factory.idle.wait(), _CLEANUP_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning("State Store cleanup timed out waiting for borrowers")
                borrowers = set(factory.borrowers)
                for borrower in borrowers:
                    borrower.cancel()
                if borrowers:
                    try:
                        await asyncio.wait_for(
                            factory.idle.wait(), _CLEANUP_TIMEOUT_SECONDS
                        )
                    except TimeoutError:
                        logger.warning(
                            "Closing State Store clients with active borrowers"
                        )
            finally:
                factory.closed = True
                await factory.stack.aclose()

        cleanup = asyncio.create_task(close())
        try:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
            await cleanup
        except Exception:
            if cancellation is None:
                raise
            logger.exception("State Store cleanup failed during cancellation")
        finally:
            if cancellation is not None:
                raise cancellation


@asynccontextmanager
async def get_state_store(
    owner: object,
    store_type: type[FoundryStateStore],
    name: str,
    *args: Any,
    **options: Any,
) -> AsyncIterator[FoundryStateStore]:
    """Borrow a handler client or close an operation-owned standalone client."""
    factory = _current_factory.get()
    if factory is not None and asyncio.get_running_loop() is not factory.loop:
        raise RuntimeError(
            "State Store client factory must use its original event loop"
        )
    if factory is not None and factory.closing:
        raise RuntimeError("State Store client factory has already ended")
    # Custom pipelines/sessions retain their existing ownership and proxy settings.
    if factory is None or any(
        key in options
        for key in (
            "transport",
            "pipeline",
            "session",
            "session_owner",
            "use_env_settings",
        )
    ):
        store = await store_type.get_or_create(name, *args, **options)
        async with store:
            yield store
        return
    borrower = asyncio.current_task()
    assert borrower is not None
    factory.borrowers[borrower] = factory.borrowers.get(borrower, 0) + 1
    factory.idle.clear()
    try:
        store = await factory.acquire(owner, store_type, name, *args, **options)
        if factory.closed:
            raise RuntimeError("State Store client factory has already ended")
        yield store
    finally:
        factory.borrowers[borrower] -= 1
        if not factory.borrowers[borrower]:
            del factory.borrowers[borrower]
        if not factory.borrowers:
            factory.idle.set()
