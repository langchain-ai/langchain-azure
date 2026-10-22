# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Handler-owned State Store clients; standalone operations keep local ownership."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
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

_current_factory: ContextVar[StateStoreClientFactory | None] = ContextVar(
    "langchain_azure_state_store_client_factory", default=None
)


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
        self.active = 0
        self.closing = False
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
            store = self.clients.get(key)
            if store is not None:
                try:
                    # Replace get_or_create's metadata read, never cache descriptors.
                    await store.get()
                    return store
                except FoundryStorageNotFoundError:
                    pass
            async with self.locks.setdefault(key, asyncio.Lock()):
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
                        self.stack.push_async_callback(self.session.close)
                    options["transport"] = AioHttpTransport(
                        session=self.session, session_owner=False, **options
                    )
                store = await store_type.get_or_create(name, *args, **options)
                self.stack.push_async_callback(store.aclose)
                self.clients[key] = store
                return store


@asynccontextmanager
async def state_store_client_scope() -> AsyncIterator[None]:
    """Own reusable clients until the handler and active borrowers settle."""
    factory = StateStoreClientFactory()
    token = _current_factory.set(factory)
    try:
        yield
    finally:
        factory.closing = True
        _current_factory.reset(token)

        async def close() -> None:
            await factory.idle.wait()
            await factory.stack.aclose()

        cleanup = asyncio.create_task(close())
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        await cleanup
        if cancelled:
            raise asyncio.CancelledError()


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
    factory.active += 1
    factory.idle.clear()
    try:
        store = await factory.acquire(owner, store_type, name, *args, **options)
        yield store
    finally:
        factory.active -= 1
        if not factory.active:
            factory.idle.set()
