# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Local HTTP integration tests for State Store connection-pool reuse."""

from contextlib import asynccontextmanager
from time import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from azure.ai.agentserver.core import (
    FoundryAgentRequestContext,
    reset_request_context,
    set_request_context,
)
from azure.ai.agentserver.core.storage import FoundryStateStore, FoundryStorageEndpoint
from azure.core.credentials import AccessToken

from langchain_azure_ai.agents.hosting._storage import (
    get_state_store,
    state_store_client_scope,
)


@asynccontextmanager
async def _http_storage() -> Any:
    requests: list[Any] = []

    async def handle(request: web.Request) -> web.Response:
        requests.append(request)
        if "/items/" in request.path:
            return web.json_response({"error": {"message": "missing"}}, status=404)
        return web.json_response(
            {
                "id": "store",
                "name": "store",
                "object": "state_store",
                "user_isolation": True,
                "item_ttl_seconds": 600,
                "created_at": 1,
                "updated_at": 1,
                "tags": {},
            }
        )

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    # Keep the listener alive until aiohttp finishes deregistering it.
    import socket

    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        await web.SockSite(runner, listener).start()
        yield f"http://127.0.0.1:{listener.getsockname()[1]}/", requests
    finally:
        try:
            await runner.cleanup()
        finally:
            listener.close()


async def test_real_transport_pool_headers_and_borrower_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "test")
    # A loopback server needs no TLS; production keeps Azure Core's HTTPS check.
    monkeypatch.setattr(
        "azure.core.pipeline.policies._authentication_async._enforce_https",
        lambda request: None,
    )
    credential = SimpleNamespace(
        get_token=AsyncMock(return_value=AccessToken("test", int(time()) + 3600)),
        close=AsyncMock(),
    )
    user_agent = ["first"]
    owner = object()
    async with _http_storage() as (url, requests):
        endpoint = FoundryStorageEndpoint(storage_base_url=url)
        async with state_store_client_scope():
            first = None
            for call_id in ("call-1", "call-2"):
                token = set_request_context(
                    FoundryAgentRequestContext(user_id="user", call_id=call_id)
                )
                try:
                    async with get_state_store(
                        owner,
                        FoundryStateStore,
                        "store",
                        credential,
                        endpoint,
                        user_isolation=True,
                        item_ttl_seconds=600,
                        user_id="delegated",
                        get_server_version=lambda: user_agent[0],
                    ) as store:
                        if first is None:
                            first = store
                        assert store is first
                        assert await store.get_item("missing") is None
                finally:
                    reset_request_context(token)
                user_agent[0] = "second"
            # Retiring one store's pipeline must not close the shared session.
            assert first is not None
            await first.aclose()
            async with get_state_store(
                object(),
                FoundryStateStore,
                "other",
                credential,
                endpoint,
            ) as other:
                await other.get()
        assert len({request.transport for request in requests}) == 1
        assert [r.headers.get("x-agent-foundry-call-id") for r in requests[:4]] == [
            "call-1",
            "call-1",
            "call-2",
            "call-2",
        ]
        assert requests[1].headers["x-ms-user-id"] == "delegated"
        assert requests[1].headers["User-Agent"] == "first"
        assert requests[3].headers["User-Agent"] == "second"
        credential.close.assert_not_awaited()
