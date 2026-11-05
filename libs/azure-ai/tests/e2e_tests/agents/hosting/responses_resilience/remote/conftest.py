# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Foundry configuration for remote Responses resilience tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from .run_case import AZURE_AI_SCOPE


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("responses-resilience")
    group.addoption(
        "--responses-endpoint",
        help="Full Foundry Responses endpoint URL",
    )
    group.addoption(
        "--reconnect-timeout",
        default=120.0,
        type=float,
        help="Seconds to wait for a crashed Foundry server to recover",
    )


@pytest.fixture(scope="session")
def responses_endpoint(request: pytest.FixtureRequest) -> str:
    endpoint = request.config.getoption("--responses-endpoint")
    if not endpoint:
        raise pytest.UsageError("--responses-endpoint is required for remote tests")
    return str(endpoint)


@pytest.fixture(scope="session")
def reconnect_timeout(request: pytest.FixtureRequest) -> float:
    return float(request.config.getoption("--reconnect-timeout"))


@pytest.fixture(scope="session")
async def foundry_api_key() -> AsyncIterator[Callable[[], Awaitable[str]]]:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, AZURE_AI_SCOPE)

    async def api_key() -> str:
        return await asyncio.to_thread(token_provider)

    try:
        yield api_key
    finally:
        credential.close()
