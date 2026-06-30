"""Shared fixtures for Deep Agents backend integration tests.

These tests run against the Azurite storage emulator. ``deepagents`` is only
imported inside fixture bodies so this module stays importable on Python 3.10
(where the ``[deepagents]`` extra is not installed); the test modules that use
these fixtures are skipped there via ``collect_ignore_glob`` in the root
conftest.
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, AsyncIterator

import pytest

if TYPE_CHECKING:
    from langchain_azure_storage.deepagents import AzureBlobBackend

# Default to the well-known Azurite development connection string. Override via
# AZURE_STORAGE_CONNECTION_STRING to point at a different emulator/account.
AZURITE_CONN_STR = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING",
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/"
    "K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;",
)

TEST_CONTAINER = "test-deepagents"


@pytest.fixture(scope="session")
async def blob_container() -> AsyncIterator[str]:
    """Create a test container in Azurite (session-scoped)."""
    from azure.core.exceptions import ResourceExistsError
    from azure.storage.blob.aio import BlobServiceClient

    async with BlobServiceClient.from_connection_string(AZURITE_CONN_STR) as client:
        try:
            await client.create_container(TEST_CONTAINER)
        except ResourceExistsError:
            pass  # Container may already exist.
        yield TEST_CONTAINER


@pytest.fixture
def backend(blob_container: str) -> AzureBlobBackend:
    """Create a fresh AzureBlobBackend per test with a unique prefix."""
    from langchain_azure_storage.deepagents import AzureBlobBackend

    return AzureBlobBackend(
        container_name=blob_container,
        connection_string=AZURITE_CONN_STR,
        prefix=f"test-{uuid.uuid4().hex[:8]}/",
    )
