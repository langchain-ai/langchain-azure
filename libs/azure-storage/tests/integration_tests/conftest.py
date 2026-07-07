"""Shared fixtures for Azure Storage integration tests.

The integration tests can run against a live Azure Storage account or a local
emulator such as Azurite:

* Set ``AZURE_STORAGE_ACCOUNT_URL`` to run against a live account,
  authenticated with ``DefaultAzureCredential``.
* Set ``AZURE_STORAGE_CONNECTION_STRING`` to run against an emulator or any
  other account reachable with a connection string.

When both are set, the account URL takes precedence. The document loader tests
require ``AZURE_STORAGE_ACCOUNT_URL`` (the document loaders do not support
connection strings).
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Iterator, Optional

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

if TYPE_CHECKING:
    from langchain_azure_storage.deepagents import AzureBlobBackend

_DEEPAGENTS_CONTAINER = "test-deepagents"


@pytest.fixture(scope="session")
def account_url() -> Optional[str]:
    return os.environ.get("AZURE_STORAGE_ACCOUNT_URL")


@pytest.fixture(scope="session")
def connection_string() -> Optional[str]:
    return os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


@pytest.fixture(scope="session")
def blob_service_client(
    account_url: Optional[str], connection_string: Optional[str]
) -> Iterator[BlobServiceClient]:
    if account_url:
        client = BlobServiceClient(account_url, credential=DefaultAzureCredential())
    elif connection_string:
        client = BlobServiceClient.from_connection_string(connection_string)
    else:
        raise ValueError(
            "Set AZURE_STORAGE_ACCOUNT_URL (live account) or "
            "AZURE_STORAGE_CONNECTION_STRING (e.g. Azurite) to run the "
            "integration tests."
        )
    with client:
        yield client


@pytest.fixture(scope="session")
def deepagents_container(blob_service_client: BlobServiceClient) -> Iterator[str]:
    """Create the Deep Agents test container (session-scoped)."""
    try:
        blob_service_client.create_container(_DEEPAGENTS_CONTAINER)
    except ResourceExistsError:
        pass  # Left over from an interrupted run.
    yield _DEEPAGENTS_CONTAINER
    try:
        blob_service_client.delete_container(_DEEPAGENTS_CONTAINER)
    except ResourceNotFoundError:
        pass


@pytest.fixture
def backend(
    deepagents_container: str,
    account_url: Optional[str],
    connection_string: Optional[str],
) -> AzureBlobBackend:
    """Create a fresh AzureBlobBackend per test with a unique key prefix."""
    from langchain_azure_storage.deepagents import AzureBlobBackend

    prefix = f"test-{uuid.uuid4().hex[:8]}/"
    if account_url:
        return AzureBlobBackend(
            account_url=account_url,
            container_name=deepagents_container,
            prefix=prefix,
        )
    return AzureBlobBackend(
        container_name=deepagents_container,
        connection_string=connection_string,
        prefix=prefix,
    )
