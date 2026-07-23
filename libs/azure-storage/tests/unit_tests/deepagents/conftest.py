"""Shared fixtures for the AzureBlobBackend unit tests (mocked, no I/O).

The backend talks to Azure through the sync ``ContainerClient`` /
``BlobClient`` and their ``azure.storage.blob.aio`` async twins. Every mock
here is built with ``spec=`` against the real class so a mis-stubbed or
nonexistent method fails loudly instead of silently returning a fresh
``MagicMock``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
# Guarding the conftest as well skips the whole directory on older Pythons.
pytest.importorskip("deepagents")

from azure.storage.blob import (  # noqa: E402
    BlobClient,
    BlobPrefix,
    BlobProperties,
    ContainerClient,
)
from azure.storage.blob._download import StorageStreamDownloader  # noqa: E402
from azure.storage.blob.aio import BlobClient as AsyncBlobClient  # noqa: E402
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient  # noqa: E402
from azure.storage.blob.aio import (  # noqa: E402
    StorageStreamDownloader as AsyncStorageStreamDownloader,
)

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

# Patch-target base and the fake connection string every backend is built from.
_BACKEND = "langchain_azure_storage.deepagents.backend"
_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"


# ------------------------------------------------------------------
# Backend construction
# ------------------------------------------------------------------


@pytest.fixture
def backend() -> AzureBlobBackend:
    """A connection-string backend with prefix ``pfx/``."""
    return AzureBlobBackend.from_connection_string(_CONN_STR, "test", prefix="pfx/")


@pytest.fixture
def make_acct_backend() -> Callable[..., AzureBlobBackend]:
    """Factory for an account-URL backend (optionally with a credential)."""

    def _make(credential: Any = None) -> AzureBlobBackend:
        return AzureBlobBackend(
            account_url="https://x.blob.core.windows.net",
            container_name="test",
            prefix="pfx/",
            credential=credential,
        )

    return _make


# ------------------------------------------------------------------
# Patched container clients
#
# ``_get_sync_container`` builds the client via ``from_connection_string`` on
# the connection-string path and via the class constructor on the account-URL
# path, so both call sites are wired to the same spec'd container instance.
# ------------------------------------------------------------------


@pytest.fixture
def sync_container() -> MagicMock:
    return MagicMock(spec=ContainerClient)


@pytest.fixture
def patched_sync(
    sync_container: MagicMock,
) -> Iterator[tuple[MagicMock, MagicMock]]:
    """Patch ``backend.ContainerClient``; yield ``(class_mock, container)``."""
    with patch(f"{_BACKEND}.ContainerClient") as mock_cls:
        mock_cls.return_value = sync_container
        mock_cls.from_connection_string.return_value = sync_container
        yield mock_cls, sync_container


@pytest.fixture
def async_container() -> MagicMock:
    return MagicMock(spec=AsyncContainerClient)


@pytest.fixture
def patched_async(
    async_container: MagicMock,
) -> Iterator[tuple[MagicMock, MagicMock]]:
    """Patch ``backend.AsyncContainerClient``; yield ``(class_mock, container)``."""
    with patch(f"{_BACKEND}.AsyncContainerClient") as mock_cls:
        mock_cls.return_value = async_container
        mock_cls.from_connection_string.return_value = async_container
        yield mock_cls, async_container


# ------------------------------------------------------------------
# Blob-listing item factories (``BlobProperties`` / ``BlobPrefix`` entries)
# ------------------------------------------------------------------


@pytest.fixture
def make_blob() -> Callable[..., MagicMock]:
    """A ``list_blobs`` / ``walk_blobs`` file entry (``BlobProperties``)."""

    def _make(name: str, size: int = 0, last_modified: Any = None) -> MagicMock:
        blob = MagicMock(spec=BlobProperties)
        blob.name = name
        blob.size = size
        blob.last_modified = last_modified
        return blob

    return _make


@pytest.fixture
def make_prefix() -> Callable[[str], MagicMock]:
    """A ``walk_blobs`` subdirectory entry (matches ``isinstance(_, BlobPrefix)``)."""

    def _make(name: str) -> MagicMock:
        prefix = MagicMock(spec=BlobPrefix)
        prefix.name = name
        return prefix

    return _make


@pytest.fixture
def async_list() -> Callable[[list[Any]], Callable[..., AsyncIterator[Any]]]:
    """Turn a list into a ``walk_blobs`` / ``list_blobs`` async-iterator stand-in."""

    def _make(blobs: list[Any]) -> Callable[..., AsyncIterator[Any]]:
        async def _gen(**kwargs: Any) -> AsyncIterator[Any]:
            for b in blobs:
                yield b

        return _gen

    return _make


# ------------------------------------------------------------------
# Download-stream + blob-client factories
#
# ``StorageStreamDownloader.properties`` is a genuine instance attribute set in
# __init__, so it is absent from the class-level spec; the etag branch assigns
# it back explicitly (a direct setattr is allowed even under spec).
# ------------------------------------------------------------------


@pytest.fixture
def make_async_download_stream() -> Callable[..., AsyncMock]:
    def _make(content: Any, *, etag: str | None = None) -> AsyncMock:
        stream = AsyncMock(spec=AsyncStorageStreamDownloader)
        stream.readall.return_value = content
        if etag is not None:
            stream.properties = MagicMock()
            stream.properties.etag = etag
        return stream

    return _make


@pytest.fixture
def make_async_download_blob(
    make_async_download_stream: Callable[..., AsyncMock],
) -> Callable[..., AsyncMock]:
    """Async blob client whose ``download_blob().readall()`` yields *content*."""

    def _make(content: Any, *, etag: str | None = None) -> AsyncMock:
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.download_blob.return_value = make_async_download_stream(content, etag=etag)
        return blob

    return _make


@pytest.fixture
def make_sync_download_stream() -> Callable[..., MagicMock]:
    def _make(content: Any, *, etag: str | None = None) -> MagicMock:
        stream = MagicMock(spec=StorageStreamDownloader)
        stream.readall.return_value = content
        if etag is not None:
            stream.properties = MagicMock()
            stream.properties.etag = etag
        return stream

    return _make


@pytest.fixture
def make_sync_download_blob(
    make_sync_download_stream: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    """Sync blob client whose ``download_blob().readall()`` yields *content*."""

    def _make(content: Any, *, etag: str | None = None) -> MagicMock:
        blob = MagicMock(spec=BlobClient)
        blob.download_blob.return_value = make_sync_download_stream(content, etag=etag)
        return blob

    return _make


# ------------------------------------------------------------------
# grep container setup (configures the patched container in place)
# ------------------------------------------------------------------


@pytest.fixture
def setup_async_grep(
    async_list: Callable[[list[Any]], Callable[..., AsyncIterator[Any]]],
    make_async_download_blob: Callable[..., AsyncMock],
) -> Callable[[MagicMock, list[Any], Any], None]:
    def _setup(container: MagicMock, blobs: list[Any], content: Any) -> None:
        container.list_blobs = async_list(blobs)
        container.get_blob_client.return_value = make_async_download_blob(content)

    return _setup


@pytest.fixture
def setup_sync_grep(
    make_sync_download_blob: Callable[..., MagicMock],
) -> Callable[[MagicMock, list[Any], Any], None]:
    def _setup(container: MagicMock, blobs: list[Any], content: Any) -> None:
        container.list_blobs.return_value = blobs
        container.get_blob_client.return_value = make_sync_download_blob(content)

    return _setup
