"""Unit tests for ``AzureBlobBackend.awrite()`` / ``write()`` (mocked, no I/O).

The async and sync methods are independent forks, so every case is covered
against both. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from azure.core.exceptions import ResourceExistsError  # noqa: E402
from azure.storage.blob import BlobClient  # noqa: E402
from azure.storage.blob.aio import BlobClient as AsyncBlobClient  # noqa: E402

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

# Every test constructs a backend, so silence the beta warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestAWrite:
    async def test_write_new_file(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob
        result = await backend.awrite("/new.txt", "hello")
        assert result.error is None
        assert result.path == "/new.txt"
        # Path-to-blob-key resolution and the create-only upload condition.
        container.get_blob_client.assert_called_once_with("pfx/new.txt")
        blob.upload_blob.assert_called_once_with(b"hello", overwrite=False)

    async def test_write_existing_file_fails(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.upload_blob.side_effect = ResourceExistsError("exists")
        container.get_blob_client.return_value = blob
        result = await backend.awrite("/exists.txt", "hello")
        assert result.error is not None
        assert "already exists" in result.error

    async def test_write_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = await backend.awrite("/src/../bad.txt", "x")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


class TestWrite:
    def test_write_new_file(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.write("/new.txt", "hello")
        assert result.error is None
        assert result.path == "/new.txt"
        # Path-to-blob-key resolution and the create-only upload condition.
        container.get_blob_client.assert_called_once_with("pfx/new.txt")
        blob.upload_blob.assert_called_once_with(b"hello", overwrite=False)

    def test_write_existing_file_fails(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.upload_blob.side_effect = ResourceExistsError("exists")
        container.get_blob_client.return_value = blob
        result = backend.write("/exists.txt", "x")
        assert result.error is not None
        assert "already exists" in result.error

    def test_write_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.write("/src/../bad.txt", "x")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
