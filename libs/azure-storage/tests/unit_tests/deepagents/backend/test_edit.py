"""Unit tests for ``AzureBlobBackend.aedit()`` / ``edit()`` (mocked, no I/O).

The async and sync methods are independent forks, so every case is covered
against both. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from azure.core import MatchConditions  # noqa: E402
from azure.core.exceptions import (  # noqa: E402
    ResourceModifiedError,
    ResourceNotFoundError,
)
from azure.storage.blob import BlobClient  # noqa: E402
from azure.storage.blob.aio import BlobClient as AsyncBlobClient  # noqa: E402

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

# Every test constructs a backend, so silence the beta warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestAEdit:
    async def test_edit_success(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(
            "hello world", etag="etag-1"
        )
        result = await backend.aedit("/f.txt", "hello", "bye")
        assert result.error is None
        assert result.path == "/f.txt"
        assert result.occurrences == 1
        # Path-to-blob-key resolution: the backend prefix is applied.
        container.get_blob_client.assert_called_once_with("pfx/f.txt")

    async def test_edit_uploads_with_etag_condition(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        blob = make_async_download_blob("hello world", etag="etag-42")
        container.get_blob_client.return_value = blob
        await backend.aedit("/f.txt", "hello", "bye")
        args, kwargs = blob.upload_blob.call_args
        assert args == (b"bye world",)
        assert kwargs["overwrite"] is True
        assert kwargs["etag"] == "etag-42"
        assert kwargs["match_condition"] == MatchConditions.IfNotModified

    async def test_edit_concurrent_modification(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        blob = make_async_download_blob("hello world", etag="etag-1")
        blob.upload_blob.side_effect = ResourceModifiedError("etag mismatch")
        container.get_blob_client.return_value = blob
        result = await backend.aedit("/f.txt", "hello", "bye")
        assert result.error is not None
        assert "modified concurrently" in result.error

    async def test_edit_not_found(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        result = await backend.aedit("/missing.txt", "a", "b")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_edit_replace_all(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(
            "aaa", etag="etag-1"
        )
        result = await backend.aedit("/f.txt", "a", "b", replace_all=True)
        assert result.occurrences == 3

    async def test_edit_multiple_without_replace_all_fails(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(
            "aaa", etag="etag-1"
        )
        result = await backend.aedit("/f.txt", "a", "b")
        assert result.error is not None

    async def test_edit_no_match(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(
            "hello", etag="etag-1"
        )
        result = await backend.aedit("/f.txt", "nope", "x")
        assert result.error is not None

    async def test_edit_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = await backend.aedit("/src/../bad.txt", "a", "b")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


class TestEdit:
    def test_edit_success(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(
            "hello world", etag="etag-1"
        )
        result = backend.edit("/f.txt", "hello", "bye")
        assert result.error is None
        assert result.path == "/f.txt"
        assert result.occurrences == 1
        # Path-to-blob-key resolution: the backend prefix is applied.
        container.get_blob_client.assert_called_once_with("pfx/f.txt")

    def test_edit_uploads_with_etag_condition(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = make_sync_download_blob("hello world", etag="etag-42")
        container.get_blob_client.return_value = blob
        backend.edit("/f.txt", "hello", "bye")
        args, kwargs = blob.upload_blob.call_args
        assert args == (b"bye world",)
        assert kwargs["overwrite"] is True
        assert kwargs["etag"] == "etag-42"
        assert kwargs["match_condition"] == MatchConditions.IfNotModified

    def test_edit_concurrent_modification(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = make_sync_download_blob("hello world", etag="etag-1")
        blob.upload_blob.side_effect = ResourceModifiedError("etag mismatch")
        container.get_blob_client.return_value = blob
        result = backend.edit("/f.txt", "hello", "bye")
        assert result.error is not None
        assert "modified concurrently" in result.error

    def test_edit_not_found(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        result = backend.edit("/missing.txt", "a", "b")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_replace_all(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(
            "aaa", etag="etag-1"
        )
        result = backend.edit("/f.txt", "a", "b", replace_all=True)
        assert result.occurrences == 3

    def test_edit_multiple_without_replace_all_fails(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(
            "aaa", etag="etag-1"
        )
        result = backend.edit("/f.txt", "a", "b")
        assert result.error is not None

    def test_edit_no_match(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(
            "hello", etag="etag-1"
        )
        result = backend.edit("/f.txt", "nope", "x")
        assert result.error is not None

    def test_edit_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.edit("/src/../bad.txt", "a", "b")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
