"""Unit tests for ``AzureBlobBackend.aread()`` / ``read()`` (mocked, no I/O).

The async and sync methods are independent forks, so every case is covered
against both. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

import base64
from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from azure.core.exceptions import ResourceNotFoundError  # noqa: E402
from azure.storage.blob import BlobClient  # noqa: E402
from azure.storage.blob.aio import BlobClient as AsyncBlobClient  # noqa: E402

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

# Every test constructs a backend, so silence the beta warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestARead:
    async def test_read_success_returns_raw_unformatted(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        blob = make_async_download_blob("alpha\nbeta")
        container.get_blob_client.return_value = blob
        result = await backend.aread("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "alpha\nbeta"
        assert result.file_data["encoding"] == "utf-8"
        assert "1\t" not in result.file_data["content"]
        # Path-to-blob-key resolution: the backend prefix is applied.
        container.get_blob_client.assert_called_once_with("pfx/file.txt")
        blob.download_blob.assert_called_once_with()

    async def test_read_with_offset_and_limit(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(
            "l1\nl2\nl3\nl4\nl5\n"
        )
        result = await backend.aread("/f.txt", offset=1, limit=2)
        assert result.file_data is not None
        assert result.file_data["content"] == "l2\nl3\n"

    async def test_read_offset_out_of_range(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob("l1\n")
        result = await backend.aread("/f.txt", offset=100)
        assert result.error is not None
        assert "offset" in result.error.lower()

    async def test_read_empty_file(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob("")
        result = await backend.aread("/empty.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

    async def test_read_binary_returns_base64(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        raw = b"\x89PNG\r\n\x1a\n\xff\xfe\x00"
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(raw)
        result = await backend.aread("/img.png")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw

    async def test_read_binary_extension_utf8_bytes_returns_base64(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        # A non-text extension is routed to base64 even when its bytes happen to
        # be valid UTF-8, so binary content is never mistaken for text.
        raw = b"GIF89a plain ascii that decodes fine"
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(raw)
        result = await backend.aread("/logo.gif")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw

    async def test_read_unknown_extension_defaults_to_text(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        # Extensions absent from the classifier default to text (try UTF-8).
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob("hello\n")
        result = await backend.aread("/notes")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "utf-8"
        assert result.file_data["content"] == "hello\n"

    async def test_read_not_found(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        result = await backend.aread("/missing.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_read_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = await backend.aread("/src/../bad.txt")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


class TestRead:
    def test_read_success_returns_raw_unformatted(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = make_sync_download_blob("alpha\nbeta")
        container.get_blob_client.return_value = blob
        result = backend.read("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "alpha\nbeta"
        assert result.file_data["encoding"] == "utf-8"
        assert "1\t" not in result.file_data["content"]
        # Path-to-blob-key resolution: the backend prefix is applied.
        container.get_blob_client.assert_called_once_with("pfx/file.txt")
        blob.download_blob.assert_called_once_with()

    def test_read_with_offset_and_limit(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(
            "l1\nl2\nl3\nl4\nl5\n"
        )
        result = backend.read("/f.txt", offset=1, limit=2)
        assert result.file_data is not None
        assert result.file_data["content"] == "l2\nl3\n"

    def test_read_offset_out_of_range(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob("l1\n")
        result = backend.read("/f.txt", offset=100)
        assert result.error is not None
        assert "offset" in result.error.lower()

    def test_read_empty_file(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob("")
        result = backend.read("/empty.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

    def test_read_binary_returns_base64(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        raw = b"\x89PNG\r\n\x1a\n\xff\xfe\x00"
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(raw)
        result = backend.read("/img.png")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw

    def test_read_binary_extension_utf8_bytes_returns_base64(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        # A non-text extension is routed to base64 even when its bytes happen to
        # be valid UTF-8, so binary content is never mistaken for text.
        raw = b"GIF89a plain ascii that decodes fine"
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(raw)
        result = backend.read("/logo.gif")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw

    def test_read_unknown_extension_defaults_to_text(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        # Extensions absent from the classifier default to text (try UTF-8).
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob("hello\n")
        result = backend.read("/notes")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "utf-8"
        assert result.file_data["content"] == "hello\n"

    def test_read_not_found(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        result = backend.read("/missing.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_read_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.read("/src/../bad.txt")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
