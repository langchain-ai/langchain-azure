"""Unit tests for ``AzureBlobBackend`` batch upload/download (mocked, no I/O).

Covers ``aupload_files()`` / ``upload_files()`` and ``adownload_files()`` /
``download_files()``. The async and sync methods are independent forks, so
every case is covered against both. Fixtures live in the parent
``conftest.py``.
"""

from __future__ import annotations

from typing import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from azure.core.exceptions import (  # noqa: E402
    HttpResponseError,
    ResourceNotFoundError,
)
from azure.storage.blob import BlobClient  # noqa: E402
from azure.storage.blob.aio import BlobClient as AsyncBlobClient  # noqa: E402

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

# Every test constructs a backend, so silence the beta warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


def _forbidden_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 403
    return response


class TestAUploadFiles:
    async def test_upload_success(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob
        result = await backend.aupload_files([("/f.bin", b"data")])
        assert result[0].path == "/f.bin"
        assert result[0].error is None
        # Path-to-blob-key resolution and the overwriting upload.
        container.get_blob_client.assert_called_once_with("pfx/f.bin")
        blob.upload_blob.assert_called_once_with(b"data", overwrite=True)

    async def test_upload_multiple(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.aupload_files([("/a.bin", b"a"), ("/b.bin", b"b")])
        assert [r.error for r in result] == [None, None]

    async def test_upload_invalid_path(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.aupload_files([("/src/../bad.bin", b"x")])
        assert result[0].error == "invalid_path"

    async def test_upload_failure_generic_returns_exception_message(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.upload_blob.side_effect = Exception("boom")
        container.get_blob_client.return_value = blob
        result = await backend.aupload_files([("/f.bin", b"data")])
        assert result[0].error == "boom"

    async def test_upload_failure_forbidden(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.upload_blob.side_effect = HttpResponseError(response=_forbidden_response())
        container.get_blob_client.return_value = blob
        result = await backend.aupload_files([("/f.bin", b"data")])
        assert result[0].error == "permission_denied"


class TestUploadFiles:
    def test_upload_success(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.upload_files([("/f.bin", b"data")])
        assert result[0].path == "/f.bin"
        assert result[0].error is None
        # Path-to-blob-key resolution and the overwriting upload.
        container.get_blob_client.assert_called_once_with("pfx/f.bin")
        blob.upload_blob.assert_called_once_with(b"data", overwrite=True)

    def test_upload_multiple(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = MagicMock(spec=BlobClient)
        result = backend.upload_files([("/a.bin", b"a"), ("/b.bin", b"b")])
        assert [r.error for r in result] == [None, None]

    def test_upload_invalid_path(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = MagicMock(spec=BlobClient)
        result = backend.upload_files([("/src/../bad.bin", b"x")])
        assert result[0].error == "invalid_path"

    def test_upload_failure_generic_returns_exception_message(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.upload_blob.side_effect = Exception("boom")
        container.get_blob_client.return_value = blob
        result = backend.upload_files([("/f.bin", b"data")])
        assert result[0].error == "boom"

    def test_upload_failure_forbidden(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.upload_blob.side_effect = HttpResponseError(response=_forbidden_response())
        container.get_blob_client.return_value = blob
        result = backend.upload_files([("/f.bin", b"data")])
        assert result[0].error == "permission_denied"


class TestADownloadFiles:
    async def test_download_success(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        blob = make_async_download_blob(b"file content")
        container.get_blob_client.return_value = blob
        result = await backend.adownload_files(["/f.txt"])
        assert result[0].content == b"file content"
        assert result[0].error is None
        # Path-to-blob-key resolution: the backend prefix is applied.
        container.get_blob_client.assert_called_once_with("pfx/f.txt")
        blob.download_blob.assert_called_once_with()

    async def test_download_string_content_encoded(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob("text")
        result = await backend.adownload_files(["/f.txt"])
        assert result[0].content == b"text"

    async def test_download_not_found(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        result = await backend.adownload_files(["/missing.txt"])
        assert result[0].error == "file_not_found"
        assert result[0].content is None

    async def test_download_invalid_path(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.adownload_files(["/src/../bad.txt"])
        assert result[0].error == "invalid_path"

    async def test_download_failure_generic_returns_exception_message(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.download_blob.side_effect = Exception("boom")
        container.get_blob_client.return_value = blob
        result = await backend.adownload_files(["/f.bin"])
        assert result[0].error == "boom"
        assert result[0].content is None

    async def test_download_failure_forbidden(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.download_blob.side_effect = HttpResponseError(
            response=_forbidden_response()
        )
        container.get_blob_client.return_value = blob
        result = await backend.adownload_files(["/f.bin"])
        assert result[0].error == "permission_denied"

    async def test_download_failure_isolated_per_file(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        # One file erroring must not abort the whole batch.
        _, container = patched_async
        good = make_async_download_blob(b"ok")
        bad = AsyncMock(spec=AsyncBlobClient)
        bad.download_blob.side_effect = Exception("boom")
        # Key by blob name so the result is independent of concurrent call order.
        clients = {"pfx/good.bin": good, "pfx/bad.bin": bad}
        container.get_blob_client.side_effect = lambda name: clients[name]
        result = await backend.adownload_files(["/good.bin", "/bad.bin"])
        assert result[0].content == b"ok"
        assert result[0].error is None
        assert result[1].content is None
        assert result[1].error == "boom"


class TestDownloadFiles:
    def test_download_success(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = make_sync_download_blob(b"file content")
        container.get_blob_client.return_value = blob
        result = backend.download_files(["/f.txt"])
        assert result[0].content == b"file content"
        assert result[0].error is None
        # Path-to-blob-key resolution: the backend prefix is applied.
        container.get_blob_client.assert_called_once_with("pfx/f.txt")
        blob.download_blob.assert_called_once_with()

    def test_download_string_content_encoded(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob("text")
        result = backend.download_files(["/f.txt"])
        assert result[0].content == b"text"

    def test_download_not_found(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        result = backend.download_files(["/missing.bin"])
        assert result[0].error == "file_not_found"
        assert result[0].content is None

    def test_download_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.download_files(["/src/../bad.bin"])
        assert result[0].error == "invalid_path"

    def test_download_failure_generic_returns_exception_message(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.download_blob.side_effect = Exception("kaboom")
        container.get_blob_client.return_value = blob
        result = backend.download_files(["/f.bin"])
        assert result[0].error == "kaboom"
        assert result[0].content is None

    def test_download_failure_forbidden(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.download_blob.side_effect = HttpResponseError(
            response=_forbidden_response()
        )
        container.get_blob_client.return_value = blob
        result = backend.download_files(["/f.bin"])
        assert result[0].error == "permission_denied"

    def test_download_failure_isolated_per_file(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        # One file erroring must not abort the whole batch.
        _, container = patched_sync
        good = make_sync_download_blob(b"ok")
        bad = MagicMock(spec=BlobClient)
        bad.download_blob.side_effect = Exception("boom")
        # Key by blob name so the result is independent of concurrent call order.
        clients = {"pfx/good.bin": good, "pfx/bad.bin": bad}
        container.get_blob_client.side_effect = lambda name: clients[name]
        result = backend.download_files(["/good.bin", "/bad.bin"])
        assert result[0].content == b"ok"
        assert result[0].error is None
        assert result[1].content is None
        assert result[1].error == "boom"
