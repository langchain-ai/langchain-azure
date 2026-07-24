"""Unit tests for ``AzureBlobBackend.agrep()`` / ``grep()`` (mocked, no I/O).

The async and sync methods are independent forks, so every case is covered
against both. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

from typing import Any, Callable
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


class TestAGrep:
    async def test_grep_finds_matches(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        setup_async_grep(
            container, [make_blob("pfx/f.py", 5)], "hello\nbye\nhello again\n"
        )
        result = await backend.agrep("hello")
        assert result.error is None
        assert result.matches is not None
        assert [m["line"] for m in result.matches] == [1, 3]
        # A recursive listing under the backend prefix; each candidate is then
        # downloaded by its full blob key.
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/")
        container.get_blob_client.assert_called_once_with("pfx/f.py")

    async def test_grep_with_glob_filter(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        setup_async_grep(
            container,
            [make_blob("pfx/f.py", 5), make_blob("pfx/f.txt", 5)],
            "match\n",
        )
        result = await backend.agrep("match", glob="*.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/f.py"]

    async def test_grep_glob_without_slash_matches_nested_names(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # rg --glob semantics: a slash-less pattern matches names at any depth.
        _, container = patched_async
        setup_async_grep(
            container,
            [make_blob("pfx/a/b/target.py", 5), make_blob("pfx/a/ignore.txt", 5)],
            "needle\n",
        )
        result = await backend.agrep("needle", glob="*.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/a/b/target.py"]

    async def test_grep_no_matches(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        setup_async_grep(container, [make_blob("pfx/f.py", 5)], "nothing\n")
        result = await backend.agrep("missing")
        assert result.matches == []

    async def test_grep_with_path_scope(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        setup_async_grep(container, [make_blob("pfx/src/f.py", 5)], "match\n")
        result = await backend.agrep("match", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/f.py"]

    async def test_grep_skips_blobs_outside_path(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        setup_async_grep(container, [make_blob("pfx/other/f.py", 5)], "match\n")
        result = await backend.agrep("match", path="/src")
        assert result.matches == []

    async def test_grep_read_failure(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([make_blob("pfx/f.py", 5)])
        blob_client = AsyncMock(spec=AsyncBlobClient)
        blob_client.download_blob.side_effect = ResourceNotFoundError("read")
        container.get_blob_client.return_value = blob_client
        result = await backend.agrep("x")
        assert result.error is not None
        assert "could not read 1 file" in result.error.lower()

    async def test_grep_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = await backend.agrep("x", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


class TestGrep:
    def test_grep_finds_matches(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(
            container, [make_blob("pfx/f.py", 5)], "hello\nbye\nhello again\n"
        )
        result = backend.grep("hello")
        assert result.error is None
        assert result.matches is not None
        assert [m["line"] for m in result.matches] == [1, 3]
        # A recursive listing under the backend prefix; each candidate is then
        # downloaded by its full blob key.
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/")
        container.get_blob_client.assert_called_once_with("pfx/f.py")

    def test_grep_with_glob_filter(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(
            container,
            [make_blob("pfx/f.py", 5), make_blob("pfx/f.txt", 5)],
            "match\n",
        )
        result = backend.grep("match", glob="*.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/f.py"]

    def test_grep_glob_without_slash_matches_nested_names(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # rg --glob semantics: a slash-less pattern matches names at any depth.
        _, container = patched_sync
        setup_sync_grep(
            container,
            [make_blob("pfx/a/b/target.py", 5), make_blob("pfx/a/ignore.txt", 5)],
            "needle\n",
        )
        result = backend.grep("needle", glob="*.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/a/b/target.py"]

    def test_grep_no_matches(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/f.py", 5)], "nothing\n")
        result = backend.grep("missing")
        assert result.matches == []

    def test_grep_with_path_scope(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/src/f.py", 5)], "match\n")
        result = backend.grep("match", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/f.py"]

    def test_grep_skips_blobs_outside_path(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/other/f.py", 5)], "match\n")
        result = backend.grep("match", path="/src")
        assert result.matches == []

    def test_grep_read_failure(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = [make_blob("pfx/f.py", 5)]
        blob_client = MagicMock(spec=BlobClient)
        blob_client.download_blob.side_effect = ResourceNotFoundError("read")
        container.get_blob_client.return_value = blob_client
        result = backend.grep("x")
        assert result.error is not None
        assert "could not read 1 file" in result.error.lower()

    def test_grep_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.grep("x", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
