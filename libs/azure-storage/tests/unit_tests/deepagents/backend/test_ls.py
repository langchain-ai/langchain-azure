"""Unit tests for ``AzureBlobBackend.als()`` / ``ls()`` (mocked, no I/O).

The async and sync methods are independent forks, so every case is covered
against both. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

# Every test constructs a backend, so silence the beta warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

_MODIFIED = datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestALs:
    async def test_ls_files(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.walk_blobs = async_list(
            [
                make_blob("pfx/src/a.py", 10, last_modified=_MODIFIED),
                make_blob("pfx/src/b.py", 20),
            ]
        )
        result = await backend.als("/src")
        assert result.entries is not None
        assert {e["path"] for e in result.entries} == {"/src/a.py", "/src/b.py"}
        assert result.entries[0]["size"] == 10
        assert result.entries[0]["modified_at"] == _MODIFIED.isoformat()
        # Path-to-prefix resolution: a non-recursive delimited walk under the
        # backend prefix plus the requested directory.
        container.walk_blobs.assert_called_once_with(
            name_starts_with="pfx/src/", delimiter="/"
        )

    async def test_ls_synthesizes_directories(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
        make_prefix: Callable[[str], MagicMock],
    ) -> None:
        # walk_blobs collapses subdirectories into BlobPrefix entries server-side.
        _, container = patched_async
        container.walk_blobs = async_list(
            [make_prefix("pfx/src/sub/"), make_blob("pfx/src/b.py", 10)]
        )
        result = await backend.als("/src")
        assert result.entries is not None
        dirs = [e for e in result.entries if e["is_dir"]]
        assert [d["path"] for d in dirs] == ["/src/sub/"]

    async def test_ls_empty(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        _, container = patched_async
        container.walk_blobs = async_list([])
        result = await backend.als("/empty")
        assert result.entries == []

    async def test_ls_path_without_leading_slash(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.walk_blobs = async_list([make_blob("pfx/src/a.py", 1)])
        result = await backend.als("src")
        assert result.entries is not None
        assert result.entries[0]["path"] == "/src/a.py"

    async def test_ls_skips_marker_blobs(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # A pseudo-directory marker blob (name ending in "/") is not a file.
        _, container = patched_async
        container.walk_blobs = async_list([make_blob("pfx/src/", 0)])
        result = await backend.als("/src")
        assert result.entries == []

    async def test_ls_invalid_path_returns_error(
        self, backend: AzureBlobBackend
    ) -> None:
        result = await backend.als("/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.entries is None


class TestLs:
    def test_ls_files(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.walk_blobs.return_value = [
            make_blob("pfx/src/a.py", 10, last_modified=_MODIFIED),
            make_blob("pfx/src/b.py", 20),
        ]
        result = backend.ls("/src")
        assert result.entries is not None
        assert {e["path"] for e in result.entries} == {"/src/a.py", "/src/b.py"}
        assert result.entries[0]["size"] == 10
        assert result.entries[0]["modified_at"] == _MODIFIED.isoformat()
        # Path-to-prefix resolution: a non-recursive delimited walk under the
        # backend prefix plus the requested directory.
        container.walk_blobs.assert_called_once_with(
            name_starts_with="pfx/src/", delimiter="/"
        )

    def test_ls_synthesizes_directories(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
        make_prefix: Callable[[str], MagicMock],
    ) -> None:
        # walk_blobs collapses subdirectories into BlobPrefix entries server-side.
        _, container = patched_sync
        container.walk_blobs.return_value = [
            make_prefix("pfx/src/sub/"),
            make_blob("pfx/src/b.py", 10),
        ]
        result = backend.ls("/src")
        assert result.entries is not None
        dirs = [e for e in result.entries if e["is_dir"]]
        assert [d["path"] for d in dirs] == ["/src/sub/"]

    def test_ls_empty(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        container.walk_blobs.return_value = []
        result = backend.ls("/empty")
        assert result.entries == []

    def test_ls_path_without_leading_slash(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.walk_blobs.return_value = [make_blob("pfx/src/a.py", 1)]
        result = backend.ls("src")
        assert result.entries is not None
        assert result.entries[0]["path"] == "/src/a.py"

    def test_ls_skips_marker_blobs(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # A pseudo-directory marker blob (name ending in "/") is not a file.
        _, container = patched_sync
        container.walk_blobs.return_value = [make_blob("pfx/src/", 0)]
        result = backend.ls("/src")
        assert result.entries == []

    def test_ls_invalid_path_returns_error(self, backend: AzureBlobBackend) -> None:
        result = backend.ls("/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.entries is None
