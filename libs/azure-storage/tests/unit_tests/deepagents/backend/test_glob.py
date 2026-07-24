"""Unit tests for ``AzureBlobBackend.aglob()`` / ``glob()`` (mocked, no I/O).

The async and sync methods are independent forks, so every case is covered
against both. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from azure.storage.blob import BlobClient  # noqa: E402
from azure.storage.blob.aio import BlobClient as AsyncBlobClient  # noqa: E402

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402

# Every test constructs a backend, so silence the beta warning module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestAGlob:
    async def test_glob_pattern(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        container.list_blobs = async_list(
            [make_blob("pfx/src/a.py", 10), make_blob("pfx/src/b.txt", 5)]
        )
        result = await backend.aglob("*.py", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/a.py"]
        # Path-to-prefix resolution: a recursive listing under the backend
        # prefix plus the search directory.
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/src/")

    async def test_glob_recursive(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list(
            [make_blob("pfx/a.py", 1), make_blob("pfx/sub/b.py", 2)]
        )
        result = await backend.aglob("**/*.py", path="/")
        assert result.matches is not None
        assert {m["path"] for m in result.matches} == {"/a.py", "/sub/b.py"}

    async def test_glob_bare_pattern_is_not_recursive(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Shell-glob semantics: a slash-less pattern matches only the immediate
        # children of *path*, not nested files (use ``**/*.py`` for that).
        _, container = patched_async
        container.list_blobs = async_list(
            [make_blob("pfx/a.py", 1), make_blob("pfx/sub/deep/b.py", 2)]
        )
        result = await backend.aglob("*.py", path="/")
        assert result.matches is not None
        assert {m["path"] for m in result.matches} == {"/a.py"}

    async def test_glob_skips_blobs_outside_base(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        container.list_blobs = async_list([make_blob("pfx/other/a.py", 1)])
        result = await backend.aglob("*.py", path="/src")
        assert result.matches == []

    async def test_glob_path_treated_as_directory(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        # *path* is always a directory prefix: an exact-file path lists what is
        # *under* it (nothing here), never matching the file itself, and no
        # extra get_blob_properties() probe is issued.
        _, container = patched_async
        container.list_blobs = async_list([])
        blob_client = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob_client
        result = await backend.aglob("main.py", path="/src/main.py")
        assert result.matches == []
        blob_client.get_blob_properties.assert_not_called()

    async def test_glob_invalid_path_returns_error(
        self, backend: AzureBlobBackend
    ) -> None:
        result = await backend.aglob("*.py", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.matches is None


class TestGlob:
    def test_glob_pattern(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = MagicMock(spec=BlobClient)
        container.list_blobs.return_value = [
            make_blob("pfx/src/a.py", 10),
            make_blob("pfx/src/b.txt", 5),
        ]
        result = backend.glob("*.py", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/a.py"]
        # Path-to-prefix resolution: a recursive listing under the backend
        # prefix plus the search directory.
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/src/")

    def test_glob_recursive(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob("pfx/a.py", 1),
            make_blob("pfx/sub/b.py", 2),
        ]
        result = backend.glob("**/*.py", path="/")
        assert result.matches is not None
        assert {m["path"] for m in result.matches} == {"/a.py", "/sub/b.py"}

    def test_glob_bare_pattern_is_not_recursive(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Shell-glob semantics: a slash-less pattern matches only the immediate
        # children of *path*, not nested files (use ``**/*.py`` for that).
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob("pfx/a.py", 1),
            make_blob("pfx/sub/deep/b.py", 2),
        ]
        result = backend.glob("*.py", path="/")
        assert result.matches is not None
        assert {m["path"] for m in result.matches} == {"/a.py"}

    def test_glob_skips_blobs_outside_base(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = MagicMock(spec=BlobClient)
        container.list_blobs.return_value = [make_blob("pfx/other/a.py", 1)]
        result = backend.glob("*.py", path="/src")
        assert result.matches == []

    def test_glob_path_treated_as_directory(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        # *path* is always a directory prefix: an exact-file path lists what is
        # *under* it (nothing here), never matching the file itself, and no
        # extra get_blob_properties() probe is issued.
        _, container = patched_sync
        container.list_blobs.return_value = []
        blob_client = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob_client
        result = backend.glob("main.py", path="/src/main.py")
        assert result.matches == []
        blob_client.get_blob_properties.assert_not_called()

    def test_glob_invalid_path_returns_error(self, backend: AzureBlobBackend) -> None:
        result = backend.glob("*.py", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.matches is None
