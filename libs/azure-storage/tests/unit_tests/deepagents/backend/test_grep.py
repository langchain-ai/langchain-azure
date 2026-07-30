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

    async def test_grep_glob_leading_slash_anchors_to_root(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # A leading "/" narrows the include filter to the search root.
        _, container = patched_async
        setup_async_grep(
            container,
            [make_blob("pfx/top.py", 5), make_blob("pfx/sub/deep.py", 5)],
            "needle\n",
        )
        result = await backend.agrep("needle", glob="/*.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/top.py"]

    async def test_grep_invalid_glob_returns_error(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
    ) -> None:
        _, container = patched_async
        setup_async_grep(container, [], "needle\n")
        result = await backend.agrep("needle", glob="{a,b}" * 12 + "x.py")
        assert result.matches is None
        assert result.error is not None
        assert "invalid glob pattern" in result.error.lower()

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

    async def test_grep_max_count_truncates(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        setup_async_grep(container, [make_blob("pfx/f.py", 5)], "hit\nhit\nhit\n")
        result = await backend.agrep("hit", max_count=2)
        assert result.error is None
        assert result.matches is not None
        assert [m["line"] for m in result.matches] == [1, 2]
        assert result.truncated is True

    async def test_grep_max_count_exactly_reached_is_not_truncated(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Hitting the cap with nothing dropped is a complete result, so the
        # model is not told to narrow a search that already returned everything.
        _, container = patched_async
        setup_async_grep(container, [make_blob("pfx/f.py", 5)], "hit\nhit\n")
        result = await backend.agrep("hit", max_count=2)
        assert result.matches is not None
        assert len(result.matches) == 2
        assert result.truncated is False

    async def test_grep_without_max_count_is_uncapped(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        setup_async_grep(container, [make_blob("pfx/f.py", 5)], "hit\n" * 50)
        result = await backend.agrep("hit")
        assert result.matches is not None
        assert len(result.matches) == 50
        assert result.truncated is False

    @pytest.mark.parametrize("max_count", [0, -1])
    async def test_grep_non_positive_max_count_returns_no_matches(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        setup_async_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
        max_count: int,
    ) -> None:
        # `max_count` arrives straight off the grep tool schema, which sets no
        # lower bound, so a model can send 0 or a negative. Left unclamped, a
        # negative would be read as a slice-from-the-end and silently drop a
        # match while reporting success. Clamping to 0 matches how the
        # reference backends' `grep_matches_from_files` treats the same input.
        _, container = patched_async
        setup_async_grep(container, [make_blob("pfx/f.py", 5)], "hit\nhit\nhit\n")
        result = await backend.agrep("hit", max_count=max_count)
        assert result.error is None
        assert result.matches == []
        assert result.truncated is True

    async def test_grep_uncapped_scans_every_candidate(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # An uncapped scan has nothing to stop for, so it dispatches every
        # candidate in one chunk rather than paying for a barrier every
        # `_MAX_CONCURRENCY` blobs. Concurrency stays bounded by the semaphore.
        blobs = [make_blob(f"pfx/f{i}.py", 5) for i in range(100)]
        _, container = patched_async
        container.list_blobs = async_list(blobs)
        container.get_blob_client.return_value = make_async_download_blob("hit\n")
        result = await backend.agrep("hit")
        assert result.truncated is False
        assert result.matches is not None
        assert len(result.matches) == 100
        assert container.get_blob_client.call_count == 100

    async def test_grep_max_count_stops_downloading_further_blobs(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # The cap exists to bound egress, not just output: once it is exceeded
        # the scan must stop fetching the remaining candidates. Candidates are
        # dispatched in windows of `_MAX_CONCURRENCY`, so with far more blobs
        # than that only the first window should be downloaded.
        blobs = [make_blob(f"pfx/f{i}.py", 5) for i in range(100)]
        _, container = patched_async
        container.list_blobs = async_list(blobs)
        container.get_blob_client.return_value = make_async_download_blob("hit\n")
        result = await backend.agrep("hit", max_count=1)
        assert result.truncated is True
        assert result.matches is not None
        assert len(result.matches) == 1
        assert container.get_blob_client.call_count == backend._MAX_CONCURRENCY


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

    def test_grep_max_count_truncates(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/f.py", 5)], "hit\nhit\nhit\n")
        result = backend.grep("hit", max_count=2)
        assert result.error is None
        assert result.matches is not None
        assert [m["line"] for m in result.matches] == [1, 2]
        assert result.truncated is True

    def test_grep_max_count_exactly_reached_is_not_truncated(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Hitting the cap with nothing dropped is a complete result, so the
        # model is not told to narrow a search that already returned everything.
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/f.py", 5)], "hit\nhit\n")
        result = backend.grep("hit", max_count=2)
        assert result.matches is not None
        assert len(result.matches) == 2
        assert result.truncated is False

    def test_grep_without_max_count_is_uncapped(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/f.py", 5)], "hit\n" * 50)
        result = backend.grep("hit")
        assert result.matches is not None
        assert len(result.matches) == 50
        assert result.truncated is False

    @pytest.mark.parametrize("max_count", [0, -1])
    def test_grep_non_positive_max_count_returns_no_matches(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
        max_count: int,
    ) -> None:
        # `max_count` arrives straight off the grep tool schema, which sets no
        # lower bound, so a model can send 0 or a negative. Left unclamped, a
        # negative would be read as a slice-from-the-end and silently drop a
        # match while reporting success. Clamping to 0 matches how the
        # reference backends' `grep_matches_from_files` treats the same input.
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/f.py", 5)], "hit\nhit\nhit\n")
        result = backend.grep("hit", max_count=max_count)
        assert result.error is None
        assert result.matches == []
        assert result.truncated is True

    def test_grep_uncapped_scans_every_candidate(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # An uncapped scan has nothing to stop for, so it dispatches every
        # candidate in one chunk rather than paying for a barrier every
        # `_MAX_CONCURRENCY` blobs. Concurrency stays bounded by the pool size.
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob(f"pfx/f{i}.py", 5) for i in range(100)
        ]
        container.get_blob_client.return_value = make_sync_download_blob("hit\n")
        result = backend.grep("hit")
        assert result.truncated is False
        assert result.matches is not None
        assert len(result.matches) == 100
        assert container.get_blob_client.call_count == 100

    def test_grep_max_count_stops_downloading_further_blobs(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # The cap exists to bound egress, not just output: once it is exceeded
        # the scan must stop fetching the remaining candidates. Candidates are
        # dispatched in windows of `_MAX_CONCURRENCY`, so with far more blobs
        # than that only the first window should be downloaded.
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob(f"pfx/f{i}.py", 5) for i in range(100)
        ]
        container.get_blob_client.return_value = make_sync_download_blob("hit\n")
        result = backend.grep("hit", max_count=1)
        assert result.truncated is True
        assert result.matches is not None
        assert len(result.matches) == 1
        assert container.get_blob_client.call_count == backend._MAX_CONCURRENCY
