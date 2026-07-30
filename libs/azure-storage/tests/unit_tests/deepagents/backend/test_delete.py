"""Unit tests for ``AzureBlobBackend.adelete()`` / ``delete()`` (mocked, no I/O).

The async and sync methods are independent forks, so every case is covered
against both. Fixtures live in the parent ``conftest.py``.
"""

from __future__ import annotations

from typing import Any, Callable
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


def _assert_deleted(container: MagicMock, blob: Any, expected: list[str]) -> None:
    """Assert exactly *expected* keys were resolved **and** actually deleted.

    Checking the resolved keys alone would pass a backend that lists correctly
    and then never calls ``delete_blob``, so the call count on the shared blob
    mock is asserted too.
    """
    resolved = sorted(call.args[0] for call in container.get_blob_client.call_args_list)
    assert resolved == sorted(expected)
    assert blob.delete_blob.call_count == len(expected)


class TestADelete:
    async def test_delete_single_file(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([make_blob("pfx/f.txt", 5)])
        blob = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob
        result = await backend.adelete("/f.txt")
        assert result.error is None
        assert result.path == "/f.txt"
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/f.txt")
        _assert_deleted(container, blob, ["pfx/f.txt"])

    async def test_delete_directory_is_recursive(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list(
            [
                make_blob("pfx/src", 5),
                make_blob("pfx/src/a.py", 5),
                make_blob("pfx/src/deep/b.py", 5),
            ]
        )
        blob = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob
        result = await backend.adelete("/src")
        assert result.error is None
        _assert_deleted(
            container,
            blob,
            [
                "pfx/src",
                "pfx/src/a.py",
                "pfx/src/deep/b.py",
            ],
        )

    async def test_delete_does_not_touch_sibling_sharing_name_stem(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # The server-side listing prefix "pfx/src" also returns "pfx/src-backup".
        # Only the exact key and keys under "pfx/src/" may be deleted.
        _, container = patched_async
        container.list_blobs = async_list(
            [
                make_blob("pfx/src/a.py", 5),
                make_blob("pfx/src-backup/a.py", 5),
                make_blob("pfx/srcx.py", 5),
            ]
        )
        blob = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob
        result = await backend.adelete("/src")
        assert result.error is None
        _assert_deleted(container, blob, ["pfx/src/a.py"])

    async def test_delete_root_removes_only_prefix_namespace(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Deleting "/" is a recursive wipe, but the configured prefix still
        # bounds it: the listing is scoped to "pfx/" so blobs elsewhere in the
        # container are never enumerated, let alone deleted.
        _, container = patched_async
        container.list_blobs = async_list(
            [make_blob("pfx/a.txt", 5), make_blob("pfx/d/b.txt", 5)]
        )
        blob = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob
        result = await backend.adelete("/")
        assert result.error is None
        assert result.path == "/"
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/")
        _assert_deleted(container, blob, ["pfx/a.txt", "pfx/d/b.txt"])

    async def test_delete_missing_path_returns_not_found(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([])
        result = await backend.adelete("/nope.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()
        container.get_blob_client.assert_not_called()

    async def test_delete_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = await backend.adelete("/src/../bad.txt")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    async def test_delete_tolerates_blob_removed_concurrently(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([make_blob("pfx/f.txt", 5)])
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.delete_blob.side_effect = ResourceNotFoundError("gone")
        container.get_blob_client.return_value = blob
        result = await backend.adelete("/f.txt")
        assert result.error is None
        assert result.path == "/f.txt"

    async def test_delete_reports_partial_failure(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list(
            [make_blob("pfx/d/a.txt", 5), make_blob("pfx/d/b.txt", 5)]
        )

        def _client(key: str) -> AsyncMock:
            blob = AsyncMock(spec=AsyncBlobClient)
            if key.endswith("b.txt"):
                blob.delete_blob.side_effect = HttpResponseError("boom")
            return blob

        container.get_blob_client.side_effect = _client
        result = await backend.adelete("/d")
        assert result.error is not None
        assert "removed 1 file(s) but failed on 1" in result.error
        assert "/d/b.txt" in result.error

    async def test_delete_reports_total_failure_distinctly(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Nothing was removed, so the error must not imply a partial delete --
        # retrying a wholly failed delete is safe, a half-done one is not.
        _, container = patched_async
        container.list_blobs = async_list([make_blob("pfx/d/a.txt", 5)])
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.delete_blob.side_effect = HttpResponseError("boom")
        container.get_blob_client.return_value = blob
        result = await backend.adelete("/d")
        assert result.error is not None
        assert "failed on all 1 file(s)" in result.error
        assert "removed" not in result.error

    async def test_delete_removes_pseudo_directory_marker_blob(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # `ls` skips marker blobs (keys ending in "/"), but they are real blobs
        # inside the subtree, so a recursive delete has to take them with it.
        # This is the one place the two paths deliberately disagree.
        _, container = patched_async
        container.list_blobs = async_list(
            [make_blob("pfx/src/", 0), make_blob("pfx/src/a.py", 5)]
        )
        blob = AsyncMock(spec=AsyncBlobClient)
        container.get_blob_client.return_value = blob
        result = await backend.adelete("/src")
        assert result.error is None
        _assert_deleted(container, blob, ["pfx/src/", "pfx/src/a.py"])


class TestDelete:
    def test_delete_single_file(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = [make_blob("pfx/f.txt", 5)]
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.delete("/f.txt")
        assert result.error is None
        assert result.path == "/f.txt"
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/f.txt")
        _assert_deleted(container, blob, ["pfx/f.txt"])

    def test_delete_directory_is_recursive(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob("pfx/src", 5),
            make_blob("pfx/src/a.py", 5),
            make_blob("pfx/src/deep/b.py", 5),
        ]
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.delete("/src")
        assert result.error is None
        _assert_deleted(
            container,
            blob,
            [
                "pfx/src",
                "pfx/src/a.py",
                "pfx/src/deep/b.py",
            ],
        )

    def test_delete_does_not_touch_sibling_sharing_name_stem(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # The server-side listing prefix "pfx/src" also returns "pfx/src-backup".
        # Only the exact key and keys under "pfx/src/" may be deleted.
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob("pfx/src/a.py", 5),
            make_blob("pfx/src-backup/a.py", 5),
            make_blob("pfx/srcx.py", 5),
        ]
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.delete("/src")
        assert result.error is None
        _assert_deleted(container, blob, ["pfx/src/a.py"])

    def test_delete_root_removes_only_prefix_namespace(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Deleting "/" is a recursive wipe, but the configured prefix still
        # bounds it: the listing is scoped to "pfx/" so blobs elsewhere in the
        # container are never enumerated, let alone deleted.
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob("pfx/a.txt", 5),
            make_blob("pfx/d/b.txt", 5),
        ]
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.delete("/")
        assert result.error is None
        assert result.path == "/"
        container.list_blobs.assert_called_once_with(name_starts_with="pfx/")
        _assert_deleted(container, blob, ["pfx/a.txt", "pfx/d/b.txt"])

    def test_delete_missing_path_returns_not_found(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = []
        result = backend.delete("/nope.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()
        container.get_blob_client.assert_not_called()

    def test_delete_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.delete("/src/../bad.txt")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    def test_delete_tolerates_blob_removed_concurrently(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = [make_blob("pfx/f.txt", 5)]
        blob = MagicMock(spec=BlobClient)
        blob.delete_blob.side_effect = ResourceNotFoundError("gone")
        container.get_blob_client.return_value = blob
        result = backend.delete("/f.txt")
        assert result.error is None
        assert result.path == "/f.txt"

    def test_delete_reports_partial_failure(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob("pfx/d/a.txt", 5),
            make_blob("pfx/d/b.txt", 5),
        ]

        def _client(key: str) -> MagicMock:
            blob = MagicMock(spec=BlobClient)
            if key.endswith("b.txt"):
                blob.delete_blob.side_effect = HttpResponseError("boom")
            return blob

        container.get_blob_client.side_effect = _client
        result = backend.delete("/d")
        assert result.error is not None
        assert "removed 1 file(s) but failed on 1" in result.error
        assert "/d/b.txt" in result.error

    def test_delete_reports_total_failure_distinctly(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # Nothing was removed, so the error must not imply a partial delete --
        # retrying a wholly failed delete is safe, a half-done one is not.
        _, container = patched_sync
        container.list_blobs.return_value = [make_blob("pfx/d/a.txt", 5)]
        blob = MagicMock(spec=BlobClient)
        blob.delete_blob.side_effect = HttpResponseError("boom")
        container.get_blob_client.return_value = blob
        result = backend.delete("/d")
        assert result.error is not None
        assert "failed on all 1 file(s)" in result.error
        assert "removed" not in result.error

    def test_delete_removes_pseudo_directory_marker_blob(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # `ls` skips marker blobs (keys ending in "/"), but they are real blobs
        # inside the subtree, so a recursive delete has to take them with it.
        # This is the one place the two paths deliberately disagree.
        _, container = patched_sync
        container.list_blobs.return_value = [
            make_blob("pfx/src/", 0),
            make_blob("pfx/src/a.py", 5),
        ]
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.delete("/src")
        assert result.error is None
        _assert_deleted(container, blob, ["pfx/src/", "pfx/src/a.py"])


class TestDeleteWithoutPrefix:
    """A prefix-less backend deletes container-wide -- the documented footgun."""

    def test_delete_root_without_prefix_lists_whole_container(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        backend = AzureBlobBackend.from_connection_string(
            "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;",
            "test",
        )
        _, container = patched_sync
        container.list_blobs.return_value = [make_blob("a.txt", 5)]
        blob = MagicMock(spec=BlobClient)
        container.get_blob_client.return_value = blob
        result = backend.delete("/")
        assert result.error is None
        container.list_blobs.assert_called_once_with(name_starts_with=None)
        _assert_deleted(container, blob, ["a.txt"])
