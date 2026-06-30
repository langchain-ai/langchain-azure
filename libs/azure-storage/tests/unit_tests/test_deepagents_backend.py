"""Unit tests for AzureBlobBackend (mocked, no I/O)."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

from langchain_azure_storage.deepagents import AzureBlobBackend
from langchain_azure_storage.deepagents._path import (
    from_blob_key,
    get_prefix_for_path,
    normalize_path,
    to_blob_key,
)
from langchain_azure_storage.deepagents._utils import build_file_info

_BACKEND = "langchain_azure_storage.deepagents.backend"
_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"


# ------------------------------------------------------------------
# Path utility tests
# ------------------------------------------------------------------


class TestNormalizePath:
    def test_strips_leading_slash(self) -> None:
        assert normalize_path("/src/main.py") == "src/main.py"

    def test_root(self) -> None:
        assert normalize_path("/") == ""

    def test_double_slashes(self) -> None:
        assert normalize_path("//src//main.py") == "src/main.py"

    def test_rejects_path_traversal(self) -> None:
        with pytest.raises(ValueError, match="Path traversal"):
            normalize_path("/src/../secrets.txt")

    def test_rejects_windows_absolute_path(self) -> None:
        with pytest.raises(ValueError, match="Windows absolute paths"):
            normalize_path("C:/temp/file.txt")


class TestToBlobKey:
    def test_with_prefix(self) -> None:
        assert to_blob_key("workspace/", "/src/main.py") == "workspace/src/main.py"

    def test_without_prefix(self) -> None:
        assert to_blob_key("", "/src/main.py") == "src/main.py"

    def test_prefix_no_trailing_slash(self) -> None:
        assert to_blob_key("workspace", "/src/main.py") == "workspace/src/main.py"


class TestFromBlobKey:
    def test_with_prefix(self) -> None:
        assert from_blob_key("workspace/", "workspace/src/main.py") == "/src/main.py"

    def test_without_prefix(self) -> None:
        assert from_blob_key("", "src/main.py") == "/src/main.py"


class TestGetPrefixForPath:
    def test_root_with_prefix(self) -> None:
        assert get_prefix_for_path("workspace/", "/") == "workspace/"

    def test_subdir_with_prefix(self) -> None:
        assert get_prefix_for_path("workspace/", "/src") == "workspace/src/"

    def test_subdir_no_prefix(self) -> None:
        assert get_prefix_for_path("", "/src") == "src/"


class TestBuildFileInfo:
    def test_defaults(self) -> None:
        assert build_file_info("/src/main.py") == {
            "path": "/src/main.py",
            "is_dir": False,
            "size": 0,
            "modified_at": "",
        }

    def test_directory(self) -> None:
        assert build_file_info("/src/", is_dir=True)["is_dir"] is True


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestConstructor:
    def test_is_backend_protocol(self) -> None:
        from deepagents.backends.protocol import BackendProtocol

        backend = AzureBlobBackend(
            account_url="https://x.blob.core.windows.net", container_name="c"
        )
        assert isinstance(backend, BackendProtocol)

    def test_prefix_defaults_to_empty(self) -> None:
        backend = AzureBlobBackend(
            account_url="https://x.blob.core.windows.net", container_name="c"
        )
        assert backend._prefix == ""

    def test_prefix_none_normalized(self) -> None:
        backend = AzureBlobBackend(
            account_url="https://x.blob.core.windows.net",
            container_name="c",
            prefix=None,
        )
        assert backend._prefix == ""


# ------------------------------------------------------------------
# Mock helpers for the sync / async container clients
# ------------------------------------------------------------------


def _make_backend(prefix: str = "pfx/") -> AzureBlobBackend:
    return AzureBlobBackend(
        container_name="test", prefix=prefix, connection_string=_CONN_STR
    )


def _make_blob(name: str, size: int = 0, metadata: dict | None = None) -> MagicMock:
    blob = MagicMock()
    blob.name = name
    blob.size = size
    blob.metadata = metadata
    return blob


@contextmanager
def _patch_async(container: MagicMock) -> Iterator[MagicMock]:
    """Patch AsyncContainerClient so ``_async_container`` yields *container*."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=container)
    cm.__aexit__ = AsyncMock(return_value=None)
    with patch(f"{_BACKEND}.AsyncContainerClient") as mock_cls:
        mock_cls.from_connection_string.return_value = cm
        yield mock_cls


@contextmanager
def _patch_sync(container: MagicMock) -> Iterator[MagicMock]:
    """Patch ContainerClient so ``_sync_container`` yields *container*."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=container)
    cm.__exit__ = MagicMock(return_value=None)
    with patch(f"{_BACKEND}.ContainerClient") as mock_cls:
        mock_cls.from_connection_string.return_value = cm
        yield mock_cls


def _async_list(blobs: list[Any]) -> Any:
    async def _gen(**kwargs: Any) -> AsyncIterator[Any]:
        for b in blobs:
            yield b

    return _gen


def _download_blob_mock(content: Any) -> AsyncMock:
    """Async blob client whose download_blob().readall() yields *content*."""
    blob = AsyncMock()
    stream = AsyncMock()
    stream.readall.return_value = content
    blob.download_blob.return_value = stream
    return blob


# ------------------------------------------------------------------
# user agent
# ------------------------------------------------------------------


class TestUserAgent:
    async def test_user_agent_passed_to_async_client(self) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        container = MagicMock()
        container.list_blobs = _async_list([])
        with _patch_async(container) as mock_cls:
            await _make_backend().als("/")
        assert (
            mock_cls.from_connection_string.call_args.kwargs["user_agent"] == USER_AGENT
        )

    def test_user_agent_passed_to_sync_client(self) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        container = MagicMock()
        container.list_blobs.return_value = []
        with _patch_sync(container) as mock_cls:
            _make_backend().ls("/")
        assert (
            mock_cls.from_connection_string.call_args.kwargs["user_agent"] == USER_AGENT
        )


# ------------------------------------------------------------------
# read
# ------------------------------------------------------------------


class TestARead:
    async def test_read_success_returns_raw_unformatted(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _download_blob_mock("alpha\nbeta")
        with _patch_async(container):
            result = await _make_backend().aread("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "alpha\nbeta"
        assert result.file_data["encoding"] == "utf-8"
        assert "1\t" not in result.file_data["content"]

    async def test_read_with_offset_and_limit(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _download_blob_mock(
            "l1\nl2\nl3\nl4\nl5\n"
        )
        with _patch_async(container):
            result = await _make_backend().aread("/f.txt", offset=1, limit=2)
        assert result.file_data is not None
        assert result.file_data["content"] == "l2\nl3\n"

    async def test_read_empty_file(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _download_blob_mock("")
        with _patch_async(container):
            result = await _make_backend().aread("/empty.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == ""

    async def test_read_binary_returns_base64(self) -> None:
        raw = b"\x89PNG\r\n\x1a\n\xff\xfe\x00"
        container = MagicMock()
        container.get_blob_client.return_value = _download_blob_mock(raw)
        with _patch_async(container):
            result = await _make_backend().aread("/img.png")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw

    async def test_read_not_found(self) -> None:
        container = MagicMock()
        blob = AsyncMock()
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            result = await _make_backend().aread("/missing.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_read_invalid_path(self) -> None:
        result = await _make_backend().aread("/src/../bad.txt")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    def test_read_sync(self) -> None:
        container = MagicMock()
        blob = MagicMock()
        blob.download_blob.return_value.readall.return_value = "hello\n"
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().read("/file.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "hello\n"


# ------------------------------------------------------------------
# write
# ------------------------------------------------------------------


class TestAWrite:
    async def test_write_new_file(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = AsyncMock()
        with _patch_async(container):
            result = await _make_backend().awrite("/new.txt", "hello")
        assert result.error is None
        assert result.path == "/new.txt"

    async def test_write_existing_file_fails(self) -> None:
        container = MagicMock()
        blob = AsyncMock()
        blob.upload_blob.side_effect = ResourceExistsError("exists")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            result = await _make_backend().awrite("/exists.txt", "hello")
        assert result.error is not None
        assert "already exists" in result.error

    async def test_write_invalid_path(self) -> None:
        result = await _make_backend().awrite("/src/../bad.txt", "x")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    def test_write_sync(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = MagicMock()
        with _patch_sync(container):
            result = _make_backend().write("/new.txt", "hello")
        assert result.error is None
        assert result.path == "/new.txt"


# ------------------------------------------------------------------
# edit
# ------------------------------------------------------------------


def _edit_blob_mock(content: str, metadata: dict | None = None) -> AsyncMock:
    blob = AsyncMock()
    stream = AsyncMock()
    stream.readall.return_value = content
    blob.download_blob.return_value = stream
    props = MagicMock()
    props.metadata = metadata or {}
    blob.get_blob_properties.return_value = props
    return blob


class TestAEdit:
    async def test_edit_success(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _edit_blob_mock(
            "hello world", {"created_at": "t1"}
        )
        with _patch_async(container):
            result = await _make_backend().aedit("/f.txt", "hello", "bye")
        assert result.error is None
        assert result.path == "/f.txt"
        assert result.occurrences == 1

    async def test_edit_not_found(self) -> None:
        container = MagicMock()
        blob = AsyncMock()
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            result = await _make_backend().aedit("/missing.txt", "a", "b")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_edit_replace_all(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _edit_blob_mock("aaa")
        with _patch_async(container):
            result = await _make_backend().aedit("/f.txt", "a", "b", replace_all=True)
        assert result.occurrences == 3

    async def test_edit_multiple_without_replace_all_fails(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _edit_blob_mock("aaa")
        with _patch_async(container):
            result = await _make_backend().aedit("/f.txt", "a", "b")
        assert result.error is not None


# ------------------------------------------------------------------
# ls
# ------------------------------------------------------------------


class TestALs:
    async def test_ls_files(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list(
            [
                _make_blob("pfx/src/a.py", 10, {"modified_at": "t1"}),
                _make_blob("pfx/src/b.py", 20),
            ]
        )
        with _patch_async(container):
            result = await _make_backend().als("/src")
        assert result.entries is not None
        assert {e["path"] for e in result.entries} == {"/src/a.py", "/src/b.py"}

    async def test_ls_synthesizes_directories(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list(
            [_make_blob("pfx/src/sub/a.py", 5), _make_blob("pfx/src/b.py", 10)]
        )
        with _patch_async(container):
            result = await _make_backend().als("/src")
        assert result.entries is not None
        dirs = [e for e in result.entries if e["is_dir"]]
        assert [d["path"] for d in dirs] == ["/src/sub/"]

    async def test_ls_empty(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([])
        with _patch_async(container):
            result = await _make_backend().als("/empty")
        assert result.entries == []

    async def test_ls_invalid_path_returns_empty(self) -> None:
        result = await _make_backend().als("/src/../bad")
        assert result.entries == []

    def test_ls_sync(self) -> None:
        container = MagicMock()
        container.list_blobs.return_value = [
            _make_blob("pfx/src/a.py", 10, {"modified_at": "t1"})
        ]
        with _patch_sync(container):
            result = _make_backend().ls("/src")
        assert result.entries is not None
        assert result.entries[0]["path"] == "/src/a.py"


# ------------------------------------------------------------------
# glob
# ------------------------------------------------------------------


class TestAGlob:
    async def test_glob_pattern(self) -> None:
        container = MagicMock()
        blob_client = AsyncMock()
        blob_client.get_blob_properties.side_effect = ResourceNotFoundError("dir")
        container.get_blob_client.return_value = blob_client
        container.list_blobs = _async_list(
            [_make_blob("pfx/src/a.py", 10), _make_blob("pfx/src/b.txt", 5)]
        )
        with _patch_async(container):
            result = await _make_backend().aglob("*.py", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/a.py"]

    async def test_glob_recursive(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list(
            [_make_blob("pfx/a.py", 1), _make_blob("pfx/sub/b.py", 2)]
        )
        with _patch_async(container):
            result = await _make_backend().aglob("**/*.py", path="/")
        assert result.matches is not None
        assert {m["path"] for m in result.matches} == {"/a.py", "/sub/b.py"}

    async def test_glob_invalid_path_returns_empty(self) -> None:
        result = await _make_backend().aglob("*.py", path="/src/../bad")
        assert result.matches == []


# ------------------------------------------------------------------
# grep
# ------------------------------------------------------------------


def _grep_container(blobs: list[Any], content: Any) -> MagicMock:
    container = MagicMock()
    container.list_blobs = _async_list(blobs)
    blob_client = AsyncMock()
    stream = AsyncMock()
    stream.readall.return_value = content
    blob_client.download_blob.return_value = stream
    blob_client.get_blob_properties.side_effect = ResourceNotFoundError("dir")
    container.get_blob_client.return_value = blob_client
    return container


class TestAGrep:
    async def test_grep_finds_matches(self) -> None:
        container = _grep_container(
            [_make_blob("pfx/f.py", 5)], "hello\nbye\nhello again\n"
        )
        with _patch_async(container):
            result = await _make_backend().agrep("hello")
        assert result.error is None
        assert result.matches is not None
        assert [m["line"] for m in result.matches] == [1, 3]

    async def test_grep_with_glob_filter(self) -> None:
        container = _grep_container(
            [_make_blob("pfx/f.py", 5), _make_blob("pfx/f.txt", 5)], "match\n"
        )
        with _patch_async(container):
            result = await _make_backend().agrep("match", glob="*.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/f.py"]

    async def test_grep_no_matches(self) -> None:
        container = _grep_container([_make_blob("pfx/f.py", 5)], "nothing\n")
        with _patch_async(container):
            result = await _make_backend().agrep("missing")
        assert result.matches == []

    async def test_grep_read_failure(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([_make_blob("pfx/f.py", 5)])
        blob_client = AsyncMock()
        blob_client.download_blob.side_effect = ResourceNotFoundError("read")
        blob_client.get_blob_properties.side_effect = ResourceNotFoundError("dir")
        container.get_blob_client.return_value = blob_client
        with _patch_async(container):
            result = await _make_backend().agrep("x")
        assert result.error is not None
        assert "could not read 1 file" in result.error.lower()

    async def test_grep_invalid_path(self) -> None:
        result = await _make_backend().agrep("x", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


# ------------------------------------------------------------------
# upload / download
# ------------------------------------------------------------------


class TestAUploadDownload:
    async def test_upload_success(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = AsyncMock()
        with _patch_async(container):
            result = await _make_backend().aupload_files([("/f.bin", b"data")])
        assert result[0].path == "/f.bin"
        assert result[0].error is None

    async def test_upload_failure(self) -> None:
        container = MagicMock()
        blob = AsyncMock()
        blob.upload_blob.side_effect = Exception("boom")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            result = await _make_backend().aupload_files([("/f.bin", b"data")])
        assert result[0].error == "permission_denied"

    async def test_upload_invalid_path(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = AsyncMock()
        with _patch_async(container):
            result = await _make_backend().aupload_files([("/src/../bad.bin", b"x")])
        assert result[0].error == "invalid_path"

    async def test_download_success(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _download_blob_mock(b"file content")
        with _patch_async(container):
            result = await _make_backend().adownload_files(["/f.txt"])
        assert result[0].content == b"file content"
        assert result[0].error is None

    async def test_download_string_content_encoded(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _download_blob_mock("text")
        with _patch_async(container):
            result = await _make_backend().adownload_files(["/f.txt"])
        assert result[0].content == b"text"

    async def test_download_not_found(self) -> None:
        container = MagicMock()
        blob = AsyncMock()
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            result = await _make_backend().adownload_files(["/missing.txt"])
        assert result[0].error == "file_not_found"
        assert result[0].content is None

    def test_download_sync(self) -> None:
        container = MagicMock()
        blob = MagicMock()
        blob.download_blob.return_value.readall.return_value = b"bytes"
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().download_files(["/f.bin"])
        assert result[0].content == b"bytes"
