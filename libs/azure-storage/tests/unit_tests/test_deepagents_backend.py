"""Unit tests for AzureBlobBackend (mocked, no I/O)."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The backend needs the optional [deepagents] extra (Python >= 3.11 only).
pytest.importorskip("deepagents")

from azure.core import MatchConditions  # noqa: E402
from azure.core.exceptions import (  # noqa: E402
    HttpResponseError,
    ResourceExistsError,
    ResourceModifiedError,
    ResourceNotFoundError,
)

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402
from langchain_azure_storage.deepagents._utils import (  # noqa: E402
    build_file_info,
    from_blob_key,
    get_prefix_for_path,
    normalize_path,
    to_blob_key,
)

_BACKEND = "langchain_azure_storage.deepagents.backend"
_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"

# Nearly every test constructs a backend, so silence the beta warning at the
# module level; TestFromConnectionString::test_emits_beta_warning still
# asserts it (pytest.warns overrides this filter).
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


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

    def test_missing_account_url_raises(self) -> None:
        with pytest.raises(TypeError):
            AzureBlobBackend(container_name="c")  # type: ignore[call-arg]

    def test_missing_container_name_raises(self) -> None:
        with pytest.raises(TypeError):
            AzureBlobBackend("https://x.blob.core.windows.net")  # type: ignore[call-arg]


class TestFromConnectionString:
    def test_sets_connection_string_container_and_prefix(self) -> None:
        backend = AzureBlobBackend.from_connection_string(
            _CONN_STR, "test", prefix="pfx/"
        )
        assert backend._connection_string == _CONN_STR
        assert backend._container_name == "test"
        assert backend._prefix == "pfx/"

    def test_is_backend_protocol(self) -> None:
        from deepagents.backends.protocol import BackendProtocol

        backend = AzureBlobBackend.from_connection_string(_CONN_STR, "test")
        assert isinstance(backend, BackendProtocol)

    def test_emits_beta_warning(self) -> None:
        # The @beta wrapper on __init__ suppresses its warning for callers
        # inside langchain* packages, which includes this classmethod, so the
        # classmethod emits it explicitly.
        from langchain_core._api import LangChainBetaWarning

        with pytest.warns(LangChainBetaWarning, match="public preview"):
            AzureBlobBackend.from_connection_string(_CONN_STR, "test")


# ------------------------------------------------------------------
# Mock helpers for the sync / async container clients
# ------------------------------------------------------------------


def _make_backend(prefix: str = "pfx/") -> AzureBlobBackend:
    return AzureBlobBackend.from_connection_string(_CONN_STR, "test", prefix=prefix)


def _make_blob(name: str, size: int = 0, last_modified: Any = None) -> MagicMock:
    blob = MagicMock()
    blob.name = name
    blob.size = size
    blob.last_modified = last_modified
    return blob


@contextmanager
def _patch_async(container: MagicMock) -> Iterator[MagicMock]:
    """Patch AsyncContainerClient so ``_get_async_container`` returns *container*."""
    with patch(f"{_BACKEND}.AsyncContainerClient") as mock_cls:
        mock_cls.from_connection_string.return_value = container
        yield mock_cls


@contextmanager
def _patch_sync(container: MagicMock) -> Iterator[MagicMock]:
    """Patch ContainerClient so ``_get_sync_container`` returns *container*."""
    with patch(f"{_BACKEND}.ContainerClient") as mock_cls:
        mock_cls.from_connection_string.return_value = container
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


def _edit_blob_mock(content: str, etag: str = "etag-1") -> AsyncMock:
    blob = AsyncMock()
    stream = AsyncMock()
    stream.readall.return_value = content
    stream.properties.etag = etag
    blob.download_blob.return_value = stream
    return blob


class TestAEdit:
    async def test_edit_success(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _edit_blob_mock("hello world")
        with _patch_async(container):
            result = await _make_backend().aedit("/f.txt", "hello", "bye")
        assert result.error is None
        assert result.path == "/f.txt"
        assert result.occurrences == 1

    async def test_edit_uploads_with_etag_condition(self) -> None:
        container = MagicMock()
        blob = _edit_blob_mock("hello world", etag="etag-42")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            await _make_backend().aedit("/f.txt", "hello", "bye")
        kwargs = blob.upload_blob.call_args.kwargs
        assert kwargs["etag"] == "etag-42"
        assert kwargs["match_condition"] == MatchConditions.IfNotModified

    async def test_edit_concurrent_modification(self) -> None:
        container = MagicMock()
        blob = _edit_blob_mock("hello world")
        blob.upload_blob.side_effect = ResourceModifiedError("etag mismatch")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            result = await _make_backend().aedit("/f.txt", "hello", "bye")
        assert result.error is not None
        assert "modified concurrently" in result.error

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

    async def test_edit_invalid_path(self) -> None:
        result = await _make_backend().aedit("/src/../bad.txt", "a", "b")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


# ------------------------------------------------------------------
# ls
# ------------------------------------------------------------------


class TestALs:
    async def test_ls_files(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list(
            [
                _make_blob("pfx/src/a.py", 10),
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

    async def test_ls_invalid_path_returns_error(self) -> None:
        result = await _make_backend().als("/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.entries is None

    def test_ls_sync(self) -> None:
        container = MagicMock()
        from datetime import datetime, timezone

        modified = datetime(2026, 1, 1, tzinfo=timezone.utc)
        container.list_blobs.return_value = [
            _make_blob("pfx/src/a.py", 10, last_modified=modified)
        ]
        with _patch_sync(container):
            result = _make_backend().ls("/src")
        assert result.entries is not None
        assert result.entries[0]["path"] == "/src/a.py"
        assert result.entries[0]["modified_at"] == modified.isoformat()


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

    async def test_glob_invalid_path_returns_error(self) -> None:
        result = await _make_backend().aglob("*.py", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.matches is None

    async def test_glob_skips_blobs_outside_base(self) -> None:
        container = MagicMock()
        blob_client = AsyncMock()
        blob_client.get_blob_properties.side_effect = ResourceNotFoundError("dir")
        container.get_blob_client.return_value = blob_client
        container.list_blobs = _async_list([_make_blob("pfx/other/a.py", 1)])
        with _patch_async(container):
            result = await _make_backend().aglob("*.py", path="/src")
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

    async def test_grep_glob_without_slash_matches_nested_names(self) -> None:
        # rg --glob semantics: a slash-less pattern matches names at any depth.
        container = _grep_container(
            [_make_blob("pfx/a/b/target.py", 5), _make_blob("pfx/a/ignore.txt", 5)],
            "needle\n",
        )
        with _patch_async(container):
            result = await _make_backend().agrep("needle", glob="*.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/a/b/target.py"]

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

    async def test_upload_failure_generic_returns_exception_message(self) -> None:
        container = MagicMock()
        blob = AsyncMock()
        blob.upload_blob.side_effect = Exception("boom")
        container.get_blob_client.return_value = blob
        with _patch_async(container):
            result = await _make_backend().aupload_files([("/f.bin", b"data")])
        assert result[0].error == "boom"

    async def test_upload_failure_forbidden(self) -> None:
        response = MagicMock()
        response.status_code = 403
        container = MagicMock()
        blob = AsyncMock()
        blob.upload_blob.side_effect = HttpResponseError(response=response)
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

    async def test_upload_multiple(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = AsyncMock()
        with _patch_async(container):
            result = await _make_backend().aupload_files(
                [("/a.bin", b"a"), ("/b.bin", b"b")]
            )
        assert [r.error for r in result] == [None, None]

    async def test_download_invalid_path(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = AsyncMock()
        with _patch_async(container):
            result = await _make_backend().adownload_files(["/src/../bad.txt"])
        assert result[0].error == "invalid_path"


# ------------------------------------------------------------------
# Extra path-utility edge cases
# ------------------------------------------------------------------


class TestPathEdges:
    def test_normalize_empty_string(self) -> None:
        assert normalize_path("") == ""

    def test_normalize_trailing_slash(self) -> None:
        assert normalize_path("/src/") == "src"

    def test_get_prefix_root_no_prefix(self) -> None:
        assert get_prefix_for_path("", "/") == ""

    def test_to_blob_key_root(self) -> None:
        assert to_blob_key("pfx/", "/") == "pfx/"


# ------------------------------------------------------------------
# Credential resolution helpers
# ------------------------------------------------------------------


def _acct_backend(credential: Any = None) -> AzureBlobBackend:
    return AzureBlobBackend(
        account_url="https://x.blob.core.windows.net",
        container_name="test",
        prefix="pfx/",
        credential=credential,
    )


class TestCredentialHelpers:
    def test_sync_default_credential(self) -> None:
        from azure.identity import DefaultAzureCredential

        cred = _acct_backend()._resolve_sync_credential(None)
        assert isinstance(cred, DefaultAzureCredential)

    def test_sync_rejects_async_credential(self) -> None:
        from azure.identity.aio import DefaultAzureCredential as AioCred

        with pytest.raises(ValueError, match="synchronous"):
            _acct_backend()._resolve_sync_credential(AioCred())

    def test_sync_passthrough(self) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        assert _acct_backend()._resolve_sync_credential(cred) is cred

    async def test_async_default_credential(self) -> None:
        from azure.identity.aio import DefaultAzureCredential as AioCred

        cred = await _acct_backend()._resolve_async_credential(None)
        assert isinstance(cred, AioCred)

    async def test_async_rejects_sync_credential(self) -> None:
        from azure.identity import DefaultAzureCredential

        with pytest.raises(ValueError, match="asynchronous"):
            await _acct_backend()._resolve_async_credential(DefaultAzureCredential())

    async def test_async_passthrough_sas(self) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        got = await _acct_backend()._resolve_async_credential(cred)
        assert got is cred


# ------------------------------------------------------------------
# Container construction and caching (account_url path, not connection_string)
# ------------------------------------------------------------------


class TestContainerConstruction:
    def test_sync_creates_default_credential(self) -> None:
        container = MagicMock()
        container.list_blobs.return_value = []
        fake_cred = MagicMock()
        with (
            patch(f"{_BACKEND}.ContainerClient", return_value=container) as mock_cc,
            patch("azure.identity.DefaultAzureCredential", return_value=fake_cred),
        ):
            _acct_backend().ls("/")
        assert mock_cc.call_args.kwargs["credential"] is fake_cred

    def test_sync_uses_provided_credential(self) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        container = MagicMock()
        container.list_blobs.return_value = []
        with patch(f"{_BACKEND}.ContainerClient", return_value=container) as mock_cc:
            _acct_backend(cred).ls("/")
        assert mock_cc.call_args.kwargs["credential"] is cred

    def test_sync_container_reused_across_calls(self) -> None:
        container = MagicMock()
        container.list_blobs.return_value = []
        with patch(f"{_BACKEND}.ContainerClient", return_value=container) as mock_cc:
            backend = _acct_backend()
            backend.ls("/")
            backend.ls("/")
        assert mock_cc.call_count == 1

    async def test_async_creates_default_credential(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([])
        fake_cred = MagicMock()
        fake_cred.close = AsyncMock()
        with (
            patch(
                f"{_BACKEND}.AsyncContainerClient", return_value=container
            ) as mock_cc,
            patch("azure.identity.aio.DefaultAzureCredential", return_value=fake_cred),
        ):
            await _acct_backend().als("/")
        assert mock_cc.call_args.kwargs["credential"] is fake_cred

    async def test_async_uses_provided_credential(self) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        container = MagicMock()
        container.list_blobs = _async_list([])
        with patch(
            f"{_BACKEND}.AsyncContainerClient", return_value=container
        ) as mock_cc:
            await _acct_backend(cred).als("/")
        assert mock_cc.call_args.kwargs["credential"] is cred

    async def test_async_container_reused_across_calls(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([])
        with patch(
            f"{_BACKEND}.AsyncContainerClient", return_value=container
        ) as mock_cc:
            backend = _acct_backend()
            await backend.als("/")
            await backend.als("/")
        assert mock_cc.call_count == 1


# ------------------------------------------------------------------
# close() / aclose()
# ------------------------------------------------------------------


class TestCloseAndAclose:
    def test_close_closes_owned_credential_and_container(self) -> None:
        container = MagicMock()
        container.list_blobs.return_value = []
        fake_cred = MagicMock()
        with (
            patch(f"{_BACKEND}.ContainerClient", return_value=container),
            patch("azure.identity.DefaultAzureCredential", return_value=fake_cred),
        ):
            backend = _acct_backend()
            backend.ls("/")
            fake_cred.close.assert_not_called()
            container.close.assert_not_called()
            backend.close()
        fake_cred.close.assert_called_once()
        container.close.assert_called_once()

    def test_close_does_not_close_caller_supplied_credential(self) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        container = MagicMock()
        container.list_blobs.return_value = []
        with patch(f"{_BACKEND}.ContainerClient", return_value=container):
            backend = _acct_backend(cred)
            backend.ls("/")
            backend.close()
        container.close.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        container = MagicMock()
        container.list_blobs.return_value = []
        with patch(f"{_BACKEND}.ContainerClient", return_value=container):
            backend = _acct_backend()
            backend.ls("/")
            backend.close()
            backend.close()
        container.close.assert_called_once()

    def test_close_without_prior_use_is_a_noop(self) -> None:
        _acct_backend().close()

    def test_context_manager_calls_close(self) -> None:
        container = MagicMock()
        container.list_blobs.return_value = []
        with patch(f"{_BACKEND}.ContainerClient", return_value=container):
            with _acct_backend() as backend:
                backend.ls("/")
        container.close.assert_called_once()

    async def test_aclose_closes_owned_credential_and_container(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([])
        container.close = AsyncMock()
        fake_cred = MagicMock()
        fake_cred.close = AsyncMock()
        with (
            patch(f"{_BACKEND}.AsyncContainerClient", return_value=container),
            patch("azure.identity.aio.DefaultAzureCredential", return_value=fake_cred),
        ):
            backend = _acct_backend()
            await backend.als("/")
            fake_cred.close.assert_not_called()
            await backend.aclose()
        fake_cred.close.assert_awaited_once()
        container.close.assert_awaited_once()

    async def test_aclose_without_prior_use_is_a_noop(self) -> None:
        await _acct_backend().aclose()

    async def test_async_context_manager_calls_aclose(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([])
        container.close = AsyncMock()
        with patch(f"{_BACKEND}.AsyncContainerClient", return_value=container):
            async with _acct_backend() as backend:
                await backend.als("/")
        container.close.assert_awaited_once()


# ------------------------------------------------------------------
# read offset-out-of-range + ls branches + exact-path listing
# ------------------------------------------------------------------


class TestReadLsGlobGrepBranches:
    async def test_read_offset_out_of_range(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _download_blob_mock("l1\n")
        with _patch_async(container):
            result = await _make_backend().aread("/f.txt", offset=100)
        assert result.error is not None
        assert "offset" in result.error.lower()

    async def test_ls_path_without_leading_slash(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([_make_blob("pfx/src/a.py", 1)])
        with _patch_async(container):
            result = await _make_backend().als("src")
        assert result.entries is not None
        assert result.entries[0]["path"] == "/src/a.py"

    async def test_ls_skips_blobs_outside_path(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([_make_blob("pfx/other/a.py", 1)])
        with _patch_async(container):
            result = await _make_backend().als("/src")
        assert result.entries == []

    async def test_ls_skips_empty_relative(self) -> None:
        container = MagicMock()
        container.list_blobs = _async_list([_make_blob("pfx/src/", 0)])
        with _patch_async(container):
            result = await _make_backend().als("/src")
        assert result.entries == []

    async def test_glob_exact_file_path(self) -> None:
        container = MagicMock()
        blob_client = AsyncMock()
        props = MagicMock()
        props.size = 10
        props.last_modified = None
        blob_client.get_blob_properties.return_value = props
        container.get_blob_client.return_value = blob_client
        container.list_blobs = MagicMock()  # must not be used
        with _patch_async(container):
            result = await _make_backend().aglob("main.py", path="/src/main.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/main.py"]
        container.list_blobs.assert_not_called()

    async def test_grep_with_path_scope(self) -> None:
        container = _grep_container([_make_blob("pfx/src/f.py", 5)], "match\n")
        with _patch_async(container):
            result = await _make_backend().agrep("match", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/f.py"]

    async def test_grep_skips_blobs_outside_path(self) -> None:
        container = _grep_container([_make_blob("pfx/other/f.py", 5)], "match\n")
        with _patch_async(container):
            result = await _make_backend().agrep("match", path="/src")
        assert result.matches == []


# ------------------------------------------------------------------
# Sync method branches (sync error paths and remaining sync operations)
# ------------------------------------------------------------------


def _sync_edit_blob(content: str, etag: str = "etag-1") -> MagicMock:
    blob = MagicMock()
    blob.download_blob.return_value.readall.return_value = content
    blob.download_blob.return_value.properties.etag = etag
    return blob


def _sync_grep_container(blobs: list[Any], content: Any) -> MagicMock:
    container = MagicMock()
    container.list_blobs.return_value = blobs
    bc = MagicMock()
    bc.download_blob.return_value.readall.return_value = content
    bc.get_blob_properties.side_effect = ResourceNotFoundError("dir")
    container.get_blob_client.return_value = bc
    return container


class TestSyncBranches:
    def test_read_not_found(self) -> None:
        container = MagicMock()
        blob = MagicMock()
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().read("/missing.txt")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_read_invalid_path(self) -> None:
        result = _make_backend().read("/src/../bad.txt")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    def test_write_existing_fails(self) -> None:
        container = MagicMock()
        blob = MagicMock()
        blob.upload_blob.side_effect = ResourceExistsError("exists")
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().write("/exists.txt", "x")
        assert result.error is not None
        assert "already exists" in result.error

    def test_write_invalid_path(self) -> None:
        result = _make_backend().write("/src/../bad.txt", "x")
        assert result.error is not None

    def test_edit_success(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _sync_edit_blob("hello world")
        with _patch_sync(container):
            result = _make_backend().edit("/f.txt", "hello", "bye")
        assert result.error is None
        assert result.occurrences == 1

    def test_edit_concurrent_modification(self) -> None:
        container = MagicMock()
        blob = _sync_edit_blob("hello world")
        blob.upload_blob.side_effect = ResourceModifiedError("etag mismatch")
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().edit("/f.txt", "hello", "bye")
        assert result.error is not None
        assert "modified concurrently" in result.error

    def test_edit_not_found(self) -> None:
        container = MagicMock()
        blob = MagicMock()
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().edit("/missing.txt", "a", "b")
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_edit_no_match(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = _sync_edit_blob("hello")
        with _patch_sync(container):
            result = _make_backend().edit("/f.txt", "nope", "x")
        assert result.error is not None

    def test_edit_invalid_path(self) -> None:
        result = _make_backend().edit("/src/../bad.txt", "a", "b")
        assert result.error is not None

    def test_glob(self) -> None:
        container = MagicMock()
        bc = MagicMock()
        bc.get_blob_properties.side_effect = ResourceNotFoundError("dir")
        container.get_blob_client.return_value = bc
        container.list_blobs.return_value = [
            _make_blob("pfx/src/a.py", 1),
            _make_blob("pfx/src/b.txt", 1),
        ]
        with _patch_sync(container):
            result = _make_backend().glob("*.py", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/a.py"]

    def test_glob_invalid_path_returns_error(self) -> None:
        result = _make_backend().glob("*.py", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.matches is None

    def test_glob_exact_file_path(self) -> None:
        container = MagicMock()
        bc = MagicMock()
        props = MagicMock()
        props.size = 5
        props.last_modified = None
        bc.get_blob_properties.return_value = props
        container.get_blob_client.return_value = bc
        container.list_blobs = MagicMock()
        with _patch_sync(container):
            result = _make_backend().glob("main.py", path="/src/main.py")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/main.py"]
        container.list_blobs.assert_not_called()

    def test_grep(self) -> None:
        container = _sync_grep_container(
            [_make_blob("pfx/f.py", 5)], "hello\nbye\nhello\n"
        )
        with _patch_sync(container):
            result = _make_backend().grep("hello")
        assert result.error is None
        assert result.matches is not None
        assert [m["line"] for m in result.matches] == [1, 3]

    def test_grep_invalid_path(self) -> None:
        result = _make_backend().grep("x", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    def test_grep_read_failure(self) -> None:
        container = MagicMock()
        container.list_blobs.return_value = [_make_blob("pfx/f.py", 5)]
        bc = MagicMock()
        bc.download_blob.side_effect = ResourceNotFoundError("read")
        bc.get_blob_properties.side_effect = ResourceNotFoundError("dir")
        container.get_blob_client.return_value = bc
        with _patch_sync(container):
            result = _make_backend().grep("x")
        assert result.error is not None
        assert "could not read 1 file" in result.error.lower()

    def test_upload_success(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = MagicMock()
        with _patch_sync(container):
            result = _make_backend().upload_files([("/f.bin", b"data")])
        assert result[0].error is None

    def test_upload_invalid_path(self) -> None:
        container = MagicMock()
        container.get_blob_client.return_value = MagicMock()
        with _patch_sync(container):
            result = _make_backend().upload_files([("/src/../bad.bin", b"x")])
        assert result[0].error == "invalid_path"

    def test_upload_failure_generic_returns_exception_message(self) -> None:
        container = MagicMock()
        blob = MagicMock()
        blob.upload_blob.side_effect = Exception("boom")
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().upload_files([("/f.bin", b"data")])
        assert result[0].error == "boom"

    def test_upload_failure_forbidden(self) -> None:
        response = MagicMock()
        response.status_code = 403
        container = MagicMock()
        blob = MagicMock()
        blob.upload_blob.side_effect = HttpResponseError(response=response)
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().upload_files([("/f.bin", b"data")])
        assert result[0].error == "permission_denied"

    def test_download_not_found(self) -> None:
        container = MagicMock()
        blob = MagicMock()
        blob.download_blob.side_effect = ResourceNotFoundError("nope")
        container.get_blob_client.return_value = blob
        with _patch_sync(container):
            result = _make_backend().download_files(["/missing.bin"])
        assert result[0].error == "file_not_found"

    def test_download_invalid_path(self) -> None:
        result = _make_backend().download_files(["/src/../bad.bin"])
        assert result[0].error == "invalid_path"

    def test_ls_invalid_path_returns_error(self) -> None:
        result = _make_backend().ls("/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.entries is None
