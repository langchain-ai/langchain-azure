"""Unit tests for AzureBlobBackend behavior (mocked, no I/O).

Client patching, container mocks, and the blob/stream scaffolding live in
``conftest.py`` as fixtures; tests configure the fixture-provided container and
call the backend directly.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any, Callable
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
from azure.storage.blob import BlobClient  # noqa: E402
from azure.storage.blob.aio import BlobClient as AsyncBlobClient  # noqa: E402

from langchain_azure_storage.deepagents import AzureBlobBackend  # noqa: E402
from langchain_azure_storage.deepagents._utils import (  # noqa: E402
    get_prefix_for_path,
    normalize_path,
    to_blob_key,
)

_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;"

# Nearly every test constructs a backend, so silence the beta warning at the
# module level; TestFromConnectionString::test_emits_beta_warning still
# asserts it (pytest.warns overrides this filter).
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


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
# user agent
# ------------------------------------------------------------------


class TestUserAgent:
    async def test_user_agent_passed_to_async_client(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
    ) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        mock_cls, container = patched_async
        container.walk_blobs = async_list([])
        await backend.als("/")
        assert (
            mock_cls.from_connection_string.call_args.kwargs["user_agent"] == USER_AGENT
        )

    def test_user_agent_passed_to_sync_client(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        from langchain_azure_storage._user_agent import USER_AGENT

        mock_cls, container = patched_sync
        container.walk_blobs.return_value = []
        backend.ls("/")
        assert (
            mock_cls.from_connection_string.call_args.kwargs["user_agent"] == USER_AGENT
        )


# ------------------------------------------------------------------
# read
# ------------------------------------------------------------------


class TestARead:
    async def test_read_success_returns_raw_unformatted(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob("alpha\nbeta")
        result = await backend.aread("/file.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "alpha\nbeta"
        assert result.file_data["encoding"] == "utf-8"
        assert "1\t" not in result.file_data["content"]

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

    def test_read_sync(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob("hello\n")
        result = backend.read("/file.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "hello\n"


# ------------------------------------------------------------------
# write
# ------------------------------------------------------------------


class TestAWrite:
    async def test_write_new_file(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.awrite("/new.txt", "hello")
        assert result.error is None
        assert result.path == "/new.txt"

    async def test_write_existing_file_fails(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.upload_blob.side_effect = ResourceExistsError("exists")
        container.get_blob_client.return_value = blob
        result = await backend.awrite("/exists.txt", "hello")
        assert result.error is not None
        assert "already exists" in result.error

    async def test_write_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = await backend.awrite("/src/../bad.txt", "x")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    def test_write_sync(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = MagicMock(spec=BlobClient)
        result = backend.write("/new.txt", "hello")
        assert result.error is None
        assert result.path == "/new.txt"


# ------------------------------------------------------------------
# edit
# ------------------------------------------------------------------


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
        kwargs = blob.upload_blob.call_args.kwargs
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

    async def test_edit_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = await backend.aedit("/src/../bad.txt", "a", "b")
        assert result.error is not None
        assert "invalid path" in result.error.lower()


# ------------------------------------------------------------------
# ls
# ------------------------------------------------------------------


class TestALs:
    async def test_ls_files(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.walk_blobs = async_list(
            [
                make_blob("pfx/src/a.py", 10),
                make_blob("pfx/src/b.py", 20),
            ]
        )
        result = await backend.als("/src")
        assert result.entries is not None
        assert {e["path"] for e in result.entries} == {"/src/a.py", "/src/b.py"}

    async def test_ls_synthesizes_directories(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
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
        async_list: Callable[[list[Any]], Any],
    ) -> None:
        _, container = patched_async
        container.walk_blobs = async_list([])
        result = await backend.als("/empty")
        assert result.entries == []

    async def test_ls_invalid_path_returns_error(
        self, backend: AzureBlobBackend
    ) -> None:
        result = await backend.als("/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.entries is None

    def test_ls_sync(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        modified = datetime(2026, 1, 1, tzinfo=timezone.utc)
        container.walk_blobs.return_value = [
            make_blob("pfx/src/a.py", 10, last_modified=modified)
        ]
        result = backend.ls("/src")
        assert result.entries is not None
        assert result.entries[0]["path"] == "/src/a.py"
        assert result.entries[0]["modified_at"] == modified.isoformat()


# ------------------------------------------------------------------
# glob
# ------------------------------------------------------------------


class TestAGlob:
    async def test_glob_pattern(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
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

    async def test_glob_recursive(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
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
        async_list: Callable[[list[Any]], Any],
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

    async def test_glob_invalid_path_returns_error(
        self, backend: AzureBlobBackend
    ) -> None:
        result = await backend.aglob("*.py", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.matches is None

    async def test_glob_skips_blobs_outside_base(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        container.list_blobs = async_list([make_blob("pfx/other/a.py", 1)])
        result = await backend.aglob("*.py", path="/src")
        assert result.matches == []


# ------------------------------------------------------------------
# grep
# ------------------------------------------------------------------


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

    async def test_grep_read_failure(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
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


# ------------------------------------------------------------------
# upload / download
# ------------------------------------------------------------------


class TestAUploadDownload:
    async def test_upload_success(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.aupload_files([("/f.bin", b"data")])
        assert result[0].path == "/f.bin"
        assert result[0].error is None

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
        response = MagicMock()
        response.status_code = 403
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.upload_blob.side_effect = HttpResponseError(response=response)
        container.get_blob_client.return_value = blob
        result = await backend.aupload_files([("/f.bin", b"data")])
        assert result[0].error == "permission_denied"

    async def test_upload_invalid_path(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.aupload_files([("/src/../bad.bin", b"x")])
        assert result[0].error == "invalid_path"

    async def test_download_success(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        make_async_download_blob: Callable[..., AsyncMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = make_async_download_blob(
            b"file content"
        )
        result = await backend.adownload_files(["/f.txt"])
        assert result[0].content == b"file content"
        assert result[0].error is None

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

    def test_download_sync(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_sync_download_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = make_sync_download_blob(b"bytes")
        result = backend.download_files(["/f.bin"])
        assert result[0].content == b"bytes"

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
        response = MagicMock()
        response.status_code = 403
        _, container = patched_async
        blob = AsyncMock(spec=AsyncBlobClient)
        blob.download_blob.side_effect = HttpResponseError(response=response)
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

    def test_download_failure_generic_sync(
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

    async def test_upload_multiple(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.aupload_files([("/a.bin", b"a"), ("/b.bin", b"b")])
        assert [r.error for r in result] == [None, None]

    async def test_download_invalid_path(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_async
        container.get_blob_client.return_value = AsyncMock(spec=AsyncBlobClient)
        result = await backend.adownload_files(["/src/../bad.txt"])
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


class TestCredentialHelpers:
    def test_sync_default_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity import DefaultAzureCredential

        cred = make_acct_backend()._resolve_sync_credential(None)
        assert isinstance(cred, DefaultAzureCredential)

    def test_sync_rejects_async_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity.aio import DefaultAzureCredential as AioCred

        with pytest.raises(ValueError, match="synchronous"):
            make_acct_backend()._resolve_sync_credential(AioCred())

    def test_sync_passthrough(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        assert make_acct_backend()._resolve_sync_credential(cred) is cred

    async def test_async_default_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity.aio import DefaultAzureCredential as AioCred

        cred = await make_acct_backend()._resolve_async_credential(None)
        assert isinstance(cred, AioCred)

    async def test_async_rejects_sync_credential(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.identity import DefaultAzureCredential

        with pytest.raises(ValueError, match="asynchronous"):
            await make_acct_backend()._resolve_async_credential(
                DefaultAzureCredential()
            )

    async def test_async_passthrough_sas(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        got = await make_acct_backend()._resolve_async_credential(cred)
        assert got is cred


# ------------------------------------------------------------------
# Container construction and caching (account_url path, not connection_string)
# ------------------------------------------------------------------


class TestContainerConstruction:
    def test_sync_creates_default_credential(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        mock_cc, container = patched_sync
        container.list_blobs.return_value = []
        fake_cred = MagicMock()
        with patch("azure.identity.DefaultAzureCredential", return_value=fake_cred):
            make_acct_backend().ls("/")
        assert mock_cc.call_args.kwargs["credential"] is fake_cred

    def test_sync_uses_provided_credential(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        mock_cc, container = patched_sync
        container.list_blobs.return_value = []
        make_acct_backend(cred).ls("/")
        assert mock_cc.call_args.kwargs["credential"] is cred

    def test_sync_container_reused_across_calls(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        mock_cc, container = patched_sync
        container.list_blobs.return_value = []
        backend = make_acct_backend()
        backend.ls("/")
        backend.ls("/")
        assert mock_cc.call_count == 1

    async def test_async_creates_default_credential(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], Any],
    ) -> None:
        mock_cc, container = patched_async
        container.list_blobs = async_list([])
        fake_cred = MagicMock()
        fake_cred.close = AsyncMock()
        with patch("azure.identity.aio.DefaultAzureCredential", return_value=fake_cred):
            await make_acct_backend().als("/")
        assert mock_cc.call_args.kwargs["credential"] is fake_cred

    async def test_async_uses_provided_credential(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], Any],
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        mock_cc, container = patched_async
        container.list_blobs = async_list([])
        await make_acct_backend(cred).als("/")
        assert mock_cc.call_args.kwargs["credential"] is cred

    async def test_async_container_reused_across_calls(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], Any],
    ) -> None:
        mock_cc, container = patched_async
        container.list_blobs = async_list([])
        backend = make_acct_backend()
        await backend.als("/")
        await backend.als("/")
        assert mock_cc.call_count == 1


# ------------------------------------------------------------------
# close() / aclose()
# ------------------------------------------------------------------


class TestCloseAndAclose:
    def test_close_closes_owned_credential_and_container(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = []
        fake_cred = MagicMock()
        with patch("azure.identity.DefaultAzureCredential", return_value=fake_cred):
            backend = make_acct_backend()
            backend.ls("/")
            fake_cred.close.assert_not_called()
            container.close.assert_not_called()
            backend.close()
        fake_cred.close.assert_called_once()
        container.close.assert_called_once()

    def test_close_does_not_close_caller_supplied_credential(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        from azure.core.credentials import AzureSasCredential

        cred = AzureSasCredential("sig")
        _, container = patched_sync
        container.list_blobs.return_value = []
        backend = make_acct_backend(cred)
        backend.ls("/")
        backend.close()
        container.close.assert_called_once()

    def test_close_is_idempotent(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = []
        backend = make_acct_backend()
        backend.ls("/")
        backend.close()
        backend.close()
        container.close.assert_called_once()

    def test_close_without_prior_use_is_a_noop(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        make_acct_backend().close()

    def test_context_manager_calls_close(
        self,
        patched_sync: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = []
        with make_acct_backend() as backend:
            backend.ls("/")
        container.close.assert_called_once()

    async def test_aclose_closes_owned_credential_and_container(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], Any],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([])
        fake_cred = MagicMock()
        fake_cred.close = AsyncMock()
        with patch("azure.identity.aio.DefaultAzureCredential", return_value=fake_cred):
            backend = make_acct_backend()
            await backend.als("/")
            fake_cred.close.assert_not_called()
            await backend.aclose()
        fake_cred.close.assert_awaited_once()
        container.close.assert_awaited_once()

    async def test_aclose_without_prior_use_is_a_noop(
        self, make_acct_backend: Callable[..., AzureBlobBackend]
    ) -> None:
        await make_acct_backend().aclose()

    async def test_async_context_manager_calls_aclose(
        self,
        patched_async: tuple[MagicMock, MagicMock],
        make_acct_backend: Callable[..., AzureBlobBackend],
        async_list: Callable[[list[Any]], Any],
    ) -> None:
        _, container = patched_async
        container.list_blobs = async_list([])
        async with make_acct_backend() as backend:
            await backend.als("/")
        container.close.assert_awaited_once()


# ------------------------------------------------------------------
# read offset-out-of-range + ls branches + exact-path listing
# ------------------------------------------------------------------


class TestReadLsGlobGrepBranches:
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

    async def test_ls_path_without_leading_slash(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
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
        async_list: Callable[[list[Any]], Any],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        # A pseudo-directory marker blob (name ending in "/") is not a file.
        _, container = patched_async
        container.walk_blobs = async_list([make_blob("pfx/src/", 0)])
        result = await backend.als("/src")
        assert result.entries == []

    async def test_glob_path_treated_as_directory(
        self,
        backend: AzureBlobBackend,
        patched_async: tuple[MagicMock, MagicMock],
        async_list: Callable[[list[Any]], Any],
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


# ------------------------------------------------------------------
# Sync method branches (sync error paths and remaining sync operations)
# ------------------------------------------------------------------


class TestSyncBranches:
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

    def test_write_existing_fails(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.upload_blob.side_effect = ResourceExistsError("exists")
        container.get_blob_client.return_value = blob
        result = backend.write("/exists.txt", "x")
        assert result.error is not None
        assert "already exists" in result.error

    def test_write_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.write("/src/../bad.txt", "x")
        assert result.error is not None

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
        assert result.occurrences == 1

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

    def test_glob(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = MagicMock(spec=BlobClient)
        container.list_blobs.return_value = [
            make_blob("pfx/src/a.py", 1),
            make_blob("pfx/src/b.txt", 1),
        ]
        result = backend.glob("*.py", path="/src")
        assert result.matches is not None
        assert [m["path"] for m in result.matches] == ["/src/a.py"]

    def test_glob_invalid_path_returns_error(self, backend: AzureBlobBackend) -> None:
        result = backend.glob("*.py", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.matches is None

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

    def test_grep(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        setup_sync_grep: Callable[[MagicMock, list[Any], Any], None],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        setup_sync_grep(container, [make_blob("pfx/f.py", 5)], "hello\nbye\nhello\n")
        result = backend.grep("hello")
        assert result.error is None
        assert result.matches is not None
        assert [m["line"] for m in result.matches] == [1, 3]

    def test_grep_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.grep("x", path="/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()

    def test_grep_read_failure(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
        make_blob: Callable[..., MagicMock],
    ) -> None:
        _, container = patched_sync
        container.list_blobs.return_value = [make_blob("pfx/f.py", 5)]
        bc = MagicMock(spec=BlobClient)
        bc.download_blob.side_effect = ResourceNotFoundError("read")
        container.get_blob_client.return_value = bc
        result = backend.grep("x")
        assert result.error is not None
        assert "could not read 1 file" in result.error.lower()

    def test_upload_success(
        self,
        backend: AzureBlobBackend,
        patched_sync: tuple[MagicMock, MagicMock],
    ) -> None:
        _, container = patched_sync
        container.get_blob_client.return_value = MagicMock(spec=BlobClient)
        result = backend.upload_files([("/f.bin", b"data")])
        assert result[0].error is None

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
        response = MagicMock()
        response.status_code = 403
        _, container = patched_sync
        blob = MagicMock(spec=BlobClient)
        blob.upload_blob.side_effect = HttpResponseError(response=response)
        container.get_blob_client.return_value = blob
        result = backend.upload_files([("/f.bin", b"data")])
        assert result[0].error == "permission_denied"

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

    def test_download_invalid_path(self, backend: AzureBlobBackend) -> None:
        result = backend.download_files(["/src/../bad.bin"])
        assert result[0].error == "invalid_path"

    def test_ls_invalid_path_returns_error(self, backend: AzureBlobBackend) -> None:
        result = backend.ls("/src/../bad")
        assert result.error is not None
        assert "invalid path" in result.error.lower()
        assert result.entries is None
