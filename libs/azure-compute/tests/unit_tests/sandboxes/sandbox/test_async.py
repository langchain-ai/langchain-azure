"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.containerapps.sandbox import (
    ExecResult,  # noqa: E402
)
from azure.core.exceptions import (  # noqa: E402
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from deepagents.backends.protocol import (  # noqa: E402
    IS_DIRECTORY,
    ReadResult,
)
from deepagents.backends.sandbox import BaseSandbox  # noqa: E402

from langchain_azure_compute.sandboxes import ACASandbox  # noqa: E402
from langchain_azure_compute.sandboxes.sandbox import (  # noqa: E402
    _READ_DOWNLOAD_CAP,
)

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

from tests.unit_tests.sandboxes.sandbox._helpers import (  # noqa: E402
    _http_error,
    stat_info,
)


class TestAsync:
    async def test_aexecute_uses_async_client_when_given(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.exec.return_value = ExecResult(
            exit_code=0, stdout="async", stderr=""
        )
        sandbox = ACASandbox(client, async_client=async_client)
        result = await sandbox.aexecute("echo hi")
        assert result.output == "async"
        async_client.exec.assert_awaited_once()
        client.exec.assert_not_called()

    async def test_aexecute_falls_back_to_thread_offload(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.exec.return_value = ExecResult(exit_code=0, stdout="sync", stderr="")
        result = await sandbox.aexecute("echo hi")
        assert result.output == "sync"
        client.exec.assert_called_once()

    async def test_aexecute_transport_failure_on_the_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.exec.side_effect = ServiceResponseError("read timed out")
        sandbox = ACASandbox(client, async_client=async_client)
        result = await sandbox.aexecute("sleep 600")
        assert result.exit_code is None
        assert "may still be running" in result.output
        assert "read timed out" not in result.output

    async def test_aexecute_transport_failure_without_an_async_client(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # The thread offload must surface the sync path's error response, not
        # re-raise the transport failure into the event loop.
        client.exec.side_effect = ServiceRequestError("connection reset")
        result = await sandbox.aexecute("sleep 600")
        assert result.exit_code is None
        assert "may still be running" in result.output

    async def test_aclose_closes_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        await ACASandbox(client, async_client=async_client).aclose()
        async_client.close.assert_awaited_once()

    async def test_aclose_is_safe_without_async_client(
        self, sandbox: ACASandbox
    ) -> None:
        await sandbox.aclose()


class TestAsyncFilesystemParity:
    """Async file operations must take the same SDK path as their sync twins.

    `BaseSandbox` routes `aread`/`awrite` through `aexecute`, i.e. the shell
    script that `read`/`write` are overridden to avoid. Without these overrides
    an async caller silently gets different semantics and the transport limits
    the sync path exists to dodge.
    """

    async def test_aread_uses_the_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.read_file.return_value = b"one\ntwo\n"
        sandbox = ACASandbox(client, async_client=async_client)
        result = await sandbox.aread("/f.txt")
        assert result.file_data["content"] == "one\ntwo"
        async_client.read_file.assert_awaited_once_with("/f.txt")
        client.exec.assert_not_called()

    async def test_aread_offloads_the_sync_client_without_an_async_one(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"one\ntwo\n"
        result = await sandbox.aread("/f.txt")
        assert result.file_data["content"] == "one\ntwo"
        client.read_file.assert_called_once_with("/f.txt")
        client.exec.assert_not_called()

    async def test_aread_matches_sync_read(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"a\nb\nc\nd\n"
        assert await sandbox.aread("/f.txt", offset=1, limit=2) == sandbox.read(
            "/f.txt", offset=1, limit=2
        )

    async def test_aread_missing_file(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # Stat-first: a missing file fails at stat_file, before any download.
        client.stat_file.side_effect = ResourceNotFoundError()
        assert (await sandbox.aread("/f.txt")).error == "File '/f.txt': file_not_found"
        client.read_file.assert_not_called()

    async def test_aread_http_error_is_classified(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.side_effect = _http_error("token=sk-secret")
        error = (await sandbox.aread("/f.txt")).error
        assert error is not None
        assert "sk-secret" not in error

    async def test_aread_directory_errors_without_downloading(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = lambda path: stat_info(path, is_directory=True)
        result = await sandbox.aread("/data")
        assert result.error == f"File '/data': {IS_DIRECTORY}"
        client.read_file.assert_not_called()

    async def test_aread_directory_errors_on_the_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.stat_file.side_effect = lambda path: stat_info(
            path, is_directory=True
        )
        sandbox = ACASandbox(client, async_client=async_client)
        result = await sandbox.aread("/data")
        assert result.error == f"File '/data': {IS_DIRECTORY}"
        async_client.read_file.assert_not_awaited()

    async def test_aread_large_file_falls_back_to_server_side_pagination(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = lambda path: stat_info(
            path, size=_READ_DOWNLOAD_CAP + 1
        )
        paginated = ReadResult(error="from the base class")
        with mock.patch.object(
            BaseSandbox, "aread", return_value=paginated
        ) as base_aread:
            result = await sandbox.aread("/huge.log", offset=3, limit=7)
        assert result is paginated
        base_aread.assert_awaited_once_with("/huge.log", 3, 7)
        client.read_file.assert_not_called()

    async def test_aread_large_file_falls_back_on_the_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.stat_file.side_effect = lambda path: stat_info(
            path, size=_READ_DOWNLOAD_CAP + 1
        )
        sandbox = ACASandbox(client, async_client=async_client)
        paginated = ReadResult(error="from the base class")
        with mock.patch.object(
            BaseSandbox, "aread", return_value=paginated
        ) as base_aread:
            result = await sandbox.aread("/huge.log")
        assert result is paginated
        base_aread.assert_awaited_once()
        async_client.read_file.assert_not_awaited()

    async def test_awrite_uses_the_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.stat_file.side_effect = ResourceNotFoundError()
        sandbox = ACASandbox(client, async_client=async_client)
        assert (await sandbox.awrite("/f.txt", "hi")).path == "/f.txt"
        async_client.write_file.assert_awaited_once_with(
            "/f.txt", "hi", create_dirs=True
        )
        client.exec.assert_not_called()

    async def test_awrite_offloads_the_sync_client_without_an_async_one(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = ResourceNotFoundError()
        assert (await sandbox.awrite("/f.txt", "hi")).path == "/f.txt"
        client.write_file.assert_called_once_with("/f.txt", "hi", create_dirs=True)
        client.exec.assert_not_called()

    async def test_awrite_refuses_to_overwrite(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        result = await sandbox.awrite("/f.txt", "hi")
        assert result.error == (
            "File '/f.txt' already exists. Use edit to modify it, or delete it first."
        )
        client.write_file.assert_not_called()

    async def test_awrite_refuses_to_overwrite_a_directory(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.stat_file.side_effect = lambda path: stat_info(
            path, is_directory=True
        )
        sandbox = ACASandbox(client, async_client=async_client)
        result = await sandbox.awrite("/data", "hi")
        assert result.error == "Path '/data' is a directory"
        async_client.write_file.assert_not_called()

    async def test_awrite_stat_error_is_classified(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = _http_error("token=sk-secret")
        error = (await sandbox.awrite("/f.txt", "hi")).error
        assert error is not None
        assert "sk-secret" not in error

    async def test_awrite_write_error_is_classified(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = ResourceNotFoundError()
        client.write_file.side_effect = _http_error("token=sk-secret")
        error = (await sandbox.awrite("/f.txt", "hi")).error
        assert error is not None
        assert "sk-secret" not in error

    async def test_aupload_uses_the_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        sandbox = ACASandbox(client, async_client=async_client)
        results = await sandbox.aupload_files([("/a", b"a"), ("rel", b"b")])
        assert [r.error for r in results] == [None, "invalid_path"]
        async_client.write_file.assert_awaited_once_with("/a", b"a", create_dirs=True)

    async def test_aupload_captures_failures_per_file(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.write_file.side_effect = [None, ServiceRequestError("reset"), None]
        sandbox = ACASandbox(client, async_client=async_client)
        results = await sandbox.aupload_files(
            [("/a", b"a"), ("/b", b"b"), ("/c", b"c")]
        )
        assert [r.error for r in results] == [
            None,
            "request failed: ServiceRequestError",
            None,
        ]

    async def test_aupload_falls_back_to_the_sync_path(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        results = await sandbox.aupload_files([("/a", b"a")])
        assert [r.error for r in results] == [None]
        client.write_file.assert_called_once_with("/a", b"a", create_dirs=True)

    async def test_adownload_uses_the_async_client(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.read_file.return_value = b"one"
        sandbox = ACASandbox(client, async_client=async_client)
        results = await sandbox.adownload_files(["/a", "rel"])
        assert [(r.content, r.error) for r in results] == [
            (b"one", None),
            (None, "invalid_path"),
        ]

    async def test_adownload_captures_failures_per_file(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.read_file.side_effect = [b"one", ServiceRequestError("reset")]
        sandbox = ACASandbox(client, async_client=async_client)
        results = await sandbox.adownload_files(["/a", "/b"])
        assert [(r.content, r.error) for r in results] == [
            (b"one", None),
            (None, "request failed: ServiceRequestError"),
        ]

    async def test_adownload_falls_back_to_the_sync_path(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"one"
        results = await sandbox.adownload_files(["/a"])
        assert [(r.content, r.error) for r in results] == [(b"one", None)]
        client.read_file.assert_called_once_with("/a")
