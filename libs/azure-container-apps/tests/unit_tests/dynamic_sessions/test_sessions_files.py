"""Unit tests for the file-transfer methods on both dynamic-sessions tools.

The two tools talk to different session-pool APIs and so have genuinely
different URL shapes and response payloads -- Python pools nest metadata under
``properties``, Shell pools return it flat -- which is why these are asserted
per tool rather than shared.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest

from langchain_azure_container_apps.dynamic_sessions import (
    SessionsBashTool,
    SessionsPythonREPLTool,
)
from langchain_azure_container_apps.dynamic_sessions._common import (
    USER_AGENT,
    auth_headers,
    build_session_url,
)
from langchain_azure_container_apps.dynamic_sessions.tools.sessions import (
    RemoteFileMetadata,
)
from tests.unit_tests.dynamic_sessions._helpers import stub_response

if TYPE_CHECKING:
    from collections.abc import Iterator

ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/rg/sessionPools/pool"


def _tool_kwargs() -> dict:
    return {
        "pool_management_endpoint": ENDPOINT,
        "access_token_provider": lambda: "token_value",
    }


@pytest.fixture
def python_tool() -> SessionsPythonREPLTool:
    return SessionsPythonREPLTool(**_tool_kwargs())


@pytest.fixture
def bash_tool() -> SessionsBashTool:
    return SessionsBashTool(**_tool_kwargs())


@pytest.fixture
def http_request() -> Iterator[mock.MagicMock]:
    with mock.patch("requests.request") as request:
        yield request


@pytest.fixture
def http_get() -> Iterator[mock.MagicMock]:
    with mock.patch("requests.get") as get:
        yield get


def _path_of(mock_call: mock.Mock, *, arg_index: int = 0) -> str:
    return urlparse(mock_call.call_args.args[arg_index]).path


def _path_of_str(url: str) -> str:
    return urlparse(url).path


def _capture_upload(request: mock.MagicMock) -> dict[str, Any]:
    """Record the multipart payload at call time.

    A handle opened by `upload_file` is closed before it returns, so the bytes
    have to be read inside the call rather than from `call_args` afterwards.
    """
    seen: dict[str, Any] = {}

    def record(*_args: Any, **kwargs: Any) -> mock.Mock:
        ((field, (name, handle, _mime)),) = kwargs["files"]
        assert field == "file"
        seen["filename"] = name
        seen["content"] = handle.read()
        seen["handle"] = handle
        return cast(mock.Mock, request.return_value)

    request.side_effect = record
    return seen


class TestRemoteFileMetadata:
    def test_full_path_is_under_mnt_data(self) -> None:
        meta = RemoteFileMetadata(filename="report.csv", size_in_bytes=10)
        assert meta.full_path == "/mnt/data/report.csv"

    def test_from_dict_reads_nested_properties(self) -> None:
        meta = RemoteFileMetadata.from_dict(
            {"properties": {"filename": "a.txt", "size": 42}}
        )
        assert meta == RemoteFileMetadata(filename="a.txt", size_in_bytes=42)

    def test_from_dict_tolerates_a_missing_properties_block(self) -> None:
        meta = RemoteFileMetadata.from_dict({})
        assert meta.filename is None
        assert meta.size_in_bytes is None


class TestPythonToolUpload:
    def test_uploads_binary_data(
        self, python_tool: SessionsPythonREPLTool, http_request: mock.MagicMock
    ) -> None:
        stub_response(
            http_request, {"value": [{"properties": {"filename": "a.txt", "size": 3}}]}
        )
        meta = python_tool.upload_file(data=BytesIO(b"abc"), remote_file_path="a.txt")

        assert meta == RemoteFileMetadata(filename="a.txt", size_in_bytes=3)
        assert http_request.call_args.args[0] == "POST"
        assert _path_of(http_request, arg_index=1).endswith("/files/upload")
        headers = http_request.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer token_value"

    def test_uploads_from_a_local_path(
        self,
        python_tool: SessionsPythonREPLTool,
        http_request: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        local = tmp_path / "local.txt"
        local.write_bytes(b"hello")
        stub_response(
            http_request,
            {"value": [{"properties": {"filename": "local.txt", "size": 5}}]},
        )
        seen = _capture_upload(http_request)
        python_tool.upload_file(local_file_path=str(local))

        assert seen["filename"] == "local.txt"
        assert seen["content"] == b"hello"
        assert seen["handle"].closed, "upload_file leaked the file handle it opened"

    def test_local_path_remote_name_can_be_overridden(
        self,
        python_tool: SessionsPythonREPLTool,
        http_request: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        local = tmp_path / "local.txt"
        local.write_bytes(b"hello")
        stub_response(
            http_request,
            {"value": [{"properties": {"filename": "renamed.txt", "size": 5}}]},
        )
        seen = _capture_upload(http_request)
        python_tool.upload_file(
            local_file_path=str(local), remote_file_path="renamed.txt"
        )

        assert seen["filename"] == "renamed.txt"
        assert seen["handle"].closed

    def test_data_and_local_path_together_are_rejected(
        self, python_tool: SessionsPythonREPLTool, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="cannot be provided together"):
            python_tool.upload_file(data=BytesIO(b"x"), local_file_path=str(tmp_path))

    def test_neither_data_nor_local_path_is_rejected(
        self, python_tool: SessionsPythonREPLTool
    ) -> None:
        # Without this guard the method falls through to an unbound file_data
        # and dies with UnboundLocalError instead of an actionable message.
        with pytest.raises(ValueError, match="must be provided"):
            python_tool.upload_file(remote_file_path="a.txt")

    def test_data_without_remote_path_is_rejected(
        self, python_tool: SessionsPythonREPLTool
    ) -> None:
        # There is no filename to default to. The old code passed
        # `filename=None` through to the multipart request; an `assert` here
        # would vanish under `python -O` and silently restore that.
        with pytest.raises(ValueError, match="remote_file_path is required"):
            python_tool.upload_file(data=BytesIO(b"x"))

    def test_caller_supplied_stream_is_left_open(
        self, python_tool: SessionsPythonREPLTool, http_request: mock.MagicMock
    ) -> None:
        # The caller owns `data` and may still need it; only a handle opened
        # inside upload_file is closed.
        stub_response(
            http_request, {"value": [{"properties": {"filename": "a.txt", "size": 1}}]}
        )
        stream = BytesIO(b"abc")
        python_tool.upload_file(data=stream, remote_file_path="a.txt")
        assert not stream.closed

    def test_opened_handle_is_closed_even_when_the_request_fails(
        self,
        python_tool: SessionsPythonREPLTool,
        http_request: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        local = tmp_path / "local.txt"
        local.write_bytes(b"hello")
        handles: list[Any] = []

        def explode(*_args: Any, **kwargs: Any) -> mock.Mock:
            ((field, (_name, handle, _mime)),) = kwargs["files"]
            assert field == "file"
            handles.append(handle)
            raise RuntimeError("connection reset")

        http_request.side_effect = explode
        with pytest.raises(RuntimeError, match="connection reset"):
            python_tool.upload_file(local_file_path=str(local))

        assert handles[0].closed

    def test_http_error_propagates(
        self, python_tool: SessionsPythonREPLTool, http_request: mock.MagicMock
    ) -> None:
        stub_response(http_request).raise_for_status.side_effect = RuntimeError("403")
        with pytest.raises(RuntimeError, match="403"):
            python_tool.upload_file(data=BytesIO(b"x"), remote_file_path="a.txt")


class TestPythonToolDownload:
    def test_returns_content_as_a_stream(
        self, python_tool: SessionsPythonREPLTool, http_get: mock.MagicMock
    ) -> None:
        http_get.return_value.content = b"file bytes"
        assert (
            python_tool.download_file(remote_file_path="a.txt").read() == b"file bytes"
        )
        assert _path_of(http_get).endswith("/files/content/a.txt")

    def test_percent_encodes_the_remote_path(
        self, python_tool: SessionsPythonREPLTool, http_get: mock.MagicMock
    ) -> None:
        http_get.return_value.content = b"x"
        python_tool.download_file(remote_file_path="my file.txt")
        assert "/files/content/my%20file.txt" in http_get.call_args.args[0]

    def test_also_writes_to_a_local_path(
        self,
        python_tool: SessionsPythonREPLTool,
        http_get: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        http_get.return_value.content = b"file bytes"
        local = tmp_path / "out.txt"
        stream = python_tool.download_file(
            remote_file_path="a.txt", local_file_path=str(local)
        )
        assert local.read_bytes() == b"file bytes"
        assert stream.read() == b"file bytes"

    def test_http_error_propagates(
        self, python_tool: SessionsPythonREPLTool, http_get: mock.MagicMock
    ) -> None:
        stub_response(http_get).raise_for_status.side_effect = RuntimeError("404")
        with pytest.raises(RuntimeError, match="404"):
            python_tool.download_file(remote_file_path="missing.txt")


class TestPythonToolListFiles:
    def test_parses_nested_properties(
        self, python_tool: SessionsPythonREPLTool, http_get: mock.MagicMock
    ) -> None:
        stub_response(
            http_get,
            {
                "value": [
                    {"properties": {"filename": "a.txt", "size": 1}},
                    {"properties": {"filename": "b.csv", "size": 2}},
                ]
            },
        )
        assert python_tool.list_files() == [
            RemoteFileMetadata(filename="a.txt", size_in_bytes=1),
            RemoteFileMetadata(filename="b.csv", size_in_bytes=2),
        ]
        assert _path_of(http_get).endswith("/files")

    def test_empty_listing(
        self, python_tool: SessionsPythonREPLTool, http_get: mock.MagicMock
    ) -> None:
        stub_response(http_get, {"value": []})
        assert python_tool.list_files() == []


class TestBashToolUpload:
    def test_uploads_binary_data_to_mnt_data(
        self, bash_tool: SessionsBashTool, http_request: mock.MagicMock
    ) -> None:
        stub_response(
            http_request,
            {
                "name": "a.txt",
                "sizeInBytes": 3,
            },
        )
        meta = bash_tool.upload_file(data=BytesIO(b"abc"), remote_file_path="a.txt")

        assert meta == RemoteFileMetadata(filename="a.txt", size_in_bytes=3)
        url = http_request.call_args.args[1]
        assert "/files?" in url
        assert parse_qs(urlparse(url).query)["path"] == ["/mnt/data"]

    def test_uploads_from_a_local_path(
        self,
        bash_tool: SessionsBashTool,
        http_request: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        local = tmp_path / "local.txt"
        local.write_bytes(b"hello")
        stub_response(
            http_request,
            {
                "name": "local.txt",
                "sizeInBytes": 5,
            },
        )
        seen = _capture_upload(http_request)
        bash_tool.upload_file(local_file_path=str(local))

        assert seen["filename"] == "local.txt"
        assert seen["content"] == b"hello"
        assert seen["handle"].closed

    def test_local_path_remote_name_can_be_overridden(
        self,
        bash_tool: SessionsBashTool,
        http_request: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        local = tmp_path / "local.txt"
        local.write_bytes(b"hello")
        stub_response(
            http_request,
            {
                "name": "renamed.txt",
                "sizeInBytes": 5,
            },
        )
        seen = _capture_upload(http_request)
        bash_tool.upload_file(
            local_file_path=str(local), remote_file_path="renamed.txt"
        )

        assert seen["filename"] == "renamed.txt"
        assert seen["handle"].closed

    def test_data_and_local_path_together_are_rejected(
        self, bash_tool: SessionsBashTool, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="cannot be provided together"):
            bash_tool.upload_file(data=BytesIO(b"x"), local_file_path=str(tmp_path))

    def test_neither_data_nor_local_path_is_rejected(
        self, bash_tool: SessionsBashTool
    ) -> None:
        with pytest.raises(ValueError, match="must be provided"):
            bash_tool.upload_file(remote_file_path="a.txt")

    def test_caller_supplied_stream_is_left_open(
        self, bash_tool: SessionsBashTool, http_request: mock.MagicMock
    ) -> None:
        stub_response(
            http_request,
            {
                "name": "a.txt",
                "sizeInBytes": 3,
            },
        )
        stream = BytesIO(b"abc")
        bash_tool.upload_file(data=stream, remote_file_path="a.txt")
        assert not stream.closed


class TestBashToolDownload:
    def test_returns_content_as_a_stream(
        self, bash_tool: SessionsBashTool, http_get: mock.MagicMock
    ) -> None:
        http_get.return_value.content = b"file bytes"
        assert bash_tool.download_file(remote_file_path="a.txt").read() == b"file bytes"
        assert _path_of(http_get).endswith("/files/a.txt/content")

    def test_percent_encodes_the_remote_path(
        self, bash_tool: SessionsBashTool, http_get: mock.MagicMock
    ) -> None:
        http_get.return_value.content = b"x"
        bash_tool.download_file(remote_file_path="my file.txt")
        assert "/files/my%20file.txt/content" in http_get.call_args.args[0]

    def test_also_writes_to_a_local_path(
        self, bash_tool: SessionsBashTool, http_get: mock.MagicMock, tmp_path: Path
    ) -> None:
        http_get.return_value.content = b"file bytes"
        local = tmp_path / "out.txt"
        stream = bash_tool.download_file(
            remote_file_path="a.txt", local_file_path=str(local)
        )
        assert local.read_bytes() == b"file bytes"
        assert stream.read() == b"file bytes"


class TestBashToolListFiles:
    def test_parses_flat_payload(
        self, bash_tool: SessionsBashTool, http_get: mock.MagicMock
    ) -> None:
        stub_response(
            http_get,
            {
                "value": [
                    {"name": "a.txt", "sizeInBytes": 1},
                    {"name": "b.csv", "sizeInBytes": 2},
                ]
            },
        )
        assert bash_tool.list_files() == [
            RemoteFileMetadata(filename="a.txt", size_in_bytes=1),
            RemoteFileMetadata(filename="b.csv", size_in_bytes=2),
        ]
        assert _path_of(http_get).endswith("/files")

    def test_empty_listing(
        self, bash_tool: SessionsBashTool, http_get: mock.MagicMock
    ) -> None:
        stub_response(http_get, {"value": []})
        assert bash_tool.list_files() == []


class TestEndpointNormalization:
    """A trailing slash on the endpoint must not produce a doubled path."""

    def test_python_tool_endpoint_with_trailing_slash(
        self, http_get: mock.MagicMock
    ) -> None:
        tool = SessionsPythonREPLTool(
            pool_management_endpoint=f"{ENDPOINT}/",
            access_token_provider=lambda: "token_value",
        )
        assert "//files" not in tool._build_url("files")
        assert _path_of_str(tool._build_url("files")).endswith("/files")

    def test_bash_tool_endpoint_with_trailing_slash(self) -> None:
        tool = SessionsBashTool(
            pool_management_endpoint=f"{ENDPOINT}/",
            access_token_provider=lambda: "token_value",
        )
        assert "//files" not in tool._build_url("files")
        assert _path_of_str(tool._build_url("files")).endswith("/files")

    def test_delete_session_url_with_trailing_slash(self) -> None:
        url = build_session_url(
            f"{ENDPOINT}/", "session", session_id="sid", api_version="v"
        )
        assert "//session" not in url
        assert _path_of_str(url).endswith("/session")

    def test_caller_params_cannot_override_the_session_identifier(self) -> None:
        # Extra params are merged first and the standard identifier/api-version
        # last, so a caller-supplied "identifier" can never replace the
        # session id.
        url = build_session_url(
            ENDPOINT,
            "files",
            session_id="real-session-id",
            api_version="v",
            params={"identifier": "evil"},
        )
        query = parse_qs(urlparse(url).query)
        assert query["identifier"] == ["real-session-id"]
        assert query["api-version"] == ["v"]


class TestAuthHeaders:
    def test_returns_bearer_and_user_agent(self) -> None:
        assert auth_headers("tok") == {
            "Authorization": "Bearer tok",
            "User-Agent": USER_AGENT,
        }

    def test_none_token_is_rejected(self) -> None:
        # Sending "Authorization: Bearer None" would only surface later as an
        # opaque service 401.
        with pytest.raises(ValueError, match="returned no token"):
            auth_headers(None)

    def test_empty_token_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="returned no token"):
            auth_headers("")
