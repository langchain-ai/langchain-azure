"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

import base64
from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.core.exceptions import (  # noqa: E402
    ResourceNotFoundError,
)
from deepagents.backends.protocol import (  # noqa: E402
    IS_DIRECTORY,
    ReadResult,
)
from deepagents.backends.sandbox import (  # noqa: E402
    MAX_BINARY_BYTES,
    MAX_OUTPUT_BYTES,
    TRUNCATION_MSG,
    BaseSandbox,
)

from langchain_azure_container_apps.sandboxes import ACASandbox  # noqa: E402
from langchain_azure_container_apps.sandboxes.sandbox import (  # noqa: E402
    _READ_DOWNLOAD_CAP,
)

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

from tests.unit_tests.sandboxes.sandbox._helpers import (  # noqa: E402
    stat_info,
)


class TestRead:
    def test_text_content_and_pagination(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"l1\nl2\nl3\nl4\n"
        result = sandbox.read("/f.txt", offset=1, limit=2)
        assert isinstance(result, ReadResult)
        assert result.file_data is not None
        assert result.file_data["content"] == "l2\nl3"
        assert result.total_lines == 4
        assert result.start_line == 2
        assert result.end_line == 3
        assert result.next_offset == 3

    def test_crlf_is_normalized(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        # Stray \r in returned content breaks a subsequent edit().
        client.read_file.return_value = b"a\r\nb\r\n"
        result = sandbox.read("/f.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "a\nb"

    def test_empty_file_reminder(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        client.read_file.return_value = b""
        result = sandbox.read("/f.txt")
        assert result.file_data is not None
        assert "empty" in result.file_data["content"].lower()

    def test_binary_returns_base64(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        raw = bytes(range(256))
        client.read_file.return_value = raw
        result = sandbox.read("/image.png")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"
        assert base64.b64decode(result.file_data["content"]) == raw

    def test_oversized_binary_errors(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"\x00" * (MAX_BINARY_BYTES + 1)
        result = sandbox.read("/big.png")
        assert result.file_data is None
        assert result.error is not None
        assert "exceeds maximum preview size" in result.error

    def test_undecodable_text_extension_falls_back_to_base64(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"\xff\xfe\x00bad"
        result = sandbox.read("/notes.txt")
        assert result.file_data is not None
        assert result.file_data["encoding"] == "base64"

    def test_missing_file(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        # Stat-first: a missing file fails at stat_file, before any download.
        client.stat_file.side_effect = ResourceNotFoundError()
        result = sandbox.read("/nope.txt")
        assert result.file_data is None
        assert result.error == "File '/nope.txt': file_not_found"
        client.read_file.assert_not_called()

    def test_offset_beyond_eof(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        client.read_file.return_value = b"a\nb\n"
        result = sandbox.read("/f.txt", offset=99)
        assert result.error is not None
        assert "exceeds file length" in result.error


class TestReadRouting:
    """Stat-first routing: what the stat decides before any download."""

    def test_directory_errors_without_downloading(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = lambda path: stat_info(path, is_directory=True)
        result = sandbox.read("/data")
        assert result.error == f"File '/data': {IS_DIRECTORY}"
        client.read_file.assert_not_called()

    def test_oversized_binary_refused_by_stat_without_downloading(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # The whole point of the pre-download stat: the refusal must not cost
        # a multi-megabyte transfer first.
        client.stat_file.side_effect = lambda path: stat_info(
            path, size=MAX_BINARY_BYTES + 1
        )
        result = sandbox.read("/big.png")
        assert result.error is not None
        assert "exceeds maximum preview size" in result.error
        client.read_file.assert_not_called()

    def test_large_text_file_falls_back_to_server_side_pagination(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # Above the download cap a paginated read must not re-transfer the
        # whole file per page; the base class paginates via execute() instead.
        client.stat_file.side_effect = lambda path: stat_info(
            path, size=_READ_DOWNLOAD_CAP + 1
        )
        paginated = ReadResult(error="from the base class")
        with mock.patch.object(
            BaseSandbox, "read", return_value=paginated
        ) as base_read:
            result = sandbox.read("/huge.log", offset=3, limit=7)
        assert result is paginated
        base_read.assert_called_once_with("/huge.log", 3, 7)
        client.read_file.assert_not_called()

    def test_size_at_the_download_cap_still_downloads(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = lambda path: stat_info(
            path, size=_READ_DOWNLOAD_CAP
        )
        client.read_file.return_value = b"ok\n"
        result = sandbox.read("/edge.log")
        assert result.file_data is not None
        assert result.file_data["content"] == "ok"
        client.read_file.assert_called_once_with("/edge.log")

    def test_unknown_stat_size_downloads_normally(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # A stat with no size must not be treated as oversized.
        client.stat_file.side_effect = lambda path: stat_info(path, size=None)
        client.read_file.return_value = b"hi\n"
        result = sandbox.read("/f.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "hi"


class TestReadPaginationEdges:
    """Branches `_read_result_from_bytes` reaches only on degenerate input."""

    @pytest.mark.parametrize("limit", [0, -1])
    def test_non_positive_limit_flags_uninspected(
        self, sandbox: ACASandbox, client: mock.Mock, limit: int
    ) -> None:
        # Models occasionally emit limit=0; it must not become an empty page
        # with an invalid start_line/end_line pair.
        client.read_file.return_value = b"a\nb\n"
        result = sandbox.read("/f.txt", limit=limit)
        assert result.no_lines_requested is True
        assert result.total_lines is None

    def test_negative_offset_is_clamped(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"a\nb\n"
        result = sandbox.read("/f.txt", offset=-5)
        assert result.start_line == 1

    def test_oversized_page_is_truncated(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        line = "x" * 999
        client.read_file.return_value = ("\n".join([line] * 2000)).encode()
        result = sandbox.read("/big.txt")

        assert result.file_data is not None
        content = result.file_data["content"]
        assert content.endswith(TRUNCATION_MSG)
        assert len(content.encode()) <= MAX_OUTPUT_BYTES
        # next_offset must not skip the partially rendered boundary line.
        assert result.next_offset == result.end_line
        assert result.end_line == content.count("\n") - TRUNCATION_MSG.count("\n")

    def test_single_line_larger_than_the_cap_still_advances(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # No newline survives the cut, so returned_lines falls back to 1
        # rather than 0, which would make a re-read loop forever.
        client.read_file.return_value = b"y" * (MAX_OUTPUT_BYTES + 10) + b"\ntail\n"
        result = sandbox.read("/huge.txt")
        assert result.end_line == 1
        assert result.next_offset == 1

    def test_no_truncation_just_under_the_cap(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.return_value = b"z" * 1000 + b"\n"
        result = sandbox.read("/f.txt")
        assert result.file_data is not None
        assert TRUNCATION_MSG not in result.file_data["content"]
        assert result.next_offset is None
