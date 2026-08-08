"""Contract tests pinning `SessionsBashBackend` to the deepagents 0.7 surface.

Every assertion here checks a return *type*, not just a method name. The bug
these guard against was silent: the backend carried `ls_info`/`glob_info`
overrides that deepagents 0.7 no longer calls, and a `read` that returned a
plain `str` where callers expect a `ReadResult`. Nothing raised -- the overrides
were simply skipped and the wrong shape was handed back.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING
from unittest import mock

import pytest

pytest.importorskip("deepagents")

from deepagents.backends.protocol import (  # noqa: E402
    EditResult,
    ExecuteResponse,
    GlobResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox  # noqa: E402

from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
    _shell,
)
from tests.unit_tests.dynamic_sessions._helpers import echo_done  # noqa: E402
from tests.unit_tests.dynamic_sessions.backend._helpers import (  # noqa: E402
    read_payload,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


@pytest.fixture
def backend() -> Iterator[SessionsBashBackend]:
    with mock.patch(
        "langchain_azure_container_apps.dynamic_sessions.backends.sessions"
        ".access_token_provider_factory",
        return_value=lambda: "token",
    ):
        yield SessionsBashBackend(
            pool_management_endpoint="https://pool.example.com/",
            session_id="test-session",
        )


def _stub(backend: SessionsBashBackend, output: str, exit_code: int = 0) -> mock.Mock:
    stub = mock.Mock(
        side_effect=lambda command, **kwargs: ExecuteResponse(
            output=echo_done(command, output), exit_code=exit_code, truncated=False
        )
    )
    backend.execute = stub  # type: ignore[method-assign]
    return stub


class TestZeroSevenMethodNames:
    """The 0.6 method names must be gone, not merely shadowed."""

    @pytest.mark.parametrize("legacy", ["ls_info", "glob_info", "grep_raw"])
    def test_legacy_override_names_absent(self, legacy: str) -> None:
        assert not hasattr(SessionsBashBackend, legacy)

    @pytest.mark.parametrize("name", ["ls", "glob", "read", "write", "edit"])
    def test_overrides_the_names_base_actually_calls(self, name: str) -> None:
        assert hasattr(BaseSandbox, name)
        assert getattr(SessionsBashBackend, name) is not getattr(BaseSandbox, name)


class TestLsContract:
    def test_returns_ls_result(self, backend: SessionsBashBackend) -> None:
        _stub(backend, "./\n../\nfile.txt\nsubdir/\n")
        result = backend.ls("/mnt/data")
        assert isinstance(result, LsResult)
        assert result.error is None

    def test_entries_are_absolute_paths(self, backend: SessionsBashBackend) -> None:
        # `ls -1ap` prints bare names; the base contract is absolute paths.
        _stub(backend, "./\n../\nfile.txt\nsubdir/\n")
        result = backend.ls("/mnt/data")
        assert result.entries == [
            {"path": "/mnt/data/file.txt", "is_dir": False},
            {"path": "/mnt/data/subdir", "is_dir": True},
        ]

    @pytest.mark.parametrize(
        "code", ["path_not_found", "not_a_directory", "permission_denied"]
    )
    def test_error_sentinel_becomes_error(
        self, backend: SessionsBashBackend, code: str
    ) -> None:
        _stub(backend, f"__ERR__{code}\n")
        result = backend.ls("/nope")
        assert result.entries is None
        assert result.error == f"Path '/nope': {code}"


class TestGlobContract:
    def test_returns_glob_result_with_relative_paths(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, "F:a.py\nD:pkg\n")
        result = backend.glob("**/*.py", path="/mnt/data")
        assert isinstance(result, GlobResult)
        assert result.matches == [
            {"path": "a.py", "is_dir": False},
            {"path": "pkg", "is_dir": True},
        ]

    def test_nonzero_exit_after_the_probes_reads_as_missing(
        self, backend: SessionsBashBackend
    ) -> None:
        """The probes catch the diagnosable root failures; a residual nonzero
        status means the root vanished between probe and `cd`."""
        _stub(backend, "", exit_code=1)
        result = backend.glob("*.py", path="/mnt/data")
        assert result.matches is None
        assert result.error == "Path '/mnt/data': path_not_found"

    def test_default_path_is_root(self, backend: SessionsBashBackend) -> None:
        stub = _stub(backend, "")
        backend.glob("*.py")
        tokens = shlex.split(stub.call_args[0][0])
        assert "p=/;" in tokens

    def test_pattern_cannot_break_out_of_quoting(
        self, backend: SessionsBashBackend
    ) -> None:
        # Tokenize the way a shell would: the whole payload must arrive as a
        # single argument to `find`, not as extra commands. Asserting on the
        # raw string instead would be meaningless -- the safety lives in the
        # `'"'"'` escaping, so any normalization of it fakes a failure.
        payload = "*.py'; id > /tmp/pwned; echo '"
        stub = _stub(backend, "")
        backend.glob(payload)
        tokens = shlex.split(stub.call_args[0][0])
        assert tokens[tokens.index("-path") + 1] == f"./{payload}"

    # The backslash case pins `has_traversal`'s separator normalization: a
    # `..` segment must not slip through spelled `..\x`.
    @pytest.mark.parametrize("pattern", ["../*.py", "a/../../etc/*", "/../x", "..\\x"])
    def test_parent_traversal_rejected(
        self, backend: SessionsBashBackend, pattern: str
    ) -> None:
        stub = _stub(backend, "")
        result = backend.glob(pattern, path="/mnt/data")
        assert result.matches is None
        assert result.error is not None
        assert "invalid_pattern" in result.error
        stub.assert_not_called()


class TestWriteEditContract:
    """`write`/`edit` must construct 0.7 result objects.

    0.7 dropped the `files_update` field these previously passed, so the
    constructor raised `TypeError` at runtime -- invisible to any test that
    only mocked `execute` without calling these methods.
    """

    def test_write_returns_write_result(self, backend: SessionsBashBackend) -> None:
        _stub(backend, "")
        result = backend.write("/mnt/data/new.txt", "content")
        assert isinstance(result, WriteResult)
        assert result.error is None
        assert result.path == "/mnt/data/new.txt"

    def test_write_existing_file_errors(self, backend: SessionsBashBackend) -> None:
        # The real conflict signal: exit status WRITE_EXISTS with no output.
        # Stubbing a nonzero exit with fabricated "already exists" text would
        # pass through the generic-failure branch instead.
        _stub(backend, "", exit_code=_shell.WRITE_EXISTS)
        result = backend.write("/mnt/data/f.txt", "content")
        assert result.error is not None
        assert "already exists" in result.error

    def test_edit_returns_edit_result(self, backend: SessionsBashBackend) -> None:
        _stub(backend, "2")
        result = backend.edit("/mnt/data/f.txt", "old", "new", replace_all=True)
        assert isinstance(result, EditResult)
        assert result.error is None
        assert result.path == "/mnt/data/f.txt"
        assert result.occurrences == 2


class TestReadContract:
    def test_returns_read_result(self, backend: SessionsBashBackend) -> None:
        _stub(backend, read_payload("line1\nline2\nline3\n", 3))
        result = backend.read("/mnt/data/f.txt")
        assert isinstance(result, ReadResult)
        assert result.error is None

    def test_content_is_raw_without_line_numbers(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, read_payload("line1\nline2\nline3\n", 3))
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "line1\nline2\nline3"
        assert result.file_data["encoding"] == "utf-8"

    def test_pagination_metadata(self, backend: SessionsBashBackend) -> None:
        _stub(backend, read_payload("line3\nline4\n", 10))
        result = backend.read("/mnt/data/f.txt", offset=2, limit=2)
        assert result.total_lines == 10
        assert result.start_line == 3
        assert result.end_line == 4
        # 0-indexed line after the last shown; ReadResult validates this.
        assert result.next_offset == 4

    def test_final_page_has_no_next_offset(self, backend: SessionsBashBackend) -> None:
        _stub(backend, read_payload("line3\nline4\n", 4))
        result = backend.read("/mnt/data/f.txt", offset=2, limit=2)
        assert result.end_line == 4
        assert result.next_offset is None

    def test_empty_file_reports_reminder(self, backend: SessionsBashBackend) -> None:
        _stub(backend, read_payload("", 0))
        result = backend.read("/mnt/data/empty.txt")
        assert result.error is None
        assert result.file_data is not None
        assert "empty" in result.file_data["content"].lower()

    def test_non_positive_limit_flags_uninspected(
        self, backend: SessionsBashBackend
    ) -> None:
        result = backend.read("/mnt/data/f.txt", limit=0)
        assert result.no_lines_requested is True
        assert result.total_lines is None

    def test_offset_beyond_eof_errors(self, backend: SessionsBashBackend) -> None:
        _stub(backend, read_payload("", 3))
        result = backend.read("/mnt/data/f.txt", offset=100)
        assert result.file_data is None
        assert result.error is not None
        assert "exceeds file length" in result.error

    @pytest.mark.parametrize(
        "code", ["file_not_found", "is_directory", "permission_denied"]
    )
    def test_error_sentinel_becomes_error(
        self, backend: SessionsBashBackend, code: str
    ) -> None:
        _stub(backend, f"__ERR__{code}\n")
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is None
        assert result.error == f"File '/mnt/data/f.txt': {code}"

    def test_line_count_uses_awk_not_wc(self, backend: SessionsBashBackend) -> None:
        # `wc -l` counts newlines, undercounting an unterminated final line.
        stub = _stub(backend, read_payload("only\n", 1))
        backend.read("/mnt/data/f.txt")
        assert "awk 'END {print NR}'" in stub.call_args[0][0]
