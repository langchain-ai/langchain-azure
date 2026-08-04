"""The data plane intermittently drops a command's final stdout write.

Measured at 1.6-4% against a live Shell pool. Without a completion marker `ls`
returns an empty listing with no error -- an agent is told a populated
directory is empty. These pin the marker handling: retry the read-only
operations, and never let a lost tail become a plausible-looking answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

pytest.importorskip("deepagents")

from deepagents.backends.protocol import ExecuteResponse  # noqa: E402

from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)
from tests.unit_tests.dynamic_sessions._helpers import echo_done  # noqa: E402
from tests.unit_tests.dynamic_sessions.backend._helpers import (  # noqa: E402
    _make_backend,
    read_payload,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

LS_OUTPUT = "a.txt\npkg/\n"
READ_OUTPUT = read_payload("hello\n", 1)
GLOB_OUTPUT = "F:a.py\n"


def _stub(
    backend: SessionsBashBackend,
    output: str,
    *,
    complete: Sequence[bool],
    exit_code: int = 0,
    truncated: bool = False,
) -> mock.Mock:
    """Stub `execute`, emitting the marker only on the attempts in `complete`."""
    calls = iter(complete)

    def run(command: str, **kwargs: object) -> ExecuteResponse:
        body = echo_done(command, output) if next(calls) else output
        return ExecuteResponse(output=body, exit_code=exit_code, truncated=truncated)

    stub = mock.Mock(side_effect=run)
    backend.execute = stub  # type: ignore[method-assign]
    return stub


class TestRetriedOnLostTail:
    """`ls`/`read`/`glob` return data, so a lost tail is a wrong answer."""

    def test_ls_retries_and_recovers(self, backend: SessionsBashBackend) -> None:
        stub = _stub(backend, LS_OUTPUT, complete=[False, True])
        result = backend.ls("/mnt/data")
        assert stub.call_count == 2
        assert result.error is None
        assert result.entries == [
            {"path": "/mnt/data/a.txt", "is_dir": False},
            {"path": "/mnt/data/pkg", "is_dir": True},
        ]

    def test_read_retries_and_recovers(self, backend: SessionsBashBackend) -> None:
        stub = _stub(backend, READ_OUTPUT, complete=[False, True])
        result = backend.read("/mnt/data/f.txt")
        assert stub.call_count == 2
        assert result.file_data is not None
        assert result.file_data["content"] == "hello"

    def test_glob_retries_and_recovers(self, backend: SessionsBashBackend) -> None:
        stub = _stub(backend, GLOB_OUTPUT, complete=[False, True])
        result = backend.glob("*.py", path="/mnt/data")
        assert stub.call_count == 2
        assert result.matches == [{"path": "a.py", "is_dir": False}]

    def test_each_attempt_uses_a_fresh_marker(
        self, backend: SessionsBashBackend
    ) -> None:
        """A replayed marker would let a stale response satisfy a later call."""
        stub = _stub(backend, LS_OUTPUT, complete=[False, True])
        backend.ls("/mnt/data")
        first, second = (c.args[0] for c in stub.call_args_list)
        assert first != second


class TestUnrecoverableLostTail:
    """Two lost attempts must not look like an empty directory or file."""

    def test_ls_reports_the_loss(self, backend: SessionsBashBackend) -> None:
        stub = _stub(backend, LS_OUTPUT, complete=[False, False])
        result = backend.ls("/mnt/data")
        assert stub.call_count == 2
        assert result.entries is None
        assert result.error is not None
        assert "retry it" in result.error

    def test_read_reports_the_loss(self, backend: SessionsBashBackend) -> None:
        _stub(backend, READ_OUTPUT, complete=[False, False])
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is None
        assert result.error is not None
        assert "retry it" in result.error

    def test_glob_reports_the_loss(self, backend: SessionsBashBackend) -> None:
        _stub(backend, GLOB_OUTPUT, complete=[False, False])
        result = backend.glob("*.py", path="/mnt/data")
        assert result.matches is None
        assert result.error is not None
        assert "retry it" in result.error

    def test_the_error_leaks_no_endpoint_or_session_id(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, LS_OUTPUT, complete=[False, False])
        error = backend.ls("/mnt/data").error or ""
        assert "dynamicsessions.io" not in error
        assert "test-session" not in error


class TestTruncatedOutput:
    """The byte cap removes the marker for a known reason: no retry -- but the
    surviving prefix must never be presented as the complete answer."""

    def test_truncated_ls_is_an_error_not_a_partial_listing(
        self, backend: SessionsBashBackend
    ) -> None:
        """`LsResult` has no truncated flag, so a capped listing must error:
        the alternative silently hides every entry past the cap."""
        stub = _stub(backend, LS_OUTPUT, complete=[False], truncated=True)
        result = backend.ls("/mnt/data")
        assert stub.call_count == 1
        assert result.entries is None
        assert result.error is not None
        assert "output cap" in result.error

    def test_truncated_glob_drops_the_partial_tail_and_flags_it(
        self, backend: SessionsBashBackend
    ) -> None:
        """The cap almost certainly cut the last line mid-name; a fabricated
        entry (`par` for `part.txt`) is worse than one fewer match."""
        _stub(backend, "F:a.py\nF:par", complete=[False], truncated=True)
        result = backend.glob("*.py", path="/mnt/data")
        assert result.error is None
        assert result.matches == [{"path": "a.py", "is_dir": False}]
        assert result.truncated is True

    def test_truncated_read_keeps_whole_lines_and_says_so(
        self, backend: SessionsBashBackend
    ) -> None:
        """The partial boundary line is neither shown nor skipped:
        `next_offset` re-reads it from its start."""
        # Above the chunked transport's floor: a cap that cannot carry the
        # read bookkeeping is refused outright and never reaches this logic.
        backend = _make_backend(max_output_bytes=200)
        first = "x" * 190
        _stub(
            backend,
            read_payload(f"{first}\nline two!!", 3, window_bytes=400),
            complete=[True],
        )
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is not None
        content = result.file_data["content"]
        assert content.startswith(f"{first}\n")
        assert "line two" not in content
        assert "truncated" in content
        assert result.end_line == 1
        assert result.next_offset == 1

    def test_truncation_before_the_separator_is_an_error(
        self, backend: SessionsBashBackend
    ) -> None:
        """The cap can land inside the line-count header; that must not read
        as 'unexpected output' from a healthy run."""
        _stub(backend, "37", complete=[False], truncated=True)
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is None
        assert result.error is not None
        assert "max_output_bytes" in result.error

    def test_truncated_read_with_no_whole_line_advances_past_it(
        self, backend: SessionsBashBackend
    ) -> None:
        """A first line that alone overflows the cap can never be rendered by
        any limit; the partial head is returned and `next_offset` advances
        past the line, matching `BaseSandbox.read`'s one-line advance."""
        backend = _make_backend(max_output_bytes=200)
        head = "partial-without-newline" + "z" * 200
        _stub(
            backend,
            read_payload(head, 3, window_bytes=400),
            complete=[True],
        )
        result = backend.read("/mnt/data/f.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"].startswith("partial-without-newline")
        assert result.start_line == 1
        assert result.end_line == 1
        assert result.next_offset == 1

    def test_truncation_cutting_the_marker_leaves_no_fragment(
        self, backend: SessionsBashBackend
    ) -> None:
        """When the cap bisects the completion marker itself, the leading
        fragment must not leak into the entries handed to the model."""
        import re

        def run(command: str, **kwargs: object) -> ExecuteResponse:
            token = re.search(r"__LCDONE[0-9a-f]{16}__", command).group(0)  # type: ignore[union-attr]
            body = "F:a.py\n" + token[: len(token) // 2]
            return ExecuteResponse(output=body, exit_code=0, truncated=True)

        backend.execute = mock.Mock(side_effect=run)  # type: ignore[method-assign]
        result = backend.glob("*.py", path="/mnt/data")
        assert result.error is None
        assert result.matches == [{"path": "a.py", "is_dir": False}]
        assert all("__LCDONE" not in m["path"] for m in result.matches or [])


class TestMarkerNotRequired:
    def test_write_is_not_gated_on_the_marker(
        self, backend: SessionsBashBackend
    ) -> None:
        """`write`'s outcome is the exit status, which the tail cannot carry off.

        Gating it would fail writes that had already succeeded, and retrying it
        would hit its own refusal to overwrite.
        """
        stub = _stub(backend, "", complete=[False], exit_code=0)
        result = backend.write("/mnt/data/f.txt", "x")
        assert stub.call_count == 1
        assert result.error is None
        assert result.path == "/mnt/data/f.txt"

    def test_edit_is_not_gated_on_the_marker(
        self, backend: SessionsBashBackend
    ) -> None:
        stub = _stub(backend, "2", complete=[False], exit_code=0)
        result = backend.edit("/mnt/data/f.txt", "a", "b", replace_all=True)
        assert stub.call_count == 1
        assert result.error is None
        assert result.occurrences == 2
