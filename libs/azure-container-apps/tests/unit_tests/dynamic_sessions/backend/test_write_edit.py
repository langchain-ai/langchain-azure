"""`write`/`edit` outcome mapping in `SessionsBashBackend`, over a stubbed
`execute()`.

Complements `test_sessions_shell_programs.py`, which runs the generated
programs under a real `/bin/sh`; here the exit-status-to-result mapping is
under test. The semantic exit codes live in `_shell` (10+, deliberately
disjoint from generic shell failures -- see `_shell.py`).
"""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")


from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
    _shell,
)
from tests.unit_tests.dynamic_sessions._helpers import echo_done  # noqa: E402

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestEditErrorPaths:
    def _stub(
        self,
        backend: SessionsBashBackend,
        output: str,
        exit_code: int,
        *,
        complete: bool = True,
    ) -> None:
        from deepagents.backends.protocol import ExecuteResponse

        def run(command: str, **kwargs: object) -> ExecuteResponse:
            body = echo_done(command, output) if complete else output
            return ExecuteResponse(output=body, exit_code=exit_code, truncated=False)

        backend.execute = mock.Mock(side_effect=run)  # type: ignore[method-assign]

    def test_string_not_found(self, backend: SessionsBashBackend) -> None:
        self._stub(backend, "", _shell.EDIT_NOT_FOUND)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error is not None
        assert "String not found" in result.error

    def test_ambiguous_match_without_replace_all(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, "3\n", _shell.EDIT_AMBIGUOUS)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error is not None
        assert "replace_all=True" in result.error

    def test_file_not_found(self, backend: SessionsBashBackend) -> None:
        self._stub(backend, "", _shell.EDIT_MISSING_FILE)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error == "Error: File '/mnt/data/f.txt' not found"

    def test_generic_failure_is_not_a_semantic_outcome(
        self, backend: SessionsBashBackend
    ) -> None:
        """Exit 1 (a failed `cat`/`mktemp`) used to collide with "string not
        found"; it must surface as a generic failure now."""
        self._stub(backend, "cat: permission denied", 1)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error == "cat: permission denied"

    def test_awk_fatal_error_is_not_ambiguity(
        self, backend: SessionsBashBackend
    ) -> None:
        """gawk exits 2 on a fatal error; that used to read as "appears
        multiple times"."""
        self._stub(backend, "awk: fatal: cannot open file", 2)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error == "awk: fatal: cannot open file"
        assert "replace_all" not in result.error

    def test_other_nonzero_exit_with_no_output(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, "", 5)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error == "Edit failed for '/mnt/data/f.txt'"

    def test_long_failure_detail_is_bounded(self, backend: SessionsBashBackend) -> None:
        """A garbled run must not ship kilobytes of file content to the model
        inside the error string."""
        self._stub(backend, "x" * 5000, 5)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error is not None
        assert len(result.error) < 300
        assert result.error.endswith("...")

    def test_empty_old_string(self, backend: SessionsBashBackend) -> None:
        self._stub(backend, "", _shell.EDIT_EMPTY_OLD)
        result = backend.edit("/mnt/data/f.txt", "", "new")
        assert result.error == "Error: old_string must not be empty"

    def test_unparseable_count_is_reported_as_unknown(
        self, backend: SessionsBashBackend
    ) -> None:
        """The count is honest or absent, never guessed: `1` would be wrong
        for a `replace_all` that replaced three."""
        self._stub(backend, "unexpected", 0)
        result = backend.edit("/mnt/data/f.txt", "old", "new")
        assert result.error is None
        assert result.occurrences is None

    def test_lost_count_is_reported_as_unknown(
        self, backend: SessionsBashBackend
    ) -> None:
        """The edit's outcome rides the exit status, so a lost output tail
        succeeds -- with the occurrence count marked unknowable."""
        self._stub(backend, "", 0, complete=False)
        result = backend.edit("/mnt/data/f.txt", "old", "new", replace_all=True)
        assert result.error is None
        assert result.path == "/mnt/data/f.txt"
        assert result.occurrences is None

    def test_surviving_count_is_used_even_without_the_marker(
        self, backend: SessionsBashBackend
    ) -> None:
        """Losing the *final* write takes the marker, not the earlier stderr
        count; a parseable count is trusted on its own."""
        self._stub(backend, "2", 0, complete=False)
        result = backend.edit("/mnt/data/f.txt", "a", "b", replace_all=True)
        assert result.error is None
        assert result.occurrences == 2

    def test_content_is_base64_encoded_not_interpolated(
        self, backend: SessionsBashBackend
    ) -> None:
        # A replacement string containing shell metacharacters must not reach
        # the command line verbatim.
        self._stub(backend, "1", 0)
        payload = "'; touch /tmp/pwned; echo '"
        backend.edit("/mnt/data/f.txt", "old", payload)
        assert payload not in backend.execute.call_args[0][0]  # type: ignore[attr-defined]

    def test_write_content_is_base64_encoded_not_interpolated(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, "", 0)
        payload = "'; touch /tmp/pwned; echo '"
        backend.write("/mnt/data/f.txt", payload)
        assert payload not in backend.execute.call_args[0][0]  # type: ignore[attr-defined]

    def test_existing_file_reports_a_conflict(
        self, backend: SessionsBashBackend
    ) -> None:
        # Signalled by exit status, so the message never carries shell output.
        # The message also tells the model how to proceed, because the
        # deepagents middleware's tool description promises overwrite.
        self._stub(backend, "", _shell.WRITE_EXISTS)
        result = backend.write("/mnt/data/f.txt", "content")
        assert result.error is not None
        assert "File '/mnt/data/f.txt' already exists" in result.error
        assert "edit" in result.error

    def test_write_generic_exit_1_is_not_a_conflict(
        self, backend: SessionsBashBackend
    ) -> None:
        """bash exits 1 on a failed redirect; that used to read as "already
        exists"."""
        self._stub(backend, "", 1)
        result = backend.write("/mnt/data/f.txt", "content")
        assert result.error == "Failed to write file '/mnt/data/f.txt': exit code 1"

    def test_mkdir_failure_is_reported(self, backend: SessionsBashBackend) -> None:
        self._stub(backend, "", 2)
        result = backend.write("/mnt/data/f.txt", "content")
        assert result.error == "Failed to write file '/mnt/data/f.txt': exit code 2"

    def test_write_failure_surfaces_shell_output(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, "sh: no space left on device", 3)
        result = backend.write("/mnt/data/f.txt", "content")
        assert result.error == (
            "Failed to write file '/mnt/data/f.txt': sh: no space left on device"
        )


class TestTransportSizeLimits:
    """Payloads whose base64 nears Linux's 128 KiB per-string cap die at exec
    inside the pool with an unhelpful E2BIG (measured live: 64 KB passes,
    120 KB fails), so oversized ones are refused client-side with advice."""

    def _unstubbed(self, backend: SessionsBashBackend) -> None:
        backend.execute = mock.MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("oversized payload reached the shell")
        )

    def test_oversized_write_is_refused_before_the_shell(
        self, backend: SessionsBashBackend
    ) -> None:
        self._unstubbed(backend)
        result = backend.write(
            "/mnt/data/big.txt", "x" * (_shell.MAX_TRANSPORT_BYTES + 1)
        )
        assert result.error is not None
        assert str(_shell.MAX_TRANSPORT_BYTES) in result.error
        assert "upload_files" in result.error

    def test_oversized_edit_string_is_refused_before_the_shell(
        self, backend: SessionsBashBackend
    ) -> None:
        self._unstubbed(backend)
        result = backend.edit(
            "/mnt/data/f.txt", "old", "y" * (_shell.MAX_TRANSPORT_BYTES + 1)
        )
        assert result.error is not None
        assert str(_shell.MAX_TRANSPORT_BYTES) in result.error

    def test_size_is_measured_in_utf8_bytes_not_characters(
        self, backend: SessionsBashBackend
    ) -> None:
        # A three-byte character puts a short string over the byte cap.
        self._unstubbed(backend)
        result = backend.edit(
            "/mnt/data/f.txt", "€" * (_shell.MAX_TRANSPORT_BYTES // 3 + 1), "new"
        )
        assert result.error is not None

    def test_two_individually_legal_edit_strings_are_refused_together(
        self, backend: SessionsBashBackend
    ) -> None:
        # Both base64 literals ride in one `sh -c` argument, so the budget is
        # their sum: each of these passes a per-string check, but together they
        # generate a 134,931-byte program against a 131,072-byte cap.
        self._unstubbed(backend)
        half = _shell.MAX_TRANSPORT_BYTES // 2 + 5_000
        result = backend.edit("/mnt/data/f.txt", "a" * half, "b" * half)
        assert result.error is not None
        assert str(2 * half) in result.error
        assert str(_shell.MAX_TRANSPORT_BYTES) in result.error

    def test_combined_payload_at_the_cap_still_reaches_the_shell(
        self, backend: SessionsBashBackend
    ) -> None:
        # The guard must not be so conservative that a legal edit is refused.
        from deepagents.backends.protocol import ExecuteResponse

        backend.execute = mock.Mock(  # type: ignore[method-assign]
            side_effect=lambda command, **kwargs: ExecuteResponse(
                output=echo_done(command, "1"), exit_code=0, truncated=False
            )
        )
        half = _shell.MAX_TRANSPORT_BYTES // 2
        result = backend.edit("/mnt/data/f.txt", "a" * half, "b" * half)
        assert result.error is None
        backend.execute.assert_called_once()
