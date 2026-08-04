"""`read` result construction in `SessionsBashBackend`, over a stubbed
`execute()`: error branches, pagination bookkeeping, and CRLF handling that
`test_sessions_backend_contract.py` does not reach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

pytest.importorskip("deepagents")


from langchain_azure_compute.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)
from tests.unit_tests.dynamic_sessions._helpers import echo_done  # noqa: E402
from tests.unit_tests.dynamic_sessions.backend._helpers import (  # noqa: E402
    read_payload,
)

if TYPE_CHECKING:
    pass

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestReadErrorPaths:
    """Branches `test_sessions_backend_contract.py` does not reach."""

    def _stub(self, backend: SessionsBashBackend, output: str, exit_code: int) -> None:
        from deepagents.backends.protocol import ExecuteResponse

        backend.execute = mock.Mock(  # type: ignore[method-assign]
            side_effect=lambda command, **kwargs: ExecuteResponse(
                output=echo_done(command, output), exit_code=exit_code, truncated=False
            )
        )

    def test_nonzero_exit_without_sentinel_surfaces_output(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, "awk: cannot open file", 2)
        result = backend.read("/mnt/data/f.txt")
        assert result.error == "File '/mnt/data/f.txt': awk: cannot open file"

    def test_nonzero_exit_with_empty_output_reports_the_code(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, "", 2)
        result = backend.read("/mnt/data/f.txt")
        assert result.error is not None
        assert "exit code 2" in result.error

    def test_missing_separator_is_reported(self, backend: SessionsBashBackend) -> None:
        self._stub(backend, "3\nno separator here\n", 0)
        result = backend.read("/mnt/data/f.txt")
        assert result.error is not None
        assert "unexpected read output" in result.error

    def test_unparseable_line_count_is_reported(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, "not-a-number\n__SEP__\n6\n__SEP__\nbGluZTEK\n", 0)
        result = backend.read("/mnt/data/f.txt")
        assert result.error is not None
        assert "could not determine line count" in result.error

    def test_negative_offset_is_clamped_to_zero(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, read_payload("a\nb\n", 2), 0)
        result = backend.read("/mnt/data/f.txt", offset=-5)
        assert result.start_line == 1

    def test_body_without_trailing_newline_keeps_last_line(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, read_payload("a\nb", 2), 0)
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "a\nb"

    def test_empty_window_for_a_non_empty_file_is_an_error(
        self, backend: SessionsBashBackend
    ) -> None:
        # Defensive: not reachable through the shell program, since an empty
        # window is excluded by the offset check. A ReadResult with
        # end_line < start_line is rejected by the protocol outright, so an
        # anomalous transport response must not produce one.
        self._stub(backend, read_payload("", 3), 0)
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is None
        assert result.error == (
            "File '/mnt/data/f.txt': read returned no lines for a 3-line file"
        )


class TestCrlfNormalization:
    """`read` returns LF-normalized content, matching `BaseSandbox.read`, so
    the model can round-trip it through `edit` on LF files."""

    def _stub(self, backend: SessionsBashBackend, output: str) -> None:
        from deepagents.backends.protocol import ExecuteResponse

        backend.execute = mock.Mock(  # type: ignore[method-assign]
            side_effect=lambda command, **kwargs: ExecuteResponse(
                output=echo_done(command, output), exit_code=0, truncated=False
            )
        )

    def test_crlf_line_endings_are_normalized(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub(backend, read_payload("alpha\r\nbeta\r\n", 2))
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "alpha\nbeta"

    def test_trailing_cr_on_the_final_line_is_stripped(
        self, backend: SessionsBashBackend
    ) -> None:
        # The window's last CRLF loses its LF to body[:-1]; the orphan CR must
        # not survive into the content.
        self._stub(backend, read_payload("hello\r\n", 1))
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "hello"

    def test_bare_cr_content_passes_through(self, backend: SessionsBashBackend) -> None:
        # A lone-CR file has no line breaks awk recognizes; documented as
        # passed through unchanged.
        self._stub(backend, read_payload("a\rb\n", 1))
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is not None
        assert result.file_data["content"] == "a\rb"


class TestChunkedRead:
    """Windows larger than one chunk are paged under the service's 4 KiB
    output cap and reassembled byte-exactly."""

    def _stub_sequence(self, backend: SessionsBashBackend, outputs: list[str]) -> None:
        from deepagents.backends.protocol import ExecuteResponse

        it = iter(outputs)

        def run(command: str, **kwargs: object) -> ExecuteResponse:
            return ExecuteResponse(
                output=echo_done(command, next(it)), exit_code=0, truncated=False
            )

        backend.execute = mock.Mock(side_effect=run)  # type: ignore[method-assign]

    @staticmethod
    def _b64(raw: bytes) -> str:
        import base64

        return base64.b64encode(raw).decode("ascii")

    def test_multi_chunk_window_is_reassembled(
        self, backend: SessionsBashBackend
    ) -> None:
        content = "".join(f"line{i:04d}\n" for i in range(800))  # 7,200 bytes
        raw = content.encode("utf-8")
        self._stub_sequence(
            backend,
            [
                f"800\n__SEP__\n{len(raw)}\n__SEP__\n{self._b64(raw[:3000])}\n",
                self._b64(raw[3000:6000]) + "\n",
                self._b64(raw[6000:]) + "\n",
            ],
        )
        result = backend.read("/mnt/data/big.txt")
        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == content[:-1]
        assert result.total_lines == 800
        assert result.end_line == 800
        assert result.next_offset is None
        # The follow-up commands page the same awk window by byte offset.
        calls = [c.args[0] for c in backend.execute.call_args_list]  # type: ignore[attr-defined]
        assert len(calls) == 3
        assert "tail -c +3001" in calls[1] and "head -c 3000" in calls[1]
        assert "tail -c +6001" in calls[2] and "head -c 1200" in calls[2]

    def test_shrinking_window_is_reported_not_stitched(
        self, backend: SessionsBashBackend
    ) -> None:
        """An empty follow-up chunk means the measured window no longer has
        those bytes: the file changed, and stitching two versions together
        would hand the model a document that never existed."""
        raw = b"x" * 3000
        self._stub_sequence(
            backend,
            [f"10\n__SEP__\n5000\n__SEP__\n{self._b64(raw)}\n", "\n"],
        )
        result = backend.read("/mnt/data/f.txt")
        assert result.file_data is None
        assert result.error == "File '/mnt/data/f.txt': file changed during read"

    def test_chunk_sentinel_propagates(self, backend: SessionsBashBackend) -> None:
        """A file deleted between chunks reports its probe code."""
        raw = b"y" * 3000
        self._stub_sequence(
            backend,
            [
                f"10\n__SEP__\n5000\n__SEP__\n{self._b64(raw)}\n",
                "__ERR__file_not_found\n",
            ],
        )
        result = backend.read("/mnt/data/f.txt")
        assert result.error == "File '/mnt/data/f.txt': file_not_found"

    def test_undecodable_chunk_is_reported(self, backend: SessionsBashBackend) -> None:
        raw = b"z" * 3000
        self._stub_sequence(
            backend,
            [f"10\n__SEP__\n5000\n__SEP__\n{self._b64(raw)}\n", "!!not-base64!!\n"],
        )
        result = backend.read("/mnt/data/f.txt")
        assert result.error is not None
        assert "unexpected read output" in result.error

    def test_lost_chunk_output_is_reported(self, backend: SessionsBashBackend) -> None:
        """A chunk whose marker never arrives, twice, is the loss condition."""
        from deepagents.backends.protocol import ExecuteResponse

        raw = b"w" * 3000
        first = f"10\n__SEP__\n5000\n__SEP__\n{self._b64(raw)}\n"
        outputs = iter([(first, True), ("", False), ("", False)])

        def run(command: str, **kwargs: object) -> ExecuteResponse:
            text, complete = next(outputs)
            return ExecuteResponse(
                output=echo_done(command, text) if complete else text,
                exit_code=0,
                truncated=False,
            )

        backend.execute = mock.Mock(side_effect=run)  # type: ignore[method-assign]
        result = backend.read("/mnt/data/f.txt")
        assert result.error is not None
        assert "returned no output" in result.error

    def test_unparseable_window_size_is_reported(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub_sequence(backend, ["3\n__SEP__\nnot-a-size\n__SEP__\nYQ==\n"])
        result = backend.read("/mnt/data/f.txt")
        assert result.error is not None
        assert "could not determine window size" in result.error

    def test_undecodable_first_chunk_is_reported(
        self, backend: SessionsBashBackend
    ) -> None:
        self._stub_sequence(backend, ["3\n__SEP__\n5\n__SEP__\n%%bad%%\n"])
        result = backend.read("/mnt/data/f.txt")
        assert result.error is not None
        assert "unexpected read output" in result.error
