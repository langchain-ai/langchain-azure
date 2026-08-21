"""`grep` result construction in `SessionsBashBackend`, over a stubbed
`execute()`.

The generated programs themselves run under a real shell in
`test_sessions_shell_programs.py`; here the parsing branches are under test:
sentinel errors, output-loss retries, truncation, the find-route path join,
and the `max_count` cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

pytest.importorskip("deepagents")

from deepagents.backends.protocol import ExecuteResponse  # noqa: E402

from langchain_azure_compute.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)
from tests.unit_tests.dynamic_sessions._helpers import echo_done  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Sequence

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

MATCHES = "/data/a.py\x001:needle here\n/data/b.py\x003:needle too\n"


def _stub(
    backend: SessionsBashBackend,
    output: str,
    *,
    complete: Sequence[bool] = (True,),
    exit_code: int | None = 0,
    truncated: bool = False,
) -> mock.Mock:
    calls = iter(complete)

    def run(command: str, **kwargs: object) -> ExecuteResponse:
        body = echo_done(command, output) if next(calls) else output
        return ExecuteResponse(output=body, exit_code=exit_code, truncated=truncated)

    stub = mock.Mock(side_effect=run)
    backend.execute = stub  # type: ignore[method-assign]
    return stub


class TestGrepParsing:
    def test_matches_are_parsed(self, backend: SessionsBashBackend) -> None:
        _stub(backend, MATCHES)
        result = backend.grep("needle", path="/data")
        assert result.error is None
        assert result.matches == [
            {"path": "/data/a.py", "line": 1, "text": "needle here"},
            {"path": "/data/b.py", "line": 3, "text": "needle too"},
        ]
        assert result.truncated is False

    def test_no_output_is_an_empty_match_set(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, "")
        result = backend.grep("needle", path="/data")
        assert result.error is None
        assert result.matches == []

    def test_sentinel_error_is_reported(self, backend: SessionsBashBackend) -> None:
        _stub(backend, "__ERR__path_not_found\n")
        result = backend.grep("needle", path="/data")
        assert result.matches is None
        assert result.error == "Path '/data': path_not_found"

    def test_nonzero_exit_is_reported_bounded(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, "x" * 5000, exit_code=2)
        result = backend.grep("needle", path="/data")
        assert result.error is not None
        assert len(result.error) < 300

    def test_nonzero_exit_with_no_output(self, backend: SessionsBashBackend) -> None:
        _stub(backend, "", exit_code=1)
        result = backend.grep("needle", path="/data")
        assert result.error == "Path '/data': search failed"

    def test_find_route_paths_join_the_root(self, backend: SessionsBashBackend) -> None:
        # The find route emits paths relative to the root it `cd`'d into.
        _stub(backend, "./src/a.py\x002:needle\n")
        result = backend.grep("needle", path="/data", glob="src/*.py")
        assert result.matches == [
            {"path": "/data/src/a.py", "line": 2, "text": "needle"}
        ]

    def test_default_root_keeps_relative_paths(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, "./a.py\x001:needle\n")
        result = backend.grep("needle", glob="src/*.py")
        assert result.matches == [{"path": "./a.py", "line": 1, "text": "needle"}]

    def test_unparseable_line_alone_is_an_error(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, "garbage without separators\n")
        result = backend.grep("needle", path="/data")
        assert result.matches is None
        assert result.error == "Path '/data': garbage without separators"

    def test_unparseable_line_number_alone_is_an_error(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, "/data/a.py\x00NaN:text\n")
        result = backend.grep("needle", path="/data")
        assert result.matches is None
        assert result.error is not None

    def test_unparseable_line_among_matches_is_dropped(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(backend, MATCHES + "garbage\n")
        result = backend.grep("needle", path="/data")
        assert result.error is None
        assert len(result.matches or []) == 2

    def test_max_count_trims_and_flags(self, backend: SessionsBashBackend) -> None:
        _stub(backend, MATCHES)
        result = backend.grep("needle", path="/data", max_count=1)
        assert result.matches == [
            {"path": "/data/a.py", "line": 1, "text": "needle here"}
        ]
        assert result.truncated is True


class TestGrepOutputLoss:
    def test_retries_and_recovers(self, backend: SessionsBashBackend) -> None:
        stub = _stub(backend, MATCHES, complete=[False, True])
        result = backend.grep("needle", path="/data")
        assert stub.call_count == 2
        assert len(result.matches or []) == 2

    def test_unrecoverable_loss_is_reported(self, backend: SessionsBashBackend) -> None:
        stub = _stub(backend, MATCHES, complete=[False, False])
        result = backend.grep("needle", path="/data")
        assert stub.call_count == 2
        assert result.matches is None
        assert result.error is not None
        assert "retry it" in result.error


class TestGrepTruncation:
    def test_partial_last_record_is_dropped_and_flagged(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub(
            backend,
            MATCHES + "/data/c.py\x005:cut mid",
            complete=[False],
            truncated=True,
        )
        result = backend.grep("needle", path="/data")
        assert result.error is None
        assert len(result.matches or []) == 2
        assert result.truncated is True

    def test_whole_final_record_is_kept(self, backend: SessionsBashBackend) -> None:
        _stub(backend, MATCHES, complete=[False], truncated=True)
        result = backend.grep("needle", path="/data")
        assert len(result.matches or []) == 2
        assert result.truncated is True
