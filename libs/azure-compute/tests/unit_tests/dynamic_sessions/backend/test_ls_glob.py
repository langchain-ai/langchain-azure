"""`ls`/`glob` parsing branches in `SessionsBashBackend`, over a stubbed
`execute()`.
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

if TYPE_CHECKING:
    pass

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestGlobParsing:
    def test_unparseable_lines_are_skipped(self, backend: SessionsBashBackend) -> None:
        from deepagents.backends.protocol import ExecuteResponse

        backend.execute = mock.Mock(  # type: ignore[method-assign]
            side_effect=lambda command, **kwargs: ExecuteResponse(
                output=echo_done(command, "F:a.py\ngarbage-without-colon\n\nD:pkg\n"),
                exit_code=0,
                truncated=False,
            )
        )
        assert backend.glob("**/*.py", path="/mnt/data").matches == [
            {"path": "a.py", "is_dir": False},
            {"path": "pkg", "is_dir": True},
        ]


def _stub_execute(
    backend: SessionsBashBackend, output: str, *, exit_code: int | None = 0
) -> None:
    from deepagents.backends.protocol import ExecuteResponse

    backend.execute = mock.Mock(  # type: ignore[method-assign]
        side_effect=lambda command, **kwargs: ExecuteResponse(
            output=echo_done(command, output),
            exit_code=exit_code,
            truncated=False,
        )
    )


class TestGlobExitCodeEdgeCases:
    def test_none_exit_code_is_not_path_not_found(
        self, backend: SessionsBashBackend
    ) -> None:
        """`None` means the service returned no status -- fabricating
        `path_not_found` from it would send the agent down a wrong path."""
        _stub_execute(backend, "F:a.py\n", exit_code=None)
        result = backend.glob("*.py", path="/mnt/data")
        assert result.matches is None
        assert result.error == "Path '/mnt/data': command returned no exit status"


class TestLsEdgeCases:
    def test_empty_output_is_an_empty_directory(
        self, backend: SessionsBashBackend
    ) -> None:
        _stub_execute(backend, "")
        result = backend.ls("/mnt/data")
        assert result.error is None
        assert result.entries == []
