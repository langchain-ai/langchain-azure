"""Unit tests for `SessionsBashBackend`'s HTTP layer against mocked `requests`.

Complements `test_sessions_backend_contract.py`, which stubs `execute()` to
check result *shapes*. Here `execute()` itself is under test, along with the
URL construction, auth headers, and the REST file-transfer methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest

pytest.importorskip("deepagents")


from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)

if TYPE_CHECKING:
    pass

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/rg/sessionPools/pool"

from tests.unit_tests.dynamic_sessions.backend._helpers import (  # noqa: E402
    _exec_body,
    _make_backend,
    _response,
)


class TestExecute:
    def test_posts_shell_command(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.post.return_value = _response(json_body=_exec_body(stdout="hi\n"))
        result = backend.execute("echo hi")

        assert result.output == "hi\n"
        assert result.exit_code == 0
        assert result.truncated is False

        url, kwargs = http.post.call_args.args[0], http.post.call_args.kwargs
        assert url.startswith(f"{ENDPOINT}/executions?")
        assert kwargs["json"] == {"shellCommand": "echo hi"}
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == "Bearer token_value"

    def test_stderr_is_appended_to_stdout(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.post.return_value = _response(
            json_body=_exec_body(stdout="out", stderr="err", status="1")
        )
        result = backend.execute("false")
        assert result.output == "outerr"
        assert result.exit_code == 1

    def test_default_timeout_is_used_when_unset(self, http: mock.MagicMock) -> None:
        http.post.return_value = _response(json_body=_exec_body())
        _make_backend(timeout=42).execute("echo hi")
        assert http.post.call_args.kwargs["timeout"] == 42

    def test_explicit_timeout_overrides_default(self, http: mock.MagicMock) -> None:
        http.post.return_value = _response(json_body=_exec_body())
        _make_backend(timeout=42).execute("echo hi", timeout=7)
        assert http.post.call_args.kwargs["timeout"] == 7

    def test_zero_timeout_is_honoured_not_treated_as_unset(
        self, http: mock.MagicMock
    ) -> None:
        # `if timeout is not None`, not a truthiness check: 0 must not fall
        # back to the instance default. It also must not reach Requests as 0 --
        # the protocol spells "no timeout" as 0, Requests spells it as None and
        # rejects 0 outright ("cannot be set to a value less than or equal
        # to 0"), so asserting the mock merely received 0 would certify a call
        # that raises against the real library.
        http.post.return_value = _response(json_body=_exec_body())
        _make_backend(timeout=42).execute("echo hi", timeout=0)
        assert http.post.call_args.kwargs["timeout"] is None

    def test_output_over_cap_is_truncated(self, http: mock.MagicMock) -> None:
        http.post.return_value = _response(json_body=_exec_body(stdout="x" * 50))
        result = _make_backend(max_output_bytes=10).execute("yes")
        assert result.truncated is True
        assert result.output == "x" * 10

    def test_output_at_cap_is_not_truncated(self, http: mock.MagicMock) -> None:
        http.post.return_value = _response(json_body=_exec_body(stdout="x" * 10))
        assert _make_backend(max_output_bytes=10).execute("yes").truncated is False

    def test_cap_counts_bytes_not_characters(self, http: mock.MagicMock) -> None:
        # Four 3-byte characters exceed a 10-byte cap despite being 4 chars.
        http.post.return_value = _response(json_body=_exec_body(stdout="→→→→"))
        result = _make_backend(max_output_bytes=10).execute("x")
        assert result.truncated is True
        # The slice lands one byte into the fourth arrow; `errors="ignore"`
        # must drop the dangling partial byte, not replace or keep it.
        assert result.output == "→→→"

    def test_missing_status_yields_no_exit_code(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.post.return_value = _response(json_body=_exec_body(status=None))
        assert backend.execute("echo hi").exit_code is None

    def test_long_command_is_truncated_in_the_log_preview(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        # The preview slice runs unconditionally, so a >120 char command must
        # not blow up the debug path.
        http.post.return_value = _response(json_body=_exec_body())
        assert backend.execute("echo " + "a" * 500).exit_code == 0

    def test_http_error_raises_with_json_detail(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.post.return_value = _response(
            ok=False, status_code=400, json_body={"error": "bad request"}
        )
        with pytest.raises(RuntimeError, match="bad request") as excinfo:
            backend.execute("echo hi")
        assert "400" in str(excinfo.value)

    def test_http_error_falls_back_to_text_when_body_is_not_json(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        resp = _response(ok=False, status_code=503, text="upstream unavailable")
        resp.json.side_effect = ValueError("no json")
        http.post.return_value = resp
        with pytest.raises(RuntimeError, match="upstream unavailable"):
            backend.execute("echo hi")


class TestNonNumericStatus:
    def test_lifecycle_state_status_yields_no_exit_code(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        """A Python-typed pool reports "Succeeded" where a Shell pool reports
        the numeric exit code; that used to crash `int()` mid-parse."""
        http.post.return_value = _response(
            json_body=_exec_body(stdout="out", status="Succeeded")
        )
        result = backend.execute("echo out")
        assert result.exit_code is None
        assert result.output == "out"


class TestServiceOutputCap:
    """The data plane truncates each stream at exactly 4,096 bytes and
    signals nothing; an at-cap stream is flagged as truncated here."""

    def test_stdout_at_the_cap_is_flagged(self, http: mock.MagicMock) -> None:
        http.post.return_value = _response(json_body=_exec_body(stdout="z" * 4096))
        result = _make_backend().execute("yes")
        assert result.truncated is True

    def test_stdout_below_the_margin_is_not_flagged(self, http: mock.MagicMock) -> None:
        # Three bytes of margin: a multi-byte character cut at the boundary is
        # replaced before it reaches this client.
        http.post.return_value = _response(json_body=_exec_body(stdout="z" * 4092))
        result = _make_backend().execute("yes")
        assert result.truncated is False

    def test_stderr_at_the_cap_is_flagged(self, http: mock.MagicMock) -> None:
        http.post.return_value = _response(json_body=_exec_body(stderr="e" * 4096))
        result = _make_backend().execute("noisy")
        assert result.truncated is True

    def test_capped_listing_reports_the_designed_ls_error(
        self, http: mock.MagicMock
    ) -> None:
        """The flag is what routes a capped `ls` into its designed truncation
        message instead of a retry-forever 'output lost' error -- the cap also
        eats the completion marker, so without the flag the loss machinery
        misdiagnoses it."""
        listing = ("name.txt\n" * 456)[:4096]
        http.post.return_value = _response(json_body=_exec_body(stdout=listing))
        result = _make_backend().ls("/mnt/data")
        assert result.entries is None
        assert result.error is not None
        assert "list a subdirectory" in result.error
