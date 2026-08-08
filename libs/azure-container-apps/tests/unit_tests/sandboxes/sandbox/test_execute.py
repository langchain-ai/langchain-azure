"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

import shlex
from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.containerapps.sandbox import ExecResult  # noqa: E402
from azure.core.exceptions import (  # noqa: E402
    ServiceRequestError,
    ServiceResponseError,
)
from deepagents.backends.protocol import (  # noqa: E402
    ExecuteResponse,
)

from langchain_azure_container_apps.sandboxes import ACASandbox  # noqa: E402

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

from tests.unit_tests.sandboxes.sandbox._helpers import (  # noqa: E402
    _command,
)


class TestExecute:
    def test_merges_stderr_into_output(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.exec.return_value = ExecResult(exit_code=1, stdout="out", stderr="err")
        result = sandbox.execute("echo hi")
        assert isinstance(result, ExecuteResponse)
        assert result.output == "out\nerr"
        assert result.exit_code == 1
        assert result.truncated is False

    def test_stderr_only(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        client.exec.return_value = ExecResult(exit_code=2, stdout="", stderr="boom")
        assert sandbox.execute("x").output == "boom"

    def test_wraps_command_with_timeout(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        sandbox.execute("echo hi", timeout=5)
        assert "timeout 5 sh -c" in _command(client)

    def test_uses_default_timeout_when_unset(self, client: mock.Mock) -> None:
        ACASandbox(client, default_timeout=99).execute("echo hi")
        assert "timeout 99 sh -c" in _command(client)

    def test_falls_back_when_timeout_binary_missing(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # Minimal images ship no coreutils `timeout`; the command must still run.
        sandbox.execute("echo hi", timeout=5)
        command = _command(client)
        assert "command -v timeout" in command
        assert "else sh -c" in command

    @pytest.mark.parametrize("timeout", [0, -1])
    def test_non_positive_timeout_is_not_wrapped(
        self, sandbox: ACASandbox, client: mock.Mock, timeout: int
    ) -> None:
        sandbox.execute("echo hi", timeout=timeout)
        assert _command(client) == "echo hi"

    def test_command_cannot_escape_the_wrapper(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # A command containing a quote must not be able to close `sh -c`'s
        # argument and disable the timeout it is wrapped in. Both the timeout
        # branch and the fallback branch must carry it quoted.
        payload = "echo hi'; touch /tmp/escaped; echo '"
        sandbox.execute(payload, timeout=5)
        assert _command(client).count(shlex.quote(payload)) == 2

    def test_timeout_exit_code_is_explained(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.exec.return_value = ExecResult(exit_code=124, stdout="", stderr="")
        result = sandbox.execute("sleep 60", timeout=1)
        assert result.exit_code == 124
        assert "timed out" in result.output.lower()

    def test_exit_124_without_the_wrapper_is_not_called_a_timeout(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # timeout=0 runs the command unwrapped, so an exit 124 is the
        # command's own status and must not be narrated as a timeout.
        client.exec.return_value = ExecResult(exit_code=124, stdout="own", stderr="")
        result = sandbox.execute("exit 124", timeout=0)
        assert result.exit_code == 124
        assert result.output == "own"


class TestExecuteTransportFailure:
    """A request that dies mid-flight must come back as a result, not raise."""

    @pytest.mark.parametrize(
        "exc",
        [
            ServiceRequestError("connection reset"),
            ServiceResponseError("read timed out"),
        ],
    )
    def test_returns_error_response_instead_of_raising(
        self, sandbox: ACASandbox, client: mock.Mock, exc: Exception
    ) -> None:
        client.exec.side_effect = exc
        result = sandbox.execute("sleep 600")
        assert isinstance(result, ExecuteResponse)
        assert result.exit_code is None
        assert result.truncated is False
        assert "may still be running" in result.output
        assert "read timeout" in result.output

    def test_error_response_does_not_leak_the_sdk_message(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # azure-core interpolates the request URL into its messages; the
        # response text is handed to the model.
        client.exec.side_effect = ServiceResponseError(
            "timeout on https://sbx.westus2.example/subscriptions/sub-123/exec"
        )
        result = sandbox.execute("sleep 600")
        assert "sbx.westus2.example" not in result.output
        assert "sub-123" not in result.output
