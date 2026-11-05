"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.core.exceptions import (  # noqa: E402
    ResourceNotFoundError,
)

from langchain_azure_compute.sandboxes import ACASandbox  # noqa: E402

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

from tests.unit_tests.sandboxes.sandbox._helpers import (  # noqa: E402
    _http_error,
)


class TestSdkErrorPaths:
    def test_read_http_error_is_reported_not_raised(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.side_effect = _http_error("boom")
        result = sandbox.read("/f.txt")
        assert result.file_data is None
        assert result.error == "File '/f.txt': request failed: HttpResponseError"

    def test_read_error_does_not_leak_the_sdk_message(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # azure-core embeds the response body in its messages; this string
        # reaches the model.
        client.read_file.side_effect = _http_error("token=sk-secret in body")
        result = sandbox.read("/f.txt")
        assert result.error is not None
        assert "sk-secret" not in result.error

    def test_write_http_error_after_stat_succeeds_is_reported(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = ResourceNotFoundError()
        client.write_file.side_effect = _http_error("disk full")
        result = sandbox.write("/f.txt", "hello")
        assert result.path is None
        assert result.error is not None
        assert "Failed to write file '/f.txt'" in result.error

    def test_unrecognised_error_is_not_guessed_as_a_code(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # A wrong code sends the agent down the wrong recovery path, so an
        # unclassifiable failure must not borrow one.
        client.read_file.side_effect = _http_error("something unexpected")
        error = sandbox.download_files(["/x"])[0].error
        assert error == "request failed: HttpResponseError"

    def test_unrecognised_error_reports_the_status_when_there_is_one(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        failure = _http_error("something unexpected")
        failure.status_code = 503
        client.read_file.side_effect = failure
        assert sandbox.download_files(["/x"])[0].error == "request failed with HTTP 503"

    def test_unrecognised_error_does_not_leak_the_sdk_message(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.side_effect = _http_error("token=sk-secret in body")
        error = sandbox.download_files(["/x"])[0].error
        assert error is not None
        assert "sk-secret" not in error

    @pytest.mark.parametrize(
        "message", ["IsADirectory: /x", "Permission denied", "access DENIED"]
    )
    def test_error_mapping_is_case_insensitive(
        self, sandbox: ACASandbox, client: mock.Mock, message: str
    ) -> None:
        client.read_file.side_effect = _http_error(message)
        assert sandbox.download_files(["/x"])[0].error in {
            "is_directory",
            "permission_denied",
        }

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_status_codes_map_to_permission_denied(
        self, sandbox: ACASandbox, client: mock.Mock, status: int
    ) -> None:
        failure = _http_error("something unexpected")
        failure.status_code = status
        client.read_file.side_effect = failure
        assert sandbox.download_files(["/x"])[0].error == "permission_denied"

    def test_status_code_outranks_the_denied_substring(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # A body that merely *contains* "denied" (a path, a quoted server
        # detail) must not steer the agent into permission-recovery when the
        # status says otherwise.
        failure = _http_error("path /tmp/denied.txt caused a server error")
        failure.status_code = 500
        client.read_file.side_effect = failure
        assert sandbox.download_files(["/x"])[0].error == "request failed with HTTP 500"


class TestNotFoundIsOperationSensitive:
    """A 404 means "missing file" only where a file was expected to exist.

    `transfer_error` is shared by exec, write, upload and download. Mapping
    every `ResourceNotFoundError` to `file_not_found` sent the model into file
    recovery when the *sandbox* was unavailable, and reported an upload 404
    against a destination that is supposed not to exist yet.
    """

    def test_exec_404_is_not_file_not_found(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.exec.side_effect = ResourceNotFoundError()
        result = sandbox.execute("echo hi")
        assert result.exit_code != 0
        assert "file_not_found" not in result.output
        assert "404" in result.output

    async def test_aexec_404_on_the_async_client_is_not_file_not_found(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        """Drives the supplied async client, not the thread fallback into the
        sync method -- that is the branch `sandbox.py` actually changed."""
        async_client.exec.side_effect = ResourceNotFoundError()
        sandbox = ACASandbox(client, async_client=async_client)
        result = await sandbox.aexecute("echo hi")
        async_client.exec.assert_awaited_once()
        client.exec.assert_not_called()
        assert "file_not_found" not in result.output
        assert "404" in result.output

    def test_upload_404_is_not_file_not_found(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.write_file.side_effect = ResourceNotFoundError()
        responses = sandbox.upload_files([("/dest.txt", b"x")])
        assert responses[0].error is not None
        assert "file_not_found" not in responses[0].error
        assert "404" in responses[0].error

    async def test_aupload_404_on_the_async_client_is_not_file_not_found(
        self, client: mock.Mock, async_client: mock.AsyncMock
    ) -> None:
        async_client.write_file.side_effect = ResourceNotFoundError()
        sandbox = ACASandbox(client, async_client=async_client)
        responses = await sandbox.aupload_files([("/dest.txt", b"x")])
        async_client.write_file.assert_awaited_once()
        client.write_file.assert_not_called()
        assert responses[0].error is not None
        assert "file_not_found" not in responses[0].error
        assert "404" in responses[0].error

    def test_write_404_is_not_file_not_found(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # The pre-write stat must 404 first (destination absent, the normal
        # case) or `write` returns the already-exists refusal before
        # `write_file` ever runs and the 404 under test is never raised.
        client.stat_file.side_effect = ResourceNotFoundError()
        client.write_file.side_effect = ResourceNotFoundError()
        result = sandbox.write("/dest.txt", "x")
        client.write_file.assert_called_once()
        assert result.error is not None
        assert "file_not_found" not in result.error
        assert "404" in result.error

    def test_download_404_is_still_file_not_found(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        """The one operation where a 404 really is a missing file."""
        client.read_file.side_effect = ResourceNotFoundError()
        responses = sandbox.download_files(["/src.txt"])
        assert responses[0].error == "file_not_found"

    def test_read_404_is_still_file_not_found(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = ResourceNotFoundError()
        assert sandbox.read("/src.txt").error == "File '/src.txt': file_not_found"
