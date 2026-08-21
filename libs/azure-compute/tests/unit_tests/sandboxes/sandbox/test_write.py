"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.core.exceptions import (  # noqa: E402
    HttpResponseError,
    ResourceNotFoundError,
)

from langchain_azure_compute.sandboxes import ACASandbox  # noqa: E402

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

from tests.unit_tests.sandboxes.sandbox._helpers import (  # noqa: E402
    stat_info,
)


class TestWrite:
    def test_uses_sdk_not_shell(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        # Sending content through execute() would risk ARG_MAX on large files.
        client.stat_file.side_effect = ResourceNotFoundError()
        sandbox.write("/f.txt", "hello")
        client.write_file.assert_called_once()
        assert client.write_file.call_args[0][1] == "hello"
        client.exec.assert_not_called()

    def test_creates_parent_dirs(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        client.stat_file.side_effect = ResourceNotFoundError()
        sandbox.write("/deep/nested/f.txt", "hello")
        assert client.write_file.call_args.kwargs["create_dirs"] is True

    def test_existing_file_is_not_overwritten(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # Required by the shared suite even though BaseSandbox.write describes
        # itself as overwriting.
        client.stat_file.side_effect = lambda path: stat_info(path)
        result = sandbox.write("/f.txt", "hello")
        assert result.error == (
            "File '/f.txt' already exists. Use edit to modify it, or delete it first."
        )
        client.write_file.assert_not_called()

    def test_existing_file_error_steers_to_edit(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        result = sandbox.write("/f.txt", "hello")
        assert result.error is not None
        assert "already exists" in result.error
        assert "edit" in result.error

    def test_existing_directory_is_reported_as_one(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # "already exists, use edit" would steer the model into editing a
        # directory; the refusal must name what is actually in the way.
        client.stat_file.side_effect = lambda path: stat_info(path, is_directory=True)
        result = sandbox.write("/data", "hello")
        assert result.error == "Path '/data' is a directory"
        client.write_file.assert_not_called()

    def test_stat_failure_surfaces_as_error(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.stat_file.side_effect = HttpResponseError(message="boom")
        result = sandbox.write("/f.txt", "hello")
        assert result.error is not None
        client.write_file.assert_not_called()
