"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.core.exceptions import (  # noqa: E402
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from deepagents.backends.protocol import (  # noqa: E402
    FileDownloadResponse,
    FileUploadResponse,
)

from langchain_azure_container_apps.sandboxes import ACASandbox  # noqa: E402

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

from tests.unit_tests.sandboxes.sandbox._helpers import (  # noqa: E402
    _http_error,
)


class TestUpload:
    def test_success_preserves_order(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        files = [("/a.txt", b"a"), ("/b.txt", b"b")]
        assert sandbox.upload_files(files) == [
            FileUploadResponse(path="/a.txt", error=None),
            FileUploadResponse(path="/b.txt", error=None),
        ]

    def test_creates_parent_dirs(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        sandbox.upload_files([("/deep/nested/f.txt", b"x")])
        assert client.write_file.call_args.kwargs["create_dirs"] is True

    def test_relative_path_rejected_without_calling_sdk(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        assert sandbox.upload_files([("rel.txt", b"x")]) == [
            FileUploadResponse(path="rel.txt", error="invalid_path")
        ]
        client.write_file.assert_not_called()

    def test_partial_failure_does_not_abort_batch(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.write_file.side_effect = [None, _http_error("permission denied"), None]
        results = sandbox.upload_files([("/a", b"a"), ("/b", b"b"), ("/c", b"c")])
        assert [r.error for r in results] == [None, "permission_denied", None]

    def test_transport_failure_does_not_abort_batch(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        # ServiceRequestError is an AzureError but not an HttpResponseError;
        # catching only the latter let one bad file kill the whole batch.
        client.write_file.side_effect = [
            None,
            ServiceRequestError("connection reset"),
            None,
        ]
        results = sandbox.upload_files([("/a", b"a"), ("/b", b"b"), ("/c", b"c")])
        assert [r.error for r in results] == [
            None,
            "request failed: ServiceRequestError",
            None,
        ]


class TestDownload:
    def test_success_preserves_order(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.side_effect = [b"one", b"two"]
        assert sandbox.download_files(["/a", "/b"]) == [
            FileDownloadResponse(path="/a", content=b"one", error=None),
            FileDownloadResponse(path="/b", content=b"two", error=None),
        ]

    def test_relative_path_rejected(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        assert sandbox.download_files(["rel.txt"]) == [
            FileDownloadResponse(path="rel.txt", content=None, error="invalid_path")
        ]
        client.read_file.assert_not_called()

    def test_transport_failure_does_not_abort_batch(
        self, sandbox: ACASandbox, client: mock.Mock
    ) -> None:
        client.read_file.side_effect = [
            b"one",
            ServiceRequestError("connection reset"),
            b"three",
        ]
        results = sandbox.download_files(["/a", "/b", "/c"])
        assert [(r.content, r.error) for r in results] == [
            (b"one", None),
            (None, "request failed: ServiceRequestError"),
            (b"three", None),
        ]

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (ResourceNotFoundError(), "file_not_found"),
            (ClientAuthenticationError(), "permission_denied"),
            (HttpResponseError(message="Is a directory"), "is_directory"),
            (HttpResponseError(message="permission denied"), "permission_denied"),
        ],
    )
    def test_error_mapping(
        self,
        sandbox: ACASandbox,
        client: mock.Mock,
        exc: Exception,
        expected: str,
    ) -> None:
        client.read_file.side_effect = exc
        assert sandbox.download_files(["/x"])[0].error == expected
