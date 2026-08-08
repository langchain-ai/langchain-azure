"""`upload_files`/`download_files` against mocked `requests`.

These are the REST file-transfer methods: the HTTP layer itself is under
test here -- URL construction, multipart shape, auth headers, and per-file
error capture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
import requests

pytest.importorskip("deepagents")

from deepagents.backends.protocol import (  # noqa: E402
    FileDownloadResponse,
    FileUploadResponse,
)

from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)

if TYPE_CHECKING:
    pass

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


from tests.unit_tests.dynamic_sessions.backend._helpers import (  # noqa: E402
    _http_error,
    _make_backend,
    _response,
)


class TestUploadFiles:
    def test_posts_each_file_to_mnt_data(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.post.return_value = _response()
        results = backend.upload_files(
            [("/mnt/data/a.txt", b"a"), ("/mnt/data/b.txt", b"bb")]
        )

        assert [(r.path, r.error) for r in results] == [
            ("/mnt/data/a.txt", None),
            ("/mnt/data/b.txt", None),
        ]
        assert http.post.call_count == 2
        url = http.post.call_args.args[0]
        # Extras precede the standard parameters, which are appended last so a
        # colliding key can never replace the session identifier.
        assert "path=%2Fmnt%2Fdata&identifier=" in url
        assert "/files?" in url

    def test_the_name_under_the_root_is_sent(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.post.return_value = _response()
        backend.upload_files([("/mnt/data/a.txt", b"a")])
        ((field, (filename, buffer, _mime)),) = http.post.call_args.kwargs["files"]
        assert field == "file"
        assert filename == "a.txt"
        assert buffer.read() == b"a"

    @pytest.mark.parametrize(
        "path",
        [
            "/deep/nested/a.txt",
            "/mnt/data/sub/a.txt",
            "/mnt/database/a.txt",
            "/mnt/data/../../etc/passwd",
            "/mnt/data",
            "a.txt",
        ],
    )
    def test_unstorable_paths_are_rejected_without_a_request(
        self, backend: SessionsBashBackend, http: mock.MagicMock, path: str
    ) -> None:
        # Reporting these as uploaded would claim the caller's path while
        # writing a different one, and would collide same-named files.
        assert backend.upload_files([(path, b"x")]) == [
            FileUploadResponse(path=path, error="invalid_path")
        ]
        http.post.assert_not_called()

    def test_failure_is_captured_per_file(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        ok, bad = _response(), _response()
        bad.raise_for_status.side_effect = _http_error(403)
        http.post.side_effect = [ok, bad, ok]
        results = backend.upload_files(
            [("/mnt/data/a", b"a"), ("/mnt/data/b", b"b"), ("/mnt/data/c", b"c")]
        )
        assert [r.error for r in results] == [None, "permission_denied", None]

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, "permission_denied"),
            (403, "permission_denied"),
            # `file_not_found` is a download-side code: an upload 404 means the
            # session or pool route resolved nowhere, not that a file is
            # missing, and the code would steer the agent wrong.
            (404, "request failed with HTTP 404"),
            (500, "request failed with HTTP 500"),
        ],
    )
    def test_status_determines_the_error_code(
        self,
        backend: SessionsBashBackend,
        http: mock.MagicMock,
        status: int,
        expected: str,
    ) -> None:
        bad = _response()
        bad.raise_for_status.side_effect = _http_error(status)
        http.post.return_value = bad
        assert backend.upload_files([("/mnt/data/a", b"a")])[0].error == expected

    def test_non_http_failure_is_not_guessed_as_a_code(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        # A wrong code sends the agent down the wrong recovery path, so a
        # failure with no status must not borrow one.
        http.post.side_effect = requests.ConnectionError("connection reset")
        error = backend.upload_files([("/mnt/data/a", b"a")])[0].error
        assert error == "request failed: ConnectionError"

    def test_error_does_not_leak_the_request_url(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        # `requests` embeds the URL -- which carries the session identifier and
        # the pool endpoint -- in its messages, and this string reaches the
        # model.
        bad = _response()
        bad.raise_for_status.side_effect = _http_error(500)
        http.post.return_value = bad
        error = backend.upload_files([("/mnt/data/a", b"a")])[0].error
        assert error is not None
        assert "test-session" not in error
        assert "dynamicsessions.io" not in error

    def test_empty_batch_makes_no_requests(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        assert backend.upload_files([]) == []
        http.post.assert_not_called()

    def test_zero_timeout_reaches_requests_as_none(self, http: mock.MagicMock) -> None:
        # Same translation `execute` performs: Requests rejects `timeout=0`
        # with a bare ValueError -- not a RequestException -- which would
        # escape the per-file capture and abort the whole batch.
        http.post.return_value = _response()
        results = _make_backend(timeout=0).upload_files([("/mnt/data/a", b"a")])
        assert results[0].error is None
        assert http.post.call_args.kwargs["timeout"] is None


class TestDownloadFiles:
    def test_zero_timeout_reaches_requests_as_none(self, http: mock.MagicMock) -> None:
        http.get.return_value = _response(content=b"x")
        results = _make_backend(timeout=0).download_files(["/mnt/data/a"])
        assert results[0].error is None
        assert http.get.call_args.kwargs["timeout"] is None

    def test_returns_content_in_order(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.get.side_effect = [_response(content=b"one"), _response(content=b"two")]
        results = backend.download_files(["/mnt/data/a", "/mnt/data/b"])
        assert [(r.path, r.content, r.error) for r in results] == [
            ("/mnt/data/a", b"one", None),
            ("/mnt/data/b", b"two", None),
        ]

    def test_basename_is_percent_encoded_in_the_url(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        http.get.return_value = _response(content=b"x")
        backend.download_files(["/mnt/data/my file.txt"])
        assert "/files/my%20file.txt/content?" in http.get.call_args.args[0]

    def test_failure_is_captured_per_file(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        bad = _response()
        bad.raise_for_status.side_effect = _http_error(404)
        http.get.side_effect = [_response(content=b"one"), bad]
        results = backend.download_files(["/mnt/data/a", "/mnt/data/b"])
        assert [(r.content, r.error) for r in results] == [
            (b"one", None),
            (None, "file_not_found"),
        ]

    def test_unclassifiable_failure_reports_the_status(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        bad = _response()
        bad.raise_for_status.side_effect = _http_error(503)
        http.get.return_value = bad
        error = backend.download_files(["/mnt/data/a"])[0].error
        assert error == "request failed with HTTP 503"

    def test_empty_batch_makes_no_requests(
        self, backend: SessionsBashBackend, http: mock.MagicMock
    ) -> None:
        assert backend.download_files([]) == []
        http.get.assert_not_called()


class TestDownloadPathValidation:
    @pytest.mark.parametrize(
        "path",
        [
            "/deep/nested/a.txt",
            "/mnt/data/sub/a.txt",
            "/mnt/database/a.txt",
            "/mnt/data/../../etc/passwd",
            "/mnt/data",
            "a.txt",
        ],
    )
    def test_unstorable_paths_are_rejected_without_a_request(
        self, backend: SessionsBashBackend, http: mock.MagicMock, path: str
    ) -> None:
        assert backend.download_files([path]) == [
            FileDownloadResponse(path=path, error="invalid_path")
        ]
        http.get.assert_not_called()
