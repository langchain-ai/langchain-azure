"""Shared constructors for this package's tests."""

from __future__ import annotations

from pathlib import PurePosixPath
from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.containerapps.sandbox import FileInfo  # noqa: E402
from azure.core.exceptions import (  # noqa: E402
    HttpResponseError,
)


def _command(client: mock.Mock) -> str:
    return client.exec.call_args[0][0]


def _http_error(message: str) -> HttpResponseError:
    return HttpResponseError(message=message)


def stat_info(path: str, size: int | None = 5, is_directory: bool = False) -> FileInfo:
    """A realistic `stat_file` result.

    A bare `Mock(spec=FileInfo)` must not be used instead: its `is_directory`
    is a truthy child mock, which routes every read down the directory-error
    path.
    """
    return FileInfo(
        name=PurePosixPath(path).name,
        path=path,
        size=size,
        is_directory=is_directory,
        modified_at="2026-01-01T00:00:00Z",
        mode="drwxr-xr-x" if is_directory else "-rw-r--r--",
    )
