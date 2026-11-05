"""Shared constructors for this package's tests."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any
from unittest import mock

import pytest
import requests

pytest.importorskip("deepagents")


from langchain_azure_compute.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)

if TYPE_CHECKING:
    pass

ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/rg/sessionPools/pool"


def _make_backend(endpoint: str = ENDPOINT, **kwargs: Any) -> SessionsBashBackend:
    return SessionsBashBackend(
        pool_management_endpoint=endpoint,
        session_id="test-session",
        access_token_provider=lambda: "token_value",
        **kwargs,
    )


def _response(
    *,
    ok: bool = True,
    status_code: int = 200,
    json_body: Any = None,
    text: str = "",
    content: bytes = b"",
) -> mock.Mock:
    resp = mock.Mock(spec=requests.Response)
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    resp.json.return_value = json_body if json_body is not None else {}
    return resp


def _http_error(status_code: int) -> requests.HTTPError:
    """A `raise_for_status()` failure carrying a URL, as `requests` builds it."""
    response = _response(ok=False, status_code=status_code)
    return requests.HTTPError(
        f"{status_code} Server Error for url: {ENDPOINT}/files?identifier=test-session",
        response=response,
    )


def _exec_body(stdout: str = "", stderr: str = "", status: str | None = "0") -> dict:
    return {"status": status, "result": {"stdout": stdout, "stderr": stderr}}


def read_payload(
    content: str, total_lines: int, *, window_bytes: int | None = None
) -> str:
    """The read command's wire format: line count, window bytes, base64 chunk.

    `window_bytes` defaults to the content's own size; pass a larger value to
    simulate a window the first chunk does not cover.
    """
    raw = content.encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    size = len(raw) if window_bytes is None else window_bytes
    return f"{total_lines}\n__SEP__\n{size}\n__SEP__\n{b64}\n"
