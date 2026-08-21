"""Shared stubs for the dynamic-sessions tool tests."""

from typing import Any, Optional
from unittest import mock

import requests


def stub_response(
    patched: mock.MagicMock, json_body: Optional[Any] = None
) -> mock.Mock:
    """Give a patched `requests` function a spec'd `Response` to return.

    Spec'd so a typo or a renamed attribute fails here rather than silently
    resolving to a truthy child mock.
    """
    response = mock.Mock(spec=requests.Response)
    response.json.return_value = {} if json_body is None else json_body
    patched.return_value = response
    return response
