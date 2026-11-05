"""Shared stubs for the dynamic-sessions tool tests."""

import re
from typing import Any, Optional
from unittest import mock

import requests

_DONE_RE = re.compile(r"^trap 'printf %s (__LCDONE[0-9a-f]+__)' EXIT; ")


def echo_done(command: str, output: str) -> str:
    """Append the completion marker the backend embedded in `command`.

    A stub that omits it is indistinguishable to the backend from the data
    plane losing the command's final write, which is exactly what the marker
    exists to detect -- so every stubbed `execute` has to reproduce it.
    """
    match = _DONE_RE.match(command)
    return output + match.group(1) if match else output


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
