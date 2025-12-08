"""Shared pytest configuration for integration tests with HTTP recordings."""

from __future__ import annotations

from typing import Any, Dict, MutableMapping, cast

import pytest
from vcr import VCR  # type: ignore[import-not-found, import-untyped]

FILTER_HEADERS = [
    ("authorization", "REDACTED"),
    ("api-key", "REDACTED"),
    ("x-api-key", "REDACTED"),
    ("openai-api-key", "REDACTED"),
    ("openai-organization", "REDACTED"),
    ("user-agent", "REDACTED"),
    ("x-openai-client-user-agent", "REDACTED"),
]


def _sanitize_request(request: Any) -> Any:
    headers = cast(MutableMapping[str, Any], getattr(request, "headers", {}))
    for header, replacement in FILTER_HEADERS:
        if header in headers:
            headers[header] = replacement
    return request


def _sanitize_response(response: Dict[str, Any]) -> Dict[str, Any]:
    headers = cast(MutableMapping[str, Any], response.get("headers", {}))
    for header in headers:
        headers[header] = ["REDACTED"]
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Base configuration for pytest-recording/VCR."""
    return {
        "cassette_library_dir": "tests/integration_tests/cassettes",
        "filter_headers": FILTER_HEADERS,
        "match_on": ["method", "uri", "body"],
        "decode_compressed_response": True,
        "before_record_request": _sanitize_request,
        "before_record_response": _sanitize_response,
        "path_transformer": VCR.ensure_suffix(".yaml"),
    }
