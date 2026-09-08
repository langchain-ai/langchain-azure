"""Construction, identity, and `execute()`'s HTTP layer against mocked
`requests`: URL construction, auth headers, and request/response handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("deepagents")


from langchain_azure_compute.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)

if TYPE_CHECKING:
    pass

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/rg/sessionPools/pool"

from tests.unit_tests.dynamic_sessions.backend._helpers import (  # noqa: E402
    _make_backend,
)


class TestBuildUrl:
    def test_empty_endpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="pool_management_endpoint is not set"):
            _make_backend(endpoint="")._build_url("executions")

    def test_missing_trailing_slash_is_added(
        self, backend: SessionsBashBackend
    ) -> None:
        assert backend._build_url("executions").startswith(f"{ENDPOINT}/executions?")

    def test_existing_trailing_slash_not_doubled(self) -> None:
        url = _make_backend(endpoint=f"{ENDPOINT}/")._build_url("executions")
        assert f"{ENDPOINT}/executions?" in url
        assert "//executions" not in url

    def test_query_params(self, backend: SessionsBashBackend) -> None:
        query = parse_qs(urlparse(backend._build_url("executions")).query)
        assert query["identifier"] == ["test-session"]
        assert query["api-version"] == ["2025-02-02-preview"]

    def test_endpoint_with_a_query_string_is_rejected(self) -> None:
        # Previously accepted, producing `...?foo=bar/executions&identifier=`
        # -- the resource path landed inside the query string.
        with pytest.raises(ValueError, match="must not contain a query string"):
            _make_backend(endpoint=f"{ENDPOINT}?foo=bar")._build_url("executions")

    def test_session_id_is_percent_encoded(self) -> None:
        backend = SessionsBashBackend(
            pool_management_endpoint=ENDPOINT,
            session_id="a b/c",
            access_token_provider=lambda: "t",
        )
        assert "identifier=a%20b%2Fc" in backend._build_url("executions")


class TestIdentity:
    def test_id_is_the_session_id(self, backend: SessionsBashBackend) -> None:
        assert backend.id == "test-session"

    def test_session_id_defaults_to_a_unique_uuid(self) -> None:
        a = SessionsBashBackend(ENDPOINT, access_token_provider=lambda: "t")
        b = SessionsBashBackend(ENDPOINT, access_token_provider=lambda: "t")
        assert a.id != b.id

    def test_auth_headers(self, backend: SessionsBashBackend) -> None:
        headers = backend._auth_headers()
        assert headers["Authorization"] == "Bearer token_value"
        assert headers["User-Agent"].startswith("langchain-azure-compute/")

    def test_default_token_provider_is_used_when_none_given(self) -> None:
        with mock.patch(
            "langchain_azure_compute.dynamic_sessions.backends.sessions"
            ".access_token_provider_factory",
            return_value=lambda: "factory_token",
        ) as factory:
            backend = SessionsBashBackend(ENDPOINT)
        factory.assert_called_once()
        assert backend._auth_headers()["Authorization"] == "Bearer factory_token"
