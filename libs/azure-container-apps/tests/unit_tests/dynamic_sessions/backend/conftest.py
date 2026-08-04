"""Unit tests for `SessionsBashBackend`'s HTTP layer against mocked `requests`.

Complements `test_sessions_backend_contract.py`, which stubs `execute()` to
check result *shapes*. Here `execute()` itself is under test, along with the
URL construction, auth headers, and the REST file-transfer methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import pytest
import requests

pytest.importorskip("deepagents")


from langchain_azure_container_apps.dynamic_sessions.backends import (  # noqa: E402
    SessionsBashBackend,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)

ENDPOINT = "https://westus2.dynamicsessions.io/subscriptions/00000000-0000-0000-0000-000000000000/resourceGroups/rg/sessionPools/pool"

from tests.unit_tests.dynamic_sessions.backend._helpers import (  # noqa: E402
    _make_backend,
)


@pytest.fixture
def backend() -> SessionsBashBackend:
    return _make_backend()


@pytest.fixture
def http() -> Iterator[mock.MagicMock]:
    with mock.patch(
        "langchain_azure_container_apps.dynamic_sessions.backends.sessions.requests"
    ) as requests_mod:
        # The module object is replaced wholesale, so the real exception class
        # has to be put back: the backend's `except requests.RequestException`
        # resolves through this mock and would otherwise raise TypeError.
        requests_mod.RequestException = requests.RequestException
        yield requests_mod
