"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")

from azure.containerapps.sandbox import ExecResult, SandboxClient  # noqa: E402
from azure.containerapps.sandbox.aio import (  # noqa: E402
    SandboxClient as AsyncSandboxClient,
)

from langchain_azure_container_apps.sandboxes import ACASandbox  # noqa: E402
from tests.unit_tests.sandboxes.sandbox._helpers import stat_info  # noqa: E402

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


@pytest.fixture
def client() -> mock.Mock:
    # spec'd so a method the SDK renames fails here rather than passing
    # against a mock that silently accepts anything.
    c = mock.Mock(spec=SandboxClient)
    c.sandbox_id = "sbx-123"
    c.exec.return_value = ExecResult(exit_code=0, stdout="", stderr="")
    # Reads are stat-first: default to a small text file so tests exercising
    # the download path keep their original meaning. Tests targeting the stat
    # routing override this per-path.
    c.stat_file.side_effect = lambda path: stat_info(path)
    return c


@pytest.fixture
def async_client() -> mock.AsyncMock:
    c = mock.AsyncMock(spec=AsyncSandboxClient)
    c.exec.return_value = ExecResult(exit_code=0, stdout="", stderr="")
    c.stat_file.side_effect = lambda path: stat_info(path)
    return c


@pytest.fixture
def sandbox(client: mock.Mock) -> ACASandbox:
    return ACASandbox(client)
