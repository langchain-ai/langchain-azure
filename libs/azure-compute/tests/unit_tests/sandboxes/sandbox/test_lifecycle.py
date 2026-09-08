"""Unit tests for `ACASandbox` against a mocked ACA data-plane client."""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("azure.containerapps.sandbox")


from langchain_azure_compute.sandboxes import ACASandbox  # noqa: E402

# Every test constructs a backend, so silence the beta notice module-wide.
pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_core._api.beta_decorator.LangChainBetaWarning"
)


class TestIdentity:
    def test_id_is_sandbox_id(self, sandbox: ACASandbox, client: mock.Mock) -> None:
        assert sandbox.id == "sbx-123"

    def test_capture_offload_defaults_off(self, sandbox: ACASandbox) -> None:
        # Requires coreutils in the disk image, which BYO/alpine images lack.
        assert sandbox.enable_capture_offload is False

    def test_capture_offload_opt_in(self, client: mock.Mock) -> None:
        assert ACASandbox(client, enable_capture_offload=True).enable_capture_offload
