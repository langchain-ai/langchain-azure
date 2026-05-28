"""Unit tests for AzureAIProjectToolbox."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from langchain_azure_ai.tools._toolbox import (
    _DEFAULT_FEATURES,
    _FEATURES_HEADER,
    AzureAIProjectToolbox,
    _fetch_require_approval_tools,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_azure_ai._api.base.ExperimentalWarning"
)


class TestFetchRequireApprovalTools:
    async def test_filters_tools_with_require_approval(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        observed: dict[str, Any] = {}

        class FakeResponse:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return self._payload

        class FakeAsyncClient:
            def __init__(
                self,
                *,
                auth: Any,
                headers: dict[str, str],
                timeout: float,
            ) -> None:
                observed["auth"] = auth
                observed["headers"] = headers
                observed["timeout"] = timeout

            async def __aenter__(self) -> "FakeAsyncClient":
                return self

            async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
                return None

            async def post(self, endpoint: str, json: dict[str, Any]) -> FakeResponse:
                observed["endpoint"] = endpoint
                observed["payload"] = json
                return FakeResponse(
                    {
                        "result": {
                            "tools": [
                                {
                                    "name": "send_email",
                                    "_meta": {
                                        "tool_configuration": {
                                            "require_approval": "always"
                                        }
                                    },
                                },
                                {
                                    "name": "read_calendar",
                                    "_meta": {
                                        "tool_configuration": {
                                            "require_approval": "never"
                                        }
                                    },
                                },
                                {
                                    "name": "echo",
                                },
                            ]
                        }
                    }
                )

        fake_httpx = ModuleType("httpx")
        fake_httpx.AsyncClient = FakeAsyncClient  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

        result = await _fetch_require_approval_tools(
            endpoint="https://example.test/mcp",
            auth="AUTH",
            extra_headers={"X-Test": "1"},
        )

        assert result == {"send_email": "always", "read_calendar": "never"}
        assert observed["endpoint"] == "https://example.test/mcp"
        assert observed["payload"] == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {},
        }
        assert observed["auth"] == "AUTH"
        assert observed["headers"] == {"X-Test": "1"}
        assert observed["timeout"] == 30.0


class TestAzureAIProjectToolboxApproval:
    async def test_get_tools_requiring_approval_returns_always_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
        )

        monkeypatch.setattr(
            toolbox,
            "_build_auth_and_headers",
            lambda: ("AUTH", {"X-Test": "1"}),
        )

        async def fake_fetch(
            endpoint: str,
            auth: Any,
            extra_headers: dict[str, str],
        ) -> dict[str, str]:
            assert endpoint == toolbox.toolbox_endpoint
            assert auth == "AUTH"
            assert extra_headers == {"X-Test": "1"}
            return {
                "send_email": "always",
                "list_files": "never",
                "delete_item": "always",
            }

        monkeypatch.setattr(
            "langchain_azure_ai.tools._toolbox._fetch_require_approval_tools",
            fake_fetch,
        )

        names = await toolbox.get_tools_requiring_approval()

        assert names == ["send_email", "delete_item"]

    async def test_get_tools_requiring_approval_requires_project_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        toolbox = AzureAIProjectToolbox(project_endpoint="", toolbox_name="tb")

        with pytest.raises(ValueError, match="project_endpoint is required"):
            await toolbox.get_tools_requiring_approval()


class TestAzureAIProjectToolboxAuthHeaders:
    def test_build_auth_and_headers_with_static_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeAuth:
            pass

        fake_httpx = ModuleType("httpx")
        fake_httpx.Auth = FakeAuth  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="abc123",
            extra_headers={"X-Test": "1"},
        )

        auth, headers = toolbox._build_auth_and_headers()
        request = SimpleNamespace(headers={})

        flow = auth.auth_flow(request)
        next(flow)

        assert request.headers["Authorization"] == "Bearer abc123"
        assert headers["X-Test"] == "1"
        assert headers[_FEATURES_HEADER] == _DEFAULT_FEATURES
