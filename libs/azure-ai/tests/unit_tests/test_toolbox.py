"""Unit tests for AzureAIProjectToolbox."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from langchain_azure_ai.tools._toolbox import (
    _DEFAULT_FEATURES,
    _FEATURES_HEADER,
    AzureAIProjectToolbox,
    _extract_skill_content,
    _extract_skill_name,
    _fetch_require_approval_tools,
    _fetch_resource_contents,
    _fetch_resources_list,
    _make_unique_skill_file_path,
    _normalize_skills_base_path,
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


class TestToolboxSkillHelpers:
    def test_normalize_skills_base_path(self) -> None:
        assert _normalize_skills_base_path("/skills") == "/skills/"
        assert _normalize_skills_base_path("skills//custom") == "/skills/custom/"

    def test_normalize_skills_base_path_rejects_invalid_segments(self) -> None:
        with pytest.raises(ValueError, match="cannot contain"):
            _normalize_skills_base_path("/skills/../bad")
        with pytest.raises(ValueError, match="non-empty"):
            _normalize_skills_base_path("   ")

    def test_extract_skill_name(self) -> None:
        assert _extract_skill_name("skill://langgraph-docs") == "langgraph-docs"
        assert (
            _extract_skill_name("skill://team/langgraph-docs") == "team/langgraph-docs"
        )

    def test_extract_skill_name_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="invalid path traversal"):
            _extract_skill_name("skill://../escape")
        with pytest.raises(ValueError, match="Unsupported"):
            _extract_skill_name("resource://x")
        with pytest.raises(ValueError, match="invalid characters"):
            _extract_skill_name("skill://encoded%20name")
        assert _extract_skill_name(r"skill://nested\name") == "nested/name"

    def test_extract_skill_content_prefers_text_then_blob(self) -> None:
        data_with_text = {"contents": [{"uri": "skill://a", "text": "markdown"}]}
        assert _extract_skill_content(data_with_text, "skill://a") == "markdown"

        data_with_blob = {"contents": [{"uri": "skill://a", "blob": "bWFya2Rvd24="}]}
        assert _extract_skill_content(data_with_blob, "skill://a") == "markdown"
        with pytest.raises(ValueError, match="did not return any contents"):
            _extract_skill_content({"contents": []}, "skill://a")
        with pytest.raises(ValueError, match="without text or blob"):
            _extract_skill_content({"contents": [{"uri": "skill://a"}]}, "skill://a")

    def test_make_unique_skill_file_path(self) -> None:
        used_paths = {"/skills/sample/SKILL.md"}
        assert (
            _make_unique_skill_file_path("/skills/", "sample", used_paths)
            == "/skills/sample__2/SKILL.md"
        )
        assert (
            _make_unique_skill_file_path("/skills/", "new-skill", used_paths)
            == "/skills/new-skill/SKILL.md"
        )
        used_paths_multi = {
            "/skills/sample/SKILL.md",
            "/skills/sample__2/SKILL.md",
            "/skills/sample__3/SKILL.md",
        }
        assert (
            _make_unique_skill_file_path("/skills/", "sample", used_paths_multi)
            == "/skills/sample__4/SKILL.md"
        )


class TestToolboxResourceRPC:
    async def test_fetch_resources_list_uses_expected_rpc_payload(
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
                            "resources": [
                                {"uri": "skill://langgraph-docs"},
                                {"uri": "resource://other"},
                            ]
                        }
                    }
                )

        fake_httpx = ModuleType("httpx")
        fake_httpx.AsyncClient = FakeAsyncClient  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

        resources = await _fetch_resources_list(
            endpoint="https://example.test/mcp",
            auth="AUTH",
            extra_headers={"X-Test": "1"},
        )

        assert resources == [
            {"uri": "skill://langgraph-docs"},
            {"uri": "resource://other"},
        ]
        assert observed["endpoint"] == "https://example.test/mcp"
        assert observed["payload"] == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/list",
            "params": {},
        }
        assert observed["auth"] == "AUTH"
        assert observed["headers"] == {"X-Test": "1"}
        assert observed["timeout"] == 30.0

    async def test_fetch_resource_contents_uses_expected_rpc_payload(
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
                            "contents": [
                                {"uri": "skill://langgraph-docs", "text": "# Skill"}
                            ]
                        }
                    }
                )

        fake_httpx = ModuleType("httpx")
        fake_httpx.AsyncClient = FakeAsyncClient  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

        result = await _fetch_resource_contents(
            endpoint="https://example.test/mcp",
            auth="AUTH",
            extra_headers={"X-Test": "1"},
            uri="skill://langgraph-docs",
        )

        assert result == {
            "contents": [{"uri": "skill://langgraph-docs", "text": "# Skill"}]
        }
        assert observed["endpoint"] == "https://example.test/mcp"
        assert observed["payload"] == {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "skill://langgraph-docs"},
        }
        assert observed["auth"] == "AUTH"
        assert observed["headers"] == {"X-Test": "1"}
        assert observed["timeout"] == 30.0


class TestAzureAIProjectToolboxSkills:
    async def test_get_skills_filters_skill_resources(
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

        async def fake_list(
            endpoint: str,
            auth: Any,
            extra_headers: dict[str, str],
        ) -> list[dict[str, Any]]:
            assert endpoint == toolbox.toolbox_endpoint
            assert auth == "AUTH"
            assert extra_headers == {"X-Test": "1"}
            return [
                {"uri": "skill://langgraph-docs", "name": "LangGraph docs"},
                {"uri": "resource://other", "name": "Other"},
                {"name": "no-uri"},
            ]

        monkeypatch.setattr(
            "langchain_azure_ai.tools._toolbox._fetch_resources_list",
            fake_list,
        )

        skills = await toolbox.get_skills()
        assert skills == [{"uri": "skill://langgraph-docs", "name": "LangGraph docs"}]

    async def test_get_skills_file_data_builds_virtual_files_and_skips_bad_inputs(
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

        async def fake_list(
            endpoint: str,
            auth: Any,
            extra_headers: dict[str, str],
        ) -> list[dict[str, Any]]:
            assert endpoint == toolbox.toolbox_endpoint
            assert auth == "AUTH"
            assert extra_headers == {"X-Test": "1"}
            return [
                {"uri": "skill://good-skill"},
                {"uri": "skill://good-skill"},
                {"uri": "skill://../bad"},
                {"uri": "skill://missing-content"},
            ]

        async def fake_read(
            endpoint: str,
            auth: Any,
            extra_headers: dict[str, str],
            uri: str,
        ) -> dict[str, Any]:
            assert endpoint == toolbox.toolbox_endpoint
            assert auth == "AUTH"
            assert extra_headers == {"X-Test": "1"}
            if uri == "skill://missing-content":
                return {"contents": [{}]}
            return {"contents": [{"uri": uri, "text": f"# {uri}"}]}

        monkeypatch.setattr(
            "langchain_azure_ai.tools._toolbox._fetch_resources_list",
            fake_list,
        )
        monkeypatch.setattr(
            "langchain_azure_ai.tools._toolbox._fetch_resource_contents",
            fake_read,
        )

        file_data = await toolbox.get_skills_file_data(
            path="/skills",
            file_factory=lambda content: {"content": content},
        )

        assert file_data == {
            "/skills/good-skill/SKILL.md": {"content": "# skill://good-skill"},
            "/skills/good-skill__2/SKILL.md": {"content": "# skill://good-skill"},
        }

    async def test_get_skills_file_data_empty_skills(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
        )

        async def fake_list(
            endpoint: str,
            auth: Any,
            extra_headers: dict[str, str],
        ) -> list[dict[str, Any]]:
            assert endpoint == toolbox.toolbox_endpoint
            assert auth == "AUTH"
            assert extra_headers == {"X-Test": "1"}
            return []

        monkeypatch.setattr(
            "langchain_azure_ai.tools._toolbox._fetch_resources_list",
            fake_list,
        )
        monkeypatch.setattr(
            toolbox,
            "_build_auth_and_headers",
            lambda: ("AUTH", {"X-Test": "1"}),
        )
        assert await toolbox.get_skills_file_data() == {}

    async def test_get_skills_file_data_uses_get_skills(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
        )
        observed: dict[str, Any] = {"called": False}

        async def fake_get_skills() -> list[dict[str, Any]]:
            observed["called"] = True
            return [{"uri": "skill://delegated-skill"}]

        async def fake_read(
            endpoint: str,
            auth: Any,
            extra_headers: dict[str, str],
            uri: str,
        ) -> dict[str, Any]:
            assert endpoint == toolbox.toolbox_endpoint
            assert auth == "AUTH"
            assert extra_headers == {"X-Test": "1"}
            assert uri == "skill://delegated-skill"
            return {"contents": [{"uri": uri, "text": "# delegated"}]}

        monkeypatch.setattr(toolbox, "get_skills", fake_get_skills)
        monkeypatch.setattr(
            toolbox,
            "_build_auth_and_headers",
            lambda: ("AUTH", {"X-Test": "1"}),
        )
        monkeypatch.setattr(
            "langchain_azure_ai.tools._toolbox._fetch_resource_contents",
            fake_read,
        )

        assert await toolbox.get_skills_file_data() == {
            "/skills/delegated-skill/SKILL.md": "# delegated"
        }
        assert observed["called"] is True
