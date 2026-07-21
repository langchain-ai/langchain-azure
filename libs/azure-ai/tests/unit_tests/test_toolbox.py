"""Unit tests for AzureAIProjectToolbox."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from langchain_azure_ai.tools._toolbox import (
    _DEFAULT_FEATURES,
    _FEATURES_HEADER,
    AzureAIProjectToolbox,
    _fetch_require_approval_tools,
    _normalize_scheme,
    _resource_name_from_uri,
    _run_async,
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


class TestAzureAIProjectToolboxTools:
    async def test_tool_execution_error_returns_tool_result(self) -> None:
        from langchain_core.tools import tool

        err_msg = "GitHub MCP returned non-200 status code."

        @tool
        async def github_search(query: str) -> str:
            """Search GitHub repositories."""
            raise RuntimeError(err_msg)

        class FakeClient:
            async def get_tools(self) -> list[Any]:
                return [github_search]

        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
        )

        tools = await toolbox._fetch_tools(FakeClient())

        result = await tools[0].ainvoke({"query": "langchain-ai/langchain-azure"})

        assert err_msg in result


class TestAzureAIProjectToolboxAuthHeaders:
    def _install_fake_httpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAuth:
            pass

        fake_httpx = ModuleType("httpx")
        fake_httpx.Auth = FakeAuth  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    def _install_fake_agentserver_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
        platform_headers: dict[str, str],
    ) -> None:
        class FakeRequestContext:
            def platform_headers(self) -> dict[str, str]:
                return platform_headers

        def get_request_context() -> FakeRequestContext:
            return FakeRequestContext()

        fake_core = ModuleType("azure.ai.agentserver.core")
        fake_core.get_request_context = (  # type: ignore[attr-defined]
            get_request_context
        )

        fake_agentserver = ModuleType("azure.ai.agentserver")
        fake_agentserver.core = fake_core  # type: ignore[attr-defined]

        fake_azure_ai = ModuleType("azure.ai")
        fake_azure_ai.agentserver = fake_agentserver  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "azure.ai", fake_azure_ai)
        monkeypatch.setitem(sys.modules, "azure.ai.agentserver", fake_agentserver)
        monkeypatch.setitem(sys.modules, "azure.ai.agentserver.core", fake_core)

    def test_build_auth_and_headers_with_static_token(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._install_fake_httpx(monkeypatch)

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

    def test_static_token_auth_applies_platform_headers_per_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._install_fake_httpx(monkeypatch)
        self._install_fake_agentserver_context(
            monkeypatch,
            {"x-agent-foundry-call-id": "call-123"},
        )

        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="abc123",
        )

        auth, _ = toolbox._build_auth_and_headers()
        request = SimpleNamespace(headers={})

        flow = auth.auth_flow(request)
        next(flow)

        assert request.headers["Authorization"] == "Bearer abc123"
        assert request.headers["x-agent-foundry-call-id"] == "call-123"

    def test_token_auth_applies_platform_headers_per_request(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._install_fake_httpx(monkeypatch)
        self._install_fake_agentserver_context(
            monkeypatch,
            {"x-agent-foundry-call-id": "call-456"},
        )

        fake_identity = ModuleType("azure.identity")
        fake_identity.DefaultAzureCredential = (  # type: ignore[attr-defined]
            lambda: object()
        )
        fake_identity.get_bearer_token_provider = (  # type: ignore[attr-defined]
            lambda credential, audience: lambda: "token-456"
        )
        monkeypatch.setitem(sys.modules, "azure.identity", fake_identity)

        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
        )

        auth, _ = toolbox._build_auth_and_headers()
        request = SimpleNamespace(headers={})

        flow = auth.auth_flow(request)
        next(flow)

        assert request.headers["Authorization"] == "Bearer token-456"
        assert request.headers["x-agent-foundry-call-id"] == "call-456"

    def test_platform_headers_do_not_override_existing_request_headers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._install_fake_httpx(monkeypatch)
        self._install_fake_agentserver_context(
            monkeypatch,
            {"x-agent-foundry-call-id": "context-call"},
        )

        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="abc123",
        )

        auth, _ = toolbox._build_auth_and_headers()
        request = SimpleNamespace(headers={"x-agent-foundry-call-id": "caller-call"})

        flow = auth.auth_flow(request)
        next(flow)

        assert request.headers["x-agent-foundry-call-id"] == "caller-call"

    def test_only_call_id_forwarded_not_user_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Per the Foundry outbound identity contract, only the per-request call
        ID is stamped outbound. ``x-agent-user-id`` (and any other identity
        header) must never be forwarded to 1P services, even if the request
        context exposes it.
        """
        self._install_fake_httpx(monkeypatch)
        self._install_fake_agentserver_context(
            monkeypatch,
            {
                "x-agent-foundry-call-id": "call-789",
                "x-agent-user-id": "user-789",
            },
        )

        toolbox = AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="abc123",
        )

        auth, _ = toolbox._build_auth_and_headers()
        request = SimpleNamespace(headers={})

        flow = auth.auth_flow(request)
        next(flow)

        assert request.headers["x-agent-foundry-call-id"] == "call-789"
        assert "x-agent-user-id" not in request.headers


class TestSchemeAndUriHelpers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("skill", "skill"),
            ("skills", "skill"),
            ("SKILLS", "skill"),
            ("file", "file"),
        ],
    )
    def test_normalize_scheme(self, raw: str, expected: str) -> None:
        assert _normalize_scheme(raw) == expected

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("skill://my-skill", "my-skill"),
            ("skill://my-skill/", "my-skill"),
            ("skill://group/my-skill", "group/my-skill"),
            ("no-scheme", "no-scheme"),
        ],
    )
    def test_resource_name_from_uri(self, uri: str, expected: str) -> None:
        assert _resource_name_from_uri(uri) == expected

    def test_run_async_without_running_loop(self) -> None:
        async def coro() -> int:
            return 42

        assert _run_async(coro()) == 42


def _install_fake_resources_module(
    monkeypatch: pytest.MonkeyPatch,
    load_mcp_resources: Any,
) -> None:
    """Inject a fake ``langchain_mcp_adapters.resources`` module."""
    if "langchain_mcp_adapters" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "langchain_mcp_adapters",
            ModuleType("langchain_mcp_adapters"),
        )
    fake_resources = ModuleType("langchain_mcp_adapters.resources")
    fake_resources.load_mcp_resources = load_mcp_resources  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.resources", fake_resources)


class _FakeSession:
    def __init__(self, uris: list[str]) -> None:
        self._uris = uris

    async def list_resources(self) -> Any:
        resources = [SimpleNamespace(uri=uri) for uri in self._uris]
        return SimpleNamespace(resources=resources)


class _FakeSessionCM:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeClient:
    def __init__(self, uris: list[str], observed: dict[str, Any]) -> None:
        self._uris = uris
        self._observed = observed

    def session(self, server_name: str) -> _FakeSessionCM:
        self._observed["server_name"] = server_name
        return _FakeSessionCM(_FakeSession(self._uris))


class TestAzureAIProjectToolboxResources:
    def _make_toolbox(self) -> AzureAIProjectToolbox:
        return AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
        )

    async def test_aget_resources_filters_by_scheme_and_sets_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()
        observed: dict[str, Any] = {}
        all_uris = ["skill://alpha", "skill://beta", "file://notes.txt"]
        monkeypatch.setattr(
            toolbox, "_build_mcp_client", lambda: _FakeClient(all_uris, observed)
        )

        async def fake_load(session: Any, *, uris: list[str]) -> list[Blob]:
            observed["uris"] = uris
            return [
                Blob.from_data(
                    data=f"# {uri}",
                    mime_type="text/markdown",
                    metadata={"uri": uri},
                )
                for uri in uris
            ]

        _install_fake_resources_module(monkeypatch, fake_load)

        blobs = await toolbox.aget_resources(scheme="skills")

        assert observed["server_name"] == "toolbox"
        assert observed["uris"] == ["skill://alpha", "skill://beta"]
        assert [blob.source for blob in blobs] == ["alpha", "beta"]
        assert blobs[0].as_string() == "# skill://alpha"

    async def test_aget_resources_no_scheme_returns_all(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()
        observed: dict[str, Any] = {}
        all_uris = ["skill://alpha", "file://notes.txt"]
        monkeypatch.setattr(
            toolbox, "_build_mcp_client", lambda: _FakeClient(all_uris, observed)
        )

        async def fake_load(session: Any, *, uris: list[str]) -> list[Blob]:
            observed["uris"] = uris
            return [Blob.from_data(data="x", metadata={"uri": uri}) for uri in uris]

        _install_fake_resources_module(monkeypatch, fake_load)

        await toolbox.aget_resources()

        assert observed["uris"] == ["skill://alpha", "file://notes.txt"]

    async def test_aget_resources_explicit_uris_skip_listing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()
        observed: dict[str, Any] = {}
        monkeypatch.setattr(
            toolbox,
            "_build_mcp_client",
            lambda: _FakeClient(["skill://should-not-be-listed"], observed),
        )

        async def fake_load(session: Any, *, uris: list[str]) -> list[Blob]:
            observed["uris"] = uris
            return [Blob.from_data(data="x", metadata={"uri": uri}) for uri in uris]

        _install_fake_resources_module(monkeypatch, fake_load)

        await toolbox.aget_resources(uris="skill://explicit")

        assert observed["uris"] == ["skill://explicit"]

    def test_get_resources_sync_wrapper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()
        observed: dict[str, Any] = {}
        all_uris = ["skill://alpha", "file://notes.txt"]
        monkeypatch.setattr(
            toolbox, "_build_mcp_client", lambda: _FakeClient(all_uris, observed)
        )

        async def fake_load(session: Any, *, uris: list[str]) -> list[Blob]:
            return [
                Blob.from_data(
                    data=f"# {uri}",
                    mime_type="text/markdown",
                    metadata={"uri": uri},
                )
                for uri in uris
            ]

        _install_fake_resources_module(monkeypatch, fake_load)

        blobs = toolbox.get_resources(scheme="skills")

        assert [blob.source for blob in blobs] == ["alpha"]

    async def test_aget_resources_requires_project_endpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        toolbox = AzureAIProjectToolbox(project_endpoint="", toolbox_name="tb")

        with pytest.raises(ValueError, match="project_endpoint is required"):
            await toolbox.aget_resources(scheme="skills")


def _install_fake_deepagents_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``deepagents.backends.utils.create_file_data``."""

    def create_file_data(content: str) -> dict[str, str]:
        return {"content": content, "encoding": "utf-8"}

    deepagents = ModuleType("deepagents")
    backends = ModuleType("deepagents.backends")
    utils = ModuleType("deepagents.backends.utils")
    utils.create_file_data = create_file_data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepagents", deepagents)
    monkeypatch.setitem(sys.modules, "deepagents.backends", backends)
    monkeypatch.setitem(sys.modules, "deepagents.backends.utils", utils)


class TestAzureAIProjectToolboxSkills:
    def _make_toolbox(self) -> AzureAIProjectToolbox:
        return AzureAIProjectToolbox(
            project_endpoint="https://resource.services.ai.azure.com/api/projects/p",
            toolbox_name="tb",
            credential="token",
        )

    async def test_aget_skills_builds_file_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()
        observed: dict[str, Any] = {}

        async def fake_aget_resources(
            _self: AzureAIProjectToolbox, *, scheme: str | None = None
        ) -> list[Blob]:
            observed["scheme"] = scheme
            return [
                Blob.from_data(
                    data="# alpha",
                    mime_type="text/markdown",
                    metadata={
                        "uri": "skill://alpha/SKILL.md",
                        "source": "alpha/SKILL.md",
                    },
                ),
                Blob.from_data(
                    data="# beta",
                    mime_type="text/markdown",
                    metadata={
                        "uri": "skill://beta/SKILL.md",
                        "source": "beta/SKILL.md",
                    },
                ),
            ]

        monkeypatch.setattr(
            AzureAIProjectToolbox, "aget_resources", fake_aget_resources
        )
        _install_fake_deepagents_module(monkeypatch)

        skill_files = await toolbox.aget_skills()

        assert observed["scheme"] == "skill"
        assert skill_files == {
            "/skills/alpha/SKILL.md": {"content": "# alpha", "encoding": "utf-8"},
            "/skills/beta/SKILL.md": {"content": "# beta", "encoding": "utf-8"},
        }

    async def test_aget_skills_custom_base_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()

        async def fake_aget_resources(
            _self: AzureAIProjectToolbox, *, scheme: str | None = None
        ) -> list[Blob]:
            return [
                Blob.from_data(
                    data="# alpha",
                    metadata={
                        "uri": "skill://alpha/SKILL.md",
                        "source": "alpha/SKILL.md",
                    },
                ),
            ]

        monkeypatch.setattr(
            AzureAIProjectToolbox, "aget_resources", fake_aget_resources
        )
        _install_fake_deepagents_module(monkeypatch)

        skill_files = await toolbox.aget_skills(base_path="/skills/shared/")

        assert list(skill_files) == ["/skills/shared/alpha/SKILL.md"]

    async def test_aget_skills_skips_blobs_without_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()

        async def fake_aget_resources(
            _self: AzureAIProjectToolbox, *, scheme: str | None = None
        ) -> list[Blob]:
            return [
                Blob.from_data(data="orphan", metadata={"uri": "skill://"}),
                Blob.from_data(
                    data="# alpha",
                    metadata={
                        "uri": "skill://alpha/SKILL.md",
                        "source": "alpha/SKILL.md",
                    },
                ),
            ]

        monkeypatch.setattr(
            AzureAIProjectToolbox, "aget_resources", fake_aget_resources
        )
        _install_fake_deepagents_module(monkeypatch)

        skill_files = await toolbox.aget_skills()

        assert list(skill_files) == ["/skills/alpha/SKILL.md"]

    async def test_aget_skills_excludes_index_resource(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()

        async def fake_aget_resources(
            _self: AzureAIProjectToolbox, *, scheme: str | None = None
        ) -> list[Blob]:
            return [
                Blob.from_data(
                    data='{"skills": []}',
                    metadata={
                        "uri": "skill://index.json",
                        "source": "index.json",
                    },
                ),
                Blob.from_data(
                    data="# alpha",
                    metadata={
                        "uri": "skill://alpha/SKILL.md",
                        "source": "alpha/SKILL.md",
                    },
                ),
            ]

        monkeypatch.setattr(
            AzureAIProjectToolbox, "aget_resources", fake_aget_resources
        )
        _install_fake_deepagents_module(monkeypatch)

        skill_files = await toolbox.aget_skills()

        assert list(skill_files) == ["/skills/alpha/SKILL.md"]

    async def test_aget_skills_uploads_to_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()

        async def fake_aget_resources(
            _self: AzureAIProjectToolbox, *, scheme: str | None = None
        ) -> list[Blob]:
            return [
                Blob.from_data(
                    data="# alpha",
                    metadata={
                        "uri": "skill://alpha/SKILL.md",
                        "source": "alpha/SKILL.md",
                    },
                ),
            ]

        monkeypatch.setattr(
            AzureAIProjectToolbox, "aget_resources", fake_aget_resources
        )
        _install_fake_deepagents_module(monkeypatch)

        uploaded: dict[str, Any] = {}

        class FakeBackend:
            async def aupload_files(self, files: list[tuple[str, bytes]]) -> None:
                uploaded["files"] = files

        skill_files = await toolbox.aget_skills(backend=FakeBackend())

        assert list(skill_files) == ["/skills/alpha/SKILL.md"]
        assert uploaded["files"] == [("/skills/alpha/SKILL.md", b"# alpha")]

    async def test_aget_skills_invalid_base_path(self) -> None:
        toolbox = self._make_toolbox()

        with pytest.raises(ValueError, match="base_path must start and end"):
            await toolbox.aget_skills(base_path="skills")

    async def test_aget_skills_requires_deepagents(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toolbox = self._make_toolbox()

        for mod in (
            "deepagents",
            "deepagents.backends",
            "deepagents.backends.utils",
        ):
            monkeypatch.setitem(sys.modules, mod, None)

        with pytest.raises(ImportError, match="requires 'deepagents'"):
            await toolbox.aget_skills()

    def test_get_skills_sync_wrapper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from langchain_core.documents.base import Blob

        toolbox = self._make_toolbox()

        async def fake_aget_resources(
            _self: AzureAIProjectToolbox, *, scheme: str | None = None
        ) -> list[Blob]:
            return [
                Blob.from_data(
                    data="# alpha",
                    metadata={
                        "uri": "skill://alpha/SKILL.md",
                        "source": "alpha/SKILL.md",
                    },
                ),
            ]

        monkeypatch.setattr(
            AzureAIProjectToolbox, "aget_resources", fake_aget_resources
        )
        _install_fake_deepagents_module(monkeypatch)

        skill_files = toolbox.get_skills()

        assert skill_files == {
            "/skills/alpha/SKILL.md": {"content": "# alpha", "encoding": "utf-8"},
        }
