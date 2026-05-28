"""Unit tests for AZURE_OPENAI_* environment variable support in
AzureAIOpenAIApiChatModel and AzureAIOpenAIApiEmbeddingsModel.
"""

from unittest.mock import MagicMock, patch

import pytest

from langchain_azure_ai._resources import _configure_openai_credential_values
from langchain_azure_ai.utils.env import get_project_endpoint

# ---------------------------------------------------------------------------
# get_project_endpoint helpers
# ---------------------------------------------------------------------------


class TestGetProjectEndpointHelpers:
    """Direct tests for the centralized project-endpoint resolution helpers."""

    def test_returns_none_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        assert get_project_endpoint(nullable=True) is None

    def test_reads_azure_ai_project_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT", "https://a.example.com/api/projects/p"
        )
        assert (
            get_project_endpoint(nullable=True)
            == "https://a.example.com/api/projects/p"
        )

    def test_reads_foundry_project_endpoint_as_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "FOUNDRY_PROJECT_ENDPOINT", "https://f.example.com/api/projects/p"
        )
        assert (
            get_project_endpoint(nullable=True)
            == "https://f.example.com/api/projects/p"
        )

    def test_azure_ai_wins_over_foundry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT", "https://azure.example.com/api/projects/p"
        )
        monkeypatch.setenv(
            "FOUNDRY_PROJECT_ENDPOINT", "https://foundry.example.com/api/projects/p"
        )
        assert (
            get_project_endpoint(nullable=True)
            == "https://azure.example.com/api/projects/p"
        )

    def test_reads_from_data_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        result = get_project_endpoint({"project_endpoint": "https://dict.example.com"})
        assert result == "https://dict.example.com"

    def test_data_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT", "https://env.example.com/api/projects/p"
        )
        result = get_project_endpoint({"project_endpoint": "https://dict.example.com"})
        assert result == "https://dict.example.com"

    def test_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT", "https://env.example.com/api/projects/p"
        )
        result = get_project_endpoint({})
        assert result == "https://env.example.com/api/projects/p"

    def test_nullable_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        assert get_project_endpoint(nullable=True) is None

    def test_raises_when_not_nullable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        with pytest.raises(ValueError):
            get_project_endpoint()


# ---------------------------------------------------------------------------
# _configure_openai_credential_values — env var resolution
# ---------------------------------------------------------------------------


class TestEnvVarEndpointResolution:
    """AZURE_OPENAI_ENDPOINT and AZURE_AI_OPENAI_ENDPOINT env vars."""

    def test_azure_openai_endpoint_appends_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT", "https://myresource.services.ai.azure.com"
        )
        values = {"credential": "fake-key"}
        result, _ = _configure_openai_credential_values(values)
        assert (
            result["endpoint"] == "https://myresource.services.ai.azure.com/openai/v1"
        )
        assert (
            result["openai_api_base"]
            == "https://myresource.services.ai.azure.com/openai/v1"
        )

    def test_azure_openai_endpoint_strips_trailing_slash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT", "https://myresource.services.ai.azure.com/"
        )
        values = {"credential": "fake-key"}
        result, _ = _configure_openai_credential_values(values)
        assert result["endpoint"].endswith("/openai/v1")
        assert not result["endpoint"].endswith("//openai/v1")

    def test_explicit_endpoint_overrides_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.services.ai.azure.com")
        values = {
            "endpoint": "https://explicit.services.ai.azure.com/openai/v1",
            "credential": "fake-key",
        }
        result, _ = _configure_openai_credential_values(values)
        assert (
            result["openai_api_base"]
            == "https://explicit.services.ai.azure.com/openai/v1"
        )

    def test_azure_ai_openai_endpoint_used_as_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AZURE_AI_OPENAI_ENDPOINT is used verbatim without appending a path."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "AZURE_AI_OPENAI_ENDPOINT",
            "https://myresource.services.ai.azure.com/openai/v1",
        )
        values = {"credential": "fake-key"}
        result, _ = _configure_openai_credential_values(values)
        assert (
            result["endpoint"] == "https://myresource.services.ai.azure.com/openai/v1"
        )

    def test_azure_ai_openai_endpoint_takes_priority_over_azure_openai_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AZURE_AI_OPENAI_ENDPOINT wins over AZURE_OPENAI_ENDPOINT."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "AZURE_AI_OPENAI_ENDPOINT",
            "https://ai.services.ai.azure.com/openai/v1",
        )
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT", "https://root.services.ai.azure.com"
        )
        values = {"credential": "fake-key"}
        result, _ = _configure_openai_credential_values(values)
        assert result["endpoint"] == "https://ai.services.ai.azure.com/openai/v1"

    def test_explicit_endpoint_overrides_azure_ai_openai_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Constructor endpoint wins over AZURE_AI_OPENAI_ENDPOINT."""
        monkeypatch.setenv(
            "AZURE_AI_OPENAI_ENDPOINT",
            "https://env.services.ai.azure.com/openai/v1",
        )
        values = {
            "endpoint": "https://explicit.services.ai.azure.com/openai/v1",
            "credential": "fake-key",
        }
        result, _ = _configure_openai_credential_values(values)
        assert (
            result["openai_api_base"]
            == "https://explicit.services.ai.azure.com/openai/v1"
        )


class TestEnvVarDeploymentNameResolution:
    """AZURE_OPENAI_DEPLOYMENT_NAME should populate model when not provided."""

    def test_deployment_name_sets_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-deploy")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key"}
        result, _ = _configure_openai_credential_values(values)
        assert result["model"] == "gpt-4o-deploy"

    def test_explicit_model_overrides_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deploy")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key", "model": "explicit-model"}
        result, _ = _configure_openai_credential_values(values)
        assert result["model"] == "explicit-model"

    def test_model_name_alias_prevents_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deploy")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key", "model_name": "alias-model"}
        result, _ = _configure_openai_credential_values(values)
        assert "model" not in result or result.get("model_name") == "alias-model"


class TestEnvVarApiVersionResolution:
    """AZURE_OPENAI_API_VERSION should populate api_version."""

    def test_api_version_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key"}
        result, clients = _configure_openai_credential_values(values)
        assert result["api_version"] == "2025-01-01"
        # When api_version is present with credential, clients should be built
        assert clients is not None

    def test_explicit_api_version_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "env-version")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key", "api_version": "explicit-version"}
        result, clients = _configure_openai_credential_values(values)
        # The explicit value should be in the result (it was already set)
        assert result.get("api_version") == "explicit-version"
        assert clients is not None

    def test_no_clients_built_without_api_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key"}
        result, clients = _configure_openai_credential_values(values)
        # Without api_version, clients are not pre-built
        assert clients is None


class TestEnvVarPriority:
    """AZURE_AI_PROJECT_ENDPOINT takes precedence over AZURE_OPENAI_ENDPOINT."""

    @patch("langchain_azure_ai._resources.AIProjectClient")
    def test_project_endpoint_wins_over_openai_endpoint(
        self, mock_project_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT",
            "https://res.services.ai.azure.com/api/projects/proj",
        )
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")

        mock_project = MagicMock()
        mock_sync_openai = MagicMock()
        mock_sync_openai.base_url = "https://res.services.ai.azure.com/openai/v1"
        mock_project.get_openai_client.return_value = mock_sync_openai
        mock_project_cls.return_value = mock_project

        from azure.identity import DefaultAzureCredential

        values = {"credential": DefaultAzureCredential()}
        result, clients = _configure_openai_credential_values(values)

        # Project endpoint should have been used
        assert result.get("project_endpoint") == (
            "https://res.services.ai.azure.com/api/projects/proj"
        )
        # Clients should have been built via project path
        assert clients is not None

    @patch("langchain_azure_ai._resources.AIProjectClient")
    def test_foundry_endpoint_used_as_fallback(
        self, mock_project_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FOUNDRY_PROJECT_ENDPOINT is used when AZURE_AI_PROJECT_ENDPOINT is absent."""
        monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "FOUNDRY_PROJECT_ENDPOINT",
            "https://res.services.ai.azure.com/api/projects/proj",
        )

        mock_project = MagicMock()
        mock_sync_openai = MagicMock()
        mock_sync_openai.base_url = "https://res.services.ai.azure.com/openai/v1"
        mock_project.get_openai_client.return_value = mock_sync_openai
        mock_project_cls.return_value = mock_project

        from azure.identity import DefaultAzureCredential

        values = {"credential": DefaultAzureCredential()}
        result, clients = _configure_openai_credential_values(values)

        assert result.get("project_endpoint") == (
            "https://res.services.ai.azure.com/api/projects/proj"
        )
        assert clients is not None

    @patch("langchain_azure_ai._resources.AIProjectClient")
    def test_azure_ai_project_endpoint_wins_over_foundry_endpoint(
        self, mock_project_cls: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AZURE_AI_PROJECT_ENDPOINT takes precedence when both env vars are set."""
        monkeypatch.setenv(
            "AZURE_AI_PROJECT_ENDPOINT",
            "https://azure.services.ai.azure.com/api/projects/azure-proj",
        )
        monkeypatch.setenv(
            "FOUNDRY_PROJECT_ENDPOINT",
            "https://foundry.services.ai.azure.com/api/projects/foundry-proj",
        )

        mock_project = MagicMock()
        mock_sync_openai = MagicMock()
        mock_sync_openai.base_url = "https://azure.services.ai.azure.com/openai/v1"
        mock_project.get_openai_client.return_value = mock_sync_openai
        mock_project_cls.return_value = mock_project

        from azure.identity import DefaultAzureCredential

        values = {"credential": DefaultAzureCredential()}
        result, clients = _configure_openai_credential_values(values)

        assert result.get("project_endpoint") == (
            "https://azure.services.ai.azure.com/api/projects/azure-proj"
        )
        assert clients is not None


class TestConflictValidation:
    """Providing both project_endpoint and endpoint as constructor params errors."""

    def test_both_explicit_raises_error(self) -> None:
        values = {
            "project_endpoint": "https://res.services.ai.azure.com/api/projects/p",
            "endpoint": "https://res.services.ai.azure.com/openai/v1",
            "credential": "fake-key",
        }
        with pytest.raises(ValueError, match="Both.*project_endpoint.*endpoint"):
            _configure_openai_credential_values(values)

    def test_only_project_endpoint_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Providing only project_endpoint should not raise."""
        with patch("langchain_azure_ai._resources.AIProjectClient") as mock_cls:
            mock_project = MagicMock()
            mock_sync_openai = MagicMock()
            mock_sync_openai.base_url = "https://x.services.ai.azure.com/openai/v1"
            mock_project.get_openai_client.return_value = mock_sync_openai
            mock_cls.return_value = mock_project

            from azure.identity import DefaultAzureCredential

            values = {
                "project_endpoint": "https://x.services.ai.azure.com/api/projects/p",
                "credential": DefaultAzureCredential(),
            }
            result, clients = _configure_openai_credential_values(values)
            assert clients is not None

    def test_only_endpoint_ok(self) -> None:
        """Providing only endpoint should not raise."""
        values = {
            "endpoint": "https://res.services.ai.azure.com/openai/v1",
            "credential": "fake-key",
        }
        result, _ = _configure_openai_credential_values(values)
        assert result["openai_api_base"] == values["endpoint"]


class TestForceOpenAIServiceEndpoint:
    """force_openai_service_endpoint=True resolves the direct endpoint from project."""

    def test_force_resolves_service_endpoint(self) -> None:
        """When force_openai_service_endpoint=True and project_endpoint is set,
        get_service_endpoint_from_project is called and the direct-endpoint path
        is used instead of the project-endpoint path."""
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()
        resolved_endpoint = "https://res.openai.azure.com/openai/v1"
        resolved_credential = "resolved-key"

        with patch(
            "langchain_azure_ai._resources.get_service_endpoint_from_project"
        ) as mock_get_service:
            mock_get_service.return_value = (resolved_endpoint, resolved_credential)
            values = {
                "project_endpoint": (
                    "https://res.services.ai.azure.com/api/projects/proj"
                ),
                "credential": credential,
            }
            result, clients = _configure_openai_credential_values(
                values, force_openai_service_endpoint=True
            )

            mock_get_service.assert_called_once_with(
                project_endpoint=(
                    "https://res.services.ai.azure.com/api/projects/proj"
                ),
                credential=credential,
                service="inference",
            )
            assert result.get("project_endpoint") is None
            assert result["endpoint"] == resolved_endpoint
            assert result["credential"] == resolved_credential

    def test_force_falls_back_to_project_path_on_error(self) -> None:
        """When get_service_endpoint_from_project raises, the project-endpoint
        path is used as a fallback."""
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential()

        mock_sync_project = MagicMock()
        mock_openai_client = MagicMock()
        mock_openai_client.base_url = "https://res.openai.azure.com/openai/v1"
        mock_sync_project.get_openai_client.return_value = mock_openai_client
        mock_openai_client.with_options.return_value = mock_openai_client

        with (
            patch(
                "langchain_azure_ai._resources.get_service_endpoint_from_project"
            ) as mock_get_service,
            patch("langchain_azure_ai._resources.AIProjectClient") as mock_project_cls,
        ):
            mock_get_service.side_effect = ValueError("resolution failed")
            mock_project_cls.return_value = mock_sync_project

            values = {
                "project_endpoint": (
                    "https://res.services.ai.azure.com/api/projects/proj"
                ),
                "credential": credential,
            }
            result, clients = _configure_openai_credential_values(
                values, force_openai_service_endpoint=True
            )

            # Fell back to project-endpoint path: project_endpoint is still set
            assert result.get("project_endpoint") == (
                "https://res.services.ai.azure.com/api/projects/proj"
            )
            assert clients is not None

    def test_force_without_project_endpoint_uses_direct_path(self) -> None:
        """When force_openai_service_endpoint=True but only endpoint is set,
        the direct-endpoint path is used as normal (no project resolution)."""
        with patch(
            "langchain_azure_ai._resources.get_service_endpoint_from_project"
        ) as mock_get_service:
            values = {
                "endpoint": "https://res.services.ai.azure.com/openai/v1",
                "credential": "fake-key",
            }
            result, _ = _configure_openai_credential_values(
                values, force_openai_service_endpoint=True
            )

            mock_get_service.assert_not_called()
            assert result["openai_api_base"] == values["endpoint"]


class TestApiVersionClientConstruction:
    """When api_version is present, pre-built clients should include default_query."""

    def test_clients_have_api_version_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-03-01")
        values = {
            "endpoint": "https://res.services.ai.azure.com/openai/v1",
            "credential": "fake-key",
        }
        result, clients = _configure_openai_credential_values(values)
        assert clients is not None
        sync_client, async_client = clients
        assert sync_client._custom_query["api-version"] == "2025-03-01"
        assert async_client._custom_query["api-version"] == "2025-03-01"

    def test_no_credential_no_clients(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without credential, api_version shouldn't cause client construction."""
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-03-01")
        values = {
            "endpoint": "https://res.services.ai.azure.com/openai/v1",
        }
        result, clients = _configure_openai_credential_values(values)
        assert clients is None


# ---------------------------------------------------------------------------
# Integration with model classes (mocking _configure_openai_credential_values)
# ---------------------------------------------------------------------------


class TestChatModelEnvVars:
    """AzureAIOpenAIApiChatModel picks up AZURE_OPENAI_* env vars."""

    def test_env_vars_configure_chat_model(self) -> None:
        from langchain_azure_ai.chat_models.openai import AzureAIOpenAIApiChatModel

        with patch(
            "langchain_azure_ai.chat_models.openai._configure_openai_credential_values"
        ) as mock_configure:
            sync_client = MagicMock()
            async_client = MagicMock()
            mock_configure.return_value = (
                {
                    "endpoint": "https://res.services.ai.azure.com/openai/v1",
                    "model": "gpt-4o-deploy",
                    "api_version": "2025-03-01",
                },
                (sync_client, async_client),
            )
            m = AzureAIOpenAIApiChatModel(
                endpoint="https://res.services.ai.azure.com/openai/v1",
                credential="fake-key",
                model="gpt-4o-deploy",
                api_version="2025-03-01",
            )
            assert m.model_name == "gpt-4o-deploy"
            assert m.api_version == "2025-03-01"


class TestEmbeddingsModelEnvVars:
    """AzureAIOpenAIApiEmbeddingsModel picks up AZURE_OPENAI_* env vars."""

    def test_env_vars_configure_embeddings_model(self) -> None:
        from langchain_azure_ai.embeddings.openai import (
            AzureAIOpenAIApiEmbeddingsModel,
        )

        with patch(
            "langchain_azure_ai.embeddings.openai._configure_openai_credential_values"
        ) as mock_configure:
            sync_client = MagicMock()
            async_client = MagicMock()
            mock_configure.return_value = (
                {
                    "endpoint": "https://res.services.ai.azure.com/openai/v1",
                    "model": "text-embedding-3-small",
                    "api_version": "2024-05-01",
                },
                (sync_client, async_client),
            )
            m = AzureAIOpenAIApiEmbeddingsModel(
                endpoint="https://res.services.ai.azure.com/openai/v1",
                credential="fake-key",
                model="text-embedding-3-small",
            )
            assert m.model == "text-embedding-3-small"

    def test_project_endpoint_uses_resolved_openai_connection_for_embeddings(
        self,
    ) -> None:
        from azure.identity import DefaultAzureCredential

        from langchain_azure_ai.embeddings.openai import (
            AzureAIOpenAIApiEmbeddingsModel,
        )

        resolved_sync_client = MagicMock()
        resolved_sync_client.embeddings = MagicMock()
        resolved_async_client = MagicMock()
        resolved_async_client.embeddings = MagicMock()

        credential = DefaultAzureCredential()

        with patch(
            "langchain_azure_ai.embeddings.openai._configure_openai_credential_values"
        ) as mock_configure:
            mock_configure.return_value = (
                {
                    "endpoint": "https://res.openai.azure.com/openai/v1",
                    "credential": "resolved-key",
                    "model": "text-embedding-3-small",
                },
                (resolved_sync_client, resolved_async_client),
            )

            model = AzureAIOpenAIApiEmbeddingsModel(
                project_endpoint="https://res.services.ai.azure.com/api/projects/proj",
                credential=credential,
                model="text-embedding-3-small",
            )

            assert model.client is resolved_sync_client.embeddings
            assert model.async_client is resolved_async_client.embeddings
            mock_configure.assert_called_once_with(
                {
                    "project_endpoint": (
                        "https://res.services.ai.azure.com/api/projects/proj"
                    ),
                    "credential": credential,
                    "model": "text-embedding-3-small",
                },
                force_openai_service_endpoint=True,
            )
            assert mock_configure.call_count == 1
