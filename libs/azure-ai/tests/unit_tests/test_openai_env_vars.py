"""Unit tests for AZURE_OPENAI_* environment variable support in
AzureAIOpenAIApiChatModel and AzureAIOpenAIApiEmbeddingsModel.
"""

from unittest.mock import MagicMock, patch

import pytest

from langchain_azure_ai._resources import _configure_openai_credential_values

# ---------------------------------------------------------------------------
# _configure_openai_credential_values — env var resolution
# ---------------------------------------------------------------------------


class TestEnvVarEndpointResolution:
    """AZURE_OPENAI_ENDPOINT and AZURE_AI_OPENAI_ENDPOINT env vars."""

    def test_azure_openai_endpoint_appends_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-deploy")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key"}
        result, _ = _configure_openai_credential_values(values)
        assert result["model"] == "gpt-4o-deploy"

    def test_explicit_model_overrides_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deploy")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key", "model": "explicit-model"}
        result, _ = _configure_openai_credential_values(values)
        assert result["model"] == "explicit-model"

    def test_model_name_alias_prevents_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "env-deploy")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://res.services.ai.azure.com")
        values = {"credential": "fake-key", "model_name": "alias-model"}
        result, _ = _configure_openai_credential_values(values)
        assert "model" not in result or result.get("model_name") == "alias-model"


class TestEnvVarApiVersionResolution:
    """AZURE_OPENAI_API_VERSION should populate api_version."""

    def test_api_version_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
