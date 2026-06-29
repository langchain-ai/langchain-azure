"""Unit tests for AzureAIAnthropicChatModel."""

from typing import Any
from unittest.mock import patch

import pytest
from azure.core.credentials import AccessToken, AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential

from langchain_azure_ai.chat_models import AzureAIAnthropicChatModel
from langchain_azure_ai.chat_models.anthropic import _resolve_anthropic_endpoint


class _FakeTokenCredential(TokenCredential):
    """Sync TokenCredential that returns a static token."""

    def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        return AccessToken("fake-sync-token", 9999999999)


class _FakeAsyncTokenCredential(AsyncTokenCredential):
    """Async TokenCredential that returns a static token."""

    async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:
        return AccessToken("fake-async-token", 9999999999)

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> "_FakeAsyncTokenCredential":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class TestResolveAnthropicEndpoint:
    """Tests for the endpoint resolution helper."""

    def test_appends_anthropic_path(self) -> None:
        assert (
            _resolve_anthropic_endpoint("https://r.services.ai.azure.com")
            == "https://r.services.ai.azure.com/anthropic/"
        )

    def test_trailing_slash_is_normalised(self) -> None:
        assert (
            _resolve_anthropic_endpoint("https://r.services.ai.azure.com/")
            == "https://r.services.ai.azure.com/anthropic/"
        )

    def test_preserves_explicit_anthropic_path(self) -> None:
        assert (
            _resolve_anthropic_endpoint("https://r.services.ai.azure.com/anthropic")
            == "https://r.services.ai.azure.com/anthropic/"
        )

    def test_env_fallback_resource_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "my-resource")
        assert (
            _resolve_anthropic_endpoint(None)
            == "https://my-resource.services.ai.azure.com/anthropic/"
        )

    def test_missing_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_FOUNDRY_RESOURCE"):
            _resolve_anthropic_endpoint(None)


class TestAzureAIAnthropicChatModel:
    """Tests for AzureAIAnthropicChatModel construction and client wiring."""

    def test_api_key_string_credential(self) -> None:
        model = AzureAIAnthropicChatModel(
            endpoint="https://test.services.ai.azure.com",
            credential="sk-ant-test",
            model="claude-sonnet-4-20250514",
        )

        assert (
            model.anthropic_api_url == "https://test.services.ai.azure.com/anthropic/"
        )
        client = model._client
        async_client = model._async_client

        assert type(client).__name__ == "AnthropicFoundry"
        assert type(async_client).__name__ == "AsyncAnthropicFoundry"
        assert str(client.base_url) == "https://test.services.ai.azure.com/anthropic/"
        assert client.api_key == "sk-ant-test"
        assert async_client.api_key == "sk-ant-test"

    def test_azure_key_credential(self) -> None:
        model = AzureAIAnthropicChatModel(
            endpoint="https://test.services.ai.azure.com",
            credential=AzureKeyCredential("my-key"),
            model="claude-sonnet-4-20250514",
        )

        assert model._client.api_key == "my-key"
        assert model._async_client.api_key == "my-key"

    def test_token_credential_uses_azure_ad_token_provider(self) -> None:
        cred = _FakeTokenCredential()
        model = AzureAIAnthropicChatModel(
            endpoint="https://test.services.ai.azure.com",
            credential=cred,
            model="claude-sonnet-4-20250514",
        )

        client = model._client
        # AnthropicFoundry stores the provider on a private attribute.
        provider = client._azure_ad_token_provider
        assert callable(provider)
        assert provider() == "fake-sync-token"
        # API key was injected as a placeholder so validation passes; the
        # actual auth happens via the token provider.
        assert client.api_key != "sk-ant-test"

    def test_async_token_credential_uses_async_provider(self) -> None:
        cred = _FakeAsyncTokenCredential()
        model = AzureAIAnthropicChatModel(
            endpoint="https://test.services.ai.azure.com",
            credential=cred,
            model="claude-sonnet-4-20250514",
        )

        async_client = model._async_client
        assert type(async_client).__name__ == "AsyncAnthropicFoundry"
        provider = async_client._azure_ad_token_provider
        assert callable(provider)

    def test_async_token_credential_rejected_for_sync_client(self) -> None:
        cred = _FakeAsyncTokenCredential()
        model = AzureAIAnthropicChatModel(
            endpoint="https://test.services.ai.azure.com",
            credential=cred,
            model="claude-sonnet-4-20250514",
        )

        with pytest.raises(ValueError, match="AsyncTokenCredential"):
            _ = model._client

    def test_default_credential_used_when_none(self) -> None:
        with patch(
            "langchain_azure_ai.chat_models.anthropic.DefaultAzureCredential"
        ) as default_cred_cls:
            default_cred_cls.return_value = _FakeTokenCredential()
            model = AzureAIAnthropicChatModel(
                endpoint="https://test.services.ai.azure.com",
                model="claude-sonnet-4-20250514",
            )
            _ = model._client

        default_cred_cls.assert_called_once()

    def test_async_default_credential_used_when_none(self) -> None:
        with patch(
            "langchain_azure_ai.chat_models.anthropic.AsyncDefaultAzureCredential"
        ) as async_default_cred_cls:
            async_default_cred_cls.return_value = _FakeAsyncTokenCredential()
            model = AzureAIAnthropicChatModel(
                endpoint="https://test.services.ai.azure.com",
                model="claude-sonnet-4-20250514",
            )
            _ = model._async_client

        async_default_cred_cls.assert_called_once()

    def test_endpoint_already_has_anthropic_suffix(self) -> None:
        model = AzureAIAnthropicChatModel(
            endpoint="https://test.services.ai.azure.com/anthropic",
            credential="sk-ant-test",
            model="claude-sonnet-4-20250514",
        )
        assert (
            str(model._client.base_url)
            == "https://test.services.ai.azure.com/anthropic/"
        )

    def test_env_var_resource_name_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "env-resource")
        model = AzureAIAnthropicChatModel(
            credential="sk-ant-test",
            model="claude-sonnet-4-20250514",
        )
        assert (
            str(model._client.base_url)
            == "https://env-resource.services.ai.azure.com/anthropic/"
        )

    def test_project_endpoint_derives_resource_url(self) -> None:
        model = AzureAIAnthropicChatModel(
            project_endpoint=(
                "https://test.services.ai.azure.com/api/projects/my-project"
            ),
            credential="sk-ant-test",
            model="claude-sonnet-4-20250514",
        )
        assert (
            str(model._client.base_url)
            == "https://test.services.ai.azure.com/anthropic/"
        )
        assert model.project_endpoint == (
            "https://test.services.ai.azure.com/api/projects/my-project"
        )
        assert model.endpoint is None

    def test_project_endpoint_with_token_credential(self) -> None:
        cred = _FakeTokenCredential()
        model = AzureAIAnthropicChatModel(
            project_endpoint=(
                "https://test.services.ai.azure.com/api/projects/my-project"
            ),
            credential=cred,
            model="claude-sonnet-4-20250514",
        )
        client = model._client
        assert type(client).__name__ == "AnthropicFoundry"
        assert str(client.base_url) == "https://test.services.ai.azure.com/anthropic/"
        provider = client._azure_ad_token_provider
        assert callable(provider)
        assert provider() == "fake-sync-token"

    def test_project_endpoint_and_endpoint_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="Both.*project_endpoint.*endpoint"):
            AzureAIAnthropicChatModel(
                project_endpoint=(
                    "https://test.services.ai.azure.com/api/projects/my-project"
                ),
                endpoint="https://test.services.ai.azure.com",
                credential="sk-ant-test",
                model="claude-sonnet-4-20250514",
            )
