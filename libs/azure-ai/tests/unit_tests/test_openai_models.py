"""Unit tests for the OpenAI-compatible Azure AI chat and embeddings models."""

import os
from unittest import mock

import pytest
from azure.core.credentials import AzureKeyCredential, TokenCredential
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_azure_ai._api.base import ExperimentalWarning
from langchain_azure_ai.chat_models.openai import AzureAIChatCompletionsModel
from langchain_azure_ai.embeddings.openai import AzureAIEmbeddingsModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_openai_client() -> mock.MagicMock:
    """Return a mock openai.OpenAI-style client with chat.completions."""
    client = mock.MagicMock()
    client.chat = mock.MagicMock()
    client.chat.completions = mock.MagicMock()
    client.embeddings = mock.MagicMock()
    return client


# ---------------------------------------------------------------------------
# AzureAIChatCompletionsModel – project_endpoint pattern
# ---------------------------------------------------------------------------


class TestAzureAIChatCompletionsModelProjectEndpoint:
    """Tests for the project_endpoint configuration path."""

    def test_is_subclass_of_azure_chat_openai(self) -> None:
        assert issubclass(AzureAIChatCompletionsModel, AzureChatOpenAI)

    def test_project_endpoint_configures_clients(self) -> None:
        sync_openai = _make_mock_openai_client()
        async_openai = _make_mock_openai_client()
        mock_credential = mock.MagicMock()

        # Make it look like a TokenCredential

        mock_credential.__class__ = type(
            "MockTokenCredential", (TokenCredential,), {}
        )

        with mock.patch(
            "langchain_azure_ai.chat_models.openai.AIProjectClient"
        ) as MockSync, mock.patch(
            "langchain_azure_ai.chat_models.openai.AsyncAIProjectClient"
        ) as MockAsync:
            MockSync.return_value.get_openai_client.return_value = sync_openai
            MockAsync.return_value.get_openai_client.return_value = async_openai

            with pytest.warns(ExperimentalWarning):
                model = AzureAIChatCompletionsModel(
                    project_endpoint=(
                        "https://resource.services.ai.azure.com/api/projects/proj"
                    ),
                    credential=mock_credential,
                    model="gpt-4o",
                )

        assert model.client is sync_openai.chat.completions
        assert model.async_client is async_openai.chat.completions
        assert model.root_client is sync_openai
        assert model.root_async_client is async_openai

    def test_project_endpoint_from_env_variable(self) -> None:
        sync_openai = _make_mock_openai_client()
        async_openai = _make_mock_openai_client()
        mock_credential = mock.MagicMock()


        mock_credential.__class__ = type(
            "MockTokenCredential", (TokenCredential,), {}
        )

        with mock.patch.dict(
            os.environ,
            {
                "AZURE_AI_PROJECT_ENDPOINT": (
                    "https://resource.services.ai.azure.com/api/projects/proj"
                )
            },
        ):
            with mock.patch(
                "langchain_azure_ai.chat_models.openai.AIProjectClient"
            ) as MockSync, mock.patch(
                "langchain_azure_ai.chat_models.openai.AsyncAIProjectClient"
            ) as MockAsync:
                MockSync.return_value.get_openai_client.return_value = sync_openai
                MockAsync.return_value.get_openai_client.return_value = async_openai

                with pytest.warns(ExperimentalWarning):
                    model = AzureAIChatCompletionsModel(
                        credential=mock_credential,
                        model="gpt-4o",
                    )

        assert model.client is sync_openai.chat.completions

    def test_project_endpoint_requires_token_credential(self) -> None:
        with pytest.raises(ValueError, match="TokenCredential"):
            with pytest.warns(ExperimentalWarning):
                AzureAIChatCompletionsModel(
                    project_endpoint=(
                        "https://resource.services.ai.azure.com/api/projects/proj"
                    ),
                    credential="api-key-string",
                )

    def test_project_endpoint_defaults_to_default_azure_credential(self) -> None:
        sync_openai = _make_mock_openai_client()
        async_openai = _make_mock_openai_client()


        with mock.patch(
            "langchain_azure_ai.chat_models.openai.AIProjectClient"
        ) as MockSync, mock.patch(
            "langchain_azure_ai.chat_models.openai.AsyncAIProjectClient"
        ) as MockAsync, mock.patch(
            "langchain_azure_ai.chat_models.openai.DefaultAzureCredential"
        ) as MockDAC:
            MockSync.return_value.get_openai_client.return_value = sync_openai
            MockAsync.return_value.get_openai_client.return_value = async_openai
            # Make the returned credential pass isinstance(..., TokenCredential)
            mock_dac = mock.MagicMock()
            mock_dac.__class__ = type("MockDAC", (TokenCredential,), {})
            MockDAC.return_value = mock_dac

            with pytest.warns(ExperimentalWarning):
                model = AzureAIChatCompletionsModel(
                    project_endpoint=(
                        "https://resource.services.ai.azure.com/api/projects/proj"
                    ),
                )

        MockDAC.assert_called_once()
        assert model.client is sync_openai.chat.completions


# ---------------------------------------------------------------------------
# AzureAIChatCompletionsModel – direct endpoint pattern
# ---------------------------------------------------------------------------


class TestAzureAIChatCompletionsModelDirectEndpoint:
    """Tests for the direct endpoint + credential configuration path."""

    def test_string_credential_maps_to_api_key(self) -> None:
        with mock.patch("openai.AzureOpenAI"):
            with pytest.warns(ExperimentalWarning):
                model = AzureAIChatCompletionsModel(
                    endpoint="https://resource.openai.azure.com/",
                    credential="my-secret-key",
                    api_version="2024-05-01-preview",
                    model="gpt-4o",
                )
        assert model.azure_endpoint == "https://resource.openai.azure.com/"

    def test_token_credential_maps_to_token_provider(self) -> None:
        mock_credential = mock.MagicMock()


        mock_credential.__class__ = type(
            "MockTokenCredential", (TokenCredential,), {}
        )

        with mock.patch("openai.AzureOpenAI"), mock.patch(
            "langchain_azure_ai.chat_models.openai._make_token_provider"
        ) as mock_tp:
            mock_tp.return_value = lambda: "token"
            with pytest.warns(ExperimentalWarning):
                model = AzureAIChatCompletionsModel(
                    endpoint="https://resource.openai.azure.com/",
                    credential=mock_credential,
                    api_version="2024-05-01-preview",
                    model="gpt-4o",
                )

        mock_tp.assert_called_once_with(mock_credential)

    def test_azure_key_credential_maps_to_api_key(self) -> None:

        with mock.patch("openai.AzureOpenAI"):
            with pytest.warns(ExperimentalWarning):
                model = AzureAIChatCompletionsModel(
                    endpoint="https://resource.openai.azure.com/",
                    credential=AzureKeyCredential("my-key"),
                    api_version="2024-05-01-preview",
                    model="gpt-4o",
                )
        assert model.azure_endpoint == "https://resource.openai.azure.com/"


# ---------------------------------------------------------------------------
# AzureAIEmbeddingsModel – project_endpoint pattern
# ---------------------------------------------------------------------------


class TestAzureAIEmbeddingsModelProjectEndpoint:
    """Tests for the project_endpoint configuration path."""

    def test_is_subclass_of_azure_openai_embeddings(self) -> None:
        assert issubclass(AzureAIEmbeddingsModel, AzureOpenAIEmbeddings)

    def test_project_endpoint_configures_clients(self) -> None:
        sync_openai = _make_mock_openai_client()
        async_openai = _make_mock_openai_client()
        mock_credential = mock.MagicMock()


        mock_credential.__class__ = type(
            "MockTokenCredential", (TokenCredential,), {}
        )

        with mock.patch(
            "langchain_azure_ai.embeddings.openai.AIProjectClient"
        ) as MockSync, mock.patch(
            "langchain_azure_ai.embeddings.openai.AsyncAIProjectClient"
        ) as MockAsync:
            MockSync.return_value.get_openai_client.return_value = sync_openai
            MockAsync.return_value.get_openai_client.return_value = async_openai

            with pytest.warns(ExperimentalWarning):
                embed_model = AzureAIEmbeddingsModel(
                    project_endpoint=(
                        "https://resource.services.ai.azure.com/api/projects/proj"
                    ),
                    credential=mock_credential,
                    model="text-embedding-3-small",
                )

        assert embed_model.client is sync_openai.embeddings
        assert embed_model.async_client is async_openai.embeddings

    def test_project_endpoint_requires_token_credential(self) -> None:
        with pytest.raises(ValueError, match="TokenCredential"):
            with pytest.warns(ExperimentalWarning):
                AzureAIEmbeddingsModel(
                    project_endpoint=(
                        "https://resource.services.ai.azure.com/api/projects/proj"
                    ),
                    credential="api-key-string",
                )

    def test_project_endpoint_defaults_to_default_azure_credential(self) -> None:
        sync_openai = _make_mock_openai_client()
        async_openai = _make_mock_openai_client()


        with mock.patch(
            "langchain_azure_ai.embeddings.openai.AIProjectClient"
        ) as MockSync, mock.patch(
            "langchain_azure_ai.embeddings.openai.AsyncAIProjectClient"
        ) as MockAsync, mock.patch(
            "langchain_azure_ai.embeddings.openai.DefaultAzureCredential"
        ) as MockDAC:
            MockSync.return_value.get_openai_client.return_value = sync_openai
            MockAsync.return_value.get_openai_client.return_value = async_openai
            # Make the returned credential pass isinstance(..., TokenCredential)
            mock_dac = mock.MagicMock()
            mock_dac.__class__ = type("MockDAC", (TokenCredential,), {})
            MockDAC.return_value = mock_dac

            with pytest.warns(ExperimentalWarning):
                embed_model = AzureAIEmbeddingsModel(
                    project_endpoint=(
                        "https://resource.services.ai.azure.com/api/projects/proj"
                    ),
                )

        MockDAC.assert_called_once()
        assert embed_model.client is sync_openai.embeddings


# ---------------------------------------------------------------------------
# AzureAIEmbeddingsModel – direct endpoint pattern
# ---------------------------------------------------------------------------


class TestAzureAIEmbeddingsModelDirectEndpoint:
    """Tests for the direct endpoint + credential configuration path."""

    def test_string_credential_maps_to_api_key(self) -> None:
        with mock.patch("openai.AzureOpenAI"):
            with pytest.warns(ExperimentalWarning):
                embed_model = AzureAIEmbeddingsModel(
                    endpoint="https://resource.openai.azure.com/",
                    credential="my-secret-key",
                    api_version="2024-05-01-preview",
                    model="text-embedding-3-small",
                )
        assert embed_model.azure_endpoint == "https://resource.openai.azure.com/"

    def test_azure_key_credential_maps_to_api_key(self) -> None:

        with mock.patch("openai.AzureOpenAI"):
            with pytest.warns(ExperimentalWarning):
                embed_model = AzureAIEmbeddingsModel(
                    endpoint="https://resource.openai.azure.com/",
                    credential=AzureKeyCredential("my-key"),
                    api_version="2024-05-01-preview",
                    model="text-embedding-3-small",
                )
        assert embed_model.azure_endpoint == "https://resource.openai.azure.com/"


# ---------------------------------------------------------------------------
# Deprecation tests for the inference-based classes
# ---------------------------------------------------------------------------


class TestInferenceClassDeprecation:
    """Ensure the old inference-based classes emit DeprecationWarning."""

    def test_chat_model_emits_deprecation_warning(self) -> None:
        with mock.patch(
            "langchain_azure_ai.chat_models.inference.ChatCompletionsClient",
            autospec=True,
        ), mock.patch(
            "langchain_azure_ai.chat_models.inference.ChatCompletionsClientAsync",
            autospec=True,
        ):
            with pytest.warns(DeprecationWarning, match="deprecated"):
                from langchain_azure_ai.chat_models.inference import (
                    AzureAIChatCompletionsModel as OldChatModel,
                )

                OldChatModel(
                    endpoint="https://my-endpoint.inference.ai.azure.com",
                    credential="my-api-key",
                )

    def test_embeddings_model_emits_deprecation_warning(self) -> None:
        with mock.patch(
            "langchain_azure_ai.embeddings.inference.EmbeddingsClient",
            autospec=True,
        ):
            with pytest.warns(DeprecationWarning, match="deprecated"):
                from langchain_azure_ai.embeddings.inference import (
                    AzureAIEmbeddingsModel as OldEmbedModel,
                )

                OldEmbedModel(
                    endpoint="https://my-endpoint.inference.ai.azure.com",
                    credential="my-api-key",
                    model="cohere-embed-v3-multilingual",
                )


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


def test_chat_models_package_exports_new_class() -> None:
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel as Exported

    from langchain_azure_ai.chat_models.openai import (
        AzureAIChatCompletionsModel as New,
    )

    assert Exported is New


def test_embeddings_package_exports_new_class() -> None:
    from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel as Exported

    from langchain_azure_ai.embeddings.openai import AzureAIEmbeddingsModel as New

    assert Exported is New


def test_chat_models_package_still_exports_old_class() -> None:
    from langchain_azure_ai.chat_models import (
        AzureAIInferenceChatCompletionsModel as Old,
    )

    from langchain_azure_ai.chat_models.inference import (
        AzureAIChatCompletionsModel as OriginalOld,
    )

    assert Old is OriginalOld


def test_embeddings_package_still_exports_old_class() -> None:
    from langchain_azure_ai.embeddings import (
        AzureAIInferenceEmbeddingsModel as Old,
    )

    from langchain_azure_ai.embeddings.inference import (
        AzureAIEmbeddingsModel as OriginalOld,
    )

    assert Old is OriginalOld

