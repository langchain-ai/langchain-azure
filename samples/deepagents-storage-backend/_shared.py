"""Setup helpers shared by the samples in this directory.

Every sample needs the same three things: a chat model, an
:class:`~langchain_azure_storage.deepagents.AzureBlobBackend`, and a container
to put the workspace in. All three are configured entirely through environment
variables so the samples run unchanged against a live Azure Storage account or
against the Azurite emulator (see README.md).
"""

import os
from collections.abc import Mapping
from typing import Any

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain.chat_models import init_chat_model
from langchain_azure_storage.deepagents import AzureBlobBackend
from langchain_core.language_models import BaseChatModel

CONTAINER_NAME = "agent-workspace"
"""Container the samples create (or reuse, if it already exists)."""

MODEL_ENV_VAR = "MODEL_NAME"

_AZURE_AI_ENDPOINT_ENV_VARS = (
    "AZURE_AI_PROJECT_ENDPOINT",
    "AZURE_AI_OPENAI_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT",
)


def build_model(
    *,
    credential: TokenCredential | None = None,
    model_name: str | None = None,
    model_options: Mapping[str, Any] | None = None,
) -> BaseChatModel:
    """Build the chat model the samples pass to ``create_deep_agent``.

    ``MODEL_NAME`` selects the model. If any Azure AI endpoint environment
    variable is also set, the model is built with ``langchain-azure-ai`` so the
    samples can run entirely on Azure; otherwise ``init_chat_model`` builds the
    provider-qualified model with the same options.

    Returns:
        An instantiated chat model.

    Raises:
        RuntimeError: If ``MODEL_NAME`` is not set.
    """
    model_name = model_name or os.environ.get(MODEL_ENV_VAR)
    if not model_name:
        raise RuntimeError(
            f"Set {MODEL_ENV_VAR} to the model to use — either a "
            f"'provider:model' identifier (e.g. 'anthropic:claude-sonnet-4-6') "
            f"or, when running on Azure AI, your model deployment name."
        )

    if any(os.environ.get(name) for name in _AZURE_AI_ENDPOINT_ENV_VARS):
        # Imported lazily so the samples run without langchain-azure-ai
        # configured when a non-Azure model is used.
        from langchain_azure_ai.chat_models import (  # pyright: ignore[reportMissingImports]
            AzureAIOpenAIApiChatModel,
        )

        options = dict(model_options or {})
        if credential is not None:
            options["credential"] = credential
        elif not os.environ.get("AZURE_AI_PROJECT_ENDPOINT"):
            options["credential"] = DefaultAzureCredential()
        return AzureAIOpenAIApiChatModel(model=model_name, **options)

    return init_chat_model(model_name, **dict(model_options or {}))


def build_blob_backend(
    prefix: str | None = None,
    *,
    container_name: str = CONTAINER_NAME,
    credential: TokenCredential | None = None,
    account_url: str | None = None,
    connection_string: str | None = None,
) -> AzureBlobBackend:
    """Build a backend optionally rooted at a prefix inside a container.

    Args:
        prefix: Optional Blob name prefix isolating one sample's workspace.

    Returns:
        A backend authenticated from the environment.

    Raises:
        RuntimeError: If no storage environment variable is set.
    """
    connection_string = connection_string or os.environ.get(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    if connection_string:
        return AzureBlobBackend.from_connection_string(
            connection_string, container_name, prefix=prefix
        )

    account_url = account_url or os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    if not account_url:
        raise RuntimeError(
            "Set AZURE_STORAGE_ACCOUNT_URL to your storage account (or "
            "AZURE_STORAGE_CONNECTION_STRING to use the Azurite emulator)"
        )
    return AzureBlobBackend(
        account_url=account_url,
        container_name=container_name,
        prefix=prefix,
        credential=credential,
    )


def ensure_container(
    container_name: str = CONTAINER_NAME,
    *,
    account_url: str | None = None,
    connection_string: str | None = None,
) -> str:
    """Create the requested container if it does not already exist."""
    connection_string = connection_string or os.environ.get(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        service = BlobServiceClient(
            account_url or os.environ["AZURE_STORAGE_ACCOUNT_URL"],
            credential=DefaultAzureCredential(),
        )
    with service:
        container = service.get_container_client(container_name)
        if not container.exists():
            container.create_container()
        return str(container.url)
