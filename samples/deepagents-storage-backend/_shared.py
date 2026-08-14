"""Setup helpers shared by the samples in this directory.

Every sample needs the same three things: a chat model, an
:class:`~langchain_azure_storage.deepagents.AzureBlobBackend`, and a container
to put the workspace in. All three are configured entirely through environment
variables so the samples run unchanged against a live Azure Storage account or
against the Azurite emulator (see README.md).
"""

import os
from typing import Union

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from langchain_core.language_models import BaseChatModel

from langchain_azure_storage.deepagents import AzureBlobBackend

CONTAINER_NAME = "agent-workspace"
"""Container the samples create (or reuse, if it already exists)."""

MODEL_ENV_VAR = "MODEL_NAME"

_AZURE_AI_ENDPOINT_ENV_VARS = (
    "AZURE_AI_PROJECT_ENDPOINT",
    "AZURE_AI_OPENAI_ENDPOINT",
    "AZURE_OPENAI_ENDPOINT",
)


def build_model() -> Union[str, BaseChatModel]:
    """Build the chat model the samples pass to ``create_deep_agent``.

    ``MODEL_NAME`` selects the model. If any Azure AI endpoint environment
    variable is also set, the model is built with ``langchain-azure-ai`` so the
    samples can run entirely on Azure; otherwise ``MODEL_NAME`` is returned as a
    ``provider:model`` string for ``init_chat_model`` to resolve.

    Returns:
        Either a ``provider:model`` string or an instantiated chat model.

    Raises:
        RuntimeError: If ``MODEL_NAME`` is not set.
    """
    model_name = os.environ.get(MODEL_ENV_VAR)
    if not model_name:
        raise RuntimeError(
            f"Set {MODEL_ENV_VAR} to the model to use — either a "
            f"'provider:model' identifier (e.g. 'anthropic:claude-sonnet-4-6') "
            f"or, when running on Azure AI, your model deployment name."
        )

    if any(os.environ.get(name) for name in _AZURE_AI_ENDPOINT_ENV_VARS):
        # Imported lazily so the samples run without langchain-azure-ai
        # configured when a non-Azure model is used.
        from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

        return AzureAIOpenAIApiChatModel(
            credential=DefaultAzureCredential(),
            model=model_name,
        )

    return model_name


def build_backend(prefix: str) -> AzureBlobBackend:
    """Build a backend rooted at ``prefix`` inside the samples' container.

    Args:
        prefix: Blob name prefix isolating one sample's workspace.

    Returns:
        A backend authenticated from the environment.

    Raises:
        RuntimeError: If no storage environment variable is set.
    """
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return AzureBlobBackend.from_connection_string(
            connection_string, CONTAINER_NAME, prefix=prefix
        )

    account_url = os.environ.get("AZURE_STORAGE_ACCOUNT_URL")
    if not account_url:
        raise RuntimeError(
            "Set AZURE_STORAGE_ACCOUNT_URL to your storage account (or "
            "AZURE_STORAGE_CONNECTION_STRING to use the Azurite emulator)"
        )
    return AzureBlobBackend(
        account_url=account_url, container_name=CONTAINER_NAME, prefix=prefix
    )


def ensure_container() -> str:
    """Create the samples' container if it does not already exist.

    Returns:
        The container URL, used to print where each file landed.
    """
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        service = BlobServiceClient.from_connection_string(connection_string)
    else:
        service = BlobServiceClient(
            os.environ["AZURE_STORAGE_ACCOUNT_URL"],
            credential=DefaultAzureCredential(),
        )
    with service:
        container = service.get_container_client(CONTAINER_NAME)
        if not container.exists():
            container.create_container()
        return str(container.url)
