"""Azure AI embeddings model using the OpenAI-compatible API."""

import logging
from typing import Any, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from langchain_openai import OpenAIEmbeddings
from pydantic import ConfigDict, Field, model_validator

from langchain_azure_ai._resources import _configure_openai_credential_values

logger = logging.getLogger(__name__)


class AzureAIOpenAIApiEmbeddingsModel(OpenAIEmbeddings):
    """Azure AI embeddings model using the OpenAI-compatible API.

    This class wraps :class:`langchain_openai.OpenAIEmbeddings` and adds
    support for the *project-endpoint pattern* available in Azure AI Foundry,
    in addition to the classic *endpoint + API-key* style used by Azure OpenAI.

    Use `AzureAIOpenAIApiEmbeddingsModel` with any Foundry model compatible with
    OpenAI APIs (e.g. gpt-5, Mistral, Cohere.) to get the benefits of
    unified authentication, single configuration, and seamless integration
    with other Azure services.

    **Project-endpoint pattern (recommended for Azure AI Foundry):**

    ```python
    from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel
    from azure.identity import DefaultAzureCredential

    embed_model = AzureAIOpenAIApiEmbeddingsModel(
        project_endpoint=(
            "https://resource.services.ai.azure.com/api/projects/my-project"
        ),
        credential=DefaultAzureCredential(),
        model="text-embedding-3-small",
    )
    ```

    Parameter `model` refers to the model deployment name in Azure AI Foundry,
    which may differ from the base model name depending on how the deployment
    was configured.

    If ``project_endpoint`` is omitted the value of the
    ``AZURE_AI_PROJECT_ENDPOINT`` environment variable is used.

    **Direct endpoint + API-key pattern:**

    ```python
    from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel

    embed_model = AzureAIOpenAIApiEmbeddingsModel(
        endpoint="https://resource.services.ai.azure.com/openai/v1",
        credential="your-api-key",
        model="text-embedding-3-small",
        api_version="2024-05-01-preview",
    )
    ```

    **Environment variables:**

    The following environment variables are recognised as fallbacks when the
    corresponding constructor parameters are not provided:

    * ``AZURE_AI_PROJECT_ENDPOINT`` – used as ``project_endpoint``.
    * ``AZURE_AI_OPENAI_ENDPOINT`` – direct OpenAI-compatible endpoint
      (e.g. ``https://<resource>.services.ai.azure.com/openai/v1``).
      Used as ``endpoint`` verbatim (no path is appended).
    * ``AZURE_OPENAI_ENDPOINT`` – root Azure OpenAI endpoint (e.g.
      ``https://<resource>.services.ai.azure.com``).  ``/openai/v1`` is
      appended automatically and the result is treated as ``endpoint``.
    * ``AZURE_OPENAI_DEPLOYMENT_NAME`` – model deployment name (``model``).
    * ``AZURE_OPENAI_API_VERSION`` – API version passed as the
      ``api-version`` query parameter on every request.

    **Resolution priority** (highest → lowest):

    1. Constructor parameters.
    2. ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
    3. ``AZURE_AI_OPENAI_ENDPOINT`` environment variable.
    4. ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_VERSION`` /
       ``AZURE_OPENAI_DEPLOYMENT_NAME`` environment variables.

    ``AZURE_AI_PROJECT_ENDPOINT``, ``AZURE_AI_OPENAI_ENDPOINT``, and
    ``AZURE_OPENAI_ENDPOINT`` may all be set at the same time; the project
    endpoint takes precedence, then ``AZURE_AI_OPENAI_ENDPOINT``, then
    ``AZURE_OPENAI_ENDPOINT``.  However, passing both ``project_endpoint``
    *and* ``endpoint`` as constructor parameters raises :class:`ValueError`.

    All other keyword arguments accepted by
    :class:`langchain_openai.AzureOpenAIEmbeddings` are forwarded as-is.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="ignore",
        protected_namespaces=(),
        populate_by_name=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    project_endpoint: Optional[str] = Field(default=None)
    """Azure AI Foundry project endpoint.  When provided the model is
    configured automatically via :class:`azure.ai.projects.AIProjectClient`.
    Overrides the ``AZURE_AI_PROJECT_ENDPOINT`` environment variable."""

    endpoint: Optional[str] = Field(default=None)
    """Direct Azure OpenAI endpoint (e.g.
    ``https://resource.services.ai.azure.com/openai/v1``).
    Used when ``project_endpoint`` is *not* provided."""

    credential: Optional[
        Union[str, AzureKeyCredential, TokenCredential, AsyncTokenCredential]
    ] = Field(default=None)
    """Credential for authentication.

    * A plain ``str`` or :class:`~azure.core.credentials.AzureKeyCredential`
      is treated as an API key.
    * A :class:`~azure.core.credentials.TokenCredential` (e.g.
      :class:`~azure.identity.DefaultAzureCredential`) is used with
      ``azure_ad_token_provider``.
    * A :class:`~azure.core.credentials_async.AsyncTokenCredential` (e.g.
      :class:`~azure.identity.aio.DefaultAzureCredential`) is used with
      ``azure_ad_token_provider`` for asynchronous operations.
    * ``None`` (default) falls back to
      :class:`~azure.identity.DefaultAzureCredential` when
      ``project_endpoint`` is used, or raises an error otherwise.
    """

    api_version: Optional[str] = Field(default=None)
    """API version to pass as the ``api-version`` query parameter on every
    request.  When omitted, falls back to the ``AZURE_OPENAI_API_VERSION``
    environment variable.  Only used when the helper constructs OpenAI
    clients directly (i.e. when ``endpoint`` is provided together with a
    ``credential``)."""

    @model_validator(mode="before")
    @classmethod
    def _configure_clients(cls, values: Any) -> Any:
        """Resolve project-endpoint or direct-endpoint credentials.

        When ``project_endpoint`` is provided (or available via the
        ``AZURE_AI_PROJECT_ENDPOINT`` env var) the method uses
        :class:`azure.ai.projects.AIProjectClient` to obtain pre-configured
        synchronous and asynchronous OpenAI clients, then injects them into
        the ``client`` / ``async_client`` fields so that
        :meth:`AzureOpenAIEmbeddings.validate_environment` does not attempt to
        create new Azure OpenAI clients (which would require an
        ``api_version``).

        When ``endpoint`` is provided the method maps it to
        ``azure_endpoint`` and translates ``credential`` to either
        ``api_key`` or ``azure_ad_token_provider``.
        """
        if not isinstance(values, dict):
            return values

        values, openai_clients = _configure_openai_credential_values(values)

        if openai_clients is not None:
            sync_openai, async_openai = openai_clients
            # Pre-populate the client fields. OpenAIEmbeddings.validate_environment
            # skips creating a new openai.OpenAI when these are set,
            # which avoids the mandatory api_version requirement.
            values["client"] = sync_openai.embeddings
            values["async_client"] = async_openai.embeddings

        return values
