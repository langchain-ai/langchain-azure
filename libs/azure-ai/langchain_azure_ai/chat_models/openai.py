"""Azure AI Chat Completions model using the OpenAI-compatible API."""

import logging
from typing import Any, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict, Field, model_validator

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai._resources import _configure_openai_credential_values

logger = logging.getLogger(__name__)


@experimental()
class AzureAIChatCompletionsModel(ChatOpenAI):
    """Azure AI chat completions model using the OpenAI-compatible API.

    This class wraps :class:`langchain_openai.ChatOpenAI` and adds support
    for the *project-endpoint pattern* available in Azure AI Foundry, in addition
    to the classic *endpoint + API-key* style used by OpenAI-compatible services.

    **Project-endpoint pattern (recommended for Azure AI Foundry):**

    ```python
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
    from azure.identity import DefaultAzureCredential

    model = AzureAIChatCompletionsModel(
        project_endpoint=(
            "https://resource.services.ai.azure.com/api/projects/my-project"
        ),
        credential=DefaultAzureCredential(),
        model="gpt-4o",
    )
    ```

    If ``project_endpoint`` is omitted the value of the
    ``AZURE_AI_PROJECT_ENDPOINT`` environment variable is used.

    **Direct endpoint + API-key pattern:**

    ```python
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

    model = AzureAIChatCompletionsModel(
        endpoint="https://resource.services.ai.azure.com/openai/v1",
        credential="your-api-key",
        model="gpt-4o",
    )
    ```

    All other keyword arguments accepted by
    :class:`langchain_openai.ChatOpenAI` are forwarded as-is, so you
    retain full control over temperature, max_tokens, streaming, etc.
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
    """Direct OpenAI-compatible endpoint used as the ``base_url`` for the
    underlying OpenAI client (e.g.
    ``https://resource.services.ai.azure.com/openai/v1``).  Used when
    ``project_endpoint`` is *not* provided."""

    credential: Optional[Union[str, AzureKeyCredential, TokenCredential]] = Field(
        default=None
    )
    """Credential for authentication.

    * A plain ``str`` or :class:`~azure.core.credentials.AzureKeyCredential`
      is treated as an API key.
    * A :class:`~azure.core.credentials.TokenCredential` (e.g.
      :class:`~azure.identity.DefaultAzureCredential`) is used as a callable
      token provider so that tokens are refreshed automatically.
    * ``None`` (default) falls back to
      :class:`~azure.identity.DefaultAzureCredential` when
      ``project_endpoint`` is used, or raises an error otherwise.
    """

    @model_validator(mode="before")
    @classmethod
    def _configure_clients(cls, values: Any) -> Any:
        """Resolve project-endpoint or direct-endpoint credentials.

        When ``project_endpoint`` is provided (or available via the
        ``AZURE_AI_PROJECT_ENDPOINT`` env var) the method uses
        :class:`azure.ai.projects.AIProjectClient` to obtain pre-configured
        synchronous and asynchronous OpenAI clients via
        :meth:`~azure.ai.projects.AIProjectClient.get_openai_client`, then
        injects them into the ``client`` / ``async_client`` fields so that
        :meth:`ChatOpenAI.validate_environment` does not attempt to create
        new clients.

        When ``endpoint`` is provided the method maps it to
        ``openai_api_base`` (i.e. ``base_url``) and translates ``credential``
        to either ``api_key`` or a callable ``openai_api_key`` (for
        token-based auth).
        """
        if not isinstance(values, dict):
            return values

        values, openai_clients = _configure_openai_credential_values(values)

        if openai_clients is not None:
            sync_openai, async_openai = openai_clients
            # Pre-populate the client fields. ChatOpenAI.validate_environment
            # skips creating a new openai.OpenAI when these are set.
            values["client"] = sync_openai.chat.completions
            values["async_client"] = async_openai.chat.completions
            values["root_client"] = sync_openai
            values["root_async_client"] = async_openai

        return values

