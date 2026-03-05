"""Azure AI Chat Completions model using the OpenAI-compatible API."""

import logging
import os
from typing import Any, Callable, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
from pydantic import ConfigDict, Field, model_validator

from langchain_azure_ai._api.base import experimental

logger = logging.getLogger(__name__)


@experimental()
class AzureAIChatCompletionsModel(AzureChatOpenAI):
    """Azure AI chat completions model using the OpenAI-compatible API.

    This class wraps :class:`langchain_openai.AzureChatOpenAI` and adds support
    for the *project-endpoint pattern* available in Azure AI Foundry, in addition
    to the classic *endpoint + API-key* style used by Azure OpenAI.

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
        endpoint="https://resource.openai.azure.com/",
        credential="your-api-key",
        model="gpt-4o",
        api_version="2024-05-01-preview",
    )
    ```

    All other keyword arguments accepted by
    :class:`langchain_openai.AzureChatOpenAI` are forwarded as-is, so you
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
    """Direct Azure OpenAI endpoint (e.g.
    ``https://resource.openai.azure.com/``).  Used when ``project_endpoint``
    is *not* provided."""

    credential: Optional[Union[str, AzureKeyCredential, TokenCredential]] = Field(
        default=None
    )
    """Credential for authentication.

    * A plain ``str`` or :class:`~azure.core.credentials.AzureKeyCredential`
      is treated as an API key.
    * A :class:`~azure.core.credentials.TokenCredential` (e.g.
      :class:`~azure.identity.DefaultAzureCredential`) is used with
      ``azure_ad_token_provider``.
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
        synchronous and asynchronous OpenAI clients, then injects them into
        the ``client`` / ``async_client`` fields so that
        :meth:`AzureChatOpenAI.validate_environment` does not attempt to
        create new Azure OpenAI clients (which would require an
        ``api_version``).

        When ``endpoint`` is provided the method maps it to
        ``azure_endpoint`` and translates ``credential`` to either
        ``api_key`` or ``azure_ad_token_provider``.
        """
        if not isinstance(values, dict):
            return values

        project_endpoint = values.get("project_endpoint") or os.environ.get(
            "AZURE_AI_PROJECT_ENDPOINT"
        )
        endpoint = values.get("endpoint")
        credential = values.get("credential")

        if project_endpoint:
            try:
                from azure.ai.projects import AIProjectClient
                from azure.ai.projects.aio import (
                    AIProjectClient as AsyncAIProjectClient,
                )
            except ImportError as exc:
                raise ImportError(
                    "The `azure-ai-projects` package is required when using "
                    "`project_endpoint`. Install it with "
                    "`pip install azure-ai-projects`."
                ) from exc

            if credential is None:
                logger.warning(
                    "No credential provided, using DefaultAzureCredential(). "
                    "If intentional, pass `credential=DefaultAzureCredential()`."
                )
                credential = DefaultAzureCredential()

            if not isinstance(credential, TokenCredential):
                raise ValueError(
                    "When using `project_endpoint` the `credential` must be "
                    "a `TokenCredential` (e.g. `DefaultAzureCredential()`)."
                )

            sync_project = AIProjectClient(
                endpoint=project_endpoint, credential=credential
            )
            async_project = AsyncAIProjectClient(
                endpoint=project_endpoint, credential=credential
            )

            sync_openai = sync_project.get_openai_client()
            async_openai = async_project.get_openai_client()

            # Pre-populate the client fields.  AzureChatOpenAI.validate_environment
            # skips creating a new openai.AzureOpenAI when these are set,
            # which avoids the mandatory api_version requirement.
            values["client"] = sync_openai.chat.completions
            values["async_client"] = async_openai.chat.completions
            values["root_client"] = sync_openai
            values["root_async_client"] = async_openai

            # Propagate the project_endpoint so the field is stored.
            values["project_endpoint"] = project_endpoint

        elif endpoint:
            values["azure_endpoint"] = endpoint

            if isinstance(credential, (str, AzureKeyCredential)):
                api_key = (
                    credential
                    if isinstance(credential, str)
                    else credential.key
                )
                values["api_key"] = api_key
            elif isinstance(credential, TokenCredential):
                values["azure_ad_token_provider"] = _make_token_provider(credential)

        return values


def _make_token_provider(credential: TokenCredential) -> Callable[[], str]:
    """Return a bearer-token provider callable for the given credential."""
    try:
        from azure.identity import get_bearer_token_provider
    except ImportError as exc:
        raise ImportError(
            "`azure-identity` is required. Install with `pip install azure-identity`."
        ) from exc

    return get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )

