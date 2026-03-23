"""Azure AI Chat Completions model using the OpenAI-compatible API."""

import logging
from typing import Any, Dict, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict, Field, model_validator

from langchain_azure_ai._resources import _configure_openai_credential_values

logger = logging.getLogger(__name__)


class AzureAIOpenAIApiChatModel(ChatOpenAI):
    """Azure AI chat model using the OpenAI-compatible API.

    This class wraps :class:`langchain_openai.ChatOpenAI` and adds support
    for the *project-endpoint pattern* available in Azure AI Foundry, in addition
    to the classic *endpoint + API-key* style used by OpenAI-compatible services.

    Use `AzureAIOpenAIApiChatModel` with any Foundry model compatible with
    OpenAI APIs (e.g. gpt-5, Mistral, Cohere.) to get the benefits of
    unified authentication, single configuration, and seamless integration
    with other Azure services.

    By default, this class uses Responses API. Set `use_responses_api=False`
    to disable it and use the standard chat completions API instead.

    **Project-endpoint pattern (recommended for Azure AI Foundry):**

    ```python
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
    from azure.identity import DefaultAzureCredential

    model = AzureAIOpenAIApiChatModel(
        project_endpoint=(
            "https://resource.services.ai.azure.com/api/projects/my-project"
        ),
        credential=DefaultAzureCredential(),
        model="gpt-4o",
    )
    ```

    Parameter `model` refers to the model deployment name in Azure AI Foundry,
    which may differ from the base model name (e.g. "gpt-4o") depending on how
    the deployment was configured.

    If ``project_endpoint`` is omitted the value of the
    ``AZURE_AI_PROJECT_ENDPOINT`` environment variable is used.

    **Direct endpoint + API-key pattern:**

    ```python
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

    model = AzureAIOpenAIApiChatModel(
        endpoint="https://resource.services.ai.azure.com/openai/v1",
        credential="your-api-key",
        model="gpt-4o",
    )
    ```

    Use `api_version` parameter to specify the API version when using a
    direct endpoint, or rely on the automatic API version detection
    when using a project endpoint.

    ```python
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

    model = AzureAIOpenAIApiChatModel(
        endpoint="https://resource.services.ai.azure.com/openai/v1",
        credential="your-api-key",
        model="gpt-4o",
        api_version="2025-05-12"
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

    credential: Optional[
        Union[str, AzureKeyCredential, TokenCredential, AsyncTokenCredential]
    ] = Field(default=None)
    """Credential for authentication.

    * A plain ``str`` or :class:`~azure.core.credentials.AzureKeyCredential`
      is treated as an API key.
    * A :class:`~azure.core.credentials.TokenCredential` (e.g.
      :class:`~azure.identity.DefaultAzureCredential`) or
      :class:`~azure.core.credentials_async.AsyncTokenCredential` (e.g.
      :class:`~azure.identity.aio.DefaultAzureCredential`) is used as a callable
      token provider so that tokens are refreshed automatically.
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
            values["use_responses_api"] = values.get("use_responses_api", True)

        return values

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Bind tools to this chat model.

        Extends :meth:`~langchain_openai.ChatOpenAI.bind_tools` to
        automatically collect any extra HTTP request headers declared by
        :class:`~langchain_azure_ai.tools.builtin.BuiltinTool` instances
        and forward them to the underlying OpenAI client via
        ``extra_headers``.

        Args:
            tools: A list of tool definitions.  Instances of
                :class:`~langchain_azure_ai.tools.builtin.BuiltinTool`
                are inspected for
                :attr:`~langchain_azure_ai.tools.builtin.BuiltinTool.request_headers`;
                all non-empty header maps are merged and passed as
                ``extra_headers``.
            **kwargs: Forwarded to
                :meth:`~langchain_openai.ChatOpenAI.bind_tools`.  Pass
                ``extra_headers`` here to merge with tool-defined headers
                (caller values take precedence).
        """
        from langchain_azure_ai.tools.builtin import BuiltinTool

        request_headers: Dict[str, str] = {}
        for tool in tools:
            if isinstance(tool, BuiltinTool):
                request_headers.update(tool.request_headers)
        if request_headers:
            existing: Dict[str, str] = kwargs.pop("extra_headers", {}) or {}
            kwargs["extra_headers"] = {**request_headers, **existing}

        return super().bind_tools(tools, **kwargs)

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:  # type: ignore[type-arg]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        # Azure AI Foundry's Responses API requires an explicit
        # ``type: "message"`` on every input item.  The upstream
        # ``_construct_responses_api_input`` (langchain-openai) produces
        # ``EasyInputMessageParam``-style dicts *without* ``type``, which
        # OpenAI's native endpoint accepts but Azure rejects with a 400.
        if self._use_responses_api(payload) and "input" in payload:
            for item in payload["input"]:
                if isinstance(item, dict) and "type" not in item and "role" in item:
                    item["type"] = "message"
        return payload
