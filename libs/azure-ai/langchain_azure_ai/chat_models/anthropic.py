"""Azure AI chat model using the Anthropic Messages API via Azure AI Foundry."""

import logging
import os
from functools import cached_property
from typing import Any, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_azure_ai._resources import (
    _DEFAULT_FOUNDRY_SCOPE,
    _get_base_url_from_endpoint,
    _make_async_token_provider,
    _make_token_provider,
)
from langchain_azure_ai.utils.env import get_project_endpoint

try:
    from anthropic import AnthropicFoundry, AsyncAnthropicFoundry
    from langchain_anthropic import ChatAnthropic
except ImportError as exc:  # pragma: no cover - exercised via lazy import
    raise ImportError(
        "`AzureAIAnthropicChatModel` requires the optional "
        "`langchain-anthropic` and `anthropic` packages. "
        "Install them with `pip install anthropic langchain-anthropic`."
    ) from exc

logger = logging.getLogger(__name__)

# Environment variable holding the Azure AI Foundry resource name or full URL.
# When only a resource name is provided (no ``://``), the endpoint is
# synthesized as ``https://<resource>.services.ai.azure.com``.
_ANTHROPIC_FOUNDRY_RESOURCE_ENV_VAR = "ANTHROPIC_FOUNDRY_RESOURCE"


def _resolve_anthropic_endpoint(endpoint: Optional[str]) -> str:
    """Resolve the Anthropic Foundry ``base_url`` from an endpoint value.

    Accepts the Foundry resource root (e.g.
    ``https://<resource>.services.ai.azure.com``) or a URL that already
    includes the ``/anthropic`` path; in both cases the returned value ends
    with ``/anthropic/``.

    When *endpoint* is ``None``, falls back to the
    ``ANTHROPIC_FOUNDRY_RESOURCE`` environment variable, which holds a bare
    Foundry resource name (e.g. ``my-resource``).  The endpoint is then
    synthesized as
    ``https://<ANTHROPIC_FOUNDRY_RESOURCE>.services.ai.azure.com/anthropic/``.
    """
    if not endpoint:
        resource = os.environ.get(_ANTHROPIC_FOUNDRY_RESOURCE_ENV_VAR, "").strip()
        if not resource:
            raise ValueError(
                "An `endpoint` must be provided, or the "
                f"`{_ANTHROPIC_FOUNDRY_RESOURCE_ENV_VAR}` environment variable "
                "must be set to the Foundry resource name."
            )
        endpoint = f"https://{resource}.services.ai.azure.com"

    stripped = endpoint.rstrip("/")
    if stripped.endswith("/anthropic"):
        return stripped + "/"
    return stripped + "/anthropic/"


def _extract_api_key(credential: Union[str, AzureKeyCredential]) -> str:
    """Return the raw API key from a string or :class:`AzureKeyCredential`."""
    if isinstance(credential, str):
        return credential
    return credential.key


class AzureAIAnthropicChatModel(ChatAnthropic):
    """Azure AI chat model using the Anthropic Messages API.

    Wraps :class:`langchain_anthropic.ChatAnthropic` so that Anthropic
    Claude models hosted by Azure AI Foundry can be accessed through the
    ``/anthropic`` endpoint with native Microsoft Entra ID authentication.

    Authentication uses :class:`anthropic.AnthropicFoundry` (and
    :class:`anthropic.AsyncAnthropicFoundry`) under the hood, which accepts
    an ``azure_ad_token_provider`` callable that is invoked on every
    request – so bearer tokens are refreshed automatically and the model
    can be used in long-running services without manual token rotation.

    **Microsoft Entra ID authentication (recommended):**

    ```python
    from langchain_azure_ai.chat_models import AzureAIAnthropicChatModel
    from azure.identity import DefaultAzureCredential

    model = AzureAIAnthropicChatModel(
        endpoint="https://<resource>.services.ai.azure.com",
        credential=DefaultAzureCredential(),
        model="claude-sonnet-4-20250514",
    )
    ```

    The ``endpoint`` is the Azure AI Foundry resource root; the
    ``/anthropic`` path is appended automatically.  You may also pass a URL
    that already ends in ``/anthropic`` – the result is the same.

    Alternatively, supply a Foundry *project* endpoint and the resource
    root is derived automatically:

    ```python
    model = AzureAIAnthropicChatModel(
        project_endpoint=(
            "https://<resource>.services.ai.azure.com/api/projects/my-project"
        ),
        credential=DefaultAzureCredential(),
        model="claude-sonnet-4-20250514",
    )
    ```

    ``project_endpoint`` and ``endpoint`` are mutually exclusive.

    **API-key authentication:**

    ```python
    model = AzureAIAnthropicChatModel(
        endpoint="https://<resource>.services.ai.azure.com",
        credential="your-api-key",
        model="claude-sonnet-4-20250514",
    )
    ```

    **Environment variables:**

    The following environment variables are checked when the corresponding
    constructor parameters are not provided:

    * ``AZURE_AI_PROJECT_ENDPOINT`` or ``FOUNDRY_PROJECT_ENDPOINT`` – resolved
      as ``project_endpoint``.  ``AZURE_AI_PROJECT_ENDPOINT`` takes precedence
      when both are set.
    * ``ANTHROPIC_FOUNDRY_RESOURCE`` – resolved as ``endpoint`` when neither
      ``endpoint`` nor ``project_endpoint`` is set (directly or via env var).
      May be a bare resource name (e.g. ``my-resource``) or a full URL.  When
      a bare name is provided the endpoint is synthesized as
      ``https://<ANTHROPIC_FOUNDRY_RESOURCE>.services.ai.azure.com``.
      This is the same variable used by **Claude Code** when connected to
      Azure AI Foundry, so models configured that way will work here without
      any additional configuration.

    **Resolution priority** (highest to lowest):

    1. Constructor parameters (``project_endpoint``, ``endpoint``).
    2. ``AZURE_AI_PROJECT_ENDPOINT`` environment variable (or
       ``FOUNDRY_PROJECT_ENDPOINT`` if the former is not set).
    3. ``ANTHROPIC_FOUNDRY_RESOURCE`` environment variable.

    All other keyword arguments accepted by
    :class:`langchain_anthropic.ChatAnthropic` are forwarded as-is.
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
    """Azure AI Foundry project endpoint, e.g.
    ``https://<resource>.services.ai.azure.com/api/projects/<project>``.
    The resource root is extracted automatically and ``/anthropic`` is
    appended.  Mutually exclusive with ``endpoint``.

    When omitted, falls back to the ``AZURE_AI_PROJECT_ENDPOINT`` environment
    variable (or ``FOUNDRY_PROJECT_ENDPOINT`` as a secondary alias)."""

    endpoint: Optional[str] = Field(default=None)
    """Azure AI Foundry resource endpoint, e.g.
    ``https://<resource>.services.ai.azure.com``.  ``/anthropic`` is
    appended automatically.  When omitted, falls back to the
    ``ANTHROPIC_FOUNDRY_RESOURCE`` environment variable (resource name or
    full URL).  Mutually exclusive with ``project_endpoint``."""

    credential: Optional[
        Union[str, AzureKeyCredential, TokenCredential, AsyncTokenCredential]
    ] = Field(default=None)
    """Credential for authentication.

    * A plain ``str`` or :class:`~azure.core.credentials.AzureKeyCredential`
      is treated as an Anthropic API key.
    * A :class:`~azure.core.credentials.TokenCredential` (e.g.
      :class:`~azure.identity.DefaultAzureCredential`) or
      :class:`~azure.core.credentials_async.AsyncTokenCredential` (e.g.
      :class:`~azure.identity.aio.DefaultAzureCredential`) is used as a
      callable bearer-token provider so that tokens are refreshed
      automatically.
    * ``None`` (default) falls back to
      :class:`~azure.identity.DefaultAzureCredential`.
    """

    @model_validator(mode="before")
    @classmethod
    def _configure_foundry(cls, values: Any) -> Any:
        """Resolve endpoint, credential and inject a placeholder API key.

        :class:`~langchain_anthropic.ChatAnthropic` requires a non-empty
        ``api_key`` field (it reads from ``ANTHROPIC_API_KEY`` when not
        provided).  When the user passes an Azure ``TokenCredential`` there
        is no API key, so a placeholder value is injected to satisfy
        validation; the actual ``_client`` and ``_async_client`` are
        overridden below to use :class:`anthropic.AnthropicFoundry`, which
        handles bearer-token authentication itself.
        """
        if not isinstance(values, dict):
            return values

        credential = values.get("credential")
        endpoint = values.get("endpoint")
        project_endpoint = values.get("project_endpoint")

        # Validate mutual exclusivity.
        if project_endpoint and endpoint:
            raise ValueError(
                "Both `project_endpoint` and `endpoint` were provided.  "
                "Use only one: `project_endpoint` for Azure AI Foundry "
                "projects, or `endpoint` for direct resource endpoints."
            )

        # Fall back to AZURE_AI_PROJECT_ENDPOINT / FOUNDRY_PROJECT_ENDPOINT
        # environment variables when no constructor parameter was provided —
        # the same resolution logic used by all other Azure AI models.
        if not project_endpoint and not endpoint:
            project_endpoint = get_project_endpoint(values, nullable=True)
            if project_endpoint:
                values["project_endpoint"] = project_endpoint

        # When a project endpoint is given, derive the resource base URL
        # from it so that ``_resolve_anthropic_endpoint`` can append the
        # ``/anthropic`` path correctly.
        endpoint_for_resolution = endpoint
        if project_endpoint:
            endpoint_for_resolution = _get_base_url_from_endpoint(project_endpoint)

        # Pre-populate ``base_url`` on the underlying ChatAnthropic instance
        # so introspection (e.g. ``self.anthropic_api_url``) reflects the
        # resolved Foundry endpoint.  The actual HTTP base URL used by the
        # overridden clients is computed from ``self.endpoint`` /
        # ``self.project_endpoint`` directly.
        resource_env = os.environ.get(_ANTHROPIC_FOUNDRY_RESOURCE_ENV_VAR, "").strip()
        if endpoint_for_resolution or resource_env:
            values.setdefault(
                "base_url", _resolve_anthropic_endpoint(endpoint_for_resolution)
            )

        if "api_key" not in values and "anthropic_api_key" not in values:
            if isinstance(credential, (str, AzureKeyCredential)):
                values["api_key"] = _extract_api_key(credential)
            else:
                values["api_key"] = SecretStr("placeholder-managed-by-azure")

        return values

    def _foundry_client_params(self) -> dict:
        """Build keyword arguments shared by sync and async Foundry clients."""
        params: dict = {
            "max_retries": self.max_retries,
            "default_headers": self._client_params.get("default_headers", {}),
        }
        # Determine the effective resource endpoint.  ``project_endpoint``
        # takes priority over ``endpoint``; when both are None,
        # AnthropicFoundry falls back to the ANTHROPIC_FOUNDRY_RESOURCE
        # environment variable natively (resource and base_url are mutually
        # exclusive in AnthropicFoundry).
        effective_endpoint = (
            _get_base_url_from_endpoint(self.project_endpoint)
            if self.project_endpoint is not None
            else self.endpoint
        )

        if effective_endpoint is not None:
            params["base_url"] = _resolve_anthropic_endpoint(effective_endpoint)
        if self.default_request_timeout is None or self.default_request_timeout > 0:
            params["timeout"] = self.default_request_timeout
        return params

    @cached_property
    def _client(self) -> Any:  # type: ignore[override]
        """Return an :class:`anthropic.AnthropicFoundry` client."""
        credential = self.credential
        if credential is None:
            logger.info("No credential provided, using DefaultAzureCredential().")
            credential = DefaultAzureCredential()

        params = self._foundry_client_params()

        if isinstance(credential, (str, AzureKeyCredential)):
            return AnthropicFoundry(api_key=_extract_api_key(credential), **params)
        if isinstance(credential, AsyncTokenCredential):
            raise ValueError(
                "An `AsyncTokenCredential` was provided but the synchronous "
                "client was requested.  Provide a `TokenCredential` "
                "(e.g. `DefaultAzureCredential()`) instead, or call the "
                "async API."
            )
        if isinstance(credential, TokenCredential):
            return AnthropicFoundry(
                azure_ad_token_provider=_make_token_provider(
                    credential, _DEFAULT_FOUNDRY_SCOPE
                ),
                **params,
            )

        raise ValueError(f"Unsupported credential type: {type(credential).__name__}.")

    @cached_property
    def _async_client(self) -> Any:  # type: ignore[override]
        """Return an :class:`anthropic.AsyncAnthropicFoundry` client."""
        credential = self.credential
        if credential is None:
            logger.info("No credential provided, using AsyncDefaultAzureCredential().")
            credential = AsyncDefaultAzureCredential()

        params = self._foundry_client_params()

        if isinstance(credential, (str, AzureKeyCredential)):
            return AsyncAnthropicFoundry(api_key=_extract_api_key(credential), **params)
        if isinstance(credential, (TokenCredential, AsyncTokenCredential)):
            return AsyncAnthropicFoundry(
                azure_ad_token_provider=_make_async_token_provider(
                    credential, _DEFAULT_FOUNDRY_SCOPE
                ),
                **params,
            )

        raise ValueError(f"Unsupported credential type: {type(credential).__name__}.")
