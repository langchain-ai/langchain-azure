"""Azure AI chat model using the Anthropic Messages API via Azure AI Foundry."""

import logging
import os
from functools import cached_property
from typing import Any, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity import DefaultAzureCredential
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_azure_ai._resources import (
    _make_async_token_provider,
    _make_token_provider,
)

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

# Token scope used to acquire bearer tokens for the Azure AI Foundry
# Anthropic Messages API.
_ANTHROPIC_FOUNDRY_SCOPE = "https://ai.azure.com/.default"


def _resolve_anthropic_endpoint(endpoint: Optional[str]) -> str:
    """Resolve the Anthropic Foundry ``base_url`` from an endpoint value.

    Accepts the Foundry resource root (e.g.
    ``https://<resource>.services.ai.azure.com``) or a URL that already
    includes the ``/anthropic`` path; in both cases the returned value ends
    with ``/anthropic/``.
    """
    if not endpoint:
        env_endpoint = os.environ.get("AZURE_AI_ANTHROPIC_ENDPOINT") or os.environ.get(
            "FOUNDRY_MODELS_ENDPOINT"
        )
        if not env_endpoint:
            raise ValueError(
                "An `endpoint` must be provided, or one of the "
                "`AZURE_AI_ANTHROPIC_ENDPOINT` or `FOUNDRY_MODELS_ENDPOINT` "
                "environment variables must be set."
            )
        endpoint = env_endpoint

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

    **API-key authentication:**

    ```python
    model = AzureAIAnthropicChatModel(
        endpoint="https://<resource>.services.ai.azure.com",
        credential="your-api-key",
        model="claude-sonnet-4-20250514",
    )
    ```

    **Environment variables:**

    * ``AZURE_AI_ANTHROPIC_ENDPOINT`` or ``FOUNDRY_MODELS_ENDPOINT`` – used
      as the ``endpoint`` when the constructor parameter is omitted.
      ``AZURE_AI_ANTHROPIC_ENDPOINT`` takes precedence when both are set.

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

    endpoint: Optional[str] = Field(default=None)
    """Azure AI Foundry resource endpoint, e.g.
    ``https://<resource>.services.ai.azure.com``.  ``/anthropic`` is
    appended automatically.  When omitted, falls back to the
    ``AZURE_AI_ANTHROPIC_ENDPOINT`` or ``FOUNDRY_MODELS_ENDPOINT``
    environment variable."""

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

        # Pre-populate ``base_url`` on the underlying ChatAnthropic instance
        # so introspection (e.g. ``self.anthropic_api_url``) reflects the
        # resolved Foundry endpoint.  The actual HTTP base URL used by the
        # overridden clients is computed from ``self.endpoint`` directly.
        if (
            endpoint
            or os.environ.get("AZURE_AI_ANTHROPIC_ENDPOINT")
            or os.environ.get("FOUNDRY_MODELS_ENDPOINT")
        ):
            values.setdefault("base_url", _resolve_anthropic_endpoint(endpoint))

        if "api_key" not in values and "anthropic_api_key" not in values:
            if isinstance(credential, (str, AzureKeyCredential)):
                values["api_key"] = _extract_api_key(credential)
            else:
                values["api_key"] = SecretStr("placeholder-managed-by-azure")

        return values

    def _foundry_client_params(self) -> dict:
        """Build keyword arguments shared by sync and async Foundry clients."""
        params: dict = {
            "base_url": _resolve_anthropic_endpoint(self.endpoint),
            "max_retries": self.max_retries,
            "default_headers": self._client_params.get("default_headers", {}),
        }
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
                    credential, _ANTHROPIC_FOUNDRY_SCOPE
                ),
                **params,
            )

        raise ValueError(f"Unsupported credential type: {type(credential).__name__}.")

    @cached_property
    def _async_client(self) -> Any:  # type: ignore[override]
        """Return an :class:`anthropic.AsyncAnthropicFoundry` client."""
        credential = self.credential
        if credential is None:
            logger.info("No credential provided, using DefaultAzureCredential().")
            credential = DefaultAzureCredential()

        params = self._foundry_client_params()

        if isinstance(credential, (str, AzureKeyCredential)):
            return AsyncAnthropicFoundry(api_key=_extract_api_key(credential), **params)
        if isinstance(credential, (TokenCredential, AsyncTokenCredential)):
            return AsyncAnthropicFoundry(
                azure_ad_token_provider=_make_async_token_provider(
                    credential, _ANTHROPIC_FOUNDRY_SCOPE
                ),
                **params,
            )

        raise ValueError(f"Unsupported credential type: {type(credential).__name__}.")
