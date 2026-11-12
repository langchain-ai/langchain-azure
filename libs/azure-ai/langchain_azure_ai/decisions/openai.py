"""Azure configuration for the OpenAI Decisions Runnable."""

import os
from typing import Any

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.utils import from_env
from langchain_openai.decisions import OpenAIDecisions
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai._resources import (
    _get_base_url_from_endpoint,
    _make_async_token_provider,
    _make_token_provider,
    _validate_endpoint_url,
)
from langchain_azure_ai.utils.env import get_from_dict_or_env, get_project_endpoint


@experimental(
    message="AzureAIOpenAIApiDecisions is experimental and its interface may change. "
    "It extends the beta OpenAIDecisions integration."
)
class AzureAIOpenAIApiDecisions(OpenAIDecisions):
    """Ask typed questions with the Azure OpenAI Decisions API in Foundry.

    Inherits question handling, response parsing, batching, retries, and synchronous
    and asynchronous invocation from ``OpenAIDecisions``. Its upstream API is beta.
    The ``model`` is an Azure deployment name for a supported Decisions model,
    currently ``gpt-6-luna``. This integration uses ``/openai/v1/decisions``.

    Args:
        model: Azure deployment name. Defaults to ``AZURE_OPENAI_DEPLOYMENT_NAME``.
        endpoint: Direct OpenAI-compatible resource URL ending in ``/openai/v1``.
            Defaults to ``AZURE_AI_OPENAI_ENDPOINT``, then ``AZURE_OPENAI_ENDPOINT``
            with ``/openai/v1`` appended. The inherited ``base_url`` is also accepted.
        project_endpoint: Foundry project URL. Defaults to
            ``AZURE_AI_PROJECT_ENDPOINT``, then ``FOUNDRY_PROJECT_ENDPOINT``, when
            no explicit endpoint is supplied. Its resource URL is used for
            Decisions; requests do not pass through the project endpoint.
        credential: API key, ``AzureKeyCredential``, ``TokenCredential``, or
            ``AsyncTokenCredential``. Defaults to ``AZURE_OPENAI_API_KEY`` for a
            direct endpoint, then ``DefaultAzureCredential``. Project endpoints
            require a token credential and default to ``DefaultAzureCredential``.
            An async-only credential requires ``ainvoke`` or ``abatch``.
        **kwargs: Options accepted by ``OpenAIDecisions``, including ``timeout``,
            ``max_retries``, HTTP clients, headers, and query parameters. An explicit
            ``api_key`` or token callable can be used instead of ``credential``.

    Raises:
        ValueError: If no endpoint is configured, endpoint parameters conflict,
            authentication parameters conflict, a credential type is unsupported,
            or a project uses key authentication.
        pydantic.ValidationError: If the deployment name or another field is invalid.

    Example:
        ```python
        from azure.identity import DefaultAzureCredential
        from langchain_azure_ai.decisions import AzureAIOpenAIApiDecisions, Predicate

        decisions = AzureAIOpenAIApiDecisions(
            endpoint="https://my-resource.openai.azure.com/openai/v1",
            credential=DefaultAzureCredential(),
            model="my-luna-deployment",
        )
        answer = decisions.invoke({
            "input": "Please refund the duplicate charge.",
            "questions": {"refund": Predicate(instructions="Is a refund requested?")},
        })
        print(answer.predicates["refund"].probability)
        ```
    """

    model: str = Field(
        default_factory=from_env("AZURE_OPENAI_DEPLOYMENT_NAME"), min_length=1
    )
    """Azure deployment name for a supported Decisions model."""

    model_config = ConfigDict(hide_input_in_errors=True)

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        project_endpoint: str | None = None,
        credential: (
            str | AzureKeyCredential | TokenCredential | AsyncTokenCredential | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Azure options and upstream Decisions configuration.

        Args:
            endpoint: Direct resource URL. Defaults to Azure environment variables.
            project_endpoint: Foundry project URL. Defaults to Foundry environment
                variables when no direct endpoint is supplied.
            credential: Azure key or token credential. Defaults to an Azure API key
                or ``DefaultAzureCredential``, as described in the class docstring.
            **kwargs: Options accepted by ``OpenAIDecisions``.
        """
        super().__init__(
            **{
                "endpoint": endpoint,
                "project_endpoint": project_endpoint,
                "credential": credential,
                **kwargs,
            }
        )

    @model_validator(mode="before")
    @classmethod
    def _configure_azure(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        values = values.copy()
        endpoint = values.get("endpoint") or values.get("base_url")
        project_endpoint = values.get("project_endpoint")
        if values.get("endpoint") and values.get("base_url"):
            raise ValueError("Provide only one of `endpoint` and `base_url`.")
        if endpoint and project_endpoint:
            raise ValueError("Provide only one of `endpoint` and `project_endpoint`.")
        if not endpoint:
            project_endpoint = get_project_endpoint(values, nullable=True)
        if project_endpoint:
            _validate_endpoint_url(project_endpoint, "project_endpoint")
            endpoint = _get_base_url_from_endpoint(project_endpoint) + "/openai/v1"
        if not endpoint:
            endpoint = get_from_dict_or_env(
                values, "endpoint", "AZURE_AI_OPENAI_ENDPOINT", nullable=True
            )
        if not endpoint and os.getenv("AZURE_OPENAI_ENDPOINT"):
            endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1"
        if not endpoint:
            raise ValueError("Provide an Azure `endpoint` or `project_endpoint`.")
        _validate_endpoint_url(endpoint, "endpoint")
        # Store only upstream configuration so Runnable reconstruction is idempotent.
        values.pop("endpoint", None)
        values.pop("project_endpoint", None)
        values["base_url"] = endpoint
        credential = values.pop("credential", None)
        if credential is not None and values.get("api_key") is not None:
            raise ValueError("Provide only one of `credential` and `api_key`.")
        if project_endpoint:
            if values.get("api_key") is not None:
                raise ValueError("A `project_endpoint` requires a token credential.")
            if credential is None:
                credential = DefaultAzureCredential()
            if not isinstance(credential, (TokenCredential, AsyncTokenCredential)):
                raise ValueError("A `project_endpoint` requires a token credential.")
        elif credential is None and values.get("api_key") is None:
            credential = os.getenv("AZURE_OPENAI_API_KEY") or DefaultAzureCredential()

        if isinstance(credential, (str, AzureKeyCredential)):
            key = credential if isinstance(credential, str) else credential.key
            values["api_key"] = SecretStr(key)
        elif isinstance(credential, AsyncTokenCredential):
            values["api_key"] = _make_async_token_provider(credential)
        elif isinstance(credential, TokenCredential):
            values["api_key"] = _make_token_provider(credential)
        elif credential is not None:
            raise ValueError(
                "`credential` must be a key string, AzureKeyCredential, "
                "TokenCredential, or AsyncTokenCredential."
            )
        values["default_headers"] = {
            "x-ms-useragent": "langchain-azure-ai",
            **(values.get("default_headers") or {}),
        }
        return values

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Return the namespace used for LangChain serialization.

        Returns:
            The Azure Decisions module namespace.
        """
        return ["langchain_azure_ai", "decisions"]
