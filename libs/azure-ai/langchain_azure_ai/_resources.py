"""Resources for connecting to services from Azure AI Foundry projects or endpoints."""

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Literal, Optional, Tuple, Union
from urllib.parse import urlparse

import openai
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.utils import pre_init
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict

from langchain_azure_ai.utils.env import get_from_dict_or_env
from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

logger = logging.getLogger(__name__)


def _make_token_provider(
    credential: TokenCredential, scopes: str = "https://ai.azure.com/.default"
) -> Callable[[], str]:
    """Return a bearer-token provider callable for the given credential."""
    try:
        from azure.identity import get_bearer_token_provider
    except ImportError as exc:
        raise ImportError(
            "`azure-identity` is required. Install with `pip install azure-identity`."
        ) from exc

    return get_bearer_token_provider(credential, scopes)


def _make_async_token_provider(
    credential: Union[TokenCredential, AsyncTokenCredential],
    scopes: str = "https://ai.azure.com/.default",
) -> Callable[[], Awaitable[str]]:
    """Return an async bearer-token provider for ``AsyncOpenAI``.

    If *credential* implements :class:`AsyncTokenCredential` the native
    ``get_token`` coroutine is used directly.  Otherwise the synchronous
    provider is wrapped via ``run_in_executor``.
    """
    if isinstance(credential, AsyncTokenCredential):

        async def _async_provider() -> str:
            token = await credential.get_token(scopes)
            return token.token

        return _async_provider

    sync_provider = _make_token_provider(credential, scopes)

    async def _provider() -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_provider)

    return _provider


def _has_version_segment(segments: list) -> bool:
    """Return True if any path segment looks like a version (e.g. 'v1', 'v12')."""
    return any(s.startswith("v") and s[1:].isdigit() for s in segments)


def _validate_endpoint_url(url: str, param_name: str) -> None:
    """Emit warnings when *url* looks incorrect.

    The checks are intentionally soft (warnings, not errors) so that users
    behind API-management gateways or model routers with custom URL shapes are
    not blocked.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return  # unparseable – let the underlying library report the error

    scheme = parsed.scheme.lower()
    path = (parsed.path or "").rstrip("/")

    # -----------------------------------------------------------------------
    # Non-HTTPS warning (skip for localhost / 127.x which is fine for dev)
    # -----------------------------------------------------------------------
    host = (parsed.hostname or "").lower()
    is_localhost = host in ("localhost", "127.0.0.1", "::1")
    if scheme not in ("https", "") and not is_localhost:
        logger.warning(
            "The `%s` URL '%s' does not use HTTPS. "
            "Non-HTTPS endpoints are not recommended for production use.",
            param_name,
            url,
        )

    if param_name == "endpoint":
        # ------------------------------------------------------------------
        # Detect accidental use of a project endpoint in the endpoint field.
        # ------------------------------------------------------------------
        if "/api/projects/" in path:
            logger.warning(
                "The `endpoint` value '%s' appears to be a project endpoint "
                "(it contains '/api/projects/').  "
                "Use the `project_endpoint` parameter instead, or supply the "
                "direct service URL (e.g. "
                "'https://<resource>.services.ai.azure.com/openai/v1').",
                url,
            )
            return  # no further endpoint checks needed

        # ------------------------------------------------------------------
        # Warn when the path looks bare (no version segment like /openai/v1).
        # We check for at least one non-empty path segment that starts with
        # a version-like component.  Custom routers may have different paths,
        # so this is advisory only.
        # ------------------------------------------------------------------
        segments = [s for s in path.split("/") if s]
        has_version_path = _has_version_segment(segments)
        if not segments:
            logger.warning(
                "The `endpoint` value '%s' appears to have no path.  "
                "For the Azure AI Foundry OpenAI-compatible service the URL "
                "should include the path, e.g. "
                "'https://<resource>.services.ai.azure.com/openai/v1'.",
                url,
            )
        elif not has_version_path:
            logger.warning(
                "The `endpoint` value '%s' does not contain a version segment "
                "(e.g. '/v1').  "
                "If you are connecting to the Azure AI Foundry OpenAI-compatible "
                "service, the URL should end with '/openai/v1'.  "
                "If you are using a custom gateway or model router, you can "
                "ignore this warning.",
                url,
            )

    elif param_name == "project_endpoint":
        # ------------------------------------------------------------------
        # Detect accidental use of a service endpoint in project_endpoint.
        # ------------------------------------------------------------------
        segments = [s for s in path.split("/") if s]
        has_version_path = _has_version_segment(segments)
        if has_version_path and "/api/projects/" not in path:
            logger.warning(
                "The `project_endpoint` value '%s' looks like a direct service "
                "URL (it contains a version path such as '/v1') rather than a "
                "project endpoint.  "
                "Use the `endpoint` parameter for direct service URLs, or supply "
                "the project endpoint, e.g. "
                "'https://<resource>.services.ai.azure.com/api/projects/<project>'.",
                url,
            )


def _configure_openai_credential_values(
    values: dict,
) -> Tuple[dict, Optional[Tuple[OpenAI, AsyncOpenAI]]]:
    """Shared pre-validation logic for OpenAI-based Azure AI models.

    Handles the ``project_endpoint`` path (uses :class:`AIProjectClient` to
    obtain pre-configured OpenAI clients) and the direct ``endpoint`` path
    (maps ``credential`` to ``api_key`` or ``azure_ad_token_provider``).

    The following environment variables are also supported as fallbacks when
    the corresponding constructor parameters are not provided:

    * ``AZURE_AI_PROJECT_ENDPOINT`` – resolved as ``project_endpoint``.
    * ``AZURE_AI_OPENAI_ENDPOINT`` – direct OpenAI-compatible endpoint
      (e.g. ``https://<resource>.services.ai.azure.com/openai/v1``).
      Used as ``endpoint`` verbatim (no path is appended).
    * ``AZURE_OPENAI_ENDPOINT`` – root Azure OpenAI endpoint (e.g.
      ``https://<resource>.services.ai.azure.com``).  ``/openai/v1`` is
      appended automatically and the result is used as ``endpoint``.
    * ``AZURE_OPENAI_API_VERSION`` – API version passed as a default query
      parameter (``api-version``) on every request when clients are
      constructed by this helper.
    * ``AZURE_OPENAI_DEPLOYMENT_NAME`` – model deployment name, used as
      ``model``.

    **Resolution priority** (highest to lowest):

    1. Constructor parameters (``project_endpoint``, ``endpoint``, ``model``,
       ``credential``, ``api_version``).
    2. ``AZURE_AI_PROJECT_ENDPOINT`` environment variable.
    3. ``AZURE_AI_OPENAI_ENDPOINT`` environment variable.
    4. ``AZURE_OPENAI_ENDPOINT`` / ``AZURE_OPENAI_API_VERSION`` /
       ``AZURE_OPENAI_DEPLOYMENT_NAME`` environment variables.

    Providing both ``project_endpoint`` and ``endpoint`` as constructor
    parameters is considered a configuration error and raises
    :class:`ValueError`.  However, if only environment variables are used,
    ``AZURE_AI_PROJECT_ENDPOINT`` silently takes precedence over
    ``AZURE_AI_OPENAI_ENDPOINT`` and ``AZURE_OPENAI_ENDPOINT``.

    Returns a tuple of ``(values, openai_clients)`` where ``openai_clients``
    is ``(sync_openai, async_openai)`` when pre-built clients are available,
    or ``None`` when the caller's parent class should construct them.  The
    caller is responsible for extracting the concrete sub-clients (e.g.
    ``chat.completions`` or ``embeddings``) from the returned OpenAI clients.
    """
    endpoint = values.get("endpoint")
    project_endpoint = values.get("project_endpoint")
    credential = values.get("credential")

    # -- Validate: explicit constructor params must not conflict ---------- #
    if project_endpoint and endpoint:
        raise ValueError(
            "Both `project_endpoint` and `endpoint` were provided.  "
            "Use only one: `project_endpoint` for Azure AI Foundry projects, "
            "or `endpoint` for direct OpenAI-compatible endpoints."
        )

    # -- Resolve from environment variables (constructor params win) ------ #
    if not project_endpoint and not endpoint:
        project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")

    if not project_endpoint and not endpoint:
        ai_openai_ep = os.environ.get("AZURE_AI_OPENAI_ENDPOINT")
        if ai_openai_ep:
            endpoint = ai_openai_ep
            values["endpoint"] = endpoint

    if not project_endpoint and not endpoint:
        azure_openai_ep = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if azure_openai_ep:
            endpoint = azure_openai_ep.rstrip("/") + "/openai/v1"
            values["endpoint"] = endpoint

    if not values.get("model") and not values.get("model_name"):
        deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        if deployment_name:
            values["model"] = deployment_name

    api_version = values.get("api_version") or values.get("openai_api_version")
    if not api_version:
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
        if api_version:
            values["api_version"] = api_version

    # -- Project-endpoint path ------------------------------------------- #
    if project_endpoint:
        _validate_endpoint_url(project_endpoint, "project_endpoint")

        if AIProjectClient is None:
            raise ImportError(
                "The `azure-ai-projects` package is required when using "
                "`project_endpoint`. Install it with "
                "`pip install azure-ai-projects`."
            )

        if credential is None:
            logger.warning(
                "No credential provided, using DefaultAzureCredential(). "
                "If intentional, pass `credential=DefaultAzureCredential()`."
            )
            credential = DefaultAzureCredential()

        if not isinstance(credential, (TokenCredential, AsyncTokenCredential)):
            raise ValueError(
                "When using `project_endpoint` the `credential` must be "
                "a `TokenCredential` or `AsyncTokenCredential` "
                "(e.g. `DefaultAzureCredential()`)."
            )

        sync_project = AIProjectClient(endpoint=project_endpoint, credential=credential)

        _ua_headers = {"x-ms-useragent": "langchain-azure-ai"}
        sync_openai = sync_project.get_openai_client().with_options(
            default_headers=_ua_headers
        )
        # AsyncAIProjectClient.get_openai_client() is a coroutine so it
        # cannot be called in this synchronous validator.  Build the async
        # client from the sync client's configuration instead.
        # Note: sync_openai.api_key returns an empty string even though the
        # client was constructed with a callable token provider – the openai
        # SDK does not expose the callable through the property.  Use our
        # own token provider so the async client authenticates correctly.
        # AsyncOpenAI requires Callable[[], Awaitable[str]] (not a sync
        # callable), so we use the async wrapper.
        async_openai = openai.AsyncOpenAI(
            api_key=_make_async_token_provider(credential),
            base_url=str(sync_openai.base_url),
            default_headers=_ua_headers,
        )

        values["project_endpoint"] = project_endpoint
        return values, (sync_openai, async_openai)

    # -- Direct-endpoint path -------------------------------------------- #
    elif endpoint:
        _validate_endpoint_url(endpoint, "endpoint")
        values["openai_api_base"] = endpoint

        if api_version and credential is not None:
            # When an api_version is available we construct the OpenAI
            # clients ourselves so we can inject the ``api-version`` query
            # parameter via ``default_query``.
            default_query: dict = {"api-version": api_version}
            _ua_headers = {"x-ms-useragent": "langchain-azure-ai"}

            if isinstance(credential, (str, AzureKeyCredential)):
                key = credential if isinstance(credential, str) else credential.key
                sync_openai = openai.OpenAI(
                    api_key=key,
                    base_url=endpoint,
                    default_headers=_ua_headers,
                    default_query=default_query,
                )
                async_openai = openai.AsyncOpenAI(
                    api_key=key,
                    base_url=endpoint,
                    default_headers=_ua_headers,
                    default_query=default_query,
                )
            elif isinstance(credential, AsyncTokenCredential):
                async_provider = _make_async_token_provider(credential)
                sync_openai = openai.OpenAI(
                    api_key=async_provider,  # type: ignore[arg-type]
                    base_url=endpoint,
                    default_headers=_ua_headers,
                    default_query=default_query,
                )
                async_openai = openai.AsyncOpenAI(
                    api_key=async_provider,
                    base_url=endpoint,
                    default_headers=_ua_headers,
                    default_query=default_query,
                )
            elif isinstance(credential, TokenCredential):
                sync_openai = openai.OpenAI(
                    api_key=_make_token_provider(credential),
                    base_url=endpoint,
                    default_headers=_ua_headers,
                    default_query=default_query,
                )
                async_openai = openai.AsyncOpenAI(
                    api_key=_make_async_token_provider(credential),
                    base_url=endpoint,
                    default_headers=_ua_headers,
                    default_query=default_query,
                )
            else:
                return values, None

            return values, (sync_openai, async_openai)

        # No api_version — let the parent class construct clients.
        if isinstance(credential, (str, AzureKeyCredential)):
            api_key = credential if isinstance(credential, str) else credential.key
            values["api_key"] = api_key
        elif isinstance(credential, AsyncTokenCredential):
            # Check AsyncTokenCredential before TokenCredential — async
            # credentials (e.g. azure.identity.aio.DefaultAzureCredential)
            # also implement TokenCredential but their get_token() returns a
            # coroutine, so _make_token_provider would fail.
            values["openai_api_key"] = _make_async_token_provider(credential)
        elif isinstance(credential, TokenCredential):
            # ChatOpenAI / OpenAIEmbeddings accept a callable as api_key; the
            # provider is invoked per-request so tokens are automatically refreshed.
            values["openai_api_key"] = _make_token_provider(credential)

    return values, None


class FDPResourceService(BaseModel):
    """Base class for connecting to services from Azure AI Foundry projects."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    project_endpoint: Optional[str] = None
    """The project endpoint associated with the AI project. If this is specified,
    then the `endpoint` parameter becomes optional and `credential` has to be of type
    `TokenCredential`."""

    endpoint: Optional[str] = None
    """The endpoint of the specific service to connect to. If you are connecting to a
    model, use the URL of the model deployment."""

    credential: Optional[Union[str, AzureKeyCredential, TokenCredential]] = None
    """The API key or credential to use to connect to the service. If using a project 
    endpoint, this must be of type `TokenCredential` since only Microsoft EntraID is 
    supported."""

    api_version: Optional[str] = None
    """The API version to use with Azure. If None, the 
    default version is used."""

    client_kwargs: Dict[str, Any] = {}
    """Additional keyword arguments to pass to the client."""

    @pre_init
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that required values are present in the environment."""
        values["credential"] = get_from_dict_or_env(
            values, "credential", "AZURE_AI_INFERENCE_CREDENTIAL", nullable=True
        )

        if values["credential"] is None:
            logger.warning(
                "No credential provided, using DefaultAzureCredential(). If "
                "intentional, use `credential=DefaultAzureCredential()`"
            )
            values["credential"] = DefaultAzureCredential()

        if values["endpoint"] is None:
            values["project_endpoint"] = get_from_dict_or_env(
                values,
                "project_endpoint",
                "AZURE_AI_PROJECT_ENDPOINT",
                nullable=True,
            )

        if values["project_endpoint"] is not None:
            if not isinstance(values["credential"], TokenCredential):
                raise ValueError(
                    "When using the `project_endpoint` parameter, the "
                    "`credential` parameter must be of type `TokenCredential`."
                )
            values["endpoint"], values["credential"] = (
                get_service_endpoint_from_project(
                    values["project_endpoint"],
                    values["credential"],
                    service=values["service"],
                    api_version=values["api_version"],
                )
            )
        else:
            values["endpoint"] = get_from_dict_or_env(
                values, "endpoint", "AZURE_AI_INFERENCE_ENDPOINT"
            )

        if values["api_version"]:
            values["client_kwargs"]["api_version"] = values["api_version"]

        values["client_kwargs"]["user_agent"] = "langchain-azure-ai"

        return values


class AIServicesService(FDPResourceService):
    service: Literal["cognitive_services"] = "cognitive_services"
    """The type of service to connect to. For Cognitive Services, use 
    'cognitive_services'."""


class ModelInferenceService(FDPResourceService):
    service: Literal["inference"] = "inference"
    """The type of service to connect to. For Inference Services, 
    use 'inference'."""
