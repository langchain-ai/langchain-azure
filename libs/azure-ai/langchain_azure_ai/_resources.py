"""Resources for connecting to services from Azure AI Foundry projects or endpoints."""

import logging
import os
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from azure.identity import DefaultAzureCredential
from langchain_core.utils import pre_init
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict

from langchain_azure_ai.utils.env import get_from_dict_or_env
from langchain_azure_ai.utils.utils import get_service_endpoint_from_project

logger = logging.getLogger(__name__)

try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient
except ImportError:
    AIProjectClient = None  # type: ignore[assignment,misc]
    AsyncAIProjectClient = None  # type: ignore[assignment,misc]


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


def _configure_openai_credential_values(
    values: dict,
) -> Tuple[dict, Optional[Tuple[OpenAI, AsyncOpenAI]]]:
    """Shared pre-validation logic for OpenAI-based Azure AI models.

    Handles the ``project_endpoint`` path (uses :class:`AIProjectClient` to
    obtain pre-configured OpenAI clients) and the direct ``endpoint`` path
    (maps ``credential`` to ``api_key`` or ``azure_ad_token_provider``).

    Returns a tuple of ``(values, openai_clients)`` where ``openai_clients``
    is ``(sync_openai, async_openai)`` when the project-endpoint path is used,
    or ``None`` for the direct-endpoint path.  The caller is responsible for
    extracting the concrete sub-clients (e.g. ``chat.completions`` or
    ``embeddings``) from the returned OpenAI clients.
    """
    project_endpoint = values.get("project_endpoint") or os.environ.get(
        "AZURE_AI_PROJECT_ENDPOINT"
    )
    endpoint = values.get("endpoint")
    credential = values.get("credential")

    if project_endpoint:
        if AIProjectClient is None or AsyncAIProjectClient is None:
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

        if not isinstance(credential, TokenCredential):
            raise ValueError(
                "When using `project_endpoint` the `credential` must be "
                "a `TokenCredential` (e.g. `DefaultAzureCredential()`)."
            )

        sync_project = AIProjectClient(endpoint=project_endpoint, credential=credential)
        async_project = AsyncAIProjectClient(
            endpoint=project_endpoint, credential=credential
        )

        _ua_headers = {"x-ms-useragent": "langchain-azure-ai"}
        sync_openai = sync_project.get_openai_client().with_options(
            default_headers=_ua_headers
        )
        async_openai = async_project.get_openai_client().with_options(
            default_headers=_ua_headers
        )

        values["project_endpoint"] = project_endpoint
        return values, (sync_openai, async_openai)

    elif endpoint:
        values["azure_endpoint"] = endpoint

        if isinstance(credential, (str, AzureKeyCredential)):
            api_key = credential if isinstance(credential, str) else credential.key
            values["api_key"] = api_key
        elif isinstance(credential, TokenCredential):
            values["azure_ad_token_provider"] = _make_token_provider(credential)

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
