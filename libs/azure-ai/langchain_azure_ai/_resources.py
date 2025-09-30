"""Resources for connecting to services from Azure AI Foundry projects or endpoints."""

import logging
from typing import Any, Dict, Literal, Optional, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import BaseModel, ConfigDict

from langchain_azure_ai.utils.utils import get_endpoint_from_project

logger = logging.getLogger(__name__)


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
        if values["project_endpoint"] is not None:
            if not isinstance(values["credential"], TokenCredential):
                raise ValueError(
                    "When using the `project_endpoint` parameter, the "
                    "`credential` parameter must be of type `TokenCredential`."
                )
            values["endpoint"], values["credential"] = get_endpoint_from_project(
                values["project_endpoint"],
                values["credential"],
                service=values["service"],
                api_version=values["api_version"],
            )

        values["endpoint"] = get_from_dict_or_env(
            values, "endpoint", "AZURE_AI_ENDPOINT"
        )
        values["credential"] = get_from_dict_or_env(
            values, "credential", "AZURE_AI_CREDENTIAL"
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
