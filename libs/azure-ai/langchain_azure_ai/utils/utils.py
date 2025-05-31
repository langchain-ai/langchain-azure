"""Utility functions for LangChain Azure AI package."""

import dataclasses
import json
from typing import Any, Tuple, Union

from azure.core.credentials import AzureKeyCredential, TokenCredential
from pydantic import BaseModel


class JSONObjectEncoder(json.JSONEncoder):
    """Custom JSON encoder for objects in LangChain."""

    def default(self, o: Any) -> Any:
        """Serialize the object to JSON string.

        Args:
            o (Any): Object to be serialized.
        """
        if isinstance(o, dict):
            if "callbacks" in o:
                del o["callbacks"]
                return o

        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)  # type: ignore

        if hasattr(o, "to_json"):
            return o.to_json()

        if isinstance(o, BaseModel) and hasattr(o, "model_dump_json"):
            return o.model_dump_json()

        if "__slots__" in dir(o):
            # Handle objects with __slots__ that are not dataclasses
            return {
                "__class__": o.__class__.__name__,
                **{slot: getattr(o, slot) for slot in o.__slots__},
            }

        return super().default(o)


def get_endpoint_from_project(
    project_connection_string: str, credential: TokenCredential
) -> Tuple[str, Union[AzureKeyCredential, TokenCredential]]:
    """Retrieves the default inference endpoint and credentials from a project.

    It uses the Azure AI project's connection string to retrieve the inference
    defaults. The default connection of type Azure AI Services is used to
    retrieve the endpoint and credentials.

    Args:
        project_connection_string (str): Connection string for the Azure AI project.
        credential (TokenCredential): Azure credential object. Credentials must be of
            type `TokenCredential` when using the `project_connection_string`
            parameter.

    Returns:
        Tuple[str, Union[AzureKeyCredential, TokenCredential]]: Endpoint URL and
            credentials.
    """
    try:
        from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]
        from azure.ai.projects.models import (  # type: ignore[import-untyped]
            ConnectionType,
        )
    except ImportError:
        raise ImportError(
            "The `azure.ai.projects` package is required to use the "
            "`project_connection_string` parameter. Please install it with "
            "`pip install azure-ai-projects`."
        )

    project = AIProjectClient.from_connection_string(
        conn_str=project_connection_string,
        credential=credential,
    )

    connection = project.connections.get_default(
        connection_type=ConnectionType.AZURE_AI_SERVICES, include_credentials=True
    )

    if not connection:
        raise ValueError(
            "No Azure AI Services connection found in the project. See "
            "https://aka.ms/azureai/modelinference/connection for more "
            "information."
        )

    if connection.endpoint_url.endswith("/models"):
        endpoint = connection.endpoint_url
    elif connection.endpoint_url.endswith("/"):
        endpoint = connection.endpoint_url + "models"
    else:
        endpoint = connection.endpoint_url + "/models"

    return endpoint, connection.key or connection.token_credential
