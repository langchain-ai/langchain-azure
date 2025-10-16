"""Utility functions for LangChain Azure AI package."""

import dataclasses
import json
import os
import tempfile
from typing import Any, Literal, Tuple, Union
from urllib.parse import urlparse

import requests
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
    project_endpoint: str,
    credential: TokenCredential,
    service: Union[
        Literal["inference"], Literal["cognitive_services"], str
    ] = "inference",
    api_version: str = "v1",
) -> Tuple[str, Union[AzureKeyCredential, TokenCredential]]:
    """Retrieves the endpoint and credentials required a given a project endpoint.

    It uses the Azure AI project's connection string to retrieve the inference
    defaults. The default connection of type Azure AI Services is used to
    retrieve the endpoint and credentials.

    Args:
        project_endpoint (str): Endpoint for the Azure AI project.
        credential (TokenCredential): Azure credential object. Credentials must be of
            type `TokenCredential` when using the `project_endpoint`
            parameter.
        service (str): The type of service to retrieve the endpoint for. Can be one of
            "inference", or "cognitive_services". Defaults to "inference".
        api_version (str): API version to use when retrieving the endpoint. Defaults
            to "v1".

    Returns:
        Tuple[str, Union[AzureKeyCredential, TokenCredential]]: Endpoint URL and
            credentials.
    """
    try:
        from azure.ai.projects import AIProjectClient  # type: ignore[import-untyped]
        from azure.ai.projects.models import (  # type: ignore[import-untyped]
            ApiKeyCredentials,
            Connection,
            ConnectionType,
        )
    except ImportError:
        raise ImportError(
            "The `azure.ai.projects` package is required to use the "
            "`project_endpoint` parameter. Please install it with "
            "`pip install azure-ai-projects`."
        )

    project = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential,
    )

    if service in "inference":
        try:
            # For hub projects, use connections
            connection: Connection = project.connections.get_default(
                connection_type=ConnectionType.AZURE_OPEN_AI, include_credentials=True
            )
            endpoint = connection.target + "/openai/v1"
            if connection.credentials is ApiKeyCredentials:
                return endpoint, connection.credentials.api_key  # type: ignore
            elif connection.credentials.type == "AAD":
                return endpoint, credential
            else:
                raise ValueError(
                    f"Unsupported credential type: {connection.credentials.type}"
                )
        except KeyError:
            # For non-hub projects, use OpenAI client
            endpoint = str(project.get_openai_client(api_version=api_version).base_url)
            return endpoint, credential
    elif service == "cognitive_services":
        return project_endpoint.split("/api")[0], credential
    else:
        raise ValueError(f"Service type '{service}' is not supported.")


def detect_file_src_type(file_path: str) -> str:
    """Detect if the file is local or remote."""
    if os.path.isfile(file_path):
        return "local"

    parsed_url = urlparse(file_path)
    if parsed_url.scheme and parsed_url.netloc:
        return "remote"

    return "invalid"


def download_audio_from_url(audio_url: str) -> str:
    """Download audio from url to local."""
    ext = audio_url.split(".")[-1]
    response = requests.get(audio_url, stream=True)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(mode="wb", suffix=f".{ext}", delete=False) as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return f.name
