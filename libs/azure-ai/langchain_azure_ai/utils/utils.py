"""Utility functions for LangChain Azure AI package."""

import dataclasses
import json
import os
import tempfile
from typing import Any
from urllib.parse import urlparse

import requests
from pydantic import BaseModel


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
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        if isinstance(o, BaseModel) and hasattr(o, "model_dump_json"):
            return o.model_dump_json()

        return super().default(o)
