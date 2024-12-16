"""Utility functions for LangChain Azure AI package."""

import dataclasses
import json
from typing import Any

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

        return super().default(o)
