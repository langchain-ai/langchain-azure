"""Agents integrated with LangChain and LangGraph."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents._v2.prebuilt.tools import AgentServiceBaseTool
    from langchain_azure_ai.agents._v2.prebuilt.tools import ImageGenTool, CodeInterpreterTool
    

__all__ = [
    "AgentServiceBaseTool",
    "ImageGenTool",
    "CodeInterpreterTool",
]

_module_lookup = {
    "AgentServiceBaseTool": "langchain_azure_ai.agents._v2.prebuilt.tools",
    "ImageGenTool": "langchain_azure_ai.agents._v2.prebuilt.tools",
    "CodeInterpreterTool": "langchain_azure_ai.agents._v2.prebuilt.tools",
}

def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
