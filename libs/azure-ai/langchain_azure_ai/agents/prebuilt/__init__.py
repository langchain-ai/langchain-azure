"""Agents integrated with LangChain and LangGraph."""

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from packaging.version import Version

_MIN_PROJECTS_VERSION = "2.0.0"

try:
    _projects_version = version("azure-ai-projects")
except PackageNotFoundError:
    raise ImportError(
        "The `azure-ai-projects` package is required to use "
        f"`{__name__}`. Please install it with "
        "`pip install 'azure-ai-projects>=2.0'`."
    )

if Version(_projects_version) < Version(_MIN_PROJECTS_VERSION):
    raise ImportError(
        f"`{__name__}` requires `azure-ai-projects>={_MIN_PROJECTS_VERSION}`, "
        f"but version {_projects_version} is installed. Please upgrade with "
        "`pip install 'azure-ai-projects>=2.0'`."
    )

if TYPE_CHECKING:
    from langchain_azure_ai.agents._v2.base import (
        AgentServiceAgentState,
        ResponsesAgentNode,
    )


__all__ = [
    "AgentServiceAgentState",
    "ResponsesAgentNode",
]

_module_lookup = {
    "AgentServiceAgentState": "langchain_azure_ai.agents._v2.base",
    "ResponsesAgentNode": "langchain_azure_ai.agents._v2.base",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
