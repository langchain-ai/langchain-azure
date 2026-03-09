"""Agents V1 integrated with LangChain and LangGraph."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from packaging.version import Version

_MAX_PROJECTS_VERSION = "2.0.0"

try:
    _ = version("azure-ai-agents")
except PackageNotFoundError:
    raise ImportError(
        "The `azure-ai-agents` package is required to use "
        f"`{__name__}`. Please install it with "
        "`pip install -U 'langchain-azure-ai[v1]'`."
    )

try:
    _projects_version = version("azure-ai-projects")
except PackageNotFoundError:
    raise ImportError(
        "The `azure-ai-projects` package is required to use "
        f"`{__name__}`. Please install it with "
        "`pip install -U 'langchain-azure-ai[v1]'`."
    )

if Version(_projects_version) >= Version(_MAX_PROJECTS_VERSION):
    raise ImportError(
        f"`{__name__}` requires `azure-ai-projects<{_MAX_PROJECTS_VERSION}`, "
        f"but version {_projects_version} is installed. Please install a compatible "
        "version with `pip install -U 'langchain-azure-ai[v1]'`."
    )


if TYPE_CHECKING:
    from langchain_azure_ai.agents._v1.agent_service import (
        AgentServiceFactory,
        external_tools_condition,
    )


__all__ = ["AgentServiceFactory", "external_tools_condition"]

_module_lookup = {
    "AgentServiceFactory": "langchain_azure_ai.agents._v1.agent_service",
    "external_tools_condition": "langchain_azure_ai.agents._v1.agent_service",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
