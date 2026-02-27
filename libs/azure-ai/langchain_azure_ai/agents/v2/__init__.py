"""Agents integrated with LangChain and LangGraph."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents._v2.agent_service import (
        AgentServiceFactory,
        external_tools_condition,
    )
    from langchain_azure_ai.agents._v2.prebuilt.declarative import (
        AgentServiceAgentState,
    )


__all__ = ["AgentServiceFactory", "AgentServiceAgentState", "external_tools_condition"]

_module_lookup = {
    "AgentServiceFactory": "langchain_azure_ai.agents._v2.agent_service",
    "AgentServiceAgentState": "langchain_azure_ai.agents._v2.prebuilt.declarative",
    "external_tools_condition": "langchain_azure_ai.agents._v2.agent_service",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
