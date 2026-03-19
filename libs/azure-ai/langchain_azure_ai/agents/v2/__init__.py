"""Agents integrated with LangChain and LangGraph."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentMiddleware

    from langchain_azure_ai.agents._v2.base import (
        AgentServiceAgentState,
    )
    from langchain_azure_ai.agents._v2.prebuilt.factory import (
        AgentServiceFactory,
        external_tools_condition,
    )


__all__ = [
    "AgentMiddleware",
    "AgentServiceFactory",
    "AgentServiceAgentState",
    "external_tools_condition",
]

_module_lookup = {
    "AgentMiddleware": "langchain.agents.middleware.types",
    "AgentServiceFactory": "langchain_azure_ai.agents._v2.prebuilt.factory",
    "AgentServiceAgentState": "langchain_azure_ai.agents._v2.base",
    "external_tools_condition": "langchain_azure_ai.agents._v2.prebuilt.factory",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
