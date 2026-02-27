"""Agents integrated with LangChain and LangGraph."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents._v2.prebuilt.declarative import (
        AgentServiceAgentState,
        PromptBasedAgentNode,
    )


__all__ = [
    "AgentServiceAgentState",
    "PromptBasedAgentNode",
]

_module_lookup = {
    "AgentServiceAgentState": "langchain_azure_ai.agents._v2.prebuilt.declarative",
    "PromptBasedAgentNode": "langchain_azure_ai.agents._v2.prebuilt.declarative",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
