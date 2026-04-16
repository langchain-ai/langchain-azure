"""Runtime host for deploying LangGraph graphs on Azure AI Foundry Agent Server."""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.runtime._host import (
        AzureAIAgentServerRuntime,
        default_request_parser,
        default_response_formatter,
    )

__all__ = [
    "AzureAIAgentServerRuntime",
    "default_request_parser",
    "default_response_formatter",
]

_module_lookup = {
    "AzureAIAgentServerRuntime": "langchain_azure_ai.agents.runtime._host",
    "default_request_parser": "langchain_azure_ai.agents.runtime._host",
    "default_response_formatter": "langchain_azure_ai.agents.runtime._host",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
