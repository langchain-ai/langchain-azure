"""Built-in server-side tools for OpenAI models deployed in Azure AI Foundry.

These tool classes represent server-side capabilities (web search, code
execution, image generation, etc.) that models can invoke within a single
conversational turn.  All classes inherit from :class:`BuiltinTool` (which
itself inherits from :class:`dict`) so they can be passed directly to
``model.bind_tools()`` without extra conversion.

Each class is a thin wrapper around the corresponding
`OpenAI SDK <https://github.com/openai/openai-python>`_ TypedDict from
:mod:`openai.types.responses`, so the parameter types and available options
always stay in sync with the API.

Available tools:

- :class:`CodeInterpreterTool` – run Python code in a sandboxed container
- :class:`WebSearchTool` – search the internet
- :class:`FileSearchTool` – semantic search over uploaded vector stores
- :class:`ImageGenerationTool` – generate or edit images
- :class:`McpTool` – call tools on a remote MCP server

Commonly needed SDK types are re-exported here for convenience:

- :class:`FileSearchFilters` – filter type for :class:`FileSearchTool`
- :class:`ImageGenerationInputImageMask` – mask type for :class:`ImageGenerationTool`
- :class:`McpAllowedTools` – allowed-tools type for :class:`McpTool`
- :class:`McpRequireApproval` – approval type for :class:`McpTool`
- :class:`RankingOptions` – ranking-options type for :class:`FileSearchTool`
- :class:`UserLocation` – user-location type for :class:`WebSearchTool`
- :class:`WebSearchFilters` – filters type for :class:`WebSearchTool`

Example::

    from langchain.chat_models import init_chat_model
    from langchain_azure_ai.tools.builtin import CodeInterpreterTool
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    model = init_chat_model(model="azure_ai:gpt-4.1", credential=credential)
    model_with_code = model.bind_tools([CodeInterpreterTool()])
    response = model_with_code.invoke("Use Python to tell me a joke")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.tools.builtin._tools import (
        BuiltinTool,
        CodeInterpreterTool,
        FileSearchFilters,
        FileSearchTool,
        ImageGenerationInputImageMask,
        ImageGenerationTool,
        McpAllowedTools,
        McpApprovalResponse,
        McpRequireApproval,
        McpTool,
        RankingOptions,
        UserLocation,
        WebSearchFilters,
        WebSearchTool,
    )

__all__ = [
    "BuiltinTool",
    "CodeInterpreterTool",
    "FileSearchFilters",
    "FileSearchTool",
    "ImageGenerationInputImageMask",
    "ImageGenerationTool",
    "McpAllowedTools",
    "McpRequireApproval",
    "McpTool",
    "RankingOptions",
    "UserLocation",
    "WebSearchFilters",
    "WebSearchTool",
    "McpApprovalResponse",
]

_module_lookup = {
    "BuiltinTool": "langchain_azure_ai.tools.builtin._tools",
    "CodeInterpreterTool": "langchain_azure_ai.tools.builtin._tools",
    "FileSearchFilters": "langchain_azure_ai.tools.builtin._tools",
    "FileSearchTool": "langchain_azure_ai.tools.builtin._tools",
    "ImageGenerationInputImageMask": "langchain_azure_ai.tools.builtin._tools",
    "ImageGenerationTool": "langchain_azure_ai.tools.builtin._tools",
    "McpAllowedTools": "langchain_azure_ai.tools.builtin._tools",
    "McpRequireApproval": "langchain_azure_ai.tools.builtin._tools",
    "McpTool": "langchain_azure_ai.tools.builtin._tools",
    "RankingOptions": "langchain_azure_ai.tools.builtin._tools",
    "UserLocation": "langchain_azure_ai.tools.builtin._tools",
    "WebSearchFilters": "langchain_azure_ai.tools.builtin._tools",
    "WebSearchTool": "langchain_azure_ai.tools.builtin._tools",
    "McpApprovalResponse": "langchain_azure_ai.tools.builtin._tools",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
