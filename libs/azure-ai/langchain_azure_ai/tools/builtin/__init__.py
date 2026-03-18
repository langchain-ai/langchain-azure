"""Built-in server-side tools for OpenAI models deployed in Azure AI Foundry.

These tool classes represent server-side capabilities (web search, code
execution, image generation, etc.) that models can invoke within a single
conversational turn.  All classes inherit from :class:`BuiltinTool` (which
itself inherits from :class:`dict`) so they can be passed directly to
``model.bind_tools()`` without extra conversion.

Available tools:

- :class:`CodeInterpreterTool` – run Python code in a sandboxed container
- :class:`WebSearchTool` – search the internet
- :class:`FileSearchTool` – semantic search over uploaded vector stores
- :class:`ImageGenerationTool` – generate or edit images
- :class:`ComputerUseTool` – control a virtual computer interface
- :class:`McpTool` – call tools on a remote MCP server

Example::

    from langchain_azure_ai.tools.builtin import CodeInterpreterTool

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
        ComputerUseTool,
        FileSearchTool,
        ImageGenerationTool,
        McpTool,
        WebSearchTool,
    )

__all__ = [
    "BuiltinTool",
    "CodeInterpreterTool",
    "ComputerUseTool",
    "FileSearchTool",
    "ImageGenerationTool",
    "McpTool",
    "WebSearchTool",
]

_module_lookup = {
    "BuiltinTool": "langchain_azure_ai.tools.builtin._tools",
    "CodeInterpreterTool": "langchain_azure_ai.tools.builtin._tools",
    "ComputerUseTool": "langchain_azure_ai.tools.builtin._tools",
    "FileSearchTool": "langchain_azure_ai.tools.builtin._tools",
    "ImageGenerationTool": "langchain_azure_ai.tools.builtin._tools",
    "McpTool": "langchain_azure_ai.tools.builtin._tools",
    "WebSearchTool": "langchain_azure_ai.tools.builtin._tools",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
