# Copyright (c) Microsoft. All rights reserved.

"""Host a compiled LangGraph graph inside Azure AI Foundry's Agent Service.

This module provides the *server/host* side of the Foundry Agent Service
integration.  It bridges the Responses API protocol with a LangGraph graph,
handling message translation, streaming and cancellation.

Unlike :mod:`langchain_azure_ai.agents` (which is a *client* that creates and
calls Foundry-hosted agents), this module makes *your LangGraph graph* the
agent that Foundry calls into.

Requires the ``runtime`` extras::

    pip install langchain-azure-ai[runtime]

Error handling overview:

``AzureAIInvokeAgentHost``
    Uses an HTTP request/response model. Handled parser failures are returned
    as JSON error payloads, while graph/runtime failures outside those parser
    hooks are delegated to the underlying invocation server.

``AzureAIResponsesAgentHost``
    Uses a streaming Responses API model. Custom parser failures are surfaced
    as ``response.failed`` lifecycle events on the stream, while default
    request validation and non-parser runtime failures continue through the
    underlying Responses pipeline.

Quick start::

    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_azure_ai.agents.runtime import (
        AzureAIInvokeAgentHost,
        AzureAIResponsesAgentHost,
    )

    builder = StateGraph(MessagesState)
    builder.add_node("agent", my_agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    host = AzureAIResponsesAgentHost(
        graph=graph,
    )

    if __name__ == "__main__":
        host.run()
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.runtime._invoke_host import (
        AzureAIInvokeAgentHost,
        GraphInvocationInput,
        InvokeInputParser,
        InvokeInputRequest,
        InvokeOutputParser,
        JSONValue,
    )
    from langchain_azure_ai.agents.runtime._responses_host import (
        AzureAIResponsesAgentHost,
        ResponsesInputContext,
        ResponsesInputParser,
        ResponsesInputRequest,
        ResponsesOutputItem,
        ResponsesOutputParser,
    )

__all__ = [
    "AzureAIResponsesAgentHost",
    "AzureAIInvokeAgentHost",
    "GraphInvocationInput",
    "InvokeInputRequest",
    "InvokeInputParser",
    "InvokeOutputParser",
    "JSONValue",
    "ResponsesInputRequest",
    "ResponsesInputContext",
    "ResponsesInputParser",
    "ResponsesOutputItem",
    "ResponsesOutputParser",
]

_module_lookup = {
    "AzureAIInvokeAgentHost": "langchain_azure_ai.agents.runtime._invoke_host",
    "GraphInvocationInput": "langchain_azure_ai.agents.runtime._invoke_host",
    "AzureAIResponsesAgentHost": "langchain_azure_ai.agents.runtime._responses_host",
    "InvokeInputRequest": "langchain_azure_ai.agents.runtime._invoke_host",
    "InvokeInputParser": "langchain_azure_ai.agents.runtime._invoke_host",
    "InvokeOutputParser": "langchain_azure_ai.agents.runtime._invoke_host",
    "JSONValue": "langchain_azure_ai.agents.runtime._invoke_host",
    "ResponsesInputRequest": "langchain_azure_ai.agents.runtime._responses_host",
    "ResponsesInputContext": "langchain_azure_ai.agents.runtime._responses_host",
    "ResponsesInputParser": "langchain_azure_ai.agents.runtime._responses_host",
    "ResponsesOutputItem": "langchain_azure_ai.agents.runtime._responses_host",
    "ResponsesOutputParser": "langchain_azure_ai.agents.runtime._responses_host",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
