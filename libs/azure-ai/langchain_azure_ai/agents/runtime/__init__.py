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

Quick start::

    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_azure_ai.agents.runtime import LangGraphAgentServerHost

    builder = StateGraph(MessagesState)
    builder.add_node("agent", my_agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    host = LangGraphAgentServerHost(
        graph=graph,
        system_message="You are a helpful assistant.",
    )

    if __name__ == "__main__":
        host.run()
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.runtime._host import AzureAIResponsesAgentHost

__all__ = ["AzureAIResponsesAgentHost"]

_module_lookup = {
    "LangGraphAgentServerHost": "langchain_azure_ai.agents.runtime._host",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
