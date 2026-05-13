# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host a compiled LangGraph graph inside Azure AI Foundry's Agent Service.

Agent Service can host agents created in LangChain/LangGraph and serve
them with the same platform guarantees Foundry provides.

You can serve and hook your agent using either the OpenAI Responses API
or the Invocations API. When using the Responses API, Microsoft Foundry
handles state automatically and securely stores it within the service.
The Invocations API is a more generic approach that lets you use input
and output schemas of your choice.

Requires the ``hosting`` extras::

    pip install langchain-azure-ai[hosting]

To run your agent in Foundry, use either ``LangGraphInvocationsHostServer`` or
``LangGraphResponsesHostServer`` depending on the API you want to use.

Quick start (Responses API)::

    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_azure_ai.agents.hosting import LangGraphResponsesHostServer

    builder = StateGraph(MessagesState)
    builder.add_node("agent", my_agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    host = LangGraphResponsesHostServer(graph)

    if __name__ == "__main__":
        host.run()

Quick start (Invocations API)::

    from langchain_azure_ai.agents.hosting import LangGraphInvocationsHostServer

    LangGraphInvocationsHostServer(my_compiled_graph).run()

For multi-protocol or custom-route scenarios, drop down to
``ResponsesAgentServerHost`` and ``InvocationAgentServerHost`` directly
and write your own ``@response_handler`` / ``@invoke_handler``.
"""

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_ai.agents.hosting._invoke_host import (
        LangGraphInvocationsHostServer,
    )
    from langchain_azure_ai.agents.hosting._responses_host import (
        LangGraphResponsesHostServer,
    )

__all__ = [
    "LangGraphInvocationsHostServer",
    "LangGraphResponsesHostServer",
]

_module_lookup = {
    "LangGraphInvocationsHostServer": "langchain_azure_ai.agents.hosting._invoke_host",
    "LangGraphResponsesHostServer": (
        "langchain_azure_ai.agents.hosting._responses_host"
    ),
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
