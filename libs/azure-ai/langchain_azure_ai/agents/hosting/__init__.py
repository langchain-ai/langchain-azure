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
import importlib.metadata
import os
from typing import TYPE_CHECKING, Any

from langchain_azure_ai._user_agent import (
    add_user_agent_prefix,
    get_user_agent,
    with_user_agent,
)

try:
    _HOSTING_VERSION = importlib.metadata.version("langchain-azure-ai")
except importlib.metadata.PackageNotFoundError:
    _HOSTING_VERSION = "0.0.0"

#: UA token contributed by this subpackage.
HOSTING_USER_AGENT: str = f"langchain_azure_ai.agents.hosting/{_HOSTING_VERSION}"

# Register on import so any outbound SDK client built within a process
# that uses this hosting layer (including by user code in the hosted
# graph) automatically carries the hosting prefix.
add_user_agent_prefix(HOSTING_USER_AGENT)

# Opaque UA propagation for every ``azure-core`` based SDK client in
# this process. ``UserAgentPolicy`` (the default policy attached to all
# azure-core pipelines) reads ``AZURE_HTTP_USER_AGENT`` and appends its
# value to the outbound ``User-Agent`` header, without any per-client
# wiring at the construction site. This covers AIProjectClient,
# azure-ai-inference, azure-search-documents, azure-ai-contentunderstanding,
# azure-ai-vision-*, azure-mgmt-logic, the agentserver host's internal
# Foundry-storage calls, and the Azure Monitor exporter.
#
# ``setdefault`` so a caller-supplied value wins.
os.environ.setdefault("AZURE_HTTP_USER_AGENT", HOSTING_USER_AGENT)

if TYPE_CHECKING:
    from langchain_azure_ai.agents.hosting._invoke_host import (
        LangGraphInvocationsHostServer,
    )
    from langchain_azure_ai.agents.hosting._responses_host import (
        LangGraphResponsesHostServer,
    )

__all__ = [
    "HOSTING_USER_AGENT",
    "LangGraphInvocationsHostServer",
    "LangGraphResponsesHostServer",
    "get_user_agent",
    "with_user_agent",
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
