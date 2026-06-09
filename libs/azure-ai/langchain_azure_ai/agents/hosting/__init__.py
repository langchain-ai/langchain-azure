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

To run your agent in Foundry, use either ``InvocationsHostServer`` or
``ResponsesHostServer`` depending on the API you want to use.

Quick start (Responses API)::

    import os

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    from langchain_azure_ai.agents.hosting import ResponsesHostServer

    model = ChatOpenAI(
        model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o"),
    )
    graph = create_agent(model, tools=[])

    if __name__ == "__main__":
        ResponsesHostServer(graph).run(port=int(os.environ.get("PORT", "8088")))

Quick start (Invocations API with session continuity)::

    import os

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver

    from langchain_azure_ai.agents.hosting import InvocationsHostServer

    model = ChatOpenAI(
        model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o"),
    )
    graph = create_agent(model, tools=[], checkpointer=MemorySaver())

    if __name__ == "__main__":
        InvocationsHostServer(graph).run(port=int(os.environ.get("PORT", "8088")))

Then call the local host from another process::

    curl -N -X POST http://127.0.0.1:8088/responses \
        -H 'Content-Type: application/json' \
        -d '{"input":"Hello!","stream":true}'

    curl -i -X POST http://127.0.0.1:8088/invocations \
        -H 'Content-Type: application/json' \
        -d '{"message":"My name is Alice."}'

    curl -X POST 'http://127.0.0.1:8088/invocations?agent_session_id=<id>' \
        -H 'Content-Type: application/json' \
        -d '{"message":"What is my name?"}'

See the ``samples/hosting/langgraph-hosted-agents`` directory for complete
Foundry-backed examples, Dockerfiles, and deployment manifests.

For multi-protocol or custom-route scenarios, pass a compatible
``ResponsesAgentServerHost`` or ``InvocationAgentServerHost`` object as
``app`` and write your own ``@response_handler`` / ``@invoke_handler``.
These host and option types are re-exported from this package so
applications do not need to import Azure SDK modules directly.
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

# Third leg of UA propagation: wrap the ``openai`` SDK client classes'
# ``__init__`` so any ``OpenAI`` / ``AsyncOpenAI`` / ``AzureOpenAI`` /
# ``AsyncAzureOpenAI`` instance constructed inside a process that imports
# this hosting layer automatically carries the hosting-SDK UA in its
# ``default_headers``. This is the only path that picks up ``ChatOpenAI``
# / ``AzureChatOpenAI`` (which build their own ``openai`` client and do
# not read ``AZURE_HTTP_USER_AGENT``), as well as ``init_chat_model``,
# raw ``openai`` SDK usage, eval libraries, etc.
#
# Patching at the ``openai`` SDK layer rather than the ``httpx`` layer
# scopes the mutation to OpenAI-bound traffic only — no URL filter, no
# surprise stamping of unrelated HTTP requests in the same process.
#
# ``openai`` is an optional runtime dependency of this package: it is
# resolved lazily and the patch is silently skipped if the package is
# not installed in the environment.
_OPENAI_INIT_PATCHED = False


def _install_openai_user_agent_stamp() -> None:
    global _OPENAI_INIT_PATCHED
    if _OPENAI_INIT_PATCHED:
        return

    try:
        openai = importlib.import_module("openai")
    except ImportError:
        return

    _OPENAI_INIT_PATCHED = True

    def _wrap(cls: type) -> None:
        orig_init = cls.__init__

        def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            orig_init(self, *args, **kwargs)
            prefix = get_user_agent()
            if not prefix:
                return
            # ``BaseClient.default_headers`` is the merge of platform +
            # auth + ``self.user_agent`` + ``self._custom_headers``;
            # ``_custom_headers`` wins. Combine our prefix with whatever
            # UA is currently effective so the SDK's own
            # ``AsyncOpenAI/Python <ver>`` (or a caller-supplied
            # override) is preserved as the suffix.
            try:
                custom = getattr(self, "_custom_headers", None)
                if custom is None:
                    custom = {}
                existing = custom.get("User-Agent") or getattr(
                    self, "user_agent", ""
                )
                if prefix in (existing or ""):
                    return
                custom["User-Agent"] = f"{prefix} {existing}".strip()
                self._custom_headers = custom
            except Exception:
                # Never let UA stamping break client construction.
                pass

        cls.__init__ = _patched_init  # type: ignore[method-assign]

    for cls_name in (
        "OpenAI",
        "AsyncOpenAI",
        "AzureOpenAI",
        "AsyncAzureOpenAI",
    ):
        cls = getattr(openai, cls_name, None)
        if cls is not None:
            _wrap(cls)


_install_openai_user_agent_stamp()

if TYPE_CHECKING:
    from azure.ai.agentserver.invocations import InvocationAgentServerHost
    from azure.ai.agentserver.responses import (
        CreateResponse,
        ResponseContext,
        ResponseEventStream,
        ResponseProviderProtocol,
        ResponsesAgentServerHost,
        ResponsesServerOptions,
    )

    from langchain_azure_ai.agents.hosting._invoke_host import (
        InvocationsHostServer,
    )
    from langchain_azure_ai.agents.hosting._responses_host import (
        ResponsesHostServer,
    )

__all__ = [
    "HOSTING_USER_AGENT",
    "CreateResponse",
    "InvocationsHostServer",
    "InvocationAgentServerHost",
    "ResponseContext",
    "ResponseEventStream",
    "ResponseProviderProtocol",
    "ResponsesAgentServerHost",
    "ResponsesHostServer",
    "ResponsesServerOptions",
    "get_user_agent",
    "with_user_agent",
]

_module_lookup = {
    "CreateResponse": "azure.ai.agentserver.responses",
    "InvocationsHostServer": "langchain_azure_ai.agents.hosting._invoke_host",
    "InvocationAgentServerHost": "azure.ai.agentserver.invocations",
    "ResponseContext": "azure.ai.agentserver.responses",
    "ResponseEventStream": "azure.ai.agentserver.responses",
    "ResponseProviderProtocol": "azure.ai.agentserver.responses",
    "ResponsesAgentServerHost": "azure.ai.agentserver.responses",
    "ResponsesHostServer": ("langchain_azure_ai.agents.hosting._responses_host"),
    "ResponsesServerOptions": "azure.ai.agentserver.responses",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
