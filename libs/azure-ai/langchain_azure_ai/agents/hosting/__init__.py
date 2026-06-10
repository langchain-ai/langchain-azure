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

# Third leg of UA propagation: wrap LLM SDK client classes' ``__init__``
# so any client instance constructed inside a process that imports this
# hosting layer automatically carries the hosting-SDK UA. The patch
# writes the merged value into ``self._custom_headers["User-Agent"]``,
# which takes precedence in the SDK's ``BaseClient.default_headers``
# merge over the built-in ``<ClientName>/Python <ver>`` token and over
# ``self.user_agent``, so the prefix lands on every outbound request.
#
# Two providers are stamped:
#
# * ``openai`` covers ``ChatOpenAI`` / ``AzureChatOpenAI`` (which build
#   their own ``openai`` client and do not read
#   ``AZURE_HTTP_USER_AGENT``), as well as ``init_chat_model``, raw
#   ``openai`` SDK usage, eval libraries, and any Foundry model exposed
#   via the OpenAI-compatible route (e.g. Mistral, Llama, Phi, DeepSeek).
# * ``anthropic`` covers Foundry's native Anthropic surface
#   (``https://<resource>.services.ai.azure.com/anthropic``), used by
#   ``langchain-anthropic``'s ``ChatAnthropic`` when pointed at Foundry,
#   and by direct ``anthropic`` SDK usage.
#
# Patching at the SDK layer rather than the ``httpx`` layer scopes the
# mutation to provider-bound traffic only — no URL filter, no surprise
# stamping of unrelated HTTP requests in the same process.
#
# Both SDKs are optional runtime dependencies of this package: they are
# resolved lazily and each patch is silently skipped if the package is
# not installed in the environment.


def _wrap_init_with_user_agent(cls: type) -> None:
    """Patch ``cls.__init__`` to merge the hosting UA prefix into outbound headers.

    Targets SDK client classes that follow the shared ``BaseClient`` shape
    used by both ``openai`` and ``anthropic``: an ``_custom_headers`` dict
    plus a ``user_agent`` attribute, with ``_custom_headers["User-Agent"]``
    winning in the outbound header merge. The per-instance
    ``if prefix in existing`` guard makes the patch idempotent across
    repeat installs and re-imports.
    """
    orig_init = cls.__init__  # type: ignore[misc]

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        prefix = get_user_agent()
        if not prefix:
            return
        # ``BaseClient.default_headers`` is the merge of platform + auth
        # + ``self.user_agent`` + ``self._custom_headers``;
        # ``_custom_headers`` wins. Combine our prefix with whatever UA
        # is currently effective so the SDK's own
        # ``<ClientName>/Python <ver>`` (or a caller-supplied override)
        # is preserved as the suffix.
        try:
            custom = getattr(self, "_custom_headers", None)
            if custom is None:
                custom = {}
            existing = custom.get("User-Agent") or getattr(self, "user_agent", "")
            if prefix in (existing or ""):
                return
            custom["User-Agent"] = f"{prefix} {existing}".strip()
            self._custom_headers = custom
        except Exception:
            # Never let UA stamping break client construction.
            pass

    cls.__init__ = _patched_init  # type: ignore[method-assign,misc]


def _install_sdk_user_agent_stamp(
    module_name: str, class_names: tuple[str, ...]
) -> bool:
    """Wrap the named classes from ``module_name`` if the SDK is importable.

    Returns ``True`` if the SDK was found and at least one class was
    patched, ``False`` if the SDK is not installed or no target class
    could be patched. Per-class failures are swallowed so a future SDK
    that makes ``__init__`` non-writable on one class cannot prevent
    the others from being patched and never breaks import of the
    hosting package.
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False

    patched = 0
    for cls_name in class_names:
        try:
            cls = getattr(module, cls_name, None)
            if cls is not None:
                _wrap_init_with_user_agent(cls)
                patched += 1
        except Exception:
            pass
    return patched > 0


_OPENAI_INIT_PATCHED = False
_ANTHROPIC_INIT_PATCHED = False


def _install_openai_user_agent_stamp() -> None:
    global _OPENAI_INIT_PATCHED
    if _OPENAI_INIT_PATCHED:
        return
    if _install_sdk_user_agent_stamp(
        "openai",
        ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"),
    ):
        _OPENAI_INIT_PATCHED = True


def _install_anthropic_user_agent_stamp() -> None:
    global _ANTHROPIC_INIT_PATCHED
    if _ANTHROPIC_INIT_PATCHED:
        return
    # ``AnthropicBedrock`` / ``AnthropicVertex`` are intentionally omitted:
    # those clients do not reach Foundry and stamping them would attribute
    # off-platform traffic to the hosting layer.
    if _install_sdk_user_agent_stamp(
        "anthropic",
        ("Anthropic", "AsyncAnthropic"),
    ):
        _ANTHROPIC_INIT_PATCHED = True


_install_openai_user_agent_stamp()
_install_anthropic_user_agent_stamp()

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
