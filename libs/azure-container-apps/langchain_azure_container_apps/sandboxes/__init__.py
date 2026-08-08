"""Azure Container Apps sandboxes backend for LangChain Deep Agents.

Provides :class:`ACASandbox`, an implementation of the Deep Agents
``SandboxBackendProtocol`` backed by an Azure Container Apps sandbox
(``Microsoft.App/SandboxGroups``). Requires the optional ``sandboxes`` extra
(which requires Python >= 3.11)::

    pip install "langchain-azure-container-apps[sandboxes]"

Not to be confused with Azure Container Apps dynamic sessions
(``Microsoft.App/sessionPools``), which are a separate product covered by
:mod:`langchain_azure_container_apps.dynamic_sessions`.
"""

import importlib.util

# Check the dependencies themselves rather than catching ImportError from the
# import below: that would also swallow an unrelated ImportError (e.g. a
# circular import bug) and report it as a missing extra.
if importlib.util.find_spec("deepagents") is None:
    raise ImportError(
        "The Azure Container Apps sandboxes backend requires deepagents, "
        "which is provided by the 'sandboxes' extra. Install it with: "
        "pip install 'langchain-azure-container-apps[sandboxes]' "
        "(requires Python >= 3.11)."
    )


# `find_spec` on a dotted name imports the parent package first, so when the
# SDK is absent entirely it *raises* ModuleNotFoundError for
# `azure.containerapps` rather than returning None. Both outcomes mean the
# same thing here: the extra is not installed.
try:
    _sdk_missing = importlib.util.find_spec("azure.containerapps.sandbox") is None
except ModuleNotFoundError:
    _sdk_missing = True

if _sdk_missing:
    raise ImportError(
        "The Azure Container Apps sandboxes backend requires "
        "azure-containerapps-sandbox, which is provided by the 'sandboxes' "
        "extra. Install it with: "
        "pip install 'langchain-azure-container-apps[sandboxes]'."
    )

from langchain_azure_container_apps.sandboxes.sandbox import ACASandbox

__all__ = ["ACASandbox"]
