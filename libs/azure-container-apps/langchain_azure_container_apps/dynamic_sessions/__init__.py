"""Azure Container Apps dynamic sessions integration for LangChain.

Provides :class:`SessionsPythonREPLTool` and :class:`SessionsBashTool`, tools
that run LLM-authored code in an ephemeral session
(``Microsoft.App/sessionPools``). Requires the optional ``dynamic-sessions``
extra::

    pip install "langchain-azure-container-apps[dynamic-sessions]"

The Deep Agents backend lives in
:mod:`langchain_azure_container_apps.dynamic_sessions.backends`, which
additionally requires Python >= 3.11.

Not to be confused with Azure Container Apps sandboxes
(``Microsoft.App/sandboxGroups``), which are a separate product covered by
:mod:`langchain_azure_container_apps.sandboxes`.
"""

import importlib.util

# Check the dependency itself rather than catching ImportError from the import
# below: that would also swallow an unrelated ImportError (e.g. a circular
# import bug) and report it as a missing extra. `requests` stands in for the
# whole extra: it is the only distribution the `dynamic-sessions` extra
# installs unconditionally (deepagents is marker-gated to Python >= 3.11 and
# guarded separately by the backends subpackage).
if importlib.util.find_spec("requests") is None:
    raise ImportError(
        "The Azure Container Apps dynamic sessions integration requires "
        "requests, which is provided by the 'dynamic-sessions' extra. Install "
        "it with: pip install "
        "'langchain-azure-container-apps[dynamic-sessions]'."
    )

from langchain_azure_container_apps.dynamic_sessions.tools.sessions import (
    SessionsBashTool,
    SessionsPythonREPLTool,
)

__all__ = [
    "SessionsBashTool",
    "SessionsPythonREPLTool",
]
