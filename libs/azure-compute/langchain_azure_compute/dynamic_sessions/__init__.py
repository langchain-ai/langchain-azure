"""Azure Container Apps dynamic sessions integration for LangChain.

Provides :class:`SessionsPythonREPLTool` and :class:`SessionsBashTool`, tools
that run LLM-authored code in an ephemeral session
(``Microsoft.App/sessionPools``). Requires the optional ``dynamic-sessions``
extra::

    pip install "langchain-azure-compute[dynamic-sessions]"

Not to be confused with Azure Container Apps sandboxes
(``Microsoft.App/sandboxGroups``), a separate product.
"""

import importlib.util

# Check the dependency itself rather than catching ImportError from the import
# below: that would also swallow an unrelated ImportError (e.g. a circular
# import bug) and report it as a missing extra.
if importlib.util.find_spec("requests") is None:
    raise ImportError(
        "The Azure Container Apps dynamic sessions integration requires "
        "requests, which is provided by the 'dynamic-sessions' extra. Install "
        "it with: pip install "
        "'langchain-azure-compute[dynamic-sessions]'."
    )

from langchain_azure_compute.dynamic_sessions.tools.sessions import (
    SessionsBashTool,
    SessionsPythonREPLTool,
)

__all__ = [
    "SessionsBashTool",
    "SessionsPythonREPLTool",
]
