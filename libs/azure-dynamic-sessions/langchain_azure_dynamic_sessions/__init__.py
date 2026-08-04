"""This package provides tools for managing dynamic sessions in Azure.

.. deprecated::
    Superseded by ``langchain-azure-container-apps``, which carries the
    dynamic sessions integrations forward and additionally covers Azure
    Container Apps sandboxes.
"""

import warnings

from langchain_azure_dynamic_sessions.tools.sessions import (
    SessionsBashTool,
    SessionsPythonREPLTool,
)

warnings.warn(
    "langchain-azure-dynamic-sessions is deprecated and will not receive "
    "further fixes. Use langchain-azure-container-apps instead: "
    "pip install 'langchain-azure-container-apps[dynamic-sessions]', then "
    "import from langchain_azure_container_apps.dynamic_sessions. Migrating "
    "also fixes SessionsBashBackend on deepagents >= 0.7, which is broken here.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SessionsBashTool",
    "SessionsPythonREPLTool",
]
