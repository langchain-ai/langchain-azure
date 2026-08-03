"""Azure Blob Storage filesystem backend for LangChain Deep Agents.

This subpackage provides :class:`AzureBlobBackend`, an implementation of the
Deep Agents ``BackendProtocol`` backed by Azure Blob Storage. It requires the
optional ``deepagents`` extra (which requires Python >= 3.11)::

    pip install "langchain-azure-storage[deepagents]"
"""

import importlib.util

# Check `deepagents` itself rather than catching ImportError from the
# `backend` import below: that would also swallow an unrelated ImportError
# (e.g. a circular import bug) and report it as a missing extra.
if importlib.util.find_spec("deepagents") is None:
    raise ImportError(
        "The Deep Agents Azure Blob Storage backend requires the 'deepagents' "
        "extra. Install it with: "
        "pip install 'langchain-azure-storage[deepagents]' "
        "(requires Python >= 3.11)."
    )

from langchain_azure_storage.deepagents.backend import AzureBlobBackend

__all__ = ["AzureBlobBackend"]
