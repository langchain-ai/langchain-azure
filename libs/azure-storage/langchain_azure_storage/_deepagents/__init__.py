"""Azure Blob Storage filesystem backend for LangChain Deep Agents.

This subpackage provides :class:`AzureBlobBackend`, an implementation of the
Deep Agents ``BackendProtocol`` that uses Azure Blob Storage as its virtual
filesystem. It is exposed at the top level of ``langchain_azure_storage`` and
requires the optional ``deepagents`` extra::

    pip install "langchain-azure-storage[deepagents]"
"""

from langchain_azure_storage._deepagents.backend import AzureBlobBackend
from langchain_azure_storage._deepagents.config import AzureBlobConfig

__all__ = ["AzureBlobBackend", "AzureBlobConfig"]
