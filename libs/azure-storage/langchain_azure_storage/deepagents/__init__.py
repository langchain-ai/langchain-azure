"""Azure Blob Storage filesystem backend for LangChain Deep Agents.

This subpackage provides :class:`AzureBlobBackend`, an implementation of the
Deep Agents ``BackendProtocol`` backed by Azure Blob Storage. It requires the
optional ``deepagents`` extra (which requires Python >= 3.11)::

    pip install "langchain-azure-storage[deepagents]"
"""

try:
    from langchain_azure_storage.deepagents.backend import AzureBlobBackend
    from langchain_azure_storage.deepagents.config import AzureBlobConfig
except ImportError as e:
    raise ImportError(
        "The Deep Agents Azure Blob Storage backend requires the 'deepagents' "
        "extra. Install it with: "
        "pip install 'langchain-azure-storage[deepagents]' "
        "(requires Python >= 3.11)."
    ) from e

__all__ = ["AzureBlobBackend", "AzureBlobConfig"]
