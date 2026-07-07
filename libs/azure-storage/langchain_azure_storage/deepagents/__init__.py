"""Azure Blob Storage filesystem backend for LangChain Deep Agents.

This subpackage provides :class:`AzureBlobBackend`, an implementation of the
Deep Agents ``BackendProtocol`` backed by Azure Blob Storage. It requires the
optional ``deepagents`` extra (which requires Python >= 3.11)::

    pip install "langchain-azure-storage[deepagents]"
"""

# TODO(#815): remove this import-time shim when the package minimum moves to
# 3.11 — revisit at Python 3.10 EOL (Oct 2026) or when langchain core drops 3.10.
try:
    from langchain_azure_storage.deepagents.backend import AzureBlobBackend
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The Deep Agents Azure Blob Storage backend requires the 'deepagents' "
        "extra. Install it with: "
        "pip install 'langchain-azure-storage[deepagents]' "
        "(requires Python >= 3.11)."
    ) from e

__all__ = ["AzureBlobBackend"]
