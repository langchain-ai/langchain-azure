"""LangChain integrations for Azure Storage."""

import importlib
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_azure_storage.checkpointer import AzureTableStorageSaver
    from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "AzureBlobStorageLoader",
    "AzureTableStorageSaver",
]

_module_lookup = {
    "AzureBlobStorageLoader": "langchain_azure_storage.document_loaders",
    "AzureTableStorageSaver": "langchain_azure_storage.checkpointer",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
