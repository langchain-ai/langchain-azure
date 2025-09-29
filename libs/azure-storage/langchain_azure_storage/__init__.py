"""LangChain integrations for Azure Storage."""

from importlib import metadata
from .document_loaders import AzureBlobStorageLoader

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)
