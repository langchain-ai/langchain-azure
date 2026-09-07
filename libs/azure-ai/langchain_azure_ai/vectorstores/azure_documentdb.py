"""Azure DocumentDB vector store compatibility imports.

This integration has moved to ``langchain_azure_documentdb``::

    pip install langchain-azure-documentdb
    from langchain_azure_documentdb import AzureDocumentDBVectorSearch
"""

import warnings
from typing import Any

_DEPRECATED_NAMES = {
    "AzureDocumentDBSimilarityType",
    "AzureDocumentDBVectorSearch",
    "AzureDocumentDBVectorSearchCompression",
    "AzureDocumentDBVectorSearchType",
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_NAMES:
        warnings.warn(
            f"Importing {name} from 'langchain_azure_ai.vectorstores' is "
            "deprecated. Use "
            f"'from langchain_azure_documentdb import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import langchain_azure_documentdb

            return getattr(langchain_azure_documentdb, name)
        except ImportError as exc:
            raise ImportError(
                f"langchain-azure-documentdb is required for {name}. "
                "Install it with: pip install langchain-azure-documentdb"
            ) from exc

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")