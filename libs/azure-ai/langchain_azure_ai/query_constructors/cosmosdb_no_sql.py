"""Translator for CosmosDB NoSQL — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Install and import directly from there instead::

    pip install langchain-azure-cosmosdb
    from langchain_azure_cosmosdb import AzureCosmosDbNoSQLTranslator
"""

import warnings
from typing import Any

__all__ = ["AzureCosmosDbNoSQLTranslator"]  # noqa: F822


def __getattr__(name: str) -> Any:
    if name == "AzureCosmosDbNoSQLTranslator":
        warnings.warn(
            "Importing AzureCosmosDbNoSQLTranslator from "
            "'langchain_azure_ai.query_constructors.cosmosdb_no_sql' is "
            "deprecated. Use 'from langchain_azure_cosmosdb import "
            "AzureCosmosDbNoSQLTranslator' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            from langchain_azure_cosmosdb import (
                AzureCosmosDbNoSQLTranslator,
            )

            return AzureCosmosDbNoSQLTranslator
        except ImportError:
            raise ImportError(
                "langchain-azure-cosmosdb is required for "
                "AzureCosmosDbNoSQLTranslator. "
                "Install it with: pip install langchain-azure-cosmosdb"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
