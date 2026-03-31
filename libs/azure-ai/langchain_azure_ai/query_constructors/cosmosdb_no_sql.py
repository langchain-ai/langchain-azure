"""Translator for CosmosDB NoSQL — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Import directly from there instead::

    from langchain_azure_cosmosdb import AzureCosmosDbNoSQLTranslator
"""

import warnings

from langchain_azure_cosmosdb.langchain._query_constructor import (  # noqa: F401
    AzureCosmosDbNoSQLTranslator,
)

warnings.warn(
    "Importing AzureCosmosDbNoSQLTranslator from "
    "'langchain_azure_ai.query_constructors.cosmosdb_no_sql' is deprecated. "
    "Use 'from langchain_azure_cosmosdb import "
    "AzureCosmosDbNoSQLTranslator' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["AzureCosmosDbNoSQLTranslator"]
