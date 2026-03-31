"""Vector Store for CosmosDB NoSql — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Import directly from there instead::

    from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch
"""

import warnings

from langchain_azure_cosmosdb.langchain._vectorstore import (  # noqa: F401
    AzureCosmosDBNoSqlVectorSearch,
    AzureCosmosDBNoSqlVectorStoreRetriever,
)

warnings.warn(
    "Importing AzureCosmosDBNoSqlVectorSearch from "
    "'langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql' is deprecated. "
    "Use 'from langchain_azure_cosmosdb import "
    "AzureCosmosDBNoSqlVectorSearch' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBNoSqlVectorStoreRetriever",
]
