"""Vector Store for CosmosDB NoSql — DEPRECATED.

This module has moved to ``langchain_azure_cosmosdb``.
Install and import directly from there instead::

    pip install langchain-azure-cosmosdb
    from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch
"""

import warnings

try:
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
except ImportError:
    pass

__all__ = [
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBNoSqlVectorStoreRetriever",
]
