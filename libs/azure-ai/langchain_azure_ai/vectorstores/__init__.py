"""**Vector store** stores embedded data and performs vector search.

One of the most common ways to store and search over unstructured data is to
embed it and store the resulting embedding vectors, and then query the store
and retrieve the data that are 'most similar' to the embedded query.

**Class hierarchy:**

```output
VectorStore --> <name>  # Examples: AzureSearch, FAISS, Milvus

BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: AzureAISearchRetriever
```

**Main helpers:**

```output
Embeddings, Document
```
"""  # noqa: E501

import importlib
from typing import Any

from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
    AzureCosmosDBMongoVCoreVectorSearch,
)
from langchain_azure_ai.vectorstores.azuresearch import (
    AzureSearch,
)

__all__ = [
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBMongoVCoreVectorSearch",
    "AzureSearch",
]

_module_lookup = {
    "AzureCosmosDBMongoVCoreVectorSearch": "langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore",  # noqa: E501
    "AzureCosmosDBNoSqlVectorSearch": "langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql",  # noqa: E501
    "AzureSearch": "langchain_azure_ai.vectorstores.azuresearch",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
