"""**Vector store** stores embedded data and performs vector search.

One of the most common ways to store and search over unstructured data is to
embed it and store the resulting embedding vectors, and then query the store
and retrieve the data that are 'most similar' to the embedded query.

**Class hierarchy:**

.. code-block::

    VectorStore --> <name>  # Examples: AzureSearch, FAISS, Milvus

    BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: AzureAISearchRetriever

**Main helpers:**

.. code-block::

    Embeddings, Document
"""  # noqa: E501

from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
    AzureCosmosDBMongoVCoreVectorSearch,
)
from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
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
