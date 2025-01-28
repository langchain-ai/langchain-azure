"""**Vector store** stores embedded data and performs vector search.

One of the most common ways to store and search over unstructured data is to
embed it and store the resulting embedding vectors, and then query the store
and retrieve the data that are 'most similar' to the embedded query.

**Class hierarchy:**

.. code-block::

    VectorStore --> <name>  # Examples: Annoy, FAISS, Milvus

    BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: VespaRetriever

**Main helpers:**

.. code-block::

    Embeddings, Document
"""  # noqa: E501

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
        AzureCosmosDBVectorSearch,
    )
    from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
        AzureCosmosDBNoSqlVectorSearch,
    )

__all__ = [
    "AzureCosmosDBNoSqlVectorSearch",
    "AzureCosmosDBVectorSearch",
]

_module_lookup = {
    "AzureCosmosDBVectorSearch": "langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore",
    "AzureCosmosDBNoSqlVectorSearch": "langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql",  # noqa: E501
}
