"""**Retrievers** provide an interface to search and retrieve relevant documents from a data source.

Retrievers abstract the logic of querying underlying data stores (such as vector stores, search engines, or databases)
and returning documents most relevant to a user's query. They are commonly used to power search, question answering, and RAG (Retrieval-Augmented Generation) workflows.

**Class hierarchy:**

.. code-block::

    BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: AzureAISearchRetriever

**Main helpers:**

.. code-block::

    Document, Query
"""  # noqa: E501

from langchain_azure_ai.retrievers.azure_ai_search import (
    AzureAISearchRetriever,
    AzureCognitiveSearchRetriever,
)

__all__ = ["AzureAISearchRetriever", "AzureCognitiveSearchRetriever"]

_module_lookup = {
    "AzureAISearchRetriever": "langchain_azure_ai.retrievers.azure_ai_search",
    "AzureCognitiveSearchRetriever": "langchain_azure_ai.retrievers.azure_ai_search",
}
