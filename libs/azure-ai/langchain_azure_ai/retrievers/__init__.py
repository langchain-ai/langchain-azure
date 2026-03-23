"""**Retrievers** provide an interface to search and retrieve relevant documents from a data source.

Retrievers abstract the logic of querying underlying data stores (such as vector stores, search engines, or databases)
and returning documents most relevant to a user's query. They are commonly used to power search, question answering, and RAG (Retrieval-Augmented Generation) workflows.

**Class hierarchy:**

```output
BaseRetriever --> VectorStoreRetriever --> <name>Retriever  # Example: AzureAISearchRetriever
```

**Main helpers:**

```output
Document, Query
```
"""  # noqa: E501

from langchain_azure_ai.retrievers.azure_ai_memory_retriever import (
    AzureAIMemoryRetriever,
)
from langchain_azure_ai.retrievers.azure_ai_search import (
    AzureAISearchRetriever,
    AzureCognitiveSearchRetriever,
)

__all__ = [
    "AzureAIMemoryRetriever",
    "AzureAISearchRetriever",
    "AzureCognitiveSearchRetriever",
]

_module_lookup = {
    "AzureAIMemoryRetriever": "langchain_azure_ai.retrievers.azure_ai_memory_retriever",
    "AzureAISearchRetriever": "langchain_azure_ai.retrievers.azure_ai_search",
    "AzureCognitiveSearchRetriever": "langchain_azure_ai.retrievers.azure_ai_search",
}
