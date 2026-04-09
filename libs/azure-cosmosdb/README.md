# langchain-azure-cosmosdb

Azure CosmosDB integrations for LangChain and LangGraph.

## Installation

```bash
pip install langchain-azure-cosmosdb
```

## Features

- **Vector Store**: `AzureCosmosDBNoSqlVectorSearch` — Vector, full-text, and hybrid search with Azure CosmosDB NoSQL
- **Semantic Cache**: `AzureCosmosDBNoSqlSemanticCache` — LLM response caching backed by CosmosDB NoSQL
- **Chat History**: `CosmosDBChatMessageHistory` — Persistent chat message history with CosmosDB NoSQL
- **Query Constructors**: `AzureCosmosDbNoSQLTranslator` — Structured query to CosmosDB NoSQL translation
- **LangGraph Checkpoint**: `CosmosDBSaver` / `CosmosDBSaverSync` — LangGraph state persistence with CosmosDB

## Quick Start

```python
from langchain_azure_cosmosdb import (
    AzureCosmosDBNoSqlVectorSearch,
    AzureCosmosDBNoSqlSemanticCache,
    CosmosDBChatMessageHistory,
    CosmosDBSaver,
    CosmosDBSaverSync,
)
```

See the [LangChain Azure documentation](https://github.com/langchain-ai/langchain-azure) for detailed usage guides.
