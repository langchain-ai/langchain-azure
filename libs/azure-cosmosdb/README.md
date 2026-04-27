# langchain-azure-cosmosdb

Azure CosmosDB NoSQL integrations for [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/).

## Installation

```bash
pip install langchain-azure-cosmosdb
```

## Integrations

| Integration | Sync | Async | Description |
|---|---|---|---|
| **Vector Store** | `AzureCosmosDBNoSqlVectorSearch` | `AsyncAzureCosmosDBNoSqlVectorSearch` | Vector, full-text, hybrid, and weighted hybrid search |
| **Semantic Cache** | `AzureCosmosDBNoSqlSemanticCache` | `AsyncAzureCosmosDBNoSqlSemanticCache` | LLM response caching backed by CosmosDB |
| **Chat History** | `CosmosDBChatMessageHistory` | `AsyncCosmosDBChatMessageHistory` | Persistent chat message history |
| **LangGraph Checkpointer** | `CosmosDBSaverSync` | `CosmosDBSaver` | LangGraph graph state persistence |
| **LangGraph Cache** | `CosmosDBCacheSync` | `CosmosDBCache` | LangGraph node-level result caching |
| **LangGraph Store** | `CosmosDBStore` | `AsyncCosmosDBStore` | LangGraph long-term memory with optional vector search |

## Usage

### Vector Store

```python
from azure.cosmos import CosmosClient, PartitionKey
from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch

cosmos_client = CosmosClient("<endpoint>", "<key>")

vectorstore = AzureCosmosDBNoSqlVectorSearch(
    cosmos_client=cosmos_client,
    embedding=embedding,
    vector_embedding_policy={
        "vectorEmbeddings": [{
            "path": "/embedding",
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": 1536,
        }]
    },
    indexing_policy={
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
    },
    cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    cosmos_database_properties={"id": "my-database"},
    vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
    database_name="my-database",
    container_name="my-container",
)

# Add documents
vectorstore.add_texts(["Azure CosmosDB is a multi-model database."])

# Search
results = vectorstore.similarity_search("What is CosmosDB?", k=3)
```

### Semantic Cache

```python
from azure.cosmos import CosmosClient, PartitionKey
from langchain_core.globals import set_llm_cache
from langchain_azure_cosmosdb import AzureCosmosDBNoSqlSemanticCache

cosmos_client = CosmosClient("<endpoint>", "<key>")

cache = AzureCosmosDBNoSqlSemanticCache(
    cosmos_client=cosmos_client,
    embedding=embedding,
    vector_embedding_policy=vector_embedding_policy,
    indexing_policy=indexing_policy,
    cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
    cosmos_database_properties={"id": "cache-db"},
    vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
    database_name="cache-db",
    container_name="cache-container",
)

set_llm_cache(cache)

# First call hits LLM, second call returns cached result
response = llm.invoke("What is CosmosDB?")
```

### Chat Message History

```python
from langchain_azure_cosmosdb import CosmosDBChatMessageHistory

history = CosmosDBChatMessageHistory(
    cosmos_endpoint="<endpoint>",
    credential="<key>",  # or a TokenCredential for AAD
    cosmos_database="chat-db",
    cosmos_container="chat-container",
    session_id="session-001",
    user_id="user-alice",
    ttl=3600,  # optional: messages expire after 1 hour
)
history.prepare_cosmos()

history.add_user_message("Hello!")
history.add_ai_message("Hi there!")
print(history.messages)
```

### LangGraph Checkpointer

#### Sync

```python
from langchain_azure_cosmosdb import CosmosDBSaverSync

# Sync — uses COSMOSDB_ENDPOINT / COSMOSDB_KEY env vars or explicit params
checkpointer = CosmosDBSaverSync(
    database_name="langgraph-db",
    container_name="checkpoints",
    endpoint="<endpoint>",
    key="<key>",
)

graph = workflow.compile(checkpointer=checkpointer)
result = graph.invoke(input, config={"configurable": {"thread_id": "1"}})
```

#### Async

```python
from langchain_azure_cosmosdb import CosmosDBSaver

# Async — use as a context manager
async with CosmosDBSaver.from_conn_info(
    endpoint="<endpoint>",
    key="<key>",
    database_name="langgraph-db",
    container_name="checkpoints",
) as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    result = await graph.ainvoke(input, config={"configurable": {"thread_id": "1"}})
```

### LangGraph Cache

#### Sync

```python
from langchain_azure_cosmosdb import CosmosDBCacheSync

cache = CosmosDBCacheSync(
    database_name="langgraph-db",
    container_name="cache",
    endpoint="<endpoint>",
    key="<key>",
)

graph = workflow.compile(cache=cache)
```

#### Async

```python
from langchain_azure_cosmosdb import CosmosDBCache

async with CosmosDBCache.from_conn_info(
    endpoint="<endpoint>",
    key="<key>",
    database_name="langgraph-db",
    container_name="cache",
) as cache:
    graph = workflow.compile(cache=cache)
    result = await graph.ainvoke(input, config={"configurable": {"thread_id": "1"}})
```

### LangGraph Store (Long-Term Memory)

#### Sync

```python
from langchain_azure_cosmosdb import CosmosDBStore

store = CosmosDBStore.from_endpoint(
    endpoint="<endpoint>",
    credential="<key>",
    database_name="langgraph-db",
    container_name="store",
    index={
        "dims": 1536,
        "embed": embedding,
        "fields": ["text"],
    },
)
store.setup()

# Store items under namespaces
store.put(("users", "alice", "preferences"), "coffee", {"text": "Dark roast"})
item = store.get(("users", "alice", "preferences"), "coffee")

# Semantic search
results = store.search(("users",), query="beverage preferences", limit=3)
```

#### Async

```python
from langchain_azure_cosmosdb import AsyncCosmosDBStore

async with AsyncCosmosDBStore.from_endpoint(
    endpoint="<endpoint>",
    credential="<key>",
    database_name="langgraph-db",
    container_name="store",
    index={
        "dims": 1536,
        "embed": embedding,
        "fields": ["text"],
    },
) as store:
    await store.setup()

    await store.aput(("users", "alice", "preferences"), "coffee", {"text": "Dark roast"})
    item = await store.aget(("users", "alice", "preferences"), "coffee")

    results = await store.asearch(("users",), query="beverage preferences", limit=3)
```

## Authentication

All integrations support both **access key** and **Microsoft Entra ID (AAD / Managed Identity)** authentication:

```python
# Access key
from azure.cosmos import CosmosClient
client = CosmosClient("<endpoint>", "<key>")

# AAD / Managed Identity
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
client = CosmosClient("<endpoint>", credential=DefaultAzureCredential())
```

The LangGraph integrations that manage their own client — `CosmosDBSaverSync` / `CosmosDBSaver`, `CosmosDBCacheSync` / `CosmosDBCache`, and `CosmosDBStore` / `AsyncCosmosDBStore` — fall back to `DefaultAzureCredential` automatically when no key is provided. The semantic cache (`AzureCosmosDBNoSqlSemanticCache`) and vectorstore require you to pass a `CosmosClient` explicitly.

## Samples

See the [samples/cosmosdb-nosql/](../../samples/cosmosdb-nosql/) directory for runnable end-to-end examples of every integration.

## Changelog

### 1.0.0

Initial release of `langchain-azure-cosmosdb` — a standalone package consolidating all Azure CosmosDB NoSQL integrations for LangChain and LangGraph.

**LangChain Integrations:**
- `AzureCosmosDBNoSqlVectorSearch` / `AsyncAzureCosmosDBNoSqlVectorSearch` — Vector, full-text, hybrid, and weighted hybrid search
- `AzureCosmosDBNoSqlSemanticCache` / `AsyncAzureCosmosDBNoSqlSemanticCache` — LLM semantic response caching
- `CosmosDBChatMessageHistory` / `AsyncCosmosDBChatMessageHistory` — Persistent chat message history with TTL support

**LangGraph Integrations:**
- `CosmosDBSaverSync` / `CosmosDBSaver` — Graph state checkpointing
- `CosmosDBCacheSync` / `CosmosDBCache` — Node-level result caching
- `CosmosDBStore` / `AsyncCosmosDBStore` — Long-term memory store with optional vector search

**Highlights:**
- Full sync and async support for all integrations
- Microsoft Entra ID (AAD / Managed Identity) authentication across all integrations
- User agent tracking for all CosmosDB client instances
