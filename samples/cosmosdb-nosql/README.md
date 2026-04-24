# Azure CosmosDB NoSQL Samples

Sample scripts demonstrating all `langchain-azure-cosmosdb` integrations — LangChain vector store, semantic cache, chat history, and LangGraph checkpointer, cache, and store.

## Prerequisites

- **Azure CosmosDB NoSQL** account ([create one](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/quickstart-portal))
- **Azure OpenAI** deployment with a chat model and an embedding model ([quickstart](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart))
- Python 3.10+

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your CosmosDB and Azure OpenAI credentials
```

## Samples

### LangChain Integrations

| Sample | Description | Run |
|--------|-------------|-----|
| [langchain_vectorstore.py](langchain_vectorstore.py) | Vector, full-text, hybrid, and weighted hybrid search | `python langchain_vectorstore.py` |
| [langchain_semantic_cache.py](langchain_semantic_cache.py) | LLM semantic caching with cache hit/miss demo | `python langchain_semantic_cache.py` |
| [langchain_chat_history.py](langchain_chat_history.py) | Chat message history with multi-session and TTL | `python langchain_chat_history.py` |
| [langchain_rag_chatbot.py](langchain_rag_chatbot.py) | End-to-end RAG chatbot (vectorstore + chat history) | `python langchain_rag_chatbot.py` |

### LangGraph Integrations

| Sample | Description | Run |
|--------|-------------|-----|
| [langgraph_checkpointer.py](langgraph_checkpointer.py) | Sync checkpointer with state persistence and history | `python langgraph_checkpointer.py` |
| [langgraph_checkpointer_async.py](langgraph_checkpointer_async.py) | Async checkpointer with `from_conn_info` context manager | `python langgraph_checkpointer_async.py` |
| [langgraph_cache.py](langgraph_cache.py) | Sync graph caching with TTL | `python langgraph_cache.py` |
| [langgraph_cache_async.py](langgraph_cache_async.py) | Async graph caching | `python langgraph_cache_async.py` |
| [langgraph_store.py](langgraph_store.py) | Sync long-term memory store with semantic search | `python langgraph_store.py` |
| [langgraph_store_async.py](langgraph_store_async.py) | Async long-term memory store | `python langgraph_store_async.py` |

## Sample Descriptions

### langchain_vectorstore.py
Creates an `AzureCosmosDBNoSqlVectorSearch` with Azure OpenAI embeddings, adds documents, and demonstrates all search types: vector similarity, full-text, hybrid, and weighted hybrid search.

### langchain_semantic_cache.py
Sets up `AzureCosmosDBNoSqlSemanticCache` as the global LangChain LLM cache. Shows cache misses (first call hits LLM) vs cache hits (subsequent calls return cached results) with latency comparison.

### langchain_chat_history.py
Uses `CosmosDBChatMessageHistory` to store and retrieve conversation history. Demonstrates multi-session support (independent histories per session) and TTL-based message expiration.

### langchain_rag_chatbot.py
End-to-end RAG chatbot combining the vector store for document retrieval with chat history for conversation memory. Loads sample CosmosDB knowledge base documents and runs an interactive chat loop.

### langgraph_checkpointer.py / langgraph_checkpointer_async.py
Builds a LangGraph chatbot with `CosmosDBSaverSync` (or `CosmosDBSaver` for async) as the checkpointer. Demonstrates multi-turn memory within a thread, state inspection with `get_state()`, state history traversal, and thread isolation.

### langgraph_cache.py / langgraph_cache_async.py
Compiles a LangGraph graph with `CosmosDBCacheSync` (or `CosmosDBCache` for async) to cache node results. Shows how the second invocation with the same input returns cached results significantly faster.

### langgraph_store.py / langgraph_store_async.py
Uses `CosmosDBStore` (or `AsyncCosmosDBStore` for async) for LangGraph long-term memory. Demonstrates namespace-based organization, put/get/search/delete operations, and semantic search over stored items using vector embeddings.

## Cleanup

Each sample cleans up its own CosmosDB database at the end. If a sample is interrupted, you can manually delete the database from the Azure Portal.

## Learn More

- [langchain-azure-cosmosdb package](https://pypi.org/project/langchain-azure-cosmosdb/)
- [Azure CosmosDB Vector Search](https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search)
- [Azure CosmosDB Hybrid Search](https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
