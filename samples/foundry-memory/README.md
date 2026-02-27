# Azure AI Foundry Memory Demo
This demo shows how to use Azure AI Foundry Memory with LangChain for long-term memory across chat sessions.

## Quick Start

> **Note:** This demo uses the dependencies from the main langchain-azure-ai Poetry environment. Make sure you have installed the project dependencies from `libs/azure-ai` before running the sample.

1. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Fill in the values for your project and deployments
   ```

2. **Run the demo:**
   ```bash
   python basic_usage.py
   ```

   **Sample output:**
   ```text
   === Turn 1 (Session A): Introduce a preference (will be extracted into long-term memory) ===
   ASSISTANT: Hi JT! Nice to meet you. I'll remember that you prefer dark roast coffee and enjoy budget trips.

   === Turn 3 (Session B): New session should recall coffee preference ===
   ASSISTANT: Your coffee preference is dark roast, and your travel style is budget-friendly.
   ```

## What This Demo Does

- Captures short-term chat history in your chosen history store
- Sends each turn to Foundry Memory for extraction and consolidation
- Retrieves cross-session memories using incremental search

## Key Concepts

- **Scope**: The stable identifier for memory isolation (e.g., `user:{user_id}` or `tenant:{org_id}`). Do not use `session_id` as scope.
- **Session**: The ephemeral chat thread identifier for short-term history only.
- **Memory Store**: Configured with chat and embedding models plus options like `user_profile_enabled` and `chat_summary_enabled`.

## Minimal Usage

```python
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_azure_ai.memory import (
    FoundryMemoryChatMessageHistory,
    FoundryMemoryRetriever,
)

client = AIProjectClient(
    endpoint="https://your-resource.azure.com/...",
    credential=DefaultAzureCredential(),
)

def base_history_factory(session_id: str):
    return InMemoryChatMessageHistory()

history = FoundryMemoryChatMessageHistory(
    client=client,
    store_name="my_store",
    scope="user:123",
    session_id="session_001",
    base_history_factory=base_history_factory,
    update_delay=0,
)

retriever = history.get_retriever(k=5)
docs = retriever.invoke("What are my preferences?")
```

## Learn More

- [Azure AI Foundry Memory Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/agents/concepts/what-is-memory)
- [Azure AI Projects SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-projects-readme)
- [LangChain Documentation](https://python.langchain.com/)
