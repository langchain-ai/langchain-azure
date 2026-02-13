"""Basic usage example for Azure AI Foundry Memory with LangChain."""

import os
import time
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    MemoryStoreDefaultDefinition,
    MemoryStoreDefaultOptions,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from langchain_azure_ai.memory import (
    FoundryMemoryChatMessageHistory,
    FoundryMemoryRetriever,
)

# Load environment variables from .env file
load_dotenv()

# 0) Set up Azure AI Project client (standard Azure SDK pattern)
endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
credential = DefaultAzureCredential()
client = AIProjectClient(endpoint=endpoint, credential=credential)

# 1) Ensure memory store exists (one-time setup - use infrastructure/scripts for prod)
store_name = "lc-integration-test-store"
try:
    store = client.memory_stores.get(store_name)
    print(f"✓ Memory store '{store_name}' already exists")
except ResourceNotFoundError:
    print(f"Creating memory store '{store_name}'...")
    definition = MemoryStoreDefaultDefinition(
        chat_model=os.environ["MEMORY_STORE_CHAT_MODEL_DEPLOYMENT_NAME"],
        embedding_model=os.environ["MEMORY_STORE_EMBEDDING_MODEL_DEPLOYMENT_NAME"],
        options=MemoryStoreDefaultOptions(
            user_profile_enabled=True,
            chat_summary_enabled=True,
        ),
    )
    store = client.memory_stores.create(
        name=store_name,
        description="Long-term memory store",
        definition=definition,
    )
    print(f"✓ Memory store '{store_name}' created successfully")


# 2) History factory
def base_history_factory(_: str) -> InMemoryChatMessageHistory:
    """Create a new in-memory chat message history."""
    return InMemoryChatMessageHistory()


# Session cache: CRITICAL for incremental search to work
# RunnableWithMessageHistory calls get_session_history on every invoke,
# so we must cache instances to preserve _previous_search_id state across turns.
_session_histories: dict[tuple[str, str], FoundryMemoryChatMessageHistory] = {}


# Scope should be stable per-user/tenant for long-term memory; NOT the session_id.
def get_session_history(user_id: str, session_id: str) -> FoundryMemoryChatMessageHistory:
    """Get or create a session history for a user and session.
    
    Args:
        user_id: Stable user identifier (used as scope in Foundry Memory)
        session_id: Ephemeral session identifier
        
    Returns:
        FoundryMemoryChatMessageHistory instance
    """
    cache_key = (user_id, session_id)
    if cache_key not in _session_histories:
        _session_histories[cache_key] = FoundryMemoryChatMessageHistory(
            client=client,
            store_name=store_name,
            scope=user_id,
            session_id=session_id,
            base_history_factory=base_history_factory,
            update_delay=0,  # TEST MODE: process updates immediately (default ~300s)
        )
    return _session_histories[cache_key]


def get_foundry_retriever(user_id: str, session_id: str) -> FoundryMemoryRetriever:
    """Get a retriever tied to the cached session history.
    
    This preserves incremental search state across turns.
    
    Args:
        user_id: Stable user identifier
        session_id: Ephemeral session identifier
        
    Returns:
        FoundryMemoryRetriever instance
    """
    return get_session_history(user_id, session_id).get_retriever(k=5)


# 3) Prompt & LLM
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful and concise. Use prior memories when relevant."),
        MessagesPlaceholder("history"),
        ("system", "Memories:\n{memories}"),
        ("human", "{question}"),
    ]
)


# Use Azure OpenAI with Entra ID authentication
def get_api_key() -> str:
    """Get bearer token for Azure OpenAI authentication."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return token_provider()


llm = ChatOpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=get_api_key(),
    model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    temperature=0.7,
)


def chain_for_session(user_id: str, session_id: str) -> RunnableWithMessageHistory:
    """Create a chain with message history for a specific user and session.
    
    Args:
        user_id: Stable user identifier
        session_id: Ephemeral session identifier
        
    Returns:
        Runnable chain with message history
    """
    retriever = get_foundry_retriever(user_id, session_id)

    def format_memories(x: dict[str, Any]) -> str:
        """Retrieve and format memories as text."""
        docs = retriever.invoke(x["question"])
        return (
            "\n".join([doc.page_content for doc in docs])
            if docs
            else "No relevant memories found."
        )

    # Use RunnablePassthrough.assign to add memories to the input dict
    # RunnableWithMessageHistory will inject history automatically
    chain = RunnablePassthrough.assign(memories=format_memories) | prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Unique identifier for the session.",
                default="",
                is_shared=True,
            ),
        ],
    )
    return chain_with_history


if __name__ == "__main__":
    user_id = "user_001"
    session_id = "session_2026_02_10_001"
    chain = chain_for_session(user_id, session_id)

    # 4) Session A: seed preferences (long-term memory extraction happens async)
    print(
        "\n=== Turn 1 (Session A): Introduce a preference "
        "(will be extracted into long-term memory) ==="
    )
    r1 = chain.invoke(
        {"question": "Hi! Call me JT. I prefer dark roast coffee and budget trips."},
        config={"configurable": {"user_id": user_id, "session_id": session_id}},
    )
    print("ASSISTANT:", r1.content)

    print("\n=== Turn 2 (Session A): Add another preference ===")
    r2 = chain.invoke(
        {
            "question": "Also, I usually drink green tea in the afternoon "
            "and I like staying in hostels."
        },
        config={"configurable": {"user_id": user_id, "session_id": session_id}},
    )
    print("ASSISTANT:", r2.content)

    # Because we set update_delay=0, extraction should happen immediately for demo.
    # If you use the default delay, you may need to wait before querying from new session.
    time.sleep(60)

    # 5) Cross-session test: same user_id, new session_id
    session_id_b = "session_2026_02_10_002"
    chain_b = chain_for_session(user_id, session_id_b)

    print("\n=== Turn 3 (Session B): New session should recall coffee preference ===")
    r4 = chain_b.invoke(
        {"question": "Remind me of my coffee preference and travel style."},
        config={"configurable": {"user_id": user_id, "session_id": session_id_b}},
    )
    print("ASSISTANT:", r4.content)

    print("\n=== Turn 4 (Session B): Retrieve another preference ===")
    r5 = chain_b.invoke(
        {
            "question": "What do I usually drink in the afternoon, "
            "and where do I like to stay?"
        },
        config={"configurable": {"user_id": user_id, "session_id": session_id_b}},
    )
    print("ASSISTANT:", r5.content)

    time.sleep(60)

    # 6) Ad-hoc cross-store query (no history_ref; non-incremental by default)
    adhoc = FoundryMemoryRetriever(
        client=client,
        store_name=store_name,
        scope=user_id,
        k=5,
    )
    print("\n=== Turn 5 (Ad-hoc): Direct retriever query without session history ===")
    adhoc_docs = adhoc.invoke("What are my drinking preferences?")
    for i, doc in enumerate(adhoc_docs, start=1):
        print(f"MEMORY {i}:", doc.page_content)

    # Cleanup: Delete all memories for this user scope to ensure test independence
    print(f"\n=== Cleanup: Deleting all memories for scope '{user_id}' ===")
    try:
        result = client.memory_stores.delete_scope(name=store_name, scope=user_id)
        print(
            f"✓ Successfully deleted {getattr(result, 'deleted_count', 'all')} "
            f"memories for scope '{user_id}'"
        )
    except Exception as e:
        print(f"⚠ Cleanup failed: {e}")
