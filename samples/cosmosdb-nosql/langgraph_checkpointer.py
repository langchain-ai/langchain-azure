"""LangGraph Checkpointer with CosmosDB (Sync).

End-to-end sample showing a multi-turn chatbot with state persistence
using CosmosDBSaverSync. Demonstrates thread-based memory, state
inspection, and state history.

Prerequisites:
    pip install -r requirements.txt
    cp .env.example .env  # fill in your values

Environment variables:
    COSMOSDB_ENDPOINT                  - CosmosDB account endpoint
    COSMOSDB_KEY                       - CosmosDB account key
    AZURE_OPENAI_ENDPOINT              - Azure OpenAI endpoint
    AZURE_OPENAI_API_KEY               - Azure OpenAI API key
    AZURE_OPENAI_CHAT_DEPLOYMENT       - Chat model deployment name
"""

import os
from typing import Annotated, Any

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langchain_azure_cosmosdb import CosmosDBSaverSync

load_dotenv()

DATABASE_NAME = "sample-checkpointer-db"
CONTAINER_NAME = "sample-checkpoints"


class State(TypedDict):
    """Graph state with message history."""

    messages: Annotated[list, add_messages]


def create_graph(checkpointer: CosmosDBSaverSync) -> Any:
    """Build a simple chatbot graph with CosmosDB checkpointing."""
    llm = AzureChatOpenAI(
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    )

    def chatbot(state: State) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    return graph.compile(checkpointer=checkpointer)


def main() -> None:
    """Run the checkpointer sample."""
    from azure.cosmos import CosmosClient

    cleanup_client = CosmosClient(
        os.environ["COSMOSDB_ENDPOINT"],
        os.environ["COSMOSDB_KEY"],
    )
    try:
        # --- Create checkpointer ---
        checkpointer = CosmosDBSaverSync(
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
            endpoint=os.environ["COSMOSDB_ENDPOINT"],
            key=os.environ["COSMOSDB_KEY"],
        )

        graph = create_graph(checkpointer)
        thread_config = {"configurable": {"thread_id": "demo-thread-1"}}

        # --- Turn 1 ---
        print("=== Turn 1 ===")
        result = graph.invoke(
            {"messages": [("user", "Hi! My name is Alice.")]},
            config=thread_config,
        )
        print(f"AI: {result['messages'][-1].content}\n")

        # --- Turn 2: The bot should remember the name ---
        print("=== Turn 2 (bot should remember name) ===")
        result = graph.invoke(
            {"messages": [("user", "What's my name?")]},
            config=thread_config,
        )
        print(f"AI: {result['messages'][-1].content}\n")

        # --- Inspect current state ---
        print("=== Current State ===")
        state = graph.get_state(thread_config)
        print(f"  Thread ID: {state.config['configurable']['thread_id']}")
        print(f"  Checkpoint ID: {state.config['configurable']['checkpoint_id']}")
        print(f"  Number of messages: {len(state.values['messages'])}")
        print(f"  Next nodes: {state.next}\n")

        # --- State history ---
        print("=== State History ===")
        for i, snapshot in enumerate(graph.get_state_history(thread_config)):
            print(
                f"  Checkpoint {i}: step={snapshot.metadata.get('step', '?')}, "
                f"messages={len(snapshot.values.get('messages', []))}"
            )
        print()

        # --- Different thread (no memory from thread 1) ---
        print("=== New Thread (no memory) ===")
        thread_config_2 = {"configurable": {"thread_id": "demo-thread-2"}}
        result = graph.invoke(
            {"messages": [("user", "What's my name?")]},
            config=thread_config_2,
        )
        print(f"AI: {result['messages'][-1].content}\n")
    finally:
        # --- Cleanup ---
        print("Cleaning up...")
        try:
            cleanup_client.delete_database(DATABASE_NAME)
            print("Done! Database deleted.")
        except Exception:
            print("Database may not have been created; skipping cleanup.")


if __name__ == "__main__":
    main()
