"""LangGraph Checkpointer with CosmosDB (Async).

Async version of the checkpointer sample using CosmosDBSaver with the
``from_conn_info`` async context manager.

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

import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langchain_azure_cosmosdb import CosmosDBSaver

load_dotenv()

DATABASE_NAME = "sample-async-checkpointer-db"
CONTAINER_NAME = "sample-async-checkpoints"


class State(TypedDict):
    """Graph state with message history."""

    messages: Annotated[list, add_messages]


async def main() -> None:
    """Run the async checkpointer sample."""
    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

    try:
        llm = AzureChatOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        )

        async def chatbot(state: State) -> dict:
            response = await llm.ainvoke(state["messages"])
            return {"messages": [response]}

        # --- Use async context manager ---
        async with CosmosDBSaver.from_conn_info(
            endpoint=os.environ["COSMOSDB_ENDPOINT"],
            key=os.environ["COSMOSDB_KEY"],
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
        ) as checkpointer:
            graph = StateGraph(State)
            graph.add_node("chatbot", chatbot)
            graph.add_edge(START, "chatbot")
            graph.add_edge("chatbot", END)
            app = graph.compile(checkpointer=checkpointer)

            thread_config = {"configurable": {"thread_id": "async-demo-thread"}}

            # Turn 1
            print("=== Turn 1 (async) ===")
            result = await app.ainvoke(
                {"messages": [("user", "Hi! I love hiking in the mountains.")]},
                config=thread_config,
            )
            print(f"AI: {result['messages'][-1].content}\n")

            # Turn 2: bot should remember
            print("=== Turn 2 (bot should remember) ===")
            result = await app.ainvoke(
                {"messages": [("user", "What hobby did I mention?")]},
                config=thread_config,
            )
            print(f"AI: {result['messages'][-1].content}\n")

            # Inspect state
            print("=== Current State ===")
            state = await app.aget_state(thread_config)
            print(f"  Messages: {len(state.values['messages'])}")
            print(
                f"  Checkpoint: {state.config['configurable']['checkpoint_id']}\n"
            )

            # State history
            print("=== State History ===")
            i = 0
            async for snapshot in app.aget_state_history(thread_config):
                print(
                    f"  Checkpoint {i}: "
                    f"step={snapshot.metadata.get('step', '?')}, "
                    f"messages={len(snapshot.values.get('messages', []))}"
                )
                i += 1
            print()
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            async with AsyncCosmosClient(
                os.environ["COSMOSDB_ENDPOINT"],
                os.environ["COSMOSDB_KEY"],
            ) as client:
                await client.delete_database(DATABASE_NAME)
            print("Done! Database deleted.")
        except Exception:
            print("Database may not have been created; skipping cleanup.")


if __name__ == "__main__":
    asyncio.run(main())
