"""LangGraph Cache with CosmosDB (Async).

Async version of the LangGraph cache sample using CosmosDBCache with
the ``from_conn_info`` async context manager.

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
import time
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langchain_azure_cosmosdb import CosmosDBCache

load_dotenv()

DATABASE_NAME = "sample-async-lgcache-db"
CONTAINER_NAME = "sample-async-lgcache"


class State(TypedDict):
    """Graph state."""

    messages: Annotated[list, add_messages]


async def main() -> None:
    """Run the async LangGraph cache sample."""
    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

    try:
        llm = AzureChatOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        )

        async with CosmosDBCache.from_conn_info(
            endpoint=os.environ["COSMOSDB_ENDPOINT"],
            key=os.environ["COSMOSDB_KEY"],
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
        ) as cache:

            async def chatbot(state: State) -> dict:
                response = await llm.ainvoke(state["messages"])
                return {"messages": [response]}

            graph = StateGraph(State)
            graph.add_node("chatbot", chatbot)
            graph.add_edge(START, "chatbot")
            graph.add_edge("chatbot", END)
            app = graph.compile(cache=cache)

            config = {"configurable": {"thread_id": "async-cache-demo"}}
            input_msg = {"messages": [("user", "What is the capital of France?")]}

            # First invocation: cache miss
            print("=== First Invocation (cache miss) ===")
            start = time.time()
            result = await app.ainvoke(input_msg, config=config)
            elapsed = time.time() - start
            print(f"  Response: {result['messages'][-1].content}")
            print(f"  Time: {elapsed:.2f}s\n")

            # Second invocation: cache hit
            print("=== Second Invocation (cache hit) ===")
            start = time.time()
            result = await app.ainvoke(input_msg, config=config)
            elapsed = time.time() - start
            print(f"  Response: {result['messages'][-1].content}")
            print(f"  Time: {elapsed:.2f}s (should be faster)\n")
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
