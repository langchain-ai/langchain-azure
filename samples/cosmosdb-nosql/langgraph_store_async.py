"""LangGraph Store with CosmosDB (Async) — Long-Term Memory.

Async version of the LangGraph store sample using AsyncCosmosDBStore
with the ``from_endpoint`` async context manager.

Prerequisites:
    pip install -r requirements.txt
    cp .env.example .env  # fill in your values

Environment variables:
    COSMOSDB_ENDPOINT                  - CosmosDB account endpoint
    COSMOSDB_KEY                       - CosmosDB account key
    AZURE_OPENAI_ENDPOINT              - Azure OpenAI endpoint
    AZURE_OPENAI_API_KEY               - Azure OpenAI API key
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT  - Embedding model deployment name
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from langchain_azure_cosmosdb import AsyncCosmosDBStore

load_dotenv()

DATABASE_NAME = "sample-async-lgstore-db"
CONTAINER_NAME = "sample-async-lgstore"


async def main() -> None:
    """Run the async LangGraph store sample."""
    from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

    try:
        embedding = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )

        # --- Create async store with vector search ---
        async with AsyncCosmosDBStore.from_endpoint(
            endpoint=os.environ["COSMOSDB_ENDPOINT"],
            credential=os.environ["COSMOSDB_KEY"],
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
            index={
                "dims": 1536,
                "embed": embedding,
                "fields": ["text"],
            },
        ) as store:
            await store.setup()
            print("Async store created and initialized.\n")

            # --- Put items ---
            print("=== Storing Items (async) ===")
            await store.aput(
                ("users", "carol", "preferences"),
                "food",
                {"text": "Carol is vegetarian and loves Thai food."},
            )
            await store.aput(
                ("users", "carol", "preferences"),
                "music",
                {"text": "Carol enjoys jazz and classical music."},
            )
            await store.aput(
                ("users", "carol", "notes"),
                "project-alpha",
                {"text": "Carol is leading Project Alpha, launching in Q3."},
            )
            print("  Stored 3 items.\n")

            # --- Get ---
            print("=== Get Item (async) ===")
            item = await store.aget(("users", "carol", "preferences"), "food")
            if item:
                print(f"  Key: {item.key}")
                print(f"  Value: {item.value}\n")

            # --- Search ---
            print("=== Search by Namespace (async) ===")
            results = await store.asearch(("users", "carol"), limit=10)
            print(f"  Found {len(results)} items:")
            for r in results:
                print(f"    [{'/'.join(r.namespace)}] {r.key}: {r.value}")
            print()

            # --- Semantic search ---
            print("=== Semantic Search (async) ===")
            results = await store.asearch(
                ("users",),
                query="What kind of food does someone like?",
                limit=3,
            )
            print(f"  Found {len(results)} results:")
            for r in results:
                score = getattr(r, "score", None)
                score_str = f"{score:.4f}" if score is not None else "N/A"
                print(f"    [score={score_str}] {r.key}: {r.value['text']}")
            print()

            # --- Delete ---
            print("=== Delete Item (async) ===")
            await store.adelete(("users", "carol", "preferences"), "music")
            item = await store.aget(("users", "carol", "preferences"), "music")
            print(f"  After delete, item exists: {item is not None}\n")
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
