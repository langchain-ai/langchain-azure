"""LangGraph Store with CosmosDB (Sync) — Long-Term Memory.

End-to-end sample demonstrating CosmosDBStore for LangGraph long-term memory.
Shows put/get/search/delete operations, namespace organization, and
optional semantic search over stored items.

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

import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from langchain_azure_cosmosdb import CosmosDBStore

load_dotenv()

DATABASE_NAME = "sample-lgstore-db"
CONTAINER_NAME = "sample-lgstore"


def main() -> None:
    """Run the LangGraph store sample."""
    from azure.cosmos import CosmosClient

    cleanup_client = CosmosClient(
        os.environ["COSMOSDB_ENDPOINT"],
        os.environ["COSMOSDB_KEY"],
    )
    try:
        embedding = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )

        # --- Create store with vector search enabled ---
        store = CosmosDBStore.from_endpoint(
            endpoint=os.environ["COSMOSDB_ENDPOINT"],
            credential=os.environ["COSMOSDB_KEY"],
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
            index={
                "dims": 1536,
                "embed": embedding,
                "fields": ["text"],
            },
        )
        store.setup()
        print("Store created and initialized.\n")

        # --- Put: Store items under namespaces ---
        print("=== Storing Items ===")
        store.put(
            ("users", "alice", "preferences"),
            "coffee",
            {"text": "Alice prefers dark roast coffee with oat milk."},
        )
        store.put(
            ("users", "alice", "preferences"),
            "travel",
            {"text": "Alice likes budget travel and staying in hostels."},
        )
        store.put(
            ("users", "alice", "notes"),
            "meeting-2024-01",
            {"text": "Discussed Q1 roadmap. Alice will lead the backend migration."},
        )
        store.put(
            ("users", "bob", "preferences"),
            "coffee",
            {"text": "Bob drinks green tea, no coffee."},
        )
        print("  Stored 4 items across 3 namespaces.\n")

        # --- Get: Retrieve a specific item ---
        print("=== Get Item ===")
        item = store.get(("users", "alice", "preferences"), "coffee")
        if item:
            print(f"  Key: {item.key}")
            print(f"  Value: {item.value}")
            print(f"  Namespace: {item.namespace}")
            print(f"  Updated: {item.updated_at}\n")

        # --- Search: Find items by namespace prefix ---
        print("=== Search by Namespace ===")
        results = store.search(("users", "alice"), limit=10)
        print(f"  Found {len(results)} items under ('users', 'alice'):")
        for r in results:
            print(f"    [{'/'.join(r.namespace)}] {r.key}: {r.value}")
        print()

        # --- Semantic search ---
        print("=== Semantic Search ===")
        results = store.search(
            ("users",),
            query="What kind of beverages do people like?",
            limit=3,
        )
        print(f"  Found {len(results)} results:")
        for r in results:
            score = getattr(r, "score", None)
            score_str = f"{score:.4f}" if score is not None else "N/A"
            print(f"    [score={score_str}] {r.key}: {r.value['text']}")
        print()

        # --- List namespaces ---
        print("=== List Namespaces ===")
        namespaces = store.list_namespaces(prefix=("users",))
        print(f"  Namespaces under ('users',): {namespaces}\n")

        # --- Delete ---
        print("=== Delete Item ===")
        store.delete(("users", "alice", "preferences"), "travel")
        item = store.get(("users", "alice", "preferences"), "travel")
        print(f"  After delete, item exists: {item is not None}\n")
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
