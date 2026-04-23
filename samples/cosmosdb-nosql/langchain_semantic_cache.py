"""Azure CosmosDB NoSQL Semantic Cache sample.

Demonstrates using CosmosDB as a semantic cache for LLM responses, showing
cache hits and misses with latency comparison.

Prerequisites:
    pip install -r requirements.txt
    cp .env.example .env  # fill in your values

Environment variables:
    COSMOSDB_ENDPOINT                  - CosmosDB account endpoint
    COSMOSDB_KEY                       - CosmosDB account key
    AZURE_OPENAI_ENDPOINT              - Azure OpenAI endpoint
    AZURE_OPENAI_API_KEY               - Azure OpenAI API key
    AZURE_OPENAI_CHAT_DEPLOYMENT       - Chat model deployment name
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT  - Embedding model deployment name
"""

import os
import time

from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_azure_cosmosdb import AzureCosmosDBNoSqlSemanticCache

load_dotenv()

DATABASE_NAME = "sample-cache-db"
CONTAINER_NAME = "sample-cache-container"


def main() -> None:
    """Run semantic cache sample."""
    cosmos_client = CosmosClient(
        os.environ["COSMOSDB_ENDPOINT"],
        os.environ["COSMOSDB_KEY"],
    )
    try:
        embedding = AzureOpenAIEmbeddings(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
        )

        llm = AzureChatOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        )

        # --- Set up semantic cache ---
        cosmos_container_properties = {"partition_key": PartitionKey(path="/id")}
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536,
                }
            ]
        }
        indexing_policy = {
            "indexingMode": "consistent",
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
        }

        cache = AzureCosmosDBNoSqlSemanticCache(
            cosmos_client=cosmos_client,
            embedding=embedding,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties={"id": DATABASE_NAME},
            vector_search_fields={
                "text_field": "text",
                "embedding_field": "embedding",
            },
            database_name=DATABASE_NAME,
            container_name=CONTAINER_NAME,
        )

        set_llm_cache(cache)
        print("Semantic cache configured.\n")

        # --- First call: cache miss ---
        prompt = "What is Azure CosmosDB in one sentence?"
        print(f"Prompt: '{prompt}'")

        start = time.time()
        response1 = llm.invoke(prompt)
        elapsed1 = time.time() - start
        print(f"  Response: {response1.content}")
        print(f"  Time: {elapsed1:.2f}s (cache MISS — called LLM)\n")

        # --- Second call: cache hit (exact same prompt) ---
        print(f"Prompt (same): '{prompt}'")
        start = time.time()
        response2 = llm.invoke(prompt)
        elapsed2 = time.time() - start
        print(f"  Response: {response2.content}")
        print(f"  Time: {elapsed2:.2f}s (cache HIT — no LLM call)\n")

        # --- Third call: semantically similar prompt ---
        similar_prompt = "Describe Azure Cosmos DB briefly."
        print(f"Prompt (similar): '{similar_prompt}'")
        start = time.time()
        response3 = llm.invoke(similar_prompt)
        elapsed3 = time.time() - start
        print(f"  Response: {response3.content}")
        print(f"  Time: {elapsed3:.2f}s (likely cache HIT — semantically similar)\n")

        # --- Clear cache ---
        print("Clearing cache...")
        cache.clear()
        print("Cache cleared.\n")
    finally:
        # --- Cleanup ---
        print("Cleaning up...")
        try:
            cosmos_client.delete_database(DATABASE_NAME)
            print("Done! Database deleted.")
        except Exception:
            print("Database may not have been created; skipping cleanup.")


if __name__ == "__main__":
    main()
