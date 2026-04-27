"""Azure CosmosDB NoSQL Vector Store sample.

Demonstrates vector, full-text, hybrid, and weighted hybrid search using
Azure CosmosDB NoSQL as the vector store with Azure OpenAI embeddings.

Prerequisites:
    pip install -r requirements.txt
    cp .env.example .env  # fill in your values

Environment variables:
    COSMOSDB_ENDPOINT               - CosmosDB account endpoint
    COSMOSDB_KEY                    - CosmosDB account key
    AZURE_OPENAI_ENDPOINT           - Azure OpenAI endpoint
    AZURE_OPENAI_API_KEY            - Azure OpenAI API key
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT - Embedding model deployment name
"""

import os

from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

from langchain_azure_cosmosdb import AzureCosmosDBNoSqlVectorSearch

load_dotenv()

DATABASE_NAME = "sample-vectorstore-db"
CONTAINER_NAME = "sample-vectorstore-container"


def get_embedding_model() -> AzureOpenAIEmbeddings:
    """Create an Azure OpenAI embedding model."""
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    )


def create_vectorstore(
    cosmos_client: CosmosClient,
    embedding: AzureOpenAIEmbeddings,
) -> AzureCosmosDBNoSqlVectorSearch:
    """Create a CosmosDB vector store with vector + full-text search enabled."""
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

    full_text_policy = {
        "defaultLanguage": "en-US",
        "fullTextPaths": [{"path": "/text", "language": "en-US"}],
    }

    indexing_policy = {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
        "fullTextIndexes": [{"path": "/text"}],
    }

    cosmos_container_properties = {"partition_key": PartitionKey(path="/id")}
    cosmos_database_properties = {"id": DATABASE_NAME}

    vector_search_fields = {
        "text_field": "text",
        "embedding_field": "embedding",
    }

    return AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=cosmos_client,
        embedding=embedding,
        vector_embedding_policy=vector_embedding_policy,
        full_text_policy=full_text_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties=cosmos_container_properties,
        cosmos_database_properties=cosmos_database_properties,
        vector_search_fields=vector_search_fields,
        database_name=DATABASE_NAME,
        container_name=CONTAINER_NAME,
        full_text_search_enabled=True,
    )


def main() -> None:
    """Run vector store sample."""
    cosmos_client = CosmosClient(
        os.environ["COSMOSDB_ENDPOINT"],
        os.environ["COSMOSDB_KEY"],
    )
    try:
        embedding = get_embedding_model()
        vectorstore = create_vectorstore(cosmos_client, embedding)

        # --- Add documents ---
        texts = [
            "Azure CosmosDB is a globally distributed, multi-model database service.",
            "LangChain is a framework for building applications with LLMs.",
            "Vector search enables similarity-based retrieval using embeddings.",
            "Python is a versatile programming language for data science and AI.",
            "Azure OpenAI provides access to GPT models via Azure cloud.",
        ]
        metadatas = [
            {"source": "azure-docs", "topic": "database"},
            {"source": "langchain-docs", "topic": "framework"},
            {"source": "search-docs", "topic": "search"},
            {"source": "python-docs", "topic": "language"},
            {"source": "azure-docs", "topic": "ai"},
        ]

        print("Adding documents...")
        ids = vectorstore.add_texts(texts=texts, metadatas=metadatas)
        print(f"Added {len(ids)} documents: {ids}\n")

        # --- Vector similarity search ---
        print("=== Vector Similarity Search ===")
        query = "What is CosmosDB?"
        results = vectorstore.similarity_search(query, k=3)
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
        print()

        # --- Vector similarity search with scores ---
        print("=== Vector Search with Scores ===")
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"  {i}. [score={score:.4f}] {doc.page_content}")
        print()

        # --- Full-text search ---
        print("=== Full-Text Search ===")
        results = vectorstore.similarity_search(
            "framework LLM", k=3, search_type="full_text_search"
        )
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
        print()

        # --- Hybrid search (vector + full-text with RRF) ---
        print("=== Hybrid Search ===")
        full_text_filter = [{"search_field": "text", "search_text": "Azure database"}]
        results = vectorstore.similarity_search(
            "Azure cloud database",
            k=3,
            search_type="hybrid",
            full_text_rank_filter=full_text_filter,
        )
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
        print()

        # --- Weighted hybrid search ---
        print("=== Weighted Hybrid Search (70% vector, 30% full-text) ===")
        results = vectorstore.similarity_search(
            "Azure AI services",
            k=3,
            search_type="hybrid",
            full_text_rank_filter=[
                {"search_field": "text", "search_text": "Azure AI"}
            ],
            weights=[0.3, 0.7],  # [FullTextScore weight, VectorDistance weight]
        )
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
        print()

        # --- MMR search (diverse results) ---
        print("=== MMR Search (maximal marginal relevance) ===")
        results = vectorstore.max_marginal_relevance_search(
            "Azure cloud services", k=3, fetch_k=5
        )
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
        print()

        # --- Similarity search by vector ---
        print("=== Similarity Search by Vector ===")
        query_embedding = embedding.embed_query("What is CosmosDB?")
        results = vectorstore.similarity_search_by_vector(query_embedding, k=3)
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content}")
        print()
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
