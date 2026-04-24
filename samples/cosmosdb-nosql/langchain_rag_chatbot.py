"""RAG Chatbot with CosmosDB Vector Store and Chat History.

End-to-end sample combining AzureCosmosDBNoSqlVectorSearch for retrieval
and CosmosDBChatMessageHistory for conversation memory.

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

from azure.cosmos import CosmosClient, PartitionKey
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from langchain_azure_cosmosdb import (
    AzureCosmosDBNoSqlVectorSearch,
    CosmosDBChatMessageHistory,
)

load_dotenv()

DATABASE_NAME = "sample-rag-chatbot-db"
VS_CONTAINER = "rag-vectorstore"
HISTORY_CONTAINER = "rag-chathistory"


def create_vectorstore(cosmos_client: CosmosClient) -> AzureCosmosDBNoSqlVectorSearch:
    """Create and populate a vector store with sample documents."""
    embedding = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    )

    vectorstore = AzureCosmosDBNoSqlVectorSearch(
        cosmos_client=cosmos_client,
        embedding=embedding,
        vector_embedding_policy={
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": 1536,
                }
            ]
        },
        indexing_policy={
            "indexingMode": "consistent",
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": '/"_etag"/?'}],
            "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
        },
        cosmos_container_properties={"partition_key": PartitionKey(path="/id")},
        cosmos_database_properties={"id": DATABASE_NAME},
        vector_search_fields={"text_field": "text", "embedding_field": "embedding"},
        database_name=DATABASE_NAME,
        container_name=VS_CONTAINER,
    )

    # Add sample knowledge base
    docs = [
        "Azure CosmosDB is a globally distributed, multi-model database. "
        "It supports NoSQL, MongoDB, PostgreSQL, and Apache Gremlin APIs.",
        "CosmosDB offers single-digit millisecond response times and "
        "99.999% availability SLA with turnkey global distribution.",
        "CosmosDB vector search supports DiskANN, flat, and quantized flat "
        "index types for efficient similarity search on embeddings.",
        "CosmosDB pricing is based on Request Units (RUs). You can choose "
        "between provisioned throughput and serverless modes.",
        "CosmosDB supports automatic indexing of all properties by default. "
        "You can customize indexing policy for better performance.",
    ]
    vectorstore.add_texts(texts=docs)
    print(f"Loaded {len(docs)} documents into vector store.\n")
    return vectorstore


def main() -> None:
    """Run the RAG chatbot."""
    cosmos_client = CosmosClient(
        os.environ["COSMOSDB_ENDPOINT"],
        os.environ["COSMOSDB_KEY"],
    )
    try:
        llm = AzureChatOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        )

        # Set up vector store and retriever
        vectorstore = create_vectorstore(cosmos_client)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Set up chat history
        history = CosmosDBChatMessageHistory(
            cosmos_endpoint=os.environ["COSMOSDB_ENDPOINT"],
            credential=os.environ["COSMOSDB_KEY"],
            cosmos_database=DATABASE_NAME,
            cosmos_container=HISTORY_CONTAINER,
            session_id="rag-session-001",
            user_id="demo-user",
        )
        history.prepare_cosmos()

        # Build the RAG chain
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions about "
                    "Azure CosmosDB using the provided context. Be concise.\n\n"
                    "Context:\n{context}",
                ),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )

        def format_docs(docs: list) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # Interactive chat loop
        print(
            "CosmosDB RAG Chatbot ready! Ask questions about Azure CosmosDB.\n"
            "Press Enter to quit.\n"
        )

        while True:
            user_input = input("You: ")
            if not user_input.strip():
                print("\nGoodbye!")
                break

            history.add_user_message(user_input)

            response = chain.invoke(
                {"question": user_input, "history": history.messages[:-1]}
            )

            history.add_ai_message(response)
            print(f"AI: {response}\n")
    finally:
        # Cleanup
        print("Cleaning up...")
        try:
            cosmos_client.delete_database(DATABASE_NAME)
            print("Done! Database deleted.")
        except Exception:
            print("Database may not have been created; skipping cleanup.")


if __name__ == "__main__":
    main()
