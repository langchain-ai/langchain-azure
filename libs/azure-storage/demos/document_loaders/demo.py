"""Sample demonstrating a chatbot using Azure AI and Azure Search."""

import os
import warnings

from azure.identity import DefaultAzureCredential
from langchain_azure_ai.chat_models import (
    AzureAIChatCompletionsModel,  # type: ignore[import-untyped]
)
from langchain_azure_ai.embeddings import (
    AzureAIEmbeddingsModel,  # type: ignore[import-untyped]
)
from langchain_azure_ai.vectorstores import AzureSearch  # type: ignore[import-untyped]
from langchain_community.vectorstores.azuresearch import AzureSearchVectorStoreRetriever

warnings.filterwarnings("ignore")


def get_chat_model() -> AzureAIChatCompletionsModel:
    """Initialize and return the Azure AI Chat Completions Model."""
    chat_model = AzureAIChatCompletionsModel(
        endpoint=os.environ["AZURE_CHAT_ENDPOINT"],
        credential=DefaultAzureCredential(),
        model="gpt-4.1-mini",
        client_kwargs={
            "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
        },
    )
    return chat_model


def get_azure_search() -> AzureSearch:
    """Initialize and return the Azure Search vector store."""
    embed_model = AzureAIEmbeddingsModel(
        endpoint=os.environ["AZURE_EMBEDDING_ENDPOINT"],
        credential=DefaultAzureCredential(),
        model="text-embedding-3-large",
        client_kwargs={
            "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
        },
    )
    azure_search = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        azure_search_key=None,
        azure_credential=DefaultAzureCredential(),
        index_name="demo-documents",
        embedding_function=embed_model,
        additional_search_client_options={
            "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
        },
    )

    return azure_search


def create_retriever() -> AzureSearchVectorStoreRetriever:
    """Create and return a retriever from Azure Search."""
    azure_search = get_azure_search()
    retriever = azure_search.as_retriever(
        search_type="similarity",
        k=3,
    )
    return retriever


def get_response(
    query: str,
    retriever: AzureSearchVectorStoreRetriever,
    llm: AzureAIChatCompletionsModel,
) -> str:
    """Get a response from the LLM based on the retrieved documents."""
    documents = retriever.invoke(query)
    context = "\n\n".join(
        [
            f"Document {doc.metadata["source"]}:\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ]
    )
    prompt = f"""You are an AI assistant. Use the following context to answer the
        question otherwise say you do not know. Include the URL for the document. 
        Documents: {context} Question: {query} Answer:"""
    response = llm.invoke(prompt)
    return response.content


def chatbot() -> None:
    """Main chatbot loop."""
    retriever = create_retriever()
    llm = get_chat_model()
    print("Welcome! Type 'exit' to quit.")

    while True:
        user_input = input("\n\nYou: ")
        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break
        try:
            response = get_response(user_input, retriever, llm)
            print(f"\nAI: {response}")
        except Exception as e:
            print(f"Error processing your request: {e}")


if __name__ == "__main__":
    chatbot()
