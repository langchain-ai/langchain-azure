"""Embedding model for Azure AI."""

from langchain_openai.embeddings import AzureOpenAIEmbeddings

from langchain_azure_ai.embeddings.inference import AzureAIEmbeddingsModel

__all__ = ["AzureAIEmbeddingsModel", "AzureOpenAIEmbeddings"]
