"""Embedding model for Azure AI."""

from langchain_azure_ai.embeddings.inference import AzureAIEmbeddingsModel
from langchain_openai.embeddings import AzureOpenAIEmbeddings

__all__ = ["AzureAIEmbeddingsModel", "AzureOpenAIEmbeddings"]
