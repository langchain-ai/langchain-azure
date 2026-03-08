"""Embedding model for Azure AI."""

from langchain_openai.embeddings import AzureOpenAIEmbeddings

from langchain_azure_ai.embeddings.inference import AzureAIEmbeddingsModel
from langchain_azure_ai.embeddings.openai import AzureAIOpenAIEmbeddingsModel

__all__ = ["AzureAIEmbeddingsModel", "AzureAIOpenAIEmbeddingsModel"]
