"""Embedding model for Azure AI."""

from langchain_openai.embeddings import AzureOpenAIEmbeddings

from langchain_azure_ai.embeddings import inference  # noqa: F401
from langchain_azure_ai.embeddings.openai import AzureAIEmbeddingsModel

__all__ = [
    "AzureAIEmbeddingsModel",
    "AzureOpenAIEmbeddings",
    "inference",
]
