"""Embedding model for Azure AI."""

from langchain_openai.embeddings import AzureOpenAIEmbeddings

from . import inference
from langchain_azure_ai.embeddings.openai import AzureAIEmbeddingsModel

__all__ = [
    "AzureAIEmbeddingsModel",
    "AzureOpenAIEmbeddings",
    "inference",
]
