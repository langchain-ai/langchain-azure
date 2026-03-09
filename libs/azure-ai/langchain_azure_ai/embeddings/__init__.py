"""Embedding model for Azure AI."""

from typing import TYPE_CHECKING

from langchain_openai.embeddings import AzureOpenAIEmbeddings

from langchain_azure_ai.embeddings.openai import AzureAIOpenAIApiEmbeddingsModel

if TYPE_CHECKING:
    from langchain_azure_ai.embeddings.inference import AzureAIEmbeddingsModel

__all__ = [
    "AzureOpenAIEmbeddings",
    "AzureAIOpenAIApiEmbeddingsModel",
    "AzureAIEmbeddingsModel",
]
