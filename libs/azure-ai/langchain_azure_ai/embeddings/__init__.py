"""Embedding model for Azure AI."""

from langchain_openai.embeddings import AzureOpenAIEmbeddings

from langchain_azure_ai.embeddings.openai import AzureAIOpenAIApiEmbeddingsModel

__all__ = ["AzureOpenAIEmbeddings", "AzureAIOpenAIApiEmbeddingsModel"]
