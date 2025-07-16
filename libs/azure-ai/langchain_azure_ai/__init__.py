"""LangChain integrations for Azure AI."""

from importlib import metadata

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

# Import key classes for easy access
from langchain_azure_ai.azure_ai_agents import AzureAIAgentsService
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

__all__ = ["AzureAIAgentsService", "AzureAIChatCompletionsModel"]
