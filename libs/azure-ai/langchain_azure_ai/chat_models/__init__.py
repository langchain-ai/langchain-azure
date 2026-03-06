"""Chat completions model for Azure AI."""

from langchain_openai.chat_models import AzureChatOpenAI

from . import inference
from langchain_azure_ai.chat_models.openai import AzureAIChatCompletionsModel

__all__ = [
    "AzureAIChatCompletionsModel",
    "AzureChatOpenAI",
    "inference",
]
