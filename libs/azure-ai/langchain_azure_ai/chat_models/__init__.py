"""Chat completions model for Azure AI."""

from langchain_openai.chat_models import AzureChatOpenAI

from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel
from langchain_azure_ai.chat_models.openai import AzureAIOpenAIChatCompletionsModel

__all__ = ["AzureAIChatCompletionsModel", "AzureAIOpenAIChatCompletionsModel"]
