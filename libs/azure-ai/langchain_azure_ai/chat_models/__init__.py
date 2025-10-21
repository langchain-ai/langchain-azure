"""Chat completions model for Azure AI."""

from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel
from langchain_openai.chat_models import AzureChatOpenAI

__all__ = ["AzureAIChatCompletionsModel", "AzureChatOpenAI"]
