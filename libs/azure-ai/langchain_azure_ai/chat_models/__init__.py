"""Chat completions model for Azure AI."""

from langchain_openai.chat_models import AzureChatOpenAI

from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel

__all__ = ["AzureAIChatCompletionsModel", "AzureChatOpenAI"]
