"""Chat completions model for Azure AI."""

from langchain_openai.chat_models import AzureChatOpenAI

from langchain_azure_ai.chat_models import inference  # noqa: F401
from langchain_azure_ai.chat_models.openai import AzureAIChatCompletionsModel

__all__ = [
    "AzureAIChatCompletionsModel",
    "AzureChatOpenAI",
    "inference",
]
