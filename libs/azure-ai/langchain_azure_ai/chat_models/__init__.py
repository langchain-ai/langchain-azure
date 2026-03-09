"""Chat completions model for Azure AI."""

from typing import TYPE_CHECKING

from langchain_openai.chat_models import AzureChatOpenAI

from langchain_azure_ai.chat_models.openai import AzureAIOpenAIApiChatModel

if TYPE_CHECKING:
    from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel

__all__ = [
    "AzureChatOpenAI",
    "AzureAIOpenAIApiChatModel",
    "AzureAIChatCompletionsModel",
]
