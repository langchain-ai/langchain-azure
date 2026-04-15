"""Chat completions model for Azure AI."""

from typing import TYPE_CHECKING, Any

from langchain_openai.chat_models import AzureChatOpenAI

from langchain_azure_ai.chat_models.openai import AzureAIOpenAIApiChatModel

if TYPE_CHECKING:
    from langchain_azure_ai.chat_models.inference import AzureAIChatCompletionsModel

__all__ = [
    "AzureChatOpenAI",
    "AzureAIOpenAIApiChatModel",
    "AzureAIChatCompletionsModel",
]


def __getattr__(name: str) -> Any:
    # Redirect the old inference-based class name to the new OpenAI-API class so
    # that `langchain.chat_models.init_chat_model("azure_ai:…")` resolves to
    # AzureAIOpenAIApiChatModel even with langchain versions that still reference
    # the old name in their provider registry.  When langchain ships the updated
    # mapping pointing directly at AzureAIOpenAIApiChatModel this function is
    # never called for that lookup, making the shim automatically inert.
    if name == "AzureAIChatCompletionsModel":
        return AzureAIOpenAIApiChatModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
