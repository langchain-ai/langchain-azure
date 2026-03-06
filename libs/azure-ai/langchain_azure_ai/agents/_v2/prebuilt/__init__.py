"""Prebuilt agents for Azure AI Foundry."""

try:
    from langchain_azure_ai.agents._v2.prebuilt.declarative import PromptBasedAgentNode

    __all__ = ["PromptBasedAgentNode"]
except (ImportError, SyntaxError):
    __all__ = []
