"""Prebuilt agents for Azure AI Foundry."""

try:
    from langchain_azure_ai.agents._v2.base import ResponsesAgentNode

    __all__ = ["ResponsesAgentNode"]
except (ImportError, SyntaxError):
    __all__ = []
