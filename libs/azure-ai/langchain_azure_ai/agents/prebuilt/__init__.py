"""Prebuilt agents for Azure AI Foundry."""

from langchain_azure_ai.agents.prebuilt.declarative import PromptBasedAgentNode
from langchain_azure_ai.agents.prebuilt.declarative_v2 import (
    PromptBasedAgentNodeV2,
)

__all__ = ["PromptBasedAgentNode", "PromptBasedAgentNodeV2"]
