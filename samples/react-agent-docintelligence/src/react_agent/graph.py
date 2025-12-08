"""Define a custom Reasoning and Action agent.

Works with a chat model with tool calling support.
"""

from langchain_azure_ai.agents import AgentServiceFactory
from langchain_azure_ai.tools import AzureAIDocumentIntelligenceTool

from react_agent.prompts import SYSTEM_PROMPT

service = AgentServiceFactory()
graph = service.create_prompt_agent(
    name="react-agent",
    description=(
        "A simple agent that can parse documents using Azure AI Document Intelligence."
    ),
    model="gpt-4.1",
    instructions=SYSTEM_PROMPT,
    tools=[AzureAIDocumentIntelligenceTool()],
    trace=True,
)
