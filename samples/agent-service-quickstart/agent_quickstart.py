"""Quick-start sample for Azure AI Foundry Agent Service with LangChain.

Demonstrates creating an agent via AgentServiceFactory, invoking it with
a simple prompt, and cleaning up afterwards.

Prerequisites:
    pip install langchain-azure-ai[agents]
    # or: pip install langchain-azure-ai azure-ai-agents

Environment variables:
    AZURE_AI_PROJECT_ENDPOINT  - your Azure AI project endpoint
    # Authentication uses DefaultAzureCredential (az login, managed identity, etc.)
"""

from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langchain_azure_ai.agents import AgentServiceFactory

load_dotenv()


def main() -> None:
    endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    credential = DefaultAzureCredential()

    print(f"Endpoint : {endpoint}")
    print(f"Model    : {model}")
    print()

    factory = AgentServiceFactory(
        project_endpoint=endpoint,
        credential=credential,
    )

    # Create a simple agent
    agent = factory.create_prompt_agent(
        name="quickstart-echo-agent",
        model=model,
        instructions=(
            "You are a helpful assistant. Keep your replies concise — "
            "at most two sentences."
        ),
    )

    # Invoke the agent
    messages = [HumanMessage(content="What is the capital of France?")]
    print("User: What is the capital of France?")
    state = agent.invoke({"messages": messages})

    for m in state["messages"]:
        m.pretty_print()

    # Clean up the agent in Azure
    factory.delete_agent(agent)
    print("\n✅ Agent deleted successfully.")


if __name__ == "__main__":
    main()
