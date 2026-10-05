"""Moderate a LangChain agent with a Content Safety guardrail policy.

Requires azure-ai-contentsafety >= 1.1.0b1 and a configured policy ID.
Set AZURE_AI_MODEL (for example, azure_ai:gpt-4.1),
AZURE_CONTENT_SAFETY_ENDPOINT, and AZURE_CONTENT_SAFETY_POLICY_ID.
Authenticate with Azure identity (for example, `az login`) or supply a
Content Safety key as the middleware credential.
"""

import os

from langchain.agents import create_agent
from langchain_core.tools import tool

from langchain_azure_ai.agents.middleware import AzureContentSafetyPolicyMiddleware
from langchain_azure_ai.agents.middleware.content_safety import (
    get_content_safety_annotations,
)


@tool
def lookup(query: str) -> str:
    """Return a sample lookup result for a query."""
    return f"Result for {query}"


agent = create_agent(
    model=os.environ["AZURE_AI_MODEL"],
    tools=[lookup],
    middleware=[
        AzureContentSafetyPolicyMiddleware(
            policy_id=os.environ["AZURE_CONTENT_SAFETY_POLICY_ID"],
        )
    ],
)

result = agent.invoke({"messages": [{"role": "user", "content": "Hello!"}]})
final_message = result["messages"][-1]
print(final_message.text)
print(get_content_safety_annotations(final_message))
