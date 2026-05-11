"""Sample 01 - Minimal Responses API host.

Hosts a no-tool ``create_react_agent`` graph as the Azure AI Responses
API on top of a Foundry-deployed Azure OpenAI chat model.

Required environment variables (set in `.env` or your shell):

    AZURE_AI_PROJECT_ENDPOINT       e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python sample_01_responses_basic.py

Then in another terminal:

    curl -N -X POST http://127.0.0.1:8088/responses \\
      -H 'Content-Type: application/json' \\
      -d '{"input":"Hello!","model":"gpt-4o","stream":true}'
"""
from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_azure_ai.agents.hosting import AzureAIResponsesAgentHost

load_dotenv()

_AAD_SCOPE = "https://ai.azure.com/.default"


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    credential = DefaultAzureCredential()
    token = credential.get_token(_AAD_SCOPE).token
    return ChatOpenAI(
        model=deployment,
        api_key=token,  # type: ignore[arg-type]
        base_url=f"{project_endpoint}/openai/v1",
    )


def main() -> None:
    graph = create_react_agent(_build_chat_model(), tools=[])
    port = int(os.environ.get("PORT", "8088"))
    AzureAIResponsesAgentHost(graph).run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
