"""Sample 03 - Responses API + remote MCP server tools.

Demonstrates loading tools from a **remote MCP server** via
[``langchain-mcp-adapters``](https://github.com/langchain-ai/langchain-mcp-adapters)
and hosting the resulting LangGraph agent through the Responses API
using ``langchain_azure_ai.agents.hosting.ResponsesHostServer``.

The sample connects to GitHub's remote MCP server at
``https://api.githubcopilot.com/mcp/`` using a Personal Access Token.

Because ``MultiServerMCPClient.get_tools()`` is asynchronous, we fetch
the tool list once at startup with ``asyncio.run(...)`` and then build
the ``create_agent`` graph synchronously - the same shape as
sample 04 (Foundry Toolbox), but pointed at a generic remote MCP
endpoint instead of a Foundry-managed one.

Required environment variables (set in ``.env`` or your shell):

    FOUNDRY_PROJECT_ENDPOINT        e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    GITHUB_PAT    a GitHub PAT with the scopes the agent should access
    MCP_SERVER_URL                  optional, defaults to https://api.githubcopilot.com/mcp/
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    pip install -r requirements.txt
    python main.py

Then in another terminal (replace the prompt with a question that
exercises a GitHub MCP tool, e.g. listing your repos)::

    # Non-streaming
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"List my 5 most recently updated GitHub repos.","model":"gpt-4o"}'

    # Streaming
    curl -N -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"Search GitHub issues mentioning langchain-azure.","model":"gpt-4o","stream":true}'
"""
from __future__ import annotations

import asyncio
import os
from typing import List

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

load_dotenv()

_DEFAULT_MCP_URL = "https://api.githubcopilot.com/mcp/"


def _build_chat_model() -> AzureAIOpenAIApiChatModel:
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    return AzureAIOpenAIApiChatModel(
        project_endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
        model=deployment,
    )


async def _load_mcp_tools() -> List[BaseTool]:
    """Fetch the LangChain-compatible tool list from the remote MCP server.

    Uses ``langchain_mcp_adapters.client.MultiServerMCPClient`` with the
    streamable-HTTP transport and a bearer token header sourced from
    ``GITHUB_PAT``.
    """
    mcp_url = os.environ.get("MCP_SERVER_URL", _DEFAULT_MCP_URL)
    pat = os.environ["GITHUB_PAT"]

    client = MultiServerMCPClient(
        {
            "github": {
                "transport": "http",
                "url": mcp_url,
                "headers": {"Authorization": f"Bearer {pat}"},
            }
        }
    )
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tool(s) from MCP server '{mcp_url}':")
    for t in tools:
        print(f"  - {t.name}")
    return tools


def main() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    tools = asyncio.run(_load_mcp_tools())

    graph = create_agent(_build_chat_model(), tools=tools)
    port = int(os.environ.get("PORT", "8088"))
    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
