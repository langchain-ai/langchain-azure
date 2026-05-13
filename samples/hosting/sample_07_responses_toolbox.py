"""Sample 07 - Responses API + Azure AI Foundry Toolbox tools.

Demonstrates loading tools from an **Azure AI Foundry Toolbox** via
``langchain_azure_ai.tools.AzureAIProjectToolbox`` and hosting the
resulting LangGraph agent through the Responses API using
``langchain_azure_ai.agents.hosting.LangGraphResponsesAgentHost``.

The Foundry Toolbox is a managed multi-MCP gateway: a single endpoint
aggregates many tool servers (custom MCP servers, OpenAPI-based tools,
SharePoint search, etc.) behind one URL. ``AzureAIProjectToolbox``:

- authenticates with ``DefaultAzureCredential`` (``az login`` works),
- injects the required ``Foundry-Features`` header,
- sanitizes the tool schemas, and
- returns standard LangChain ``BaseTool`` instances ready to plug into
  any LangGraph / LangChain agent.

Because ``AzureAIProjectToolbox.get_tools()`` is asynchronous, we fetch
the tools once at startup with ``asyncio.run(...)`` and then build the
prebuilt ``create_react_agent`` graph synchronously - the same shape as
sample 02, but with Foundry-managed tools instead of a local ``@tool``.

Required environment variables (set in ``.env`` or your shell):

    AZURE_AI_PROJECT_ENDPOINT       e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    FOUNDRY_AGENT_TOOLBOX_NAME      name of the toolbox configured in Foundry
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    pip install -r requirements.txt  # pulls in langchain-mcp-adapters + httpx
    python sample_07_responses_toolbox.py

Then in another terminal (replace the prompt with one that exercises a
tool in your toolbox)::

    # Non-streaming
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"What tools do you have available?","model":"gpt-4o"}'

    # Streaming - intermediate function_call / function_call_output items
    # are surfaced for every toolbox tool the agent invokes.
    curl -N -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"<a question your toolbox can answer>","model":"gpt-4o","stream":true}'
"""
from __future__ import annotations

import asyncio
import os
from typing import List

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import LangGraphResponsesAgentHost
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.tools import AzureAIProjectToolbox

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


async def _load_toolbox_tools() -> List[BaseTool]:
    """Fetch the LangChain-compatible tool list from the Foundry Toolbox.

    ``project_endpoint`` is resolved from ``AZURE_AI_PROJECT_ENDPOINT``
    automatically. The credential defaults to ``DefaultAzureCredential``
    (so ``az login`` is enough for local dev). Each call opens a fresh
    MCP session against the toolbox and closes it before returning.
    """
    toolbox_name = os.environ.get("FOUNDRY_AGENT_TOOLBOX_NAME")
    if not toolbox_name:
        raise RuntimeError(
            "FOUNDRY_AGENT_TOOLBOX_NAME is not set. Configure a toolbox in "
            "Azure AI Foundry and set its name in your .env file."
        )

    toolbox = AzureAIProjectToolbox(toolbox_name=toolbox_name)
    tools = await toolbox.get_tools()
    print(f"Loaded {len(tools)} tool(s) from Foundry toolbox '{toolbox_name}':")
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

    # Fetch toolbox tools once at startup. The tools themselves are
    # async-safe LangChain BaseTools that open their own MCP sessions
    # on each invocation, so we don't need to keep an outer event loop.
    tools = asyncio.run(_load_toolbox_tools())

    graph = create_react_agent(_build_chat_model(), tools=tools)
    port = int(os.environ.get("PORT", "8088"))
    LangGraphResponsesAgentHost(graph).run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
