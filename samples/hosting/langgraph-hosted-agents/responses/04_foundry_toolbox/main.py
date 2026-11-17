"""Sample 04 - Responses API + Azure AI Foundry Toolbox tools and skills.

Demonstrates loading tools from an **Azure AI Foundry Toolbox** via
``langchain_azure_ai.tools.AzureAIProjectToolbox`` and hosting the
resulting LangGraph agent through the Responses API using
``langchain_azure_ai.agents.hosting.ResponsesHostServer``.

The Foundry Toolbox is a managed multi-MCP gateway: a single endpoint
aggregates many tool servers (custom MCP servers, OpenAPI-based tools,
SharePoint search, etc.) behind one URL. ``AzureAIProjectToolbox``:

- authenticates with ``DefaultAzureCredential`` (``az login`` works),
- injects the required ``Foundry-Features`` header,
- sanitizes the tool schemas, and
- returns standard LangChain ``BaseTool`` instances ready to plug into
  any LangGraph / LangChain agent.

Tools and the skill catalog are fetched once at startup. Only skill metadata
enters the system prompt; the agent reads instructions and supporting text
from the Toolbox on demand through ``read_skill_resource``.

See README.md for skill setup, environment variables, and runnable requests.
"""

from __future__ import annotations

import asyncio
import os

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.tools import AzureAIProjectToolbox
from langchain_core.tools import BaseTool, ToolException, tool
from langchain_openai import ChatOpenAI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

load_dotenv()


_AZURE_AI_SCOPE = "https://ai.azure.com/.default"


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    credential = DefaultAzureCredential()
    project = AIProjectClient(endpoint=project_endpoint, credential=credential)
    openai_client = project.get_openai_client()
    token_provider = get_bearer_token_provider(credential, _AZURE_AI_SCOPE)

    return ChatOpenAI(
        model=deployment,
        base_url=str(openai_client.base_url),
        api_key=token_provider,
        use_responses_api=True,
        output_version="responses/v1",
        store=False,  # LangGraph replays the messages, including tool calls.
    )


async def _load_toolbox() -> tuple[list[BaseTool], str]:
    """Fetch tools and skill metadata, and expose an on-demand resource reader.

    ``project_endpoint`` is resolved from ``FOUNDRY_PROJECT_ENDPOINT``
    automatically. The credential defaults to ``DefaultAzureCredential``
    (so ``az login`` is enough for local dev). Each call opens a fresh
    MCP session against the toolbox and closes it before returning.
    """
    toolbox_name = os.environ["TOOLBOX_NAME"]

    toolbox = AzureAIProjectToolbox(toolbox_name=toolbox_name)
    tools = await toolbox.get_tools()
    print(f"Loaded {len(tools)} tool(s) from Foundry toolbox '{toolbox_name}':")
    for t in tools:
        print(f"  - {t.name}")
    resources = await toolbox.aget_resources(uris="skill://index.json")
    catalog = "\n\n".join(resource.as_string() for resource in resources)
    if not catalog.strip():
        raise ValueError("The Toolbox must provide skill://index.json.")
    print(f"Loaded skill catalog from Foundry toolbox '{toolbox_name}'.")

    @tool
    async def read_skill_resource(uri: str) -> str:
        """Read a skill's SKILL.md or supporting text from this Toolbox.

        Use a skill:// URI from the catalog or a loaded skill. Read SKILL.md
        before following a skill, then read references only when needed.
        """
        if not uri.startswith("skill://"):
            raise ToolException("Use a skill:// URI from this Toolbox.")
        resources = await toolbox.aget_resources(uris=uri)
        content = "\n\n".join(resource.as_string() for resource in resources)
        if not content.strip():
            raise ToolException(f"No skill content returned for {uri}.")
        return content

    read_skill_resource.handle_tool_error = True
    return [*tools, read_skill_resource], catalog


def main() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    ):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    tools, catalog = asyncio.run(_load_toolbox())
    graph = create_agent(
        _build_chat_model(),
        tools=tools,
        system_prompt=(
            "Available skills are listed in the Toolbox catalog below. "
            "When a skill matches the user's request, call read_skill_resource "
            "with its SKILL.md URI before following its procedure using the "
            "toolbox tools. Read supporting resources only when needed.\n\n" + catalog
        ),
    )
    port = int(os.environ.get("PORT", "8088"))
    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
