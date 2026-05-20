"""Sample 02 - Tool call & tool result messages over the Responses API.

Demonstrates that intermediate tool calls and tool results are surfaced
to the client as ``function_call`` / ``function_call_output`` output
items - in both non-streaming JSON responses and SSE streams.

The agent uses a Foundry-deployed Azure OpenAI chat model and one local
tool, ``get_weather``.

Required environment variables (set in `.env` or your shell):

    FOUNDRY_PROJECT_ENDPOINT        e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python main.py

Then in another terminal:

    # Non-streaming -- the JSON `output` array contains 3 items:
    #   [0] function_call(get_weather)
    #   [1] function_call_output(<weather string>)
    #   [2] message(<final assistant text>)
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"What is the weather in Seattle?","model":"gpt-4o"}'

    # Streaming -- you should see the events arrive in this order:
    #   response.output_item.added/done   (function_call)
    #   response.output_item.added/done   (function_call_output)
    #   response.output_item.added        (message)
    #   response.output_text.delta * N
    #   response.output_text.done
    #   response.output_item.done         (message)
    #   response.completed
    curl -N -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"What is the weather in Tokyo?","model":"gpt-4o","stream":true}'
"""
from __future__ import annotations

import os
from random import randint
from typing import Annotated

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

load_dotenv()

_AAD_SCOPE = "https://ai.azure.com/.default"


@tool
def get_weather(
    location: Annotated[str, "City and country, e.g. 'Seattle, US'."],
) -> str:
    """Return a fake weather snapshot for the given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return (
        f"The weather in {location} is {conditions[randint(0, 3)]} "
        f"with a high of {randint(10, 30)}C."
    )


def _build_chat_model() -> AzureChatOpenAI:
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    # AzureChatOpenAI talks to the Azure OpenAI shape
    # ({endpoint}/openai/deployments/<name>/...?api-version=...), which lives
    # on the *account* (the bit before /api/projects/<project>).
    account_endpoint = project_endpoint.split("/api/projects/", 1)[0]
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), _AAD_SCOPE)
    return AzureChatOpenAI(
        azure_endpoint=account_endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )


def main() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    graph = create_agent(_build_chat_model(), tools=[get_weather])
    port = int(os.environ.get("PORT", "8088"))
    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
