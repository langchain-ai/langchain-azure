"""Sample 04 - Invocations API with a local tool.

Same shape as ``sample_03_invocations_basic.py`` but with a ``@tool``
function attached to the graph so the agent runs a tool round-trip
before answering. The Invocations API surfaces only the final assistant
text (and intermediate token deltas when ``"stream": true``); the tool
call/result pair lives inside the graph and is consumed during the ReAct
loop.

Required environment variables (set in `.env` or your shell):

    AZURE_AI_PROJECT_ENDPOINT       e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python sample_04_invocations_tools.py

Then in another terminal:

    # Non-streaming - the tool runs server-side, the JSON response is
    # the final assistant text.
    curl -X POST http://127.0.0.1:8088/invocations \\
      -H 'Content-Type: application/json' \\
      -d '{"message":"What is the weather in Seattle?"}'
    # -> {"response":"The weather in Seattle, US is sunny ..."}

    # Streaming - per-token text deltas as event-stream `data:` lines,
    # followed by `event: done`.
    curl -N -X POST http://127.0.0.1:8088/invocations \\
      -H 'Content-Type: application/json' \\
      -d '{"message":"What is the weather in Tokyo?","stream":true}'
"""
from __future__ import annotations

import os
from random import randint
from typing import Annotated

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import AzureAIInvokeAgentHost
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
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    graph = create_react_agent(
        _build_chat_model(),
        tools=[get_weather],
        checkpointer=MemorySaver(),
    )
    port = int(os.environ.get("PORT", "8088"))
    AzureAIInvokeAgentHost(graph).run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
