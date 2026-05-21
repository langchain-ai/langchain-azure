"""Sample 01 - Minimal Responses API host.

Hosts a no-tool ``create_agent`` graph as the Azure AI Responses
API on top of a Foundry-deployed Azure OpenAI chat model.

Required environment variables (set in `.env` or your shell):

    FOUNDRY_PROJECT_ENDPOINT        e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python main.py

Then in another terminal:

    curl -N -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"Hello!","model":"gpt-4o","stream":true}'
"""
from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain.agents import create_agent

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

load_dotenv()


def _build_chat_model() -> AzureAIOpenAIApiChatModel:
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    return AzureAIOpenAIApiChatModel(
        project_endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
        model=deployment,
    )


def main() -> None:
    # Tracing destination is picked entirely by env vars (priority order):
    #   1. OTEL_EXPORTER_OTLP_ENDPOINT -> OTLP/HTTP collector
    #   2. APPLICATION_INSIGHTS_CONNECTION_STRING -> Azure Monitor directly
    #   3. FOUNDRY_PROJECT_ENDPOINT -> Foundry project's managed App Insights
    #   4. None of the above -> tracer attached but no exporter (no-op)
    # Set OTEL_SDK_DISABLED=true at any time to short-circuit the whole thing.
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    graph = create_agent(_build_chat_model(), tools=[])
    port = int(os.environ.get("PORT", "8088"))

    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
