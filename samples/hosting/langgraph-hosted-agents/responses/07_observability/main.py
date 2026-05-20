"""Sample 07 - Responses API + explicit observability.

A standalone observability sample. The graph itself is intentionally
trivial - one `create_agent` call with no custom tools - so the focus
is on the **tracing wiring** rather than the agent behavior. Every
LangGraph node, model call, and tool call emits OpenTelemetry GenAI
semantic-convention spans via
``langchain_azure_ai.callbacks.tracers.AzureAIOpenTelemetryTracer``.

Tracing destination is picked from environment variables; the same
priority order applies as in every other sample (first match wins):

    0. OTEL_SDK_DISABLED=true                          -> tracer is a no-op.
    1. OTEL_EXPORTER_OTLP_ENDPOINT (or _TRACES_)       -> OTLP/HTTP collector.
    2. APPLICATION_INSIGHTS_CONNECTION_STRING          -> Azure Monitor directly.
    3. FOUNDRY_PROJECT_ENDPOINT                        -> Foundry project's managed App Insights.
    4. none of the above                               -> tracer attached, no exporter.

Set ``AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true`` to also
record message content, tool arguments, and tool results on the spans
(default: redacted). This sample's `.env.example` enables it so you can
see full content end-to-end in App Insights.

Required environment variables (set in ``.env`` or your shell):

    FOUNDRY_PROJECT_ENDPOINT                        e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME                  e.g. gpt-4o   (defaults to "gpt-4o")
    AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED  true/false    (defaults to false)
    PORT                                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python main.py

Then in another terminal::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"Tell me a fun fact about distributed tracing.","model":"gpt-4o"}'

Open the Foundry project's managed App Insights resource (or your
configured destination) to view the spans:

    - one ``invoke_agent`` span per request (root server span)
    - nested ``chain``/``graph`` spans for each LangGraph node
    - nested ``chat <model>`` spans for each model call
"""
from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import ResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

load_dotenv()

_AAD_SCOPE = "https://ai.azure.com/.default"


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    credential = DefaultAzureCredential()
    token = credential.get_token(_AAD_SCOPE).token
    return ChatOpenAI(
        model=deployment,
        api_key=token,  # type: ignore[arg-type]
        base_url=f"{project_endpoint}/openai/v1",
    )


def main() -> None:
    # Tracing destination selection - same priority order as every other
    # sample. Centralized here so it's easy to copy into your own agent.
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    ):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        # auto_configure_azure_monitor resolves App Insights from
        # APPLICATION_INSIGHTS_CONNECTION_STRING first, then falls back
        # to FOUNDRY_PROJECT_ENDPOINT (project-managed App Insights).
        enable_auto_tracing(auto_configure_azure_monitor=True)

    graph = create_agent(_build_chat_model(), tools=[])
    port = int(os.environ.get("PORT", "8088"))
    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
