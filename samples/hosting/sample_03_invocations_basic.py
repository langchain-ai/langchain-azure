"""Sample 03 - Invocations API with multi-turn session continuity.

Hosts a ``create_agent`` graph (compiled with ``MemorySaver``) as
the Azure AI Invocations API. Multi-turn conversations work
automatically: the resolved ``agent_session_id`` is forwarded to the
graph as ``RunnableConfig.configurable.thread_id``, so the checkpointer
keeps each session's history in memory.

Required environment variables (set in `.env` or your shell):

    AZURE_AI_PROJECT_ENDPOINT       e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python sample_03_invocations_basic.py

Then in another terminal:

    # Turn 1 - capture the x-agent-session-id response header
    curl -i -X POST http://127.0.0.1:8088/invocations -H 'Content-Type: application/json' -d '{"message":"My name is Alice."}'

    # Turn 2 - reuse the same session id
    curl -X POST 'http://127.0.0.1:8088/invocations?agent_session_id=<id>' -H 'Content-Type: application/json' -d '{"message":"What is my name?"}'

    # Streaming variant
    curl -N -X POST http://127.0.0.1:8088/invocations -H 'Content-Type: application/json' -d '{"message":"Count to 5.","stream":true}'
"""
from __future__ import annotations

import os

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import LangGraphInvokeAgentHost
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

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
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    # MemorySaver keys conversations by thread_id, which the host wires
    # from agent_session_id. Replace with a durable checkpointer
    # (Redis, Cosmos, etc.) for production.
    graph = create_agent(
        _build_chat_model(), tools=[], checkpointer=MemorySaver()
    )
    port = int(os.environ.get("PORT", "8088"))
    LangGraphInvokeAgentHost(graph).run(host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
