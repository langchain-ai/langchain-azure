"""Run the native middleware agent with explicit startup resource ownership."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack

from agent import build_agent
from azure.ai.agentserver.core import AgentConfig
from azure.ai.projects import AIProjectClient
from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_azure_ai.agents.hosting import (
    FoundryCheckpointSaver,
    ResponsesHostServer,
)
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def _configure_tracing(resources: AsyncExitStack, credential: TokenCredential) -> None:
    os.environ.setdefault("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false")
    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        enable_auto_tracing(auto_configure_azure_monitor=False)
        return
    if os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT"
    ):
        provider = TracerProvider()
        resources.callback(provider.shutdown)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(credential=credential, auto_configure_azure_monitor=True)


async def main() -> None:
    """Keep credentials, model transports, and checkpoints alive until host exit."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    async with AsyncExitStack() as resources:
        credential = resources.enter_context(DefaultAzureCredential())
        _configure_tracing(resources, credential)
        with AIProjectClient(
            endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"], credential=credential
        ) as project:
            with project.get_openai_client() as endpoint_client:
                base_url = str(endpoint_client.base_url)
        sync_http = resources.enter_context(DefaultHttpxClient())
        async_http = await resources.enter_async_context(DefaultAsyncHttpxClient())
        model = ChatOpenAI(
            model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4.1-mini"),
            base_url=base_url,
            api_key=get_bearer_token_provider(
                credential, "https://ai.azure.com/.default"
            ),
            http_client=sync_http,
            http_async_client=async_http,
            use_responses_api=True,
            output_version="v1",
        )
        checkpointer: BaseCheckpointSaver = (
            await resources.enter_async_context(FoundryCheckpointSaver())
            if AgentConfig.from_env().is_hosted
            else InMemorySaver()
        )
        graph = build_agent(model, checkpointer)
        await ResponsesHostServer(graph).run_async(
            port=int(os.environ.get("PORT", "8088"))
        )


if __name__ == "__main__":
    asyncio.run(main())
