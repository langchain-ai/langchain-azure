"""Sample 08 - Foundry-managed checkpointer over the Responses API.

Demonstrates :class:`langchain_azure_ai.checkpointers.FoundryCheckpointSaver`
— a LangGraph checkpoint saver that persists graph state to Azure AI
Foundry's managed checkpoint storage service (api-version
``2025-11-15-preview``, preview).

Why this is different from samples 01-07
----------------------------------------

Samples 01-07 use ``InMemorySaver`` — checkpoints live in the host
process and disappear when the server restarts. This sample swaps in
``FoundryCheckpointSaver``, so the same ``conversation.id`` continues
the graph state **across server restarts** because every checkpoint is
written to the Foundry project rather than to local memory.

Required environment variables (set in ``.env`` or your shell):

    AZURE_AI_PROJECT_ENDPOINT       e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python sample_08_responses_foundry_checkpointer.py

Then drive a multi-turn conversation — keep the same ``conversation.id``
across requests so the agent can recall the earlier turns from the
Foundry-managed checkpoint::

    # Turn 1 — give the agent a fact to remember
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"My favourite city is Seattle. Remember that.","conversation":{"id":"demo-foundry-ckpt-1"}}'

    # Turn 2 — verify the agent remembers it (same conversation.id)
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"What is my favourite city?","conversation":{"id":"demo-foundry-ckpt-1"}}'

Now stop the process (Ctrl+C), start it again with the same command,
and re-run the second curl. The agent still recalls "Seattle" because
the checkpoint was loaded back from Foundry — not in-memory state.

Notes
-----

* ``FoundryCheckpointSaver`` is **async-only** and requires an async
  credential (``azure.identity.aio.DefaultAzureCredential``). We hold
  the credential and saver open via ``async with`` and host the agent
  through ``LangGraphResponsesHostServer(...).run_async()``.
* The feature is **experimental** while the underlying REST surface is
  in preview; expect a warning on import.
"""
from __future__ import annotations

import asyncio
import os

from azure.identity import DefaultAzureCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import LangGraphResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing
from langchain_azure_ai.checkpointers import FoundryCheckpointSaver

load_dotenv()

_AAD_SCOPE = "https://ai.azure.com/.default"


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------


def _build_chat_model() -> ChatOpenAI:
    project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"].rstrip("/")
    deployment = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o")
    # Token-style auth against the Foundry-fronted Azure OpenAI endpoint.
    # The saver uses an *async* credential — see main() — but the model
    # only needs a one-shot token, so a sync credential is fine here.
    credential = DefaultAzureCredential()
    token = credential.get_token(_AAD_SCOPE).token
    return ChatOpenAI(
        model=deployment,
        api_key=token,  # type: ignore[arg-type]
        base_url=f"{project_endpoint}/openai/v1",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _amain() -> None:
    # Tracing destination is picked entirely by env vars (priority order):
    #   1. OTEL_EXPORTER_OTLP_ENDPOINT -> OTLP/HTTP collector
    #   2. APPLICATION_INSIGHTS_CONNECTION_STRING -> Azure Monitor directly
    #   3. AZURE_AI_PROJECT_ENDPOINT -> Foundry project's managed App Insights
    #   4. None of the above -> tracer attached but no exporter (no-op)
    # Set OTEL_SDK_DISABLED=true at any time to short-circuit the whole thing.
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    project_endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    port = int(os.environ.get("PORT", "8088"))

    # Hold an async credential open for the lifetime of the server so the
    # saver's bearer-token policy can refresh tokens as needed.
    async with AsyncDefaultAzureCredential() as cred:
        async with FoundryCheckpointSaver(
            project_endpoint=project_endpoint,
            credential=cred,
        ) as saver:
            graph = create_agent(
                _build_chat_model(),
                tools=[],
                checkpointer=saver,
            )
            await LangGraphResponsesHostServer(graph).run_async(
                host="127.0.0.1", port=port
            )


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
