"""Sample 05 - All-in-one: multi-node LangGraph workflow on both APIs.

Hosts a custom LangGraph ``StateGraph`` with three named nodes and two
tools as **both** the Responses API and the Invocations API on the same
process / port.

Workflow shape::

    START -> plan ---(no tool call)---> END
              |
              +---(tool call)---> tools -> synthesize -> END

Nodes:

- ``plan`` - LLM call bound to the tools. Decides whether to call a
  tool. When the model can answer directly (e.g. "Hello") the workflow
  ends here.
- ``tools`` - ``ToolNode`` that executes any tools the planner asked for
  (here: ``get_weather`` and ``add``).
- ``synthesize`` - second LLM call with no tools, asked to summarise the
  research the planner produced into a friendly, single-paragraph reply.

Both protocols are mounted on one ``MultiProtocolHost`` via the
``app=`` constructor kwarg on ``ResponsesHostServer`` /
``InvocationsHostServer``. The Responses API surfaces every
intermediate ``function_call`` / ``function_call_output`` / ``message``
from the workflow; the Invocations API returns just the final assistant
text (or streams its tokens when ``stream=true``).

Required environment variables (set in ``.env`` or your shell):

    FOUNDRY_PROJECT_ENDPOINT        e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python main.py

Then in another terminal:

    # Responses API - tool round-trip with full trace
    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"What is the weather in Seattle?","model":"gpt-4o"}'

    # Invocations API - same graph, final-text-only shape
    curl -i -X POST http://127.0.0.1:8088/invocations -H 'Content-Type: application/json' -d '{"message":"What is 17 plus 25?"}'
"""
from __future__ import annotations

import os
from random import randint
from typing import Annotated

from azure.ai.agentserver.invocations import InvocationAgentServerHost
from azure.ai.agentserver.responses import ResponsesAgentServerHost
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import (
    InvocationsHostServer,
    ResponsesHostServer,
)
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

load_dotenv()

_AAD_SCOPE = "https://ai.azure.com/.default"


# ── Tools ────────────────────────────────────────────────────────────────


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


@tool
def add(
    a: Annotated[float, "First addend."],
    b: Annotated[float, "Second addend."],
) -> str:
    """Return the sum of two numbers."""
    return str(a + b)


_TOOLS = [get_weather, add]


# ── State + nodes ────────────────────────────────────────────────────────


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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


def _build_graph():
    base_model = _build_chat_model()
    planner = base_model.bind_tools(_TOOLS)

    _PLAN_PROMPT = SystemMessage(
        content=(
            "You are a research planner. If a tool can fetch a fact the "
            "user asked about, call it. Otherwise answer directly."
        )
    )
    _SYNTH_PROMPT = SystemMessage(
        content=(
            "Summarise the gathered facts in one short, friendly sentence "
            "for the user. Do not call any tools."
        )
    )

    async def plan(state: State) -> dict:
        result = await planner.ainvoke([_PLAN_PROMPT, *state["messages"]])
        return {"messages": [result]}

    async def synthesize(state: State) -> dict:
        result = await base_model.ainvoke([_SYNTH_PROMPT, *state["messages"]])
        return {"messages": [result]}

    builder = StateGraph(State)
    builder.add_node("plan", plan)
    builder.add_node("tools", ToolNode(_TOOLS))
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "plan")
    # tools_condition routes to "tools" when the planner emitted
    # tool_calls, otherwise to END. We feed the tool results back
    # through synthesize.
    builder.add_conditional_edges(
        "plan", tools_condition, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=MemorySaver())


# ── Multi-protocol host ──────────────────────────────────────────────────


class MultiProtocolHost(InvocationAgentServerHost, ResponsesAgentServerHost):
    """One ``AgentServerHost`` serving both protocols via cooperative MRO."""


def main() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    port = int(os.environ.get("PORT", "8088"))
    graph = _build_graph()
    app = MultiProtocolHost()

    ResponsesHostServer(graph, app=app)
    InvocationsHostServer(graph, app=app)
    app.run(port=port)


if __name__ == "__main__":
    main()
