"""Sample 08 - Human-in-the-loop over the Responses API.

This sample demonstrates an **approval-style HITL** flow using LangGraph's
``langgraph.types.interrupt``: before any tool runs, the graph pauses and
asks the client to approve the proposed tool call. The pause is
serialized to the wire as the **standard OpenAI ``mcp_approval_request``
output item**, so any Responses-API client that already supports MCP
server approvals can drive this agent without code changes.

For each pending interrupt the host emits TWO paired output items in
the same response, both keyed by the same LangGraph interrupt id:

* an ``mcp_approval_request`` item (``server_label == "langgraph"``,
  ``arguments`` JSON contains the proposed tool call) — the
  OpenAI-standard channel; clients respond with an
  ``mcp_approval_response``, and
* a ``function_call`` item with
  ``name == "__hosted_agent_adapter_interrupt__"`` — a parallel rich
  channel for callers that want to send arbitrary resume payloads
  (``{"resume", "update", "goto"}``) via ``function_call_output``.

State is persisted by an ``InMemorySaver`` checkpointer keyed by the
``conversation`` id, so the second request continues the paused run.

Required environment variables (set in ``.env`` or your shell):

    FOUNDRY_PROJECT_ENDPOINT        e.g. https://<acct>.services.ai.azure.com/api/projects/<proj>
    AZURE_AI_MODEL_DEPLOYMENT_NAME  e.g. gpt-4o   (defaults to "gpt-4o")
    PORT                            optional, defaults to 8088

Run::

    az login
    cp .env.example .env  # then edit the values
    python main.py

Then in another terminal — ask the agent a question that requires a
tool::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"input":"What is the weather in Seattle?","conversation":{"id":"demo-hitl-1"}}'

The response ``output`` will contain an ``mcp_approval_request`` whose
``arguments`` JSON describes the proposed tool call::

    {"interrupt_id": "<id>", "value": {"tool": "get_weather", "arguments": {"location": "Seattle"}}}

**Primary path — approve via the standard MCP-approval channel.** The
host resumes the graph and executes the tool::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"conversation":{"id":"demo-hitl-1"},"input":[{"type":"mcp_approval_response","approval_request_id":"<id>","approve":true}]}'

**Reject.** The turn ends with ``response.failed``
``code="interrupt_rejected"``; the pending interrupt remains in the
checkpoint so the client can retry::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"conversation":{"id":"demo-hitl-1"},"input":[{"type":"mcp_approval_response","approval_request_id":"<id>","approve":false,"reason":"user canceled"}]}'

**Advanced — rich resume via ``function_call_output``.** When you need
to inject a custom resume value or drive a LangGraph ``Command`` with
``update``/``goto`` fields, target the paired ``function_call`` item
instead (its ``call_id`` is the same interrupt id)::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"conversation":{"id":"demo-hitl-1"},"input":[{"type":"function_call_output","call_id":"<id>","output":"{\\"resume\\": {\\"tool\\":\\"get_weather\\",\\"arguments\\":{\\"location\\":\\"Vancouver\\"}}}"}]}'
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import LangGraphResponsesHostServer
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

load_dotenv()

_AAD_SCOPE = "https://ai.azure.com/.default"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(
    location: Annotated[str, "City and country, e.g. 'Seattle, US'."],
) -> str:
    """Return a fake weather snapshot for the given location."""
    return f"It's sunny and 22C in {location}."


_TOOLS_BY_NAME = {"get_weather": get_weather}


# ---------------------------------------------------------------------------
# Chat model
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _build_graph() -> "object":
    llm = _build_chat_model()
    tools = list(_TOOLS_BY_NAME.values())
    model = llm.bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    def approve_and_call_tool(state: MessagesState) -> dict:
        """Pause for human approval, then execute the proposed tool call.
        """
        last = state["messages"][-1]
        tool_call = last.tool_calls[0]  # type: ignore[attr-defined]
        proposed: dict[str, Any] = {
            "tool": tool_call["name"],
            "arguments": tool_call["args"],
        }

        # On approve=True, the resume value is the original ``proposed``
        # dict. On a ``function_call_output``-style resume the client can
        # send a different payload (e.g. to override the arguments) — we
        # use whatever the client returned for the actual invocation.
        approved: Any = interrupt(proposed)
        if not isinstance(approved, dict) or "tool" not in approved:
            approved = proposed

        tool_fn = _TOOLS_BY_NAME[approved["tool"]]
        result = tool_fn.invoke(approved.get("arguments") or {})
        return {
            "messages": [
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            ]
        }

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        return "approve_and_call_tool"

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("approve_and_call_tool", approve_and_call_tool)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        path_map=["approve_and_call_tool", END],
    )
    workflow.add_edge("approve_and_call_tool", "agent")
    return workflow.compile(checkpointer=InMemorySaver())


def main() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    graph = _build_graph()
    port = int(os.environ.get("PORT", "8088"))
    LangGraphResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
