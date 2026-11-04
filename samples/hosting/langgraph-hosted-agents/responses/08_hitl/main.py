"""Sample 08 - Human-in-the-loop over the Responses API.

This sample demonstrates an **approval-style HITL** flow using LangGraph's
``langgraph.types.interrupt``: before any tool runs, the graph pauses and
asks the client to review all tool calls from the same model turn. The pause is
serialized to the wire as the **standard OpenAI ``mcp_approval_request``
output item**, so any Responses-API client that already supports MCP
server approvals can drive this agent without code changes.

For each pending interrupt the host emits TWO paired output items in
the same response, both referring to the same LangGraph interrupt id:

* an ``mcp_approval_request`` item (``server_label == "langgraph"``,
  ``arguments`` JSON contains each call's ID, name, and arguments) — the
  OpenAI-standard channel; clients respond with an
  ``mcp_approval_response``, and
* a ``function_call`` item with
  ``name == "__hosted_agent_adapter_interrupt__"`` — a parallel rich
  channel for callers that want to send arbitrary resume payloads
  (``{"resume", "update", "goto"}``) via ``function_call_output``.

State is persisted by a checkpointer keyed by the ``conversation`` id, so the
second request continues the paused run. Local runs use ``InMemorySaver``;
Foundry-hosted runs use ``FoundryCheckpointSaver``.

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
``arguments`` JSON describes the complete proposed batch::

    {"interrupt_id": "<id>", "value": {"tool_calls": [{"id": "<tool call id>", "name": "get_weather", "arguments": {"location": "Seattle"}}]}}

**Primary path — approve via the standard MCP-approval channel.** The
host resumes the graph and executes every call in the reviewed batch::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"conversation":{"id":"demo-hitl-1"},"input":[{"type":"mcp_approval_response","approval_request_id":"<id>","approve":true}]}'

**Reject.** The turn ends with ``response.failed``
``code="interrupt_rejected"``; the pending interrupt remains in the
checkpoint so the client can retry::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"conversation":{"id":"demo-hitl-1"},"input":[{"type":"mcp_approval_response","approval_request_id":"<id>","approve":false,"reason":"user canceled"}]}'

**Advanced — rich resume via ``function_call_output``.** When you need
to inject a custom resume value or drive a LangGraph ``Command`` with
``update``/``goto`` fields, target the paired ``function_call`` item
instead (its ``call_id`` is the same interrupt id)::

    curl -X POST http://127.0.0.1:8088/responses -H 'Content-Type: application/json' -d '{"conversation":{"id":"demo-hitl-1"},"input":[{"type":"function_call_output","call_id":"<id>","output":"{\\"resume\\": {\\"decisions\\":[{\\"id\\":\\"<tool call id>\\",\\"type\\":\\"edit\\",\\"arguments\\":{\\"location\\":\\"Vancouver\\"}}]}}"}]}'
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any

from azure.ai.agentserver.core import AgentConfig
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from langchain_azure_ai.agents.hosting import (
    FoundryCheckpointSaver,
    ResponsesHostServer,
)
from langchain_azure_ai.callbacks.tracers import enable_auto_tracing

load_dotenv()


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
    )


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _build_graph(checkpointer: BaseCheckpointSaver[Any]) -> "object":
    llm = _build_chat_model()
    tools = list(_TOOLS_BY_NAME.values())
    model = llm.bind_tools(tools)

    def call_model(state: MessagesState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    def approve_and_call_tool(state: MessagesState) -> dict:
        """Review the whole model turn once, binding each decision to its call ID."""
        last = state["messages"][-1]
        calls = [
            {
                "id": tool_call["id"],
                "name": tool_call["name"],
                "arguments": tool_call["args"],
            }
            for tool_call in last.tool_calls  # type: ignore[attr-defined]
        ]
        proposed = {"tool_calls": calls}
        review = interrupt(proposed)
        if review == proposed:
            decisions = [{"id": call["id"], "type": "approve"} for call in calls]
        elif isinstance(review, dict) and isinstance(review.get("decisions"), list):
            decisions = review["decisions"]
        else:
            raise ValueError("Expected the approved batch or ID-keyed decisions")

        if any(
            not isinstance(decision, dict) or not isinstance(decision.get("id"), str)
            for decision in decisions
        ):
            raise ValueError("Every decision must include a tool call ID")
        decisions_by_id = {decision["id"]: decision for decision in decisions}
        if len(decisions) != len(calls) or set(decisions_by_id) != {
            call["id"] for call in calls
        }:
            raise ValueError("Expected exactly one decision for every proposed call ID")
        for decision in decisions:
            if decision.get("type") not in ("approve", "edit", "reject"):
                raise ValueError("Decision type must be approve, edit, or reject")
            if decision["type"] == "edit" and not isinstance(
                decision.get("arguments"), dict
            ):
                raise ValueError("An edit decision must include arguments")

        messages = []
        for call in calls:
            decision = decisions_by_id[call["id"]]
            if decision["type"] == "reject":
                result = "Tool call rejected by the reviewer."
            else:
                arguments = (
                    decision["arguments"]
                    if decision["type"] == "edit"
                    else call["arguments"]
                )
                result = _TOOLS_BY_NAME[call["name"]].invoke(arguments)
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": messages}

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
    return workflow.compile(checkpointer=checkpointer)


async def main() -> None:
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
        enable_auto_tracing()
    else:
        enable_auto_tracing(auto_configure_azure_monitor=True)

    if AgentConfig.from_env().is_hosted:
        checkpointer = FoundryCheckpointSaver(user_isolation=False)
    else:
        checkpointer = InMemorySaver()
    async with checkpointer:
        graph = _build_graph(checkpointer)
        await ResponsesHostServer(graph).run_async(
            port=int(os.environ.get("PORT", "8088"))
        )


if __name__ == "__main__":
    asyncio.run(main())
