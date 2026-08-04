"""Sample 99 - Durable Invocations with a tool-using LangGraph agent.

This is the Invocations protocol counterpart of
``responses/99_resilient/main.py``. It keeps the same trip-planning graph,
human approval, crash-simulation tool, and persistent LangGraph checkpointer.

The Invocations protocol does not provide the Responses API's background task
lease or automatic resilient re-entry. After a process restart, retry the
request with the same ``agent_session_id``. The host detects the unfinished
LangGraph checkpoint and resumes it with input ``None``.

Optional environment variables:

    PORT               optional, defaults to 8088
    CHECKPOINT_DB      optional path to the LangGraph checkpoint SQLite file.
                       Defaults to ``checkpoints.sqlite`` locally, or
                       ``$HOME/checkpoints.sqlite`` when hosted on Foundry.
    FOUNDRY_PROJECT_ENDPOINT required project endpoint for the model.
    AZURE_AI_MODEL_DEPLOYMENT_NAME required model deployment name.

Run::

    python main.py

Use a stable session id so a request can be retried after a simulated crash::

    curl -X POST \
        'http://127.0.0.1:8088/invocations?agent_session_id=trip-demo' \
        -H 'Content-Type: application/json' \
        -d '{"message":"Plan a two-night trip to Seattle."}'

When the graph pauses before ``book_trip``, approve or reject it with the same
session id::

    curl -X POST \
        'http://127.0.0.1:8088/invocations?agent_session_id=trip-demo' \
        -H 'Content-Type: application/json' \
        -d '{"message":"approve"}'

Ask the agent to call ``simulate_crash``. After restarting the process, resend
the same request with the same session id to resume from the pending tool node.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict

from azure.ai.agentserver.core import AgentConfig
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_azure_ai.agents.hosting import InvocationsHostServer
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

load_dotenv()

logger = logging.getLogger(__name__)


def _ignore_graph_lifecycle_event(_tracer: Any, _event: Any) -> None:
    return None


def _install_otel_langgraph_callback_compatibility() -> None:
    """Prevent generic Microsoft OTel tracers from receiving unknown hooks."""
    try:
        from microsoft.opentelemetry._genai._langchain._tracer import (
            LangChainTracer as MicrosoftLangChainTracer,
        )
    except ImportError:
        return

    for callback_name in ("on_interrupt", "on_resume"):
        if not hasattr(MicrosoftLangChainTracer, callback_name):
            setattr(
                MicrosoftLangChainTracer,
                callback_name,
                _ignore_graph_lifecycle_event,
            )


def _resolve_checkpoint_db() -> str:
    configured_path = os.environ.get("CHECKPOINT_DB")
    if configured_path:
        return configured_path
    if AgentConfig.from_env().is_hosted:
        return os.path.join(os.path.expanduser("~"), "checkpoints.sqlite")
    return "checkpoints.sqlite"


_CHECKPOINT_DB = _resolve_checkpoint_db()
_AZURE_AI_SCOPE = "https://ai.azure.com/.default"
_SENSITIVE_TOOLS = {"book_trip"}
_APPROVAL_MESSAGES = {"approve", "approved", "yes", "y"}
_REJECTION_MESSAGES = {"reject", "rejected", "no", "n"}
_SYSTEM_PROMPT = """You are a concise trip-planning assistant.
For a trip request, first call search_flights and search_hotels to gather options.
Then recommend a specific flight and hotel in one short paragraph and call book_trip
with that choice. The application obtains human approval before book_trip executes,
so do not ask for confirmation yourself. After booking, give a short confirmation
summary with the confirmation number.
At every step, write a brief user-visible progress message before making tool calls.
Compose that message yourself from the current context. Keep exploration,
recommendation and booking, and final confirmation visibly separated, but do not
emit a canned checklist or application-style status text.
Call simulate_crash only when the user explicitly asks you to simulate a crash.
After any tool result, continue the task and explain the result naturally.
Never claim that you used a tool unless you actually called it."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_real_model() -> BaseChatModel:
    deployment = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]
    credential = DefaultAzureCredential()
    project = AIProjectClient(
        endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/"),
        credential=credential,
    )
    openai_client = project.get_openai_client()
    return ChatOpenAI(
        model=deployment,
        base_url=str(openai_client.base_url),
        api_key=get_bearer_token_provider(credential, _AZURE_AI_SCOPE),
        streaming=True,
        use_responses_api=True,
        output_version="responses/v1",
    )


async def _sigkill_current_process() -> None:
    print("Crash trigger received; sending SIGKILL to current process.", flush=True)
    kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    os.kill(os.getpid(), kill_signal)
    await asyncio.sleep(60 * 60 * 24)


@tool
def search_flights(city: str) -> dict[str, Any]:
    """Search round-trip flight options to a destination city."""
    return {
        "city": city,
        "options": [
            {"label": f"{city} Express AA123", "price_usd": 780, "stops": 0},
            {"label": f"{city} Saver BB456", "price_usd": 540, "stops": 1},
        ],
    }


@tool
def search_hotels(city: str) -> dict[str, Any]:
    """Search hotel options in a destination city."""
    return {
        "city": city,
        "options": [
            {
                "label": f"{city} Grand Hotel",
                "price_per_night_usd": 240,
                "rating": 4.6,
            },
            {
                "label": f"{city} Budget Inn",
                "price_per_night_usd": 95,
                "rating": 4.0,
            },
        ],
    }


@tool
def book_trip(
    city: str,
    nights: int = 1,
    flight: str = "",
    hotel: str = "",
) -> dict[str, Any]:
    """Book and pay for a selected flight and hotel after human approval."""
    confirmation = f"TRIP-{abs(hash((city, nights, flight, hotel))) % 1_000_000:06d}"
    return {
        "status": "booked",
        "confirmation": confirmation,
        "city": city,
        "nights": nights,
        "flight": flight or "(cheapest)",
        "hotel": hotel or "(recommended)",
    }


@tool
async def simulate_crash(config: RunnableConfig) -> str:
    """Crash this agent process to demonstrate durable checkpoint recovery.

    Call this tool only when the user explicitly asks to simulate a crash.
    """
    if config.get("configurable", {}).get("invocation_recovery", False):
        return "Crash recovery succeeded; resumed the pending tool call."
    await _sigkill_current_process()
    return "The process did not terminate."


def build_graph(checkpointer, model: BaseChatModel):
    all_tools = [search_flights, search_hotels, book_trip, simulate_crash]
    tools_by_name = {selected_tool.name: selected_tool for selected_tool in all_tools}
    tool_model = model.bind_tools(all_tools)

    async def agent(state: AgentState, config: RunnableConfig) -> dict:
        response = await tool_model.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), *state["messages"]],
            config=config,
        )
        return {"messages": [response]}

    async def approval(state: AgentState, config: RunnableConfig) -> dict:
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            raise TypeError("Approval requires an AIMessage with tool calls")

        outputs: list[ToolMessage] = []
        for tool_call in message.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in _SENSITIVE_TOOLS:
                approved = interrupt(
                    {
                        "action": tool_name,
                        "arguments": tool_call["args"],
                        "tool_call_id": tool_call["id"],
                        "prompt": "Approve this sensitive tool call?",
                    }
                )
                if not approved:
                    outputs.append(
                        ToolMessage(
                            content=json.dumps({"status": "rejected"}),
                            name=tool_name,
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue
            result = await tools_by_name[tool_name].ainvoke(
                tool_call["args"],
                config=config,
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(result),
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

    def route_after_agent(state: AgentState) -> str:
        message = state["messages"][-1]
        if isinstance(message, AIMessage) and message.tool_calls:
            if any(
                tool_call["name"] in _SENSITIVE_TOOLS
                for tool_call in message.tool_calls
            ):
                return "approval"
            return "tools"
        return "end"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node("approval", approval)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "approval": "approval", "end": END},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("approval", "agent")
    return builder.compile(checkpointer=checkpointer)


class DurableInvocationsHostServer(InvocationsHostServer):
    """Resume unfinished checkpoints before accepting a new user turn."""

    async def _handle_invoke(self, request: Request) -> Response:
        try:
            message, stream = await self.parse_request(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        config = self.build_runnable_config(request)
        snapshot = await self.graph.aget_state(config)
        if not snapshot.next:
            graph_input: dict[str, Any] | Command | None = self.build_input(message)
        elif any(task.interrupts for task in snapshot.tasks):
            decision = message.strip().lower()
            if decision in _APPROVAL_MESSAGES:
                graph_input = Command(resume=True)
            elif decision in _REJECTION_MESSAGES:
                graph_input = Command(resume=False)
            else:
                return JSONResponse(
                    {
                        "error": (
                            "This session is waiting for approval. Send "
                            "'approve' or 'reject'."
                        )
                    },
                    status_code=409,
                )
        else:
            graph_input = None
            configurable = dict(config.get("configurable", {}))
            configurable["invocation_recovery"] = True
            config = {**config, "configurable": configurable}

        if stream:
            return StreamingResponse(
                self._stream_with_approval(graph_input, config),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            output = await self.graph.ainvoke(graph_input, config=config)
        except Exception:  # noqa: BLE001 - invocation boundary returns HTTP 500
            logger.exception("Durable LangGraph invocation failed")
            return JSONResponse(
                {"error": "Internal server error."},
                status_code=500,
            )

        body: dict[str, Any] = {"response": self.parse_output(output)}
        approval = self._approval_from_snapshot(await self.graph.aget_state(config))
        if approval is not None:
            body["status"] = "approval_required"
            body["approval"] = approval
        return JSONResponse(body)

    async def _stream_with_approval(
        self,
        graph_input: dict[str, Any] | Command | None,
        config: RunnableConfig,
    ) -> AsyncIterator[bytes]:
        done_frame = b"event: done\ndata: {}\n\n"
        saw_done = False
        async for frame in self._stream_tokens(graph_input, config):
            if frame == done_frame:
                saw_done = True
                continue
            yield frame

        if not saw_done:
            return

        approval = self._approval_from_snapshot(await self.graph.aget_state(config))
        if approval is not None:
            payload = json.dumps(approval, ensure_ascii=False)
            yield f"event: approval_required\ndata: {payload}\n\n".encode()
        yield done_frame

    @staticmethod
    def _approval_from_snapshot(snapshot: Any) -> dict[str, Any] | None:
        for task in snapshot.tasks:
            for pending in task.interrupts:
                value = pending.value if isinstance(pending.value, dict) else {}
                arguments = value.get("arguments")
                return {
                    "id": str(pending.id),
                    "action": str(value.get("action") or "sensitive tool"),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                    "prompt": str(value.get("prompt") or "Approve this tool call?"),
                }
        return None


async def amain() -> None:
    _install_otel_langgraph_callback_compatibility()
    model = build_real_model()
    async with AsyncSqliteSaver.from_conn_string(_CHECKPOINT_DB) as checkpointer:
        graph = build_graph(checkpointer, model)
        server = DurableInvocationsHostServer(graph)
        await server.run_async(port=int(os.environ.get("PORT", "8088")))


if __name__ == "__main__":
    asyncio.run(amain())