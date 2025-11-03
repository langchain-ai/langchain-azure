"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from langchain_azure_ai.callbacks.tracers import AzureAIOpenTelemetryTracer
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str



async def call_model(state: MessagesState, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    
    global model
    return {
        "messages": [
            await model.ainvoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant that always replies back to the user stating exactly the opposite of what the user said."
                    )
                ]
                + state["messages"]
            )
        ],
    }

agent_id = "negative-agent"

model = init_chat_model("openai:gpt-4.1")
tracer = AzureAIOpenTelemetryTracer(enable_content_recording=True, name=agent_id)

graph = (
    StateGraph(MessagesState, context_schema=Context)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile(name=agent_id)
    .with_config({ "callbacks": [tracer] })
)
