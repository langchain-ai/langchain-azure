# Copyright (c) Microsoft. All rights reserved.

"""Example 04: `create_react_agent` and compiled subgraphs as nodes."""

from __future__ import annotations

import os
import sys
import warnings
from operator import add
from typing import Annotated, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from langchain_azure_compute.graph import attach_durability, durable_tool, sdk_options

warnings.filterwarnings("ignore", category=DeprecationWarning)


class OfflineCalcModel(BaseChatModel):
    """Deterministic local chat model that invokes the `multiply` tool once."""

    @property
    def _llm_type(self) -> str:
        return "offline-calc"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_msgs:
            msg = AIMessage(content=f"The product is {tool_msgs[-1].content}.")
        else:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call-mul-1", "type": "tool_call"}],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@durable_tool(retries=3)
@tool
def multiply(a: int, b: int) -> str:
    """Multiply two integers inside a durable tool activity."""
    return str(a * b)


class ParentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    trace: Annotated[list[str], add]


def build_parent_with_subgraph():
    react_subgraph = create_react_agent(OfflineCalcModel(), [multiply])

    builder = StateGraph(ParentState)
    builder.add_node("prepare", lambda s: {"trace": ["prepare"]})
    builder.add_node("calculator_agent", react_subgraph)
    builder.add_node("finalize", lambda s: {"trace": ["finalize"]})
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "calculator_agent")
    builder.add_edge("calculator_agent", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()


def main() -> None:
    endpoint = (
        os.environ.get("DTS_ENDPOINT")
        or os.environ.get("DTS_EMULATOR")
        or os.environ.get("DURABLE_TASK_SCHEDULER_CONNECTION_STRING")
    )
    if not endpoint:
        import socket

        for _cand_port in (18080, 8080):
            try:
                with socket.create_connection(("127.0.0.1", _cand_port), timeout=0.3):
                    endpoint = f"127.0.0.1:{_cand_port}"
                    break
            except OSError:
                pass
    if not endpoint:
        print("Set DTS_EMULATOR=127.0.0.1:8080 (or DURABLE_TASK_SCHEDULER_CONNECTION_STRING) to run.")
        print("Start one with: docker run -d -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest")
        sys.exit(0)

    reg = attach_durability(build_parent_with_subgraph(), name="react_subgraph_demo")
    worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
    sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
    reg.register(worker)
    worker.start()
    try:
        client = reg.client(sdk)
        run = client.start({"messages": [HumanMessage(content="What is 6 * 7?")], "trace": []})
        out = run.wait_result(timeout=60)
        print("Trace:", out["trace"])
        print("Final AI reply:", out["messages"][-1].content)
    finally:
        worker.stop()
        sdk.close()


if __name__ == "__main__":
    main()
