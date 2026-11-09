# Copyright (c) Microsoft. All rights reserved.

"""Example 01: Minimal linear & conditional graph in Zero-Config mode (`builder.compile()`)."""

from __future__ import annotations

import asyncio
import operator
import os
import sys
from typing import Annotated, NotRequired, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langgraph.graph import END, START, StateGraph

from langchain_azure_compute.graph import attach_durability, sdk_options


class HelloInput(TypedDict):
    text: str


class HelloState(TypedDict):
    text: str
    intent: NotRequired[str]
    reply: NotRequired[str]
    visited: NotRequired[Annotated[list[str], operator.add]]


class HelloOutput(TypedDict):
    text: str
    reply: str
    visited: list[str]


def normalize(state: HelloState) -> dict:
    return {"text": state["text"].strip().lower(), "visited": ["normalize"]}


async def classify(state: HelloState) -> dict:
    await asyncio.sleep(0)
    intent = "greeting" if state["text"].startswith("hello") else "farewell"
    return {"intent": intent, "visited": ["classify"]}


def greeting(state: HelloState) -> dict:
    return {"reply": "Hello from a durable LangGraph workflow!", "visited": ["greeting"]}


def farewell(state: HelloState) -> dict:
    return {"reply": "Goodbye from a durable LangGraph workflow!", "visited": ["farewell"]}


def build_graph():
    builder = StateGraph(HelloState, input_schema=HelloInput, output_schema=HelloOutput)
    builder.add_node("normalize", normalize)
    builder.add_node("classify", classify)
    builder.add_node("greeting", greeting)
    builder.add_node("farewell", farewell)
    builder.add_edge(START, "normalize")
    builder.add_edge("normalize", "classify")
    builder.add_conditional_edges("classify", lambda s: s["intent"], {"greeting": "greeting", "farewell": "farewell"})
    builder.add_edge("greeting", END)
    builder.add_edge("farewell", END)
    # Zero-Config mode: no checkpointer argument required
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

    reg = attach_durability(build_graph(), name="hello_graph")
    worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
    sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
    reg.register(worker)
    worker.start()
    try:
        client = reg.client(sdk)
        for sample_text in (" Hello World ", "See you later"):
            # Omitting instance_id generates a fresh orchestration instance on each run
            # (see 02_vacation_booking.py for idempotent instance_id reuse).
            run = client.start({"text": sample_text})
            result = run.wait_result(timeout=30)
            print(f"Input: {sample_text!r} -> Output: {result}")
    finally:
        worker.stop()
        sdk.close()


if __name__ == "__main__":
    main()
