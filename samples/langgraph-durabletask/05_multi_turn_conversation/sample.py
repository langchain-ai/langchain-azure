# Copyright (c) Microsoft. All rights reserved.

"""Example 05: Multi-turn stateful conversation with `SqliteSaver` checkpointer, `SqliteStore`, and `config` / `Runtime[Context]` propagation."""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict

from durabletask.azuremanaged.client import DurableTaskSchedulerClient
from durabletask.azuremanaged.worker import DurableTaskSchedulerWorker
from langchain_azure_compute.graph import attach_durability, sdk_options
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.sqlite import SqliteStore


@dataclass
class AppContext:
    tenant_id: str


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str


def assistant_node(
    state: ChatState, config: RunnableConfig, runtime: Runtime[AppContext]
) -> dict:
    uid = state["user_id"]
    tenant = runtime.context.tenant_id if runtime.context else "default"
    channel = (config.get("configurable") or {}).get("channel", "web")
    human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    if not human.id:
        raise ValueError("Each input HumanMessage needs a stable checkpointed ID")
    last_user_msg = human.content
    ns = ("memories", tenant, uid)
    assert runtime.store is not None
    fact_ns = (*ns, "facts")
    runtime.store.put(fact_ns, human.id, {"fact": str(last_user_msg)})
    facts = [item.value["fact"] for item in runtime.store.search(fact_ns, limit=1000)]
    runtime.store.put(ns, "history", {"facts": facts, "channel": channel})

    reply = f"[{tenant}/{channel}] Turn #{len(facts)} recorded for {uid}: remembered {len(facts)} fact(s)."
    return {"messages": [AIMessage(content=reply, id=f"reply-{human.id}")]}


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
        print(
            "Set DTS_EMULATOR=127.0.0.1:8080 (or DURABLE_TASK_SCHEDULER_CONNECTION_STRING) to run."
        )
        print(
            "Start one with: docker run -d -p 8080:8080 -p 8082:8082 mcr.microsoft.com/dts/dts-emulator:latest"
        )
        sys.exit(0)

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        cp_conn = sqlite3.connect(Path(tmp) / "checkpoints.db", check_same_thread=False)
        store_conn = sqlite3.connect(
            Path(tmp) / "store.db", check_same_thread=False, isolation_level=None
        )
        saver = SqliteSaver(cp_conn)
        store = SqliteStore(store_conn)
        store.setup()

        builder = StateGraph(ChatState, context_schema=AppContext)
        builder.add_node("assistant", assistant_node)
        builder.add_edge(START, "assistant")
        builder.add_edge("assistant", END)
        graph = builder.compile(checkpointer=saver, store=store)

        reg = attach_durability(graph, name="multi_turn_chat")
        worker = DurableTaskSchedulerWorker(**sdk_options(endpoint))
        sdk = DurableTaskSchedulerClient(**sdk_options(endpoint))
        reg.register(worker)
        worker.start()
        try:
            client = reg.client(sdk)
            shared_thread = "conversation-user-42"
            ctx = AppContext(tenant_id="acme-corp")
            cfg = {"configurable": {"channel": "mobile"}, "tags": ["vip"]}

            # Turn 1 on shared_thread
            run1 = client.start(
                {
                    "messages": [
                        HumanMessage(content="I prefer window seats.", id="turn-1")
                    ],
                    "user_id": "user-42",
                },
                thread_id=shared_thread,
                config=cfg,
                context=ctx,
            )
            out1 = run1.wait_result(timeout=30)
            print("Turn 1 reply:", out1["messages"][-1].content)

            # Turn 2 on the same shared_thread (accumulates messages in SqliteSaver + facts in SqliteStore)
            run2 = client.start(
                {
                    "messages": [
                        HumanMessage(
                            content="I also like boutique hotels.", id="turn-2"
                        )
                    ],
                    "user_id": "user-42",
                },
                thread_id=shared_thread,
                config=cfg,
                context=ctx,
            )
            out2 = run2.wait_result(timeout=30)
            print("Turn 2 reply:", out2["messages"][-1].content)
            print("Total messages in thread:", len(out2["messages"]))
            print(
                "SqliteStore facts:",
                store.get(("memories", "acme-corp", "user-42"), "history").value,
            )
        finally:
            worker.stop()
            sdk.close()
            cp_conn.close()
            store_conn.close()


if __name__ == "__main__":
    main()
