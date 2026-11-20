"""LangGraph checkpointer with Azure Table Storage.

Runs a small graph twice on one thread to show state persisting across
invocations, then inspects the saved state and history. No LLM is needed.

Prerequisites:
    pip install -r requirements.txt

Environment variables (set one):
    AZURE_STORAGE_CONNECTION_STRING  - connection string, e.g. for the Azurite
                                       emulator
    AZURE_STORAGE_TABLE_ENDPOINT     - table endpoint of a live account, e.g.
                                       https://<account>.table.core.windows.net
                                       (authenticates with DefaultAzureCredential)
"""

import operator
import os
import uuid
from typing import Annotated

from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from langchain_azure_storage.checkpoint import AzureTableStorageSaver

# A unique table per run keeps the sample from touching existing data.
TABLE_NAME = f"sample{uuid.uuid4().hex[:12]}"


class State(TypedDict):
    """Graph state: a list that grows by concatenation."""

    history: Annotated[list[str], operator.add]


def audit(state: State) -> dict:
    """Record how much history this node could see."""
    return {"history": [f"node saw {len(state['history'])} entries"]}


def build_graph(saver: AzureTableStorageSaver) -> CompiledStateGraph:
    """Compile a one-node graph that checkpoints to *saver*."""
    builder = StateGraph(State)
    builder.add_node("audit", audit)
    builder.add_edge(START, "audit")
    builder.add_edge("audit", END)
    return builder.compile(checkpointer=saver)


def connect() -> tuple[AzureTableStorageSaver, TableServiceClient]:
    """Create the saver and a service client (for cleanup) from the environment."""
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if connection_string:
        return (
            AzureTableStorageSaver.from_connection_string(
                connection_string, TABLE_NAME
            ),
            TableServiceClient.from_connection_string(connection_string),
        )
    endpoint = os.environ["AZURE_STORAGE_TABLE_ENDPOINT"]
    return (
        AzureTableStorageSaver(endpoint, TABLE_NAME),
        TableServiceClient(endpoint, credential=DefaultAzureCredential()),
    )


def main() -> None:
    """Run the checkpointer sample."""
    saver, service = connect()
    try:
        with saver:
            graph = build_graph(saver)
            thread: RunnableConfig = {"configurable": {"thread_id": "demo-thread-1"}}

            print("=== Turn 1 ===")
            result = graph.invoke({"history": ["Alice: hi"]}, thread)
            print(f"  history: {result['history']}\n")

            print("=== Turn 2 (state is restored from Table Storage) ===")
            result = graph.invoke({"history": ["Alice: again"]}, thread)
            print(f"  history: {result['history']}\n")

            print("=== Current state ===")
            state = graph.get_state(thread)
            print(f"  checkpoint: {state.config['configurable']['checkpoint_id']}")
            print(f"  entries: {len(state.values['history'])}\n")

            print("=== State history (newest first) ===")
            for snapshot in graph.get_state_history(thread):
                print(
                    f"  step={(snapshot.metadata or {}).get('step')} "
                    f"entries={len(snapshot.values.get('history', []))}"
                )
            print()

            print("=== New thread (no memory of thread 1) ===")
            other: RunnableConfig = {"configurable": {"thread_id": "demo-thread-2"}}
            result = graph.invoke({"history": ["Bob: hi"]}, other)
            print(f"  history: {result['history']}\n")
    finally:
        print("Cleaning up...")
        with service:
            service.delete_table(TABLE_NAME)


if __name__ == "__main__":
    main()
