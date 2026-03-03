"""
Sample illustrating AzureAIOpenTelemetryTracer compatibility with LangGraph:
- Dataclass state
- Custom message key/path overrides
- Command-like returns that include goto
- Network-free (no model calls)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from langchain_azure_ai.callbacks.tracers.inference_tracing import AzureAIOpenTelemetryTracer


@dataclass
class State:
    chat_history: list[Any]
    final: str | None = None


class FakeCommand(dict):
    """Duck-typed Command replacement carrying update and goto."""

    def __init__(self, update: Dict[str, Any], goto: str) -> None:
        super().__init__(update)
        self.update = update
        self.goto = goto


async def analyze(state: State, runtime: Any | None = None) -> FakeCommand:
    """Simulate a node that updates chat history and jumps to review."""
    return FakeCommand(
        {"chat_history": state.chat_history + [{"role": "assistant", "content": "step1"}]},
        goto="review",
    )


async def review(state: State, runtime: Any | None = None) -> Dict[str, Any]:
    """Finalize the conversation."""
    return {
        "chat_history": state.chat_history + [{"role": "assistant", "content": "done"}],
        "final": "done",
    }


async def main() -> None:
    tracer = AzureAIOpenTelemetryTracer(
        message_keys=("messages",),
        message_paths=("chat_history",),  # messages live under chat_history
        trace_all_langgraph_nodes=True,
    )

    graph = (
        StateGraph(State)
        .add_node(
            "analyze",
            analyze,
            metadata={
                "otel_trace": True,
                "otel_messages_key": "chat_history",
                "langgraph_node": "analyze",
            },
        )
        .add_node(
            "review",
            review,
            metadata={
                "otel_trace": True,
                "otel_messages_key": "chat_history",
                "langgraph_node": "review",
            },
        )
        .add_edge(START, "analyze")
        .add_edge("analyze", "review")
        .add_edge("review", END)
        .compile(name="tracer-compat-graph")
        .with_config({"callbacks": [tracer]})
    )

    result = await graph.ainvoke(State(chat_history=[{"role": "user", "content": "hi"}]))
    print("Final state:", result)


if __name__ == "__main__":
    asyncio.run(main())
