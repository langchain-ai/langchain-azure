from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import UUID

import pytest
from langgraph.graph import END, START, StateGraph

from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    Attrs,
    AzureAIOpenTelemetryTracer,
)


@dataclass
class State:
    chat_history: List[Any]
    final: str | None = None


@dataclass
class _CompletedSpan:
    name: str
    operation: str
    run_id: str
    parent_run_id: str | None
    attributes: Dict[str, Any]


class RecordingTracer(AzureAIOpenTelemetryTracer):
    """Tracer that records completed spans for assertions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.completed_spans: List[_CompletedSpan] = []
        self._span_names: Dict[str, str] = {}

    def _start_span(  # type: ignore[override]
        self,
        run_id: UUID,
        name: str,
        *,
        operation: str,
        kind: Any,
        parent_run_id: UUID | None,
        attributes: Dict[str, Any] | None = None,
        thread_key: str | None = None,
    ) -> None:
        super()._start_span(
            run_id,
            name,
            operation=operation,
            kind=kind,
            parent_run_id=parent_run_id,
            attributes=attributes,
            thread_key=thread_key,
        )
        self._span_names[str(run_id)] = name

    def _end_span(  # type: ignore[override]
        self,
        run_id: UUID,
        *,
        status: Any = None,
        error: Any = None,
    ) -> None:
        run_key = str(run_id)
        record = self._spans.get(run_key)
        span_name = self._span_names.pop(run_key, None)
        operation = record.operation if record else ""
        run_id_str = record.run_id if record else run_key
        parent_run = record.parent_run_id if record else None
        attributes = dict(record.attributes) if record else {}
        super()._end_span(run_id, status=status, error=error)
        if span_name is not None:
            self.completed_spans.append(
                _CompletedSpan(
                    name=span_name,
                    operation=operation,
                    run_id=run_id_str,
                    parent_run_id=parent_run,
                    attributes=attributes,
                )
            )


class FakeCommand(dict):
    def __init__(self, update: Dict[str, Any], goto: str) -> None:
        super().__init__(update)
        self.update = update  # type: ignore[method-assign, assignment]
        self.goto = goto


@pytest.mark.asyncio
async def test_custom_graph_tracer_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    tracer = RecordingTracer(enable_content_recording=True, name="custom-graph")

    analyze_metadata = {
        "otel_trace": True,
        "otel_messages_key": "chat_history",
        "agent_name": "custom-graph",
        "langgraph_node": "analyze",
    }
    review_metadata = {
        "otel_trace": True,
        "otel_messages_key": "chat_history",
        "agent_name": "custom-graph",
        "langgraph_node": "review",
    }

    async def analyze(state: State, runtime: Any | None = None) -> FakeCommand:
        return FakeCommand(
            update={
                "chat_history": state.chat_history
                + [{"role": "assistant", "content": "step1"}]
            },
            goto="review",
        )

    async def review(state: State, runtime: Any | None = None) -> Dict[str, Any]:
        return {
            "chat_history": state.chat_history
            + [{"role": "assistant", "content": "done"}],
            "final": "done",
        }

    graph = (
        StateGraph(State)
        .add_node("analyze", analyze, metadata=analyze_metadata)
        .add_node("review", review, metadata=review_metadata)
        .add_edge(START, "analyze")
        .add_edge("analyze", "review")
        .add_edge("review", END)
        .compile(name="custom-graph")
        .with_config({"callbacks": [tracer]})
    )

    result = await graph.ainvoke(
        State(chat_history=[{"role": "user", "content": "hi"}])  # type: ignore[arg-type]
    )
    final_value = getattr(result, "final", None)
    if isinstance(result, dict):
        final_value = result.get("final", final_value)
    assert final_value == "done"

    assert any(
        span.attributes.get("metadata.langgraph.goto") == "review"
        for span in tracer.completed_spans
    )

    input_messages = [
        span.attributes.get(Attrs.INPUT_MESSAGES)
        for span in tracer.completed_spans
        if span.attributes.get(Attrs.INPUT_MESSAGES)
    ]
    assert input_messages
    parsed = json.loads(input_messages[0])  # type: ignore[arg-type]
    assert parsed[0]["role"] == "user"
