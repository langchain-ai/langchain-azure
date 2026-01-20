from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List
from uuid import UUID

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from pydantic import SecretStr
from typing_extensions import TypedDict

from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    AzureAIOpenTelemetryTracer,
)


class Context(TypedDict):
    my_configurable_param: str


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


def _build_negative_agent(tracer: RecordingTracer, use_azure: bool) -> Any:
    model: BaseChatModel
    if use_azure:
        model = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            temperature=0.1,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            seed=100,
        )
    else:
        model = ChatOpenAI(model="gpt-4.1")

    async def call_model(
        state: MessagesState,
        runtime: Runtime[Context],
    ) -> Dict[str, Any]:
        prompt_messages = [
            SystemMessage(
                content=(
                    "You are a helpful assistant that always replies back "
                    "to the user stating exactly the opposite of what the "
                    "user said."
                )
            ),
            *state["messages"],
        ]
        return {"messages": [await model.ainvoke(prompt_messages)]}

    graph = (
        StateGraph(MessagesState, context_schema=Context)
        .add_node(call_model)
        .add_edge(START, "call_model")
        .compile(name="negative-agent")
        .with_config({"callbacks": [tracer]})
    )
    return graph


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_negative_agent_tracer_records(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-test")

    tracer = RecordingTracer(enable_content_recording=True, name="negative-agent")
    graph = _build_negative_agent(tracer, use_azure=False)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="It is sunny today.")]},
    )

    assert result["messages"][-1].content.lower().startswith("it is not")

    relevant_spans = {span.name: span for span in tracer.completed_spans}

    assert "invoke_agent negative-agent" in relevant_spans
    llm_span = next(
        span
        for span in tracer.completed_spans
        if span.operation in {"text_completion", "chat"}
    )
    root_span = relevant_spans["invoke_agent negative-agent"]

    assert llm_span.parent_run_id == root_span.run_id
    assert root_span.operation == "invoke_agent"
    assert llm_span.operation in {"text_completion", "chat"}
    assert root_span.attributes["gen_ai.agent.name"] == "negative-agent"


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_negative_agent_tracer_records_azure() -> None:
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing = [var for var in required if var not in os.environ]
    if missing:
        pytest.skip(f"Missing Azure env vars: {', '.join(missing)}")

    tracer = RecordingTracer(enable_content_recording=True, name="negative-agent")
    graph = _build_negative_agent(tracer, use_azure=True)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="It is sunny today.")]},
    )

    assert result["messages"][-1].content.lower().startswith("it is not")

    relevant_spans = {span.name: span for span in tracer.completed_spans}

    assert "invoke_agent negative-agent" in relevant_spans
