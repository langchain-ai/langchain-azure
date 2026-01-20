"""End-to-end integration tests for agent tracing with real LLM calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

import pytest
from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

from langchain_azure_ai.callbacks.tracers.inference_tracing import (
    Attrs,
    AzureAIOpenTelemetryTracer,
)


@dataclass
class _RecordedSpan:
    name: str
    operation: str
    run_id: str
    parent_run_id: str | None
    attributes: Dict[str, Any]


class RecordingTracer(AzureAIOpenTelemetryTracer):
    """Tracer that records completed spans for assertions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.completed_spans: List[_RecordedSpan] = []
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
                _RecordedSpan(
                    name=span_name,
                    operation=operation,
                    run_id=run_id_str,
                    parent_run_id=parent_run,
                    attributes=attributes,
                )
            )


def _messages_state(*messages: AnyMessage) -> MessagesState:
    return {"messages": list(messages)}


def _get_openai_model() -> ChatOpenAI:
    """Create a ChatOpenAI model for testing."""
    return ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
        seed=42,
    )


def _get_azure_model() -> AzureChatOpenAI:
    """Create an AzureChatOpenAI model for testing."""
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        temperature=0,
        seed=42,
    )


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_basic_agent_tracing_records_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure a simple agent invocation produces agent and LLM spans."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-test")

    tracer = RecordingTracer(enable_content_recording=True, name="informational-agent")
    model = _get_openai_model()

    async def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        prompt = [
            SystemMessage(
                content="You are an informational agent. Answer questions cheerfully "
                "and concisely in one sentence."
            ),
            *state["messages"],
        ]
        response = await model.ainvoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    app = workflow.compile(name="informational-agent").with_config(
        {"callbacks": [tracer]}
    )

    input_state = _messages_state(
        HumanMessage(content="What's the weather like in general?"),
    )
    result = await app.ainvoke(cast(Any, input_state))

    assert result["messages"][-1].content
    span_names = [span.name for span in tracer.completed_spans]
    assert any(name.startswith("invoke_agent") for name in span_names)
    chat_span = next(
        span for span in tracer.completed_spans if span.operation == "chat"
    )
    assert chat_span.attributes.get(Attrs.OPERATION_NAME) == "chat"
    root_span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    )
    assert root_span.attributes.get(Attrs.AGENT_NAME) == "informational-agent"
    assert chat_span.parent_run_id == root_span.run_id


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_agent_with_tool_records_tool_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure tool execution is traced with arguments and results."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-test")

    tracer = RecordingTracer(enable_content_recording=True, name="weather-agent")

    @tool
    def get_weather(city: str) -> dict[str, Any]:
        """Return weather information for a city."""
        return {"temperature": 72, "description": "Sunny", "city": city}

    model = _get_openai_model()
    agent: Any = create_agent(model=model, tools=[get_weather])
    agent = agent.with_config({"callbacks": [tracer]})

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="What's the weather in San Francisco?")]},
    )

    assert result["messages"][-1].content
    assert (
        "San Francisco" in result["messages"][-1].content
        or "72" in result["messages"][-1].content
    )

    tool_span = next(
        (
            span
            for span in tracer.completed_spans
            if span.operation == "execute_tool"
            and span.attributes.get(Attrs.TOOL_NAME) == "get_weather"
        ),
        None,
    )
    assert tool_span is not None
    assert tool_span.attributes.get(Attrs.TOOL_CALL_ARGUMENTS)

    root_span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    )
    assert root_span.attributes.get(Attrs.AGENT_NAME) == "LangGraph"


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_langgraph_agent_loop_records_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure LangGraph agent/tool loop traces correctly."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-test")

    tracer = RecordingTracer(enable_content_recording=True, name="calculator-agent")

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @tool
    def multiply_numbers(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b

    model = _get_openai_model()

    def should_continue(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        return "continue" if getattr(last_message, "tool_calls", None) else "end"

    async def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        prompt = [
            SystemMessage(
                content="You are a calculator assistant. Use the provided tools to "
                "perform calculations. Always use the tools for math operations."
            ),
            *state["messages"],
        ]
        response = await model.bind_tools([add_numbers, multiply_numbers]).ainvoke(
            prompt
        )
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", ToolNode([add_numbers, multiply_numbers]))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "action", "end": END},
    )
    workflow.add_edge("action", "agent")
    app = workflow.compile(checkpointer=MemorySaver(), name="calculator-agent")
    app = app.with_config({"callbacks": [tracer]})

    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": "calc-thread"}},
    )
    final_message: Optional[AIMessage] = None
    initial_state: MessagesState = {
        "messages": [HumanMessage(content="What is 5 + 3?")]
    }
    async for event in app.astream(
        cast(Any, initial_state),
        config=config,
        stream_mode="values",
    ):
        final_message = cast(AIMessage, event["messages"][-1])

    assert final_message is not None
    assert "8" in str(final_message.content)

    execute_spans = [
        span for span in tracer.completed_spans if span.operation == "execute_tool"
    ]
    assert len(execute_spans) > 0
    assert execute_spans[0].attributes.get(Attrs.TOOL_NAME) == "add_numbers"

    root_span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    )
    assert root_span.attributes.get(Attrs.AGENT_NAME) == "calculator-agent"


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_multi_turn_conversation_with_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure multi-turn conversations maintain thread context."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-test")

    tracer = RecordingTracer(enable_content_recording=True, name="multi-turn-agent")
    model = _get_openai_model()

    async def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        prompt = [
            SystemMessage(
                content="You are a helpful assistant. Keep your responses brief."
            ),
            *state["messages"],
        ]
        response = await model.ainvoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    app = workflow.compile(checkpointer=MemorySaver(), name="multi-turn-agent")
    app = app.with_config({"callbacks": [tracer]})

    thread_id = "multi-turn-thread-123"
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": thread_id}},
    )

    # First turn
    input1: MessagesState = {"messages": [HumanMessage(content="Hi there!")]}
    result1 = await app.ainvoke(cast(Any, input1), config=config)
    assert result1["messages"][-1].content

    # Second turn - continue the conversation
    result2 = await app.ainvoke(
        cast(Any, {"messages": [HumanMessage(content="What can you help me with?")]}),
        config=config,
    )
    assert result2["messages"][-1].content

    # Verify spans were created for both turns
    root_spans = [
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    ]
    assert len(root_spans) >= 2
    # All root spans should have conversation ID attribute set
    for span in root_spans:
        assert span.attributes.get(Attrs.CONVERSATION_ID) == thread_id


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_agent_with_content_recording_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure content is redacted when content recording is disabled."""
    monkeypatch.setenv("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY") or "sk-test")

    tracer = RecordingTracer(enable_content_recording=False, name="redacted-agent")
    model = _get_openai_model()

    async def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        prompt = [
            SystemMessage(content="You are a helpful assistant."),
            *state["messages"],
        ]
        response = await model.ainvoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    app = workflow.compile(name="redacted-agent").with_config({"callbacks": [tracer]})

    input_state = _messages_state(HumanMessage(content="Tell me a secret"))
    result = await app.ainvoke(cast(Any, input_state))

    assert result["messages"][-1].content

    chat_span = next(
        span for span in tracer.completed_spans if span.operation == "chat"
    )
    # Input messages should be redacted
    input_messages = chat_span.attributes.get(Attrs.INPUT_MESSAGES)
    if input_messages:
        assert "[redacted]" in input_messages


@pytest.mark.asyncio
@pytest.mark.block_network()
@pytest.mark.vcr()
async def test_basic_agent_tracing_azure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure Azure OpenAI agent tracing works correctly."""
    required = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    missing = [var for var in required if var not in os.environ]
    if missing:
        pytest.skip(f"Missing Azure env vars: {', '.join(missing)}")

    tracer = RecordingTracer(enable_content_recording=True, name="azure-agent")
    model = _get_azure_model()

    async def call_model(state: MessagesState) -> Dict[str, List[BaseMessage]]:
        prompt = [
            SystemMessage(
                content="You are a helpful assistant. Answer concisely in one sentence."
            ),
            *state["messages"],
        ]
        response = await model.ainvoke(prompt)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    app = workflow.compile(name="azure-agent").with_config({"callbacks": [tracer]})

    input_state = _messages_state(HumanMessage(content="What is Python?"))
    result = await app.ainvoke(cast(Any, input_state))

    assert result["messages"][-1].content
    root_span = next(
        span
        for span in tracer.completed_spans
        if span.operation == "invoke_agent" and span.parent_run_id is None
    )
    assert root_span.attributes.get(Attrs.AGENT_NAME) == "azure-agent"
    chat_span = next(
        span for span in tracer.completed_spans if span.operation == "chat"
    )
    assert chat_span.parent_run_id == root_span.run_id
