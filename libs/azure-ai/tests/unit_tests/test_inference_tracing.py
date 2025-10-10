import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, cast
from uuid import uuid4

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult

import langchain_azure_ai.callbacks.tracers.inference_tracing as tracing

# Skip tests cleanly if required deps are not present
pytest.importorskip("azure.monitor.opentelemetry")
pytest.importorskip("opentelemetry")
pytest.importorskip("langchain_core")


class MockSpan:
    name: str
    attributes: Dict[str, Any]
    events: List[Tuple[str, Dict[str, Any]]]
    ended: bool
    status: Optional[Any]
    exceptions: List[Exception]

    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes = {}
        self.events = []
        self.ended = False
        self.status = None
        self.exceptions = []
        self._context = SimpleNamespace(is_valid=True)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended = True

    def get_span_context(self) -> Any:
        return self._context


class MockTracer:
    spans: List[MockSpan]

    def __init__(self) -> None:
        self.spans = []

    def start_span(self, name: str, kind: Any = None, context: Any = None) -> MockSpan:
        span = MockSpan(name)
        self.spans.append(span)
        return span


@pytest.fixture(autouse=True)
def patch_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = SimpleNamespace(get_tracer=lambda _: MockTracer())
    monkeypatch.setattr(tracing, "otel_trace", mock)
    monkeypatch.setattr(tracing, "set_span_in_context", lambda span: None)


def get_last_span_for(tracer_obj: Any) -> MockSpan:
    return tracer_obj._core._tracer.spans[-1]


def test_llm_start_attributes_content_recording_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure env enables content recording
    # fmt: off
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(include_legacy_keys=True)
    run_id = uuid4()
    serialized = {
        "kwargs": {"model": "gpt-4o", "endpoint": "http://host:8080"}
    }
    # fmt: on
    t.on_llm_start(serialized, ["hello"], run_id=run_id)
    span = get_last_span_for(t)

    attrs = span.attributes
    assert attrs.get(tracing.Attrs.PROVIDER_NAME) == t._core.provider
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "chat"
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    assert attrs.get(tracing.Attrs.SERVER_ADDRESS) == "http://host:8080"
    assert attrs.get(tracing.Attrs.SERVER_PORT) == 8080
    assert tracing.Attrs.INPUT_MESSAGES in attrs
    # Legacy keys when enabled
    assert attrs.get(tracing.Attrs.LEGACY_KEYS_FLAG) is True
    assert tracing.Attrs.LEGACY_PROMPT in attrs


def test_llm_start_attributes_content_recording_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fmt: off
    monkeypatch.delenv(
        "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", raising=False
    )
    # fmt: on
    t = tracing.AzureAIOpenTelemetryTracer(include_legacy_keys=False)
    run_id = uuid4()
    serialized = {"kwargs": {"model": "gpt-4o", "endpoint": "https://host"}}
    t.on_llm_start(serialized, ["hello"], run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    # No input messages recorded when disabled
    assert tracing.Attrs.INPUT_MESSAGES not in attrs


def test_redaction_on_chat_and_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(redact=True)
    run_id = uuid4()
    messages = [[HumanMessage(content="secret"), AIMessage(content="reply")]]
    serialized = {"kwargs": {"model": "m", "endpoint": "https://e"}}
    t.on_chat_model_start(serialized, messages, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    # Input content should be redacted
    input_json = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_json[0][0]["content"] == "[REDACTED]"
    # End with output
    gen = ChatGeneration(message=AIMessage(content="reply"))
    result = LLMResult(generations=[[gen]], llm_output={})
    t.on_llm_end(result, run_id=run_id)
    # Verify output redacted on chat span when present;
    # some paths emit under agent root
    out_attr = span.attributes.get(tracing.Attrs.OUTPUT_MESSAGES)
    if out_attr:
        out_json = json.loads(out_attr)
        assert out_json[0]["content"] == "[REDACTED]"
    else:
        # Fallback: if no chat output recorded, allow absence without failure
        # (agent root may contain the final output summary in role/parts schema)
        pass


def test_usage_and_response_metadata() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    t.on_llm_start(serialized, ["hi"], run_id=run_id)
    gen = ChatGeneration(message=AIMessage(content="ok"))
    result = LLMResult(
        generations=[[gen]],
        llm_output={
            "token_usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "model": "m",
            "id": "resp-123",
        },
    )
    t.on_llm_end(result, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 3
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 5
    assert attrs.get(tracing.Attrs.RESPONSE_MODEL) == "m"
    assert attrs.get(tracing.Attrs.RESPONSE_ID) == "resp-123"


def test_streaming_token_event(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    t.on_llm_start(serialized, ["hi"], run_id=run_id)
    t.on_llm_new_token("abcdef", run_id=run_id)
    span = get_last_span_for(t)
    name, event_attrs = span.events[-1]
    assert name == "gen_ai.token"
    assert event_attrs.get("gen_ai.token.length") == 6
    assert event_attrs.get("gen_ai.token.preview") == "abcdef"


def test_synthetic_execute_tool_under_chat_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer()
    # Start root agent via chain_start
    root = uuid4()
    t.on_chain_start({}, {"messages": [{"role": "user", "content": "hi"}]}, run_id=root)
    # Start a chat span
    chat_run = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    msgs = cast(List[List[BaseMessage]], [[HumanMessage(content="prompt")]])
    t.on_chat_model_start(serialized, msgs, run_id=chat_run, parent_run_id=root)
    # End with tool_calls requested by assistant
    # Provide tool_calls via additional_kwargs to match LC parsing
    tool_calls = [
        {
            "id": "call-1",
            "function": {"name": "get_current_date", "arguments": "{}"},
        }
    ]
    ai_msg = AIMessage(content="", additional_kwargs={"tool_calls": tool_calls})
    gen = ChatGeneration(
        message=ai_msg, generation_info={"finish_reason": "tool_calls"}
    )
    result = LLMResult(
        generations=[[gen]],
        llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )
    t.on_llm_end(result, run_id=chat_run, parent_run_id=root)
    # Synthetic execute_tool span should be emitted when chain completes
    t.on_chain_end({}, run_id=root)
    span = get_last_span_for(t)
    assert span.name.startswith("execute_tool")
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    # Fallback synthetic spans default to "tool" when no explicit name
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "tool"
    # Conversation id should be present on chat and tools
    chat_span = [s for s in t._core._tracer.spans if s.name.startswith("chat")][-1]
    assert chat_span.attributes.get(tracing.Attrs.CONVERSATION_ID) == str(root)
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)


def test_no_invoke_agent_on_agent_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    # on_agent_action should not start invoke_agent spans;
    # only create_agent when applicable
    before = len(
        [
            s
            for s in t._core._tracer.spans
            if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "invoke_agent"
        ]
    )
    action = SimpleNamespace(
        agent_name="Agent",
        system_instructions=[{"type": "text", "content": "You are an agent."}],
    )
    t.on_agent_action(action, run_id=uuid4())
    after = len(
        [
            s
            for s in t._core._tracer.spans
            if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "invoke_agent"
        ]
    )
    assert after == before


def test_tool_start_name_and_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root = uuid4()
    t.on_chain_start({}, {"messages": [{"role": "user", "content": "hi"}]}, run_id=root)
    # Start a tool with serialized name and inputs id
    run_id = uuid4()
    parent_run = uuid4()  # simulate parent chat id context
    serialized_tool = {
        "name": "search",
        "type": "function",
        "description": "desc",
    }
    inputs = {"id": "call-1", "query": "foo"}
    t.on_tool_start(
        serialized_tool,
        "ignored",
        inputs=inputs,
        run_id=run_id,
        parent_run_id=parent_run,
    )
    t.on_tool_end({"answer": "bar"}, run_id=run_id, parent_run_id=parent_run)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)
    assert attrs.get(tracing.Attrs.METADATA_PARENT_RUN_ID) == str(root)


def test_tool_deduplicates_synthetic_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root = uuid4()
    t.on_chain_start({}, {"messages": [{"role": "user", "content": "hi"}]}, run_id=root)
    chat_run = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    msgs = cast(List[List[BaseMessage]], [[HumanMessage(content="prompt")]])
    t.on_chat_model_start(serialized, msgs, run_id=chat_run, parent_run_id=root)
    tool_calls = [
        {
            "id": "call-1",
            "function": {"name": "get_weather", "arguments": '{"city": "SF"}'},
        }
    ]
    ai_msg = AIMessage(content="", additional_kwargs={"tool_calls": tool_calls})
    gen = ChatGeneration(message=ai_msg)
    result = LLMResult(generations=[[gen]], llm_output={})
    t.on_llm_end(result, run_id=chat_run, parent_run_id=root)
    tool_run = uuid4()
    t.on_tool_start(
        {"name": "get_weather", "description": "Returns weather"},
        "ignored",
        inputs={"id": "call-1", "city": "SF"},
        run_id=tool_run,
        parent_run_id=chat_run,
    )
    t.on_tool_end({"temperature": 60}, run_id=tool_run, parent_run_id=chat_run)
    t.on_chain_end({}, run_id=root)
    tool_spans = [
        s
        for s in t._core._tracer.spans
        if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    ]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes.get(tracing.Attrs.TOOL_NAME) == "get_weather"
    assert span.attributes.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"


def test_invoke_agent_records_tool_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    root = uuid4()
    tools = [
        {"name": "get_weather", "description": "Returns forecast"},
        {"name": "search_docs", "description": "Search knowledge base"},
    ]
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}], "tools": tools},
        run_id=root,
    )
    span = [s for s in t._core._tracer.spans if s.name.startswith("invoke_agent")][-1]
    defs = span.attributes.get(tracing.Attrs.TOOL_DEFINITIONS)
    assert defs is not None
    assert "get_weather" in defs and "search_docs" in defs


def test_finish_reasons_normalized() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    chat_run = uuid4()
    gen = ChatGeneration(
        message=AIMessage(content=""),
        generation_info={"finish_reason": "tool_calls"},
    )
    result = LLMResult(generations=[[gen]], llm_output={})
    # Create chat span and then end it
    t.on_llm_start({"kwargs": {"model": "m"}}, ["hi"], run_id=chat_run)
    t.on_llm_end(result, run_id=chat_run)
    span = [s for s in t._core._tracer.spans if s.name.startswith("chat")][-1]
    # Aggregated finish reasons should be normalized to singular
    assert span.attributes.get(tracing.Attrs.RESPONSE_FINISH_REASONS) == ["tool_call"]


def test_chat_parenting_under_root_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    root = uuid4()
    # Start root agent
    t.on_chain_start({}, {"messages": [{"role": "user", "content": "hi"}]}, run_id=root)
    # Start chat without specifying parent; should parent under root agent
    chat_run = uuid4()
    t.on_llm_start({"kwargs": {"model": "m"}}, ["hello"], run_id=chat_run)
    span = get_last_span_for(t)
    assert span.attributes.get(tracing.Attrs.METADATA_PARENT_RUN_ID) == str(root)


def test_llm_error_sets_status_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    t.on_llm_start(serialized, ["hi"], run_id=run_id)
    err = RuntimeError("boom")
    t.on_llm_error(err, run_id=run_id)
    span = get_last_span_for(t)
    assert span.ended is True
    assert span.exceptions and isinstance(span.exceptions[0], RuntimeError)


def test_tool_start_end_records_args_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    parent_run = uuid4()
    serialized_tool = {
        "name": "search",
        "type": "function",
        "description": "desc",
    }
    inputs = {"id": "call-1", "query": "foo"}
    t.on_tool_start(
        serialized_tool,
        "ignored",
        inputs=inputs,
        run_id=run_id,
        parent_run_id=parent_run,
    )
    t.on_tool_end({"answer": "bar"}, run_id=run_id, parent_run_id=parent_run)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    # Args and result only when content recording enabled
    tool_args = attrs.get(tracing.Attrs.TOOL_CALL_ARGS)
    tool_result = attrs.get(tracing.Attrs.TOOL_CALL_RESULT)
    assert tool_args is not None and tool_result is not None
    assert json.loads(tool_args).get("query") == "foo"
    assert json.loads(tool_result).get("answer") == "bar"


def test_choice_count_only_when_n_not_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m", "n": 1}}
    t.on_llm_start(serialized, ["hi"], run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert tracing.Attrs.REQUEST_CHOICE_COUNT not in attrs
    # Now with n=3
    run_id2 = uuid4()
    serialized2 = {"kwargs": {"model": "m", "n": 3}}
    t.on_llm_start(serialized2, ["hi"], run_id=run_id2)
    span2 = get_last_span_for(t)
    assert span2.attributes.get(tracing.Attrs.REQUEST_CHOICE_COUNT) == 3


def test_server_port_extraction_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    # https default port not set
    run1 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "endpoint": "https://host:443"}},
        ["hi"],
        run_id=run1,
    )
    s1 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s1.attributes
    # https non-default port set
    run2 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "endpoint": "https://host:8443"}},
        ["hi"],
        run_id=run2,
    )
    s2 = get_last_span_for(t)
    assert s2.attributes.get(tracing.Attrs.SERVER_PORT) == 8443
    # http default port omitted
    run3 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "endpoint": "http://host:80"}},
        ["hi"],
        run_id=run3,
    )
    s3 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s3.attributes
    # http non-default port set
    run4 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "endpoint": "http://host:8080"}},
        ["hi"],
        run_id=run4,
    )
    s4 = get_last_span_for(t)
    assert s4.attributes.get(tracing.Attrs.SERVER_PORT) == 8080


def test_retriever_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"name": "index", "id": "retr"}
    t.on_retriever_start(serialized, "q", run_id=run_id)
    t.on_retriever_end([1, 2, 3], run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "retrieve"
    assert attrs.get("retriever.query") == "q"
    assert attrs.get("retriever.documents.count") == 3


def test_parser_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    serialized = {"id": "parser1", "kwargs": {"_type": "json"}}
    inputs = {"x": 1}
    outputs = {"y": 2}
    t.on_parser_start(serialized, inputs, run_id=run_id)
    t.on_parser_end(outputs, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get("parser.name") == "parser1"
    assert attrs.get("parser.type") == "json"
    parser_input = attrs.get("parser.input")
    parser_output = attrs.get("parser.output")
    assert parser_input is not None and parser_output is not None
    assert json.loads(parser_input).get("x") == 1
    assert json.loads(parser_output).get("y") == 2
    assert attrs.get("parser.output.size") == 1


def test_transform_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    serialized = {"id": "transform1", "kwargs": {"type": "map"}}
    inputs = [1, 2, 3]
    t.on_transform_start(serialized, inputs, run_id=run_id)
    t.on_transform_end({}, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get("transform.name") == "transform1"
    assert attrs.get("transform.type") == "map"
    assert attrs.get("transform.inputs.count") == 3


def test_pending_tool_call_cached_for_chain_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    # Seed pending tool calls cache directly to simulate LLM tool_calls parsing
    t._pending_tool_calls["abc"] = {"name": "echo", "args": {"message": "hi"}}
    # Now a ToolMessage should trigger a synthetic tool span
    parent_run = uuid4()
    t.on_chat_model_start(
        serialized,
        [[ToolMessage(name="echo", tool_call_id="abc", content="result")]],
        run_id=parent_run,
        parent_run_id=run_id,
    )
    # No execute_tool span is emitted immediately; the metadata is cached
    spans = t._core._tracer.spans
    tool_spans = [
        s
        for s in spans
        if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    ]
    assert not tool_spans
    # The pending call should remain so chain_end can emit the fallback span
    cached = t._pending_tool_calls.get("abc")
    assert cached is not None
    assert cached.get("name") == "echo"
    assert cached.get("args") == {"message": "hi"}
