import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

# Skip tests cleanly if required deps are not present
pytest.importorskip("azure.monitor.opentelemetry")
pytest.importorskip("opentelemetry")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, ToolCall
from langchain_core.outputs import ChatGeneration, LLMResult

import langchain_azure_ai.callbacks.tracers.inference_tracing as tracing


class MockSpan:
    def __init__(self, name: str):
        self.name = name
        self.attributes = {}
        self.events = []
        self.ended = False
        self.status = None
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))

    def set_status(self, status):
        self.status = status

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def end(self):
        self.ended = True


class MockTracer:
    def __init__(self):
        self.spans = []

    def start_span(self, name: str, kind=None, context=None):
        span = MockSpan(name)
        self.spans.append(span)
        return span


@pytest.fixture(autouse=True)
def patch_otel(monkeypatch):
    mock = SimpleNamespace(get_tracer=lambda _: MockTracer())
    monkeypatch.setattr(tracing, "otel_trace", mock)
    monkeypatch.setattr(tracing, "set_span_in_context", lambda span: None)


def get_last_span_for(tracer_obj):
    return tracer_obj._core._tracer.spans[-1]


def test_llm_start_attributes_content_recording_on(monkeypatch):
    # Ensure env enables content recording
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(include_legacy_keys=True)
    run_id = uuid4()
    serialized = {"kwargs": {"model": "gpt-4o", "endpoint": "http://host:8080"}}
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


def test_llm_start_attributes_content_recording_off(monkeypatch):
    monkeypatch.delenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", raising=False)
    t = tracing.AzureAIOpenTelemetryTracer(include_legacy_keys=False)
    run_id = uuid4()
    serialized = {"kwargs": {"model": "gpt-4o", "endpoint": "https://host"}}
    t.on_llm_start(serialized, ["hello"], run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    # No input messages recorded when disabled
    assert tracing.Attrs.INPUT_MESSAGES not in attrs


def test_redaction_on_chat_and_end(monkeypatch):
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
    # Verify output redacted
    out_json = json.loads(span.attributes[tracing.Attrs.OUTPUT_MESSAGES])
    assert out_json[0]["content"] == "[REDACTED]"


def test_usage_and_response_metadata():
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


def test_streaming_token_event(monkeypatch):
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


def test_llm_error_sets_status_and_exception(monkeypatch):
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    t.on_llm_start(serialized, ["hi"], run_id=run_id)
    err = RuntimeError("boom")
    t.on_llm_error(err, run_id=run_id)
    span = get_last_span_for(t)
    assert span.ended is True
    assert span.exceptions and isinstance(span.exceptions[0], RuntimeError)


def test_tool_start_end_records_args_and_result(monkeypatch):
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    parent_run = uuid4()
    serialized_tool = {"name": "search", "type": "function", "description": "desc"}
    inputs = {"id": "call-1", "query": "foo"}
    t.on_tool_start(serialized_tool, "ignored", inputs=inputs, run_id=run_id, parent_run_id=parent_run)
    t.on_tool_end({"answer": "bar"}, run_id=run_id, parent_run_id=parent_run)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    # Args and result only when content recording enabled
    assert json.loads(attrs.get(tracing.Attrs.TOOL_CALL_ARGS)).get("query") == "foo"
    assert json.loads(attrs.get(tracing.Attrs.TOOL_CALL_RESULT)).get("answer") == "bar"


def test_choice_count_only_when_n_not_one(monkeypatch):
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


def test_server_port_extraction_variants(monkeypatch):
    t = tracing.AzureAIOpenTelemetryTracer()
    # https default port not set
    run1 = uuid4()
    t.on_llm_start({"kwargs": {"model": "m", "endpoint": "https://host:443"}}, ["hi"], run_id=run1)
    s1 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s1.attributes
    # https non-default port set
    run2 = uuid4()
    t.on_llm_start({"kwargs": {"model": "m", "endpoint": "https://host:8443"}}, ["hi"], run_id=run2)
    s2 = get_last_span_for(t)
    assert s2.attributes.get(tracing.Attrs.SERVER_PORT) == 8443
    # http default port omitted
    run3 = uuid4()
    t.on_llm_start({"kwargs": {"model": "m", "endpoint": "http://host:80"}}, ["hi"], run_id=run3)
    s3 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s3.attributes
    # http non-default port set
    run4 = uuid4()
    t.on_llm_start({"kwargs": {"model": "m", "endpoint": "http://host:8080"}}, ["hi"], run_id=run4)
    s4 = get_last_span_for(t)
    assert s4.attributes.get(tracing.Attrs.SERVER_PORT) == 8080


def test_retriever_start_end(monkeypatch):
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


def test_parser_start_end(monkeypatch):
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
    assert json.loads(attrs.get("parser.input")).get("x") == 1
    assert json.loads(attrs.get("parser.output")).get("y") == 2
    assert attrs.get("parser.output.size") == 1


def test_transform_start_end(monkeypatch):
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


def test_synthetic_tool_span_from_tool_calls(monkeypatch):
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
    # Find the most recent execute_tool span
    spans = t._core._tracer.spans
    tool_spans = [s for s in spans if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"]
    assert tool_spans, "Expected a synthetic tool span to be emitted"
    attrs = tool_spans[-1].attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "echo"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "abc"
    # arguments and result recorded
    assert "message" in json.loads(attrs.get(tracing.Attrs.TOOL_CALL_ARGS))
    assert attrs.get(tracing.Attrs.TOOL_CALL_RESULT) == json.dumps("result")
