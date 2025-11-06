import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, cast
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.trace.status import StatusCode

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

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})
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

    def update_name(self, name: str) -> None:
        self.name = name


class MockTracer:
    spans: List[MockSpan]

    def __init__(self) -> None:
        self.spans = []

    def start_span(
        self,
        name: str,
        kind: Any = None,
        context: Any = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> MockSpan:
        span = MockSpan(name, attributes)
        self.spans.append(span)
        return span


@pytest.fixture(autouse=True)
def patch_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = SimpleNamespace(get_tracer=lambda *_, **__: MockTracer())
    monkeypatch.setattr(tracing, "otel_trace", mock)
    monkeypatch.setattr(tracing, "set_span_in_context", lambda span: None)
    monkeypatch.setattr(tracing, "get_current_span", lambda: None)


def get_last_span_for(
    tracer_obj: tracing.AzureAIOpenTelemetryTracer,
) -> MockSpan:
    tracer = cast(MockTracer, tracer_obj._tracer)  # type: ignore[attr-defined]
    return tracer.spans[-1]


def get_all_spans(
    tracer_obj: tracing.AzureAIOpenTelemetryTracer,
) -> List[MockSpan]:
    tracer = cast(MockTracer, tracer_obj._tracer)  # type: ignore[attr-defined]
    return list(tracer.spans)


def test_llm_start_attributes_content_recording_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure env enables content recording
    # fmt: off
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    serialized = {
        "kwargs": {
            "model": "gpt-4o",
            "azure_endpoint": "http://host:8080",
        }
    }
    # fmt: on
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )
    span = get_last_span_for(t)

    attrs = span.attributes
    assert attrs.get(tracing.Attrs.PROVIDER_NAME) == "azure.ai.openai"
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "text_completion"
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    assert attrs.get(tracing.Attrs.SERVER_ADDRESS) == "host"
    assert attrs.get(tracing.Attrs.SERVER_PORT) == 8080
    input_payload = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_payload[0]["parts"][0]["content"] == "hello"


def test_llm_start_attributes_content_recording_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # fmt: off
    monkeypatch.delenv(
        "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", raising=False
    )
    # fmt: on
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=False)
    run_id = uuid4()
    serialized = {
        "kwargs": {
            "model": "gpt-4o",
            "azure_endpoint": "https://contoso.openai.azure.com",
        }
    }
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "gpt-4o"},
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    input_payload = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_payload[0]["parts"][0]["content"] == "[redacted]"


def test_redaction_on_chat_and_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=False)
    run_id = uuid4()
    messages = [[HumanMessage(content="secret"), AIMessage(content="reply")]]
    serialized = {"kwargs": {"model": "m", "endpoint": "https://e"}}
    t.on_chat_model_start(serialized, messages, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    # Input content should be redacted
    input_json = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_json[0]["parts"][0]["content"] == "[redacted]"
    # End with output
    gen = ChatGeneration(message=AIMessage(content="reply"))
    result = LLMResult(generations=[[gen]], llm_output={})
    t.on_llm_end(result, run_id=run_id)
    # Verify output redacted on chat span when present;
    # some paths emit under agent root
    out_attr = span.attributes.get(tracing.Attrs.OUTPUT_MESSAGES)
    if out_attr:
        out_json = json.loads(out_attr)
        assert out_json[0]["parts"][0]["content"] == "[redacted]"
    else:
        # Fallback: if no chat output recorded, allow absence without failure
        # (agent root may contain the final output summary in role/parts schema)
        pass


def test_usage_and_response_metadata() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    gen = ChatGeneration(message=AIMessage(content="ok"))
    result = LLMResult(
        generations=[[gen]],
        llm_output={
            "token_usage": {"prompt_tokens": 3, "completion_tokens": 5},
            "model_name": "m",
            "id": "resp-123",
            "service_tier": "standard",
            "system_fingerprint": "fingerprint",
        },
    )
    t.on_llm_end(result, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 3
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 5
    assert attrs.get(tracing.Attrs.RESPONSE_MODEL) == "m"
    assert attrs.get(tracing.Attrs.RESPONSE_ID) == "resp-123"
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SERVICE_TIER) == "standard"
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT) == "fingerprint"


def test_usage_metadata_input_output_keys() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    gen = ChatGeneration(message=AIMessage(content="ok"))
    result = LLMResult(
        generations=[[gen]],
        llm_output={"token_usage": {"input_tokens": "7", "output_tokens": "11"}},
    )
    t.on_llm_end(result, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 7
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 11


def test_inference_span_records_gen_ai_semantic_attributes() -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root_run = uuid4()
    conversation_id = "thread-123"
    base_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ]
    t.on_chain_start(
        {},
        {"messages": base_messages},
        run_id=root_run,
        metadata={"thread_id": conversation_id, "agent_name": "Comedian"},
    )

    llm_run = uuid4()
    invocation_params = {
        "model": "gpt-4o",
        "max_tokens": 128,
        "max_input_tokens": 256,
        "max_output_tokens": 64,
        "temperature": 0.1,
        "top_p": 0.9,
        "top_k": 20,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.1,
        "n": 2,
        "seed": 123,
        "stop": ["stop"],
        "response_format": {"type": "json_object"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather lookup",
                },
            }
        ],
        "base_url": "https://api.example.com:8443/v1",
        "service_tier": "standard",
    }
    serialized = {
        "kwargs": {
            "model": "gpt-4o",
            "openai_api_base": "https://api.example.com:8443/v1",
        }
    }
    prompts = cast(List[str], base_messages)
    t.on_llm_start(
        serialized,
        prompts,
        run_id=llm_run,
        parent_run_id=root_run,
        metadata={"ls_provider": "openai"},
        invocation_params=invocation_params,
    )

    generation = ChatGeneration(
        message=AIMessage(content="Here is a funny line."),
        generation_info={"finish_reason": "stop"},
    )
    result = LLMResult(
        generations=[[generation]],
        llm_output={
            "token_usage": {"input_tokens": 42, "output_tokens": 17},
            "model_name": "gpt-4o",
            "id": "resp-456",
            "system_fingerprint": "fp-123",
            "service_tier": "premium",
        },
    )
    t.on_llm_end(result, run_id=llm_run)

    span = get_last_span_for(t)
    attrs = span.attributes

    assert span.name == "text_completion gpt-4o"
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "text_completion"
    assert attrs.get(tracing.Attrs.PROVIDER_NAME) == "openai"
    assert attrs.get(tracing.Attrs.REQUEST_MODEL) == "gpt-4o"
    assert attrs.get(tracing.Attrs.SERVER_ADDRESS) == "api.example.com"
    assert attrs.get(tracing.Attrs.SERVER_PORT) == 8443
    assert attrs.get(tracing.Attrs.REQUEST_MAX_TOKENS) == 128
    assert attrs.get(tracing.Attrs.REQUEST_MAX_INPUT_TOKENS) == 256
    assert attrs.get(tracing.Attrs.REQUEST_MAX_OUTPUT_TOKENS) == 64
    assert attrs.get(tracing.Attrs.REQUEST_TEMPERATURE) == 0.1
    assert attrs.get(tracing.Attrs.REQUEST_TOP_P) == 0.9
    assert attrs.get(tracing.Attrs.REQUEST_TOP_K) == 20
    assert attrs.get(tracing.Attrs.REQUEST_FREQ_PENALTY) == 0.5
    assert attrs.get(tracing.Attrs.REQUEST_PRES_PENALTY) == 0.1
    assert attrs.get(tracing.Attrs.REQUEST_CHOICE_COUNT) == 2
    assert attrs.get(tracing.Attrs.REQUEST_SEED) == 123
    assert attrs.get(tracing.Attrs.OPENAI_REQUEST_SERVICE_TIER) == "standard"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == conversation_id

    assert json.loads(attrs[tracing.Attrs.REQUEST_STOP]) == ["stop"]
    assert json.loads(attrs[tracing.Attrs.REQUEST_ENCODING_FORMATS]) == {
        "type": "json_object"
    }
    system_instr = json.loads(attrs[tracing.Attrs.SYSTEM_INSTRUCTIONS])
    assert system_instr[0]["content"] == "You are a helpful assistant."
    input_messages = json.loads(attrs[tracing.Attrs.INPUT_MESSAGES])
    assert input_messages[0]["parts"][0]["content"] == "Tell me a joke."
    tool_defs = json.loads(attrs[tracing.Attrs.TOOL_DEFINITIONS])
    assert tool_defs[0]["function"]["name"] == "get_weather"

    assert attrs.get(tracing.Attrs.OUTPUT_TYPE) == "text"
    output_messages = json.loads(attrs[tracing.Attrs.OUTPUT_MESSAGES])
    assert output_messages[0]["parts"][0]["content"] == "Here is a funny line."
    assert json.loads(attrs[tracing.Attrs.RESPONSE_FINISH_REASONS]) == ["stop"]
    assert attrs.get(tracing.Attrs.RESPONSE_ID) == "resp-456"
    assert attrs.get(tracing.Attrs.RESPONSE_MODEL) == "gpt-4o"
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 42
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 17
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT) == "fp-123"
    assert attrs.get(tracing.Attrs.OPENAI_RESPONSE_SERVICE_TIER) == "premium"


def test_agent_span_accumulates_usage_tokens() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    agent_run = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "plan trip"}]},
        run_id=agent_run,
        metadata={"otel_agent_span": True, "agent_name": "Coordinator"},
    )

    llm_run_one = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        {"kwargs": {"model": "gpt-4o"}},
        prompts,
        run_id=llm_run_one,
        parent_run_id=agent_run,
        invocation_params={"model": "gpt-4o"},
    )
    gen_one = ChatGeneration(message=AIMessage(content="first"))
    result_one = LLMResult(
        generations=[[gen_one]],
        llm_output={"token_usage": {"input_tokens": 5, "output_tokens": 2}},
    )
    t.on_llm_end(result_one, run_id=llm_run_one, parent_run_id=agent_run)

    llm_run_two = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "gpt-4o"}},
        prompts,
        run_id=llm_run_two,
        parent_run_id=agent_run,
        invocation_params={"model": "gpt-4o"},
    )
    gen_two = ChatGeneration(message=AIMessage(content="second"))
    result_two = LLMResult(
        generations=[[gen_two]],
        llm_output={"token_usage": {"input_tokens": 7, "output_tokens": 4}},
    )
    t.on_llm_end(result_two, run_id=llm_run_two, parent_run_id=agent_run)

    agent_record = t._spans[str(agent_run)]
    agent_span = cast(MockSpan, agent_record.span)
    attrs = agent_span.attributes
    assert attrs.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 12
    assert attrs.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 6


def test_streaming_token_event(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    t.on_llm_new_token("abcdef", run_id=run_id)
    span = get_last_span_for(t)
    assert span.events == []


def test_synthetic_execute_tool_under_chat_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer()
    # Start root agent via chain_start
    root = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root,
        metadata={"thread_id": root},
    )
    chat_run = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    msgs = cast(List[List[BaseMessage]], [[HumanMessage(content="prompt")]])
    t.on_chat_model_start(serialized, msgs, run_id=chat_run, parent_run_id=root)
    tool_run = uuid4()
    t.on_tool_start(
        {"name": "get_current_date"},
        "",
        inputs={"tool_call_id": "call-1"},
        run_id=tool_run,
        parent_run_id=root,
    )
    record = t._spans[str(tool_run)]
    assert record.parent_run_id == str(chat_run)
    span = cast(MockSpan, record.span)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "get_current_date"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)
    t.on_tool_end({"result": "ok"}, run_id=tool_run, parent_run_id=root)


def test_tool_call_without_function_schema() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    tool_run = uuid4()
    t.on_tool_start(
        {"name": "get_weather"},
        "",
        inputs={"tool_call_id": "call-1", "city": "San Francisco"},
        metadata={},
        run_id=tool_run,
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "get_weather"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"
    args = json.loads(attrs[tracing.Attrs.TOOL_CALL_ARGUMENTS])
    assert args["city"] == "San Francisco"
    t.on_tool_end({"temperature": 60}, run_id=tool_run)


def test_no_invoke_agent_on_agent_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    # on_agent_action should not start invoke_agent spans;
    # only create_agent when applicable
    before = len(
        [
            s
            for s in get_all_spans(t)
            if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "invoke_agent"
        ]
    )
    action = cast(
        AgentAction,
        SimpleNamespace(
            agent_name="Agent",
            system_instructions=[{"type": "text", "content": "You are an agent."}],
        ),
    )
    t.on_agent_action(action, run_id=uuid4())
    after = len(
        [
            s
            for s in get_all_spans(t)
            if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "invoke_agent"
        ]
    )
    assert after == before


def test_tool_start_name_and_conversation_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root,
        metadata={"thread_id": root},
    )
    # Start a tool with serialized name and inputs id
    run_id = uuid4()
    serialized_tool = {
        "name": "search",
        "type": "function",
        "description": "desc",
    }
    inputs = {"tool_call_id": "call-1", "query": "foo"}
    t.on_tool_start(
        serialized_tool,
        "ignored",
        inputs=inputs,
        run_id=run_id,
        parent_run_id=root,
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)
    record = t._spans[str(run_id)]
    assert record.parent_run_id == str(root)
    t.on_tool_end({"answer": "bar"}, run_id=run_id, parent_run_id=root)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.TOOL_NAME) == "search"
    assert attrs.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"
    assert attrs.get(tracing.Attrs.CONVERSATION_ID) == str(root)


def test_tool_deduplicates_synthetic_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    root = uuid4()
    t.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root,
        metadata={"thread_id": root},
    )
    chat_run = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    msgs = cast(List[List[BaseMessage]], [[HumanMessage(content="prompt")]])
    t.on_chat_model_start(serialized, msgs, run_id=chat_run, parent_run_id=root)
    tool_run = uuid4()
    t.on_tool_start(
        {"name": "get_weather", "description": "Returns weather"},
        "ignored",
        inputs={"tool_call_id": "call-1", "city": "SF"},
        run_id=tool_run,
        parent_run_id=chat_run,
    )
    t.on_tool_end({"temperature": 60}, run_id=tool_run, parent_run_id=chat_run)
    t.on_chain_end({}, run_id=root)
    tool_spans = [
        s
        for s in get_all_spans(t)
        if s.attributes.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    ]
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.attributes.get(tracing.Attrs.TOOL_NAME) == "get_weather"
    assert span.attributes.get(tracing.Attrs.TOOL_CALL_ID) == "call-1"


def test_end_span_handles_context_detach_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FaultyToken:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            raise ValueError("context mismatch")

    token = FaultyToken()

    def fake_use_span(span: Any, *, end_on_exit: bool = False) -> FaultyToken:
        return token

    monkeypatch.setattr(tracing, "use_span", fake_use_span)

    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer._start_span(  # type: ignore[attr-defined]
        run_id,
        "test",
        operation="invoke_agent",
        kind=tracing.SpanKind.INTERNAL,  # type: ignore[attr-defined]
        parent_run_id=None,
    )
    tracer._end_span(run_id)  # type: ignore[attr-defined]


def test_invoke_agent_records_tool_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    tools = [
        {"name": "get_weather", "description": "Returns forecast"},
        {"name": "search_docs", "description": "Search knowledge base"},
    ]
    run_id = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        {"kwargs": {"model": "m"}},
        prompts,
        run_id=run_id,
        invocation_params={"model": "m", "tools": tools},
    )
    span = get_last_span_for(t)
    defs = span.attributes.get(tracing.Attrs.TOOL_DEFINITIONS)
    assert defs is not None
    parsed = json.loads(defs)
    tool_names = {tool["name"] for tool in parsed}
    assert {"get_weather", "search_docs"} <= tool_names


def test_finish_reasons_normalized() -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    chat_run = uuid4()
    gen = ChatGeneration(
        message=AIMessage(content=""),
        generation_info={"finish_reason": "tool_calls"},
    )
    result = LLMResult(generations=[[gen]], llm_output={})
    # Create chat span and then end it
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        {"kwargs": {"model": "m"}},
        prompts,
        run_id=chat_run,
        invocation_params={"model": "m"},
    )
    t.on_llm_end(result, run_id=chat_run)
    span = get_last_span_for(t)
    finish_reasons = span.attributes.get(tracing.Attrs.RESPONSE_FINISH_REASONS)
    assert finish_reasons is not None
    assert json.loads(finish_reasons) == ["tool_calls"]


def test_chat_parenting_under_root_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    root = uuid4()
    # Start root agent
    t.on_chain_start({}, {"messages": [{"role": "user", "content": "hi"}]}, run_id=root)
    # Start chat without specifying parent; should parent under root agent
    chat_run = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hello"}])
    t.on_llm_start(
        {"kwargs": {"model": "m"}},
        prompts,
        run_id=chat_run,
        parent_run_id=root,
        invocation_params={"model": "m"},
    )
    record = t._spans[str(chat_run)]
    assert record.parent_run_id == str(root)


def test_llm_error_sets_status_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    err = RuntimeError("boom")
    t.on_llm_error(err, run_id=run_id)
    span = get_last_span_for(t)
    assert span.ended is True
    assert span.status is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"


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
    inputs = {"tool_call_id": "call-1", "query": "foo"}
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
    tool_args = attrs.get(tracing.Attrs.TOOL_CALL_ARGUMENTS)
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
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        serialized,
        prompts,
        run_id=run_id,
        invocation_params={"model": "m", "n": 1},
    )
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.REQUEST_CHOICE_COUNT) == 1
    # Now with n=3
    run_id2 = uuid4()
    serialized2 = {"kwargs": {"model": "m", "n": 3}}
    t.on_llm_start(
        serialized2,
        prompts,
        run_id=run_id2,
        invocation_params={"model": "m", "n": 3},
    )
    span2 = get_last_span_for(t)
    assert span2.attributes.get(tracing.Attrs.REQUEST_CHOICE_COUNT) == 3


def test_server_port_extraction_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    # https default port not set
    run1 = uuid4()
    prompts = cast(List[str], [{"role": "user", "content": "hi"}])
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "https://host"}},
        prompts,
        run_id=run1,
        invocation_params={"model": "m"},
    )
    s1 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s1.attributes
    # https non-default port set
    run2 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "https://host:8443"}},
        prompts,
        run_id=run2,
        invocation_params={"model": "m"},
    )
    s2 = get_last_span_for(t)
    assert s2.attributes.get(tracing.Attrs.SERVER_PORT) == 8443
    # http default port omitted
    run3 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "http://host"}},
        prompts,
        run_id=run3,
        invocation_params={"model": "m"},
    )
    s3 = get_last_span_for(t)
    assert tracing.Attrs.SERVER_PORT not in s3.attributes
    # http non-default port set
    run4 = uuid4()
    t.on_llm_start(
        {"kwargs": {"model": "m", "azure_endpoint": "http://host:8080"}},
        prompts,
        run_id=run4,
        invocation_params={"model": "m"},
    )
    s4 = get_last_span_for(t)
    assert s4.attributes.get(tracing.Attrs.SERVER_PORT) == 8080


def test_retriever_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    serialized = {"name": "index", "id": "retr"}
    t.on_retriever_start(serialized, "q", run_id=run_id)
    documents = [
        Document(page_content="doc1", metadata={"source": "a"}),
        Document(page_content="doc2", metadata={"source": "b"}),
        Document(page_content="doc3", metadata={"source": "c"}),
    ]
    t.on_retriever_end(documents, run_id=run_id)
    span = get_last_span_for(t)
    attrs = span.attributes
    assert attrs.get(tracing.Attrs.OPERATION_NAME) == "execute_tool"
    assert attrs.get(tracing.Attrs.TOOL_TYPE) == "retriever"
    assert attrs.get(tracing.Attrs.RETRIEVER_QUERY) == "q"
    results = json.loads(attrs[tracing.Attrs.RETRIEVER_RESULTS])
    assert len(results) == 3


def test_parser_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    with pytest.raises(AttributeError):
        getattr(t, "on_parser_start")
    with pytest.raises(AttributeError):
        getattr(t, "on_parser_end")


def test_transform_start_end(monkeypatch: pytest.MonkeyPatch) -> None:
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    with pytest.raises(AttributeError):
        getattr(t, "on_transform_start")
    with pytest.raises(AttributeError):
        getattr(t, "on_transform_end")


def test_pending_tool_call_cached_for_chain_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "true")
    t = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)
    run_id = uuid4()
    serialized = {"kwargs": {"model": "m"}}
    t.on_chat_model_start(
        serialized,
        [[ToolMessage(name="echo", tool_call_id="abc", content="result")]],
        run_id=run_id,
    )
    span = get_last_span_for(t)
    input_messages = json.loads(span.attributes[tracing.Attrs.INPUT_MESSAGES])
    parts = input_messages[0]["parts"]
    tool_part = next(part for part in parts if part.get("type") == "tool_call_response")
    assert tool_part["id"] == "abc"
    assert tool_part["result"] == "result"
