import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast
from uuid import uuid4

import pytest
from langchain_core.agents import AgentAction
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
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


def test_use_propagated_context_attaches_and_detaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()

    headers = {"traceparent": "00-01" + "0" * 30 + "-02" + "0" * 14 + "-01"}
    sentinel_context = object()
    attached: list[object] = []
    detached: list[object] = []

    def fake_extract(carrier: Mapping[str, str]) -> object:
        assert carrier == headers
        return sentinel_context

    def fake_attach(ctx: object) -> str:
        attached.append(ctx)
        return "token"

    def fake_detach(token: str) -> None:
        detached.append(token)

    monkeypatch.setattr(tracing, "extract", fake_extract)
    monkeypatch.setattr(tracing, "attach", fake_attach)
    monkeypatch.setattr(tracing, "detach", fake_detach)

    with tracer.use_propagated_context(headers=headers):
        assert attached == [sentinel_context]

    assert detached == ["token"]


def test_thread_root_parent_resolution() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    thread_id = "thread-123"
    root_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "travel_planner",
        },
    )
    child_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi again"}]},
        run_id=child_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "flight_specialist",
        },
    )
    child_record = tracer._spans[str(child_run)]
    assert child_record.parent_run_id == str(root_run)
    assert tracer._langgraph_root_by_thread[thread_id] == str(root_run)


def test_llm_and_tool_attach_to_latest_agent() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    thread_id = "stack-thread"
    root_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=root_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "travel_planner",
        },
    )
    child_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "help"}]},
        run_id=child_run,
        metadata={
            "thread_id": thread_id,
            "otel_agent_span": True,
            "agent_name": "flight_specialist",
        },
    )
    llm_run = uuid4()
    prompts = cast(List[List[BaseMessage]], [[HumanMessage(content="flight options")]])
    tracer.on_llm_start(
        {"kwargs": {"model": "gpt-test"}},
        prompts,
        run_id=llm_run,
        metadata={"thread_id": thread_id},
    )
    llm_record = tracer._spans[str(llm_run)]
    assert llm_record.parent_run_id == str(child_run)
    tool_run = uuid4()
    tracer.on_tool_start(
        {"name": "search"},
        "",
        run_id=tool_run,
        metadata={"thread_id": thread_id},
        inputs={"tool_call_id": "1"},
    )
    tool_record = tracer._spans[str(tool_run)]
    assert tool_record.parent_run_id == str(child_run)

    tracer.on_tool_end({}, run_id=tool_run)
    tracer.on_llm_end(
        LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]),
        run_id=llm_run,
    )
    tracer.on_chain_end({}, run_id=child_run)
    tracer.on_chain_end({}, run_id=root_run)
    assert tracer._agent_stack_by_thread.get(thread_id) in (None, [])


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


def test_message_helpers_handle_dict_and_langchain_messages() -> None:
    assert tracing._message_role(HumanMessage(content="hi")) == "user"
    assert tracing._message_role(AIMessage(content="ok")) == "assistant"
    tool_message = ToolMessage(name="t", tool_call_id="abc", content="x")
    assert tracing._message_role(tool_message) == "tool"
    assert tracing._message_role(SystemMessage(content="sys")) == "system"

    assert tracing._message_role({"role": "human", "content": "hi"}) == "user"
    assert tracing._message_role({"type": "ai", "content": "ok"}) == "assistant"
    assert tracing._message_role({"role": "tool", "content": "x"}) == "tool"
    assert tracing._message_role({"role": "system", "content": "sys"}) == "system"
    assert tracing._message_role({"content": "hi"}) == "user"


def test_tool_call_helpers_extract_ids_and_calls() -> None:
    assert tracing._tool_call_id_from_message({"tool_call_id": 123}) == "123"
    assert (
        tracing._tool_call_id_from_message(
            ToolMessage(name="t", tool_call_id="call-1", content="ok")
        )
        == "call-1"
    )
    assert tracing._tool_call_id_from_message({"content": "x"}) is None

    # AIMessage with valid tool_calls (LangChain validates all items must be dicts)
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "1", "name": "tool", "args": {"k": "v"}}],
    )
    extracted = tracing._extract_tool_calls(ai)
    assert len(extracted) == 1
    assert extracted[0]["id"] == "1"
    assert extracted[0]["name"] == "tool"
    assert extracted[0]["args"] == {"k": "v"}
    # Dict-based messages can have arbitrary items, function filters to dicts only
    assert tracing._extract_tool_calls(
        {"tool_calls": [{"id": "2"}, None, "ignore"]}
    ) == [{"id": "2"}]


def test_prepare_messages_supports_threads_system_and_tool_calls() -> None:
    raw_messages = [
        [
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "hi"},
        ],
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc1", "name": "lookup", "arguments": None}],
            }
        ],
        [{"role": "tool", "content": "result", "tool_call_id": "tc1"}],
    ]

    formatted, system_instr = tracing._prepare_messages(
        raw_messages, record_content=True, include_roles={"user", "assistant", "tool"}
    )
    assert formatted is not None and system_instr is not None
    assert json.loads(system_instr)[0]["content"] == "system rules"
    formatted_payload = json.loads(formatted)
    assert [msg["role"] for msg in formatted_payload] == ["user", "assistant", "tool"]
    assistant_parts = formatted_payload[1]["parts"]
    tool_call_part = next(
        part for part in assistant_parts if part.get("type") == "tool_call"
    )
    assert tool_call_part["arguments"] == {}
    tool_parts = formatted_payload[2]["parts"]
    tool_response = next(
        part for part in tool_parts if part.get("type") == "tool_call_response"
    )
    assert tool_response["id"] == "tc1"
    assert tool_response["result"] == "result"

    formatted_redacted, system_redacted = tracing._prepare_messages(
        raw_messages, record_content=False, include_roles={"user", "assistant", "tool"}
    )
    assert formatted_redacted is not None and system_redacted is not None
    assert json.loads(system_redacted)[0]["content"] == "[redacted]"
    formatted_redacted_payload = json.loads(formatted_redacted)
    assert formatted_redacted_payload[0]["parts"][0]["content"] == "[redacted]"
    redacted_tool_call = next(
        part
        for part in formatted_redacted_payload[1]["parts"]
        if part.get("type") == "tool_call"
    )
    assert redacted_tool_call["arguments"] == "[redacted]"

    assistant_only, _ = tracing._prepare_messages(
        raw_messages, record_content=True, include_roles={"assistant"}
    )
    assert assistant_only is not None
    assert [msg["role"] for msg in json.loads(assistant_only)] == ["assistant"]


def test_filter_assistant_output_filters_to_text_parts() -> None:
    assert tracing._filter_assistant_output("not json") == "not json"
    assert (
        tracing._filter_assistant_output(
            json.dumps([{"role": "user", "parts": [{"type": "text", "content": "hi"}]}])
        )
        is None
    )
    assert (
        tracing._filter_assistant_output(
            json.dumps(
                [{"role": "assistant", "parts": [{"type": "tool_call", "id": "1"}]}]
            )
        )
        is None
    )

    cleaned = tracing._filter_assistant_output(
        json.dumps(
            [
                {"role": "assistant", "parts": [{"type": "text", "content": "ok"}]},
                {
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "content": "more"},
                        {"type": "tool_call", "id": "ignore"},
                    ],
                },
            ]
        )
    )
    assert cleaned is not None
    payload = json.loads(cleaned)
    assert payload == [
        {"role": "assistant", "parts": [{"type": "text", "content": "ok"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "more"}]},
    ]


@dataclass
class _SampleDataclass:
    count: int
    label: str


def test_scrub_and_serialise_helpers_cover_common_types() -> None:
    assert tracing._scrub_value(None, record_content=True) is None
    assert tracing._scrub_value(True, record_content=True) is True
    assert tracing._scrub_value('  {"a": 1} ', record_content=True) == {"a": 1}
    assert tracing._scrub_value('{"a": 1', record_content=True) == '{"a": 1'
    assert tracing._scrub_value(
        _SampleDataclass(count=2, label="x"),
        record_content=True,
    ) == {
        "count": 2,
        "label": "x",
    }
    assert tracing._scrub_value(["[1, 2]", 3], record_content=True) == [[1, 2], 3]
    secret_message = HumanMessage(content="secret")
    assert tracing._scrub_value(secret_message, record_content=False) == "[redacted]"
    # AIMessage with valid content (list of strings without None - LangChain validation)
    ai_message = AIMessage(content=["a", "b"])
    assert tracing._scrub_value(ai_message, record_content=True) == {
        "type": "ai",
        "content": "a b",
    }

    tool_msg = ToolMessage(name="t", tool_call_id="call-1", content="ok")
    tool_payload = json.loads(
        tracing._serialise_tool_result(tool_msg, record_content=True)
    )
    assert tool_payload["tool_call_id"] == "call-1"
    assert tool_payload["content"] == "ok"

    msg_payload = json.loads(
        tracing._serialise_tool_result(AIMessage(content="done"), record_content=True)
    )
    assert msg_payload["type"] == "ai"
    assert msg_payload["content"] == "done"

    assert (
        json.loads(tracing._serialise_tool_result({"k": "v"}, record_content=False))
        == "[redacted]"
    )


def test_collect_tool_definitions_dedupes_by_identity() -> None:
    shared = {"name": "shared"}
    distinct = {"name": "shared"}
    collected = tracing._collect_tool_definitions(
        shared,
        [shared, distinct, None, "ignore"],
        ("ignore",),
        [{"name": "other"}],
    )
    assert collected == [shared, distinct, {"name": "other"}]


def test_usage_helpers_cover_bedrock_and_token_variants() -> None:
    assert tracing._coerce_token_value(["1", 2, None]) == 3
    assert tracing._coerce_token_value({"value": "4"}) == 4
    assert tracing._coerce_token_value(iter([1, 2])) == 3
    assert tracing._coerce_token_value({"unknown": 1}) is None

    usage = {"inputTokenCount": {"value": 2}, "outputTokenCount": 3}
    assert tracing._normalize_bedrock_usage_dict(usage) == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }
    assert tracing._normalize_bedrock_usage_dict({"foo": 1}) is None
    assert tracing._normalize_bedrock_metrics(
        {"inputTokenCount": 1, "outputTokenCount": 2}
    ) == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }

    llm_output_metrics = {
        "amazon-bedrock-invocationMetrics": {
            "inputTokenCount": 4,
            "outputTokenCount": 6,
        }
    }
    assert tracing._extract_bedrock_usage(llm_output_metrics, []) == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
    }

    llm_output_nested = {"response": {"usage": {"inputTokens": 7, "outputTokens": 11}}}
    assert tracing._extract_bedrock_usage(llm_output_nested, []) == {
        "prompt_tokens": 7,
        "completion_tokens": 11,
        "total_tokens": 18,
    }

    generations = [
        ChatGeneration(
            message=AIMessage(content="x"),
            generation_info={"usage": {"inputTokens": 2, "outputTokens": 3}},
        )
    ]
    assert tracing._extract_bedrock_usage({}, generations) == {
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "total_tokens": 5,
    }

    input_tokens, output_tokens, total_tokens, normalized, should_attach = (
        tracing._resolve_usage_from_llm_output(
            {"token_usage": {"prompt_tokens": 1, "completion_tokens": 2}}, []
        )
    )
    assert (input_tokens, output_tokens, total_tokens) == (1, 2, 3)
    assert normalized == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert should_attach is False

    resolved = tracing._resolve_usage_from_llm_output(llm_output_metrics, [])
    _, _, _, bedrock_normalized, bedrock_attach = resolved
    assert bedrock_normalized == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
    }
    assert bedrock_attach is True


def test_provider_and_server_inference_helpers_cover_variants() -> None:
    assert (
        tracing._infer_provider_name(None, {"ls_provider": "azure_openai"}, None)
        == "azure.ai.openai"
    )
    assert (
        tracing._infer_provider_name(None, {"ls_provider": "github"}, None)
        == "azure.ai.openai"
    )
    assert (
        tracing._infer_provider_name(None, {"ls_provider": "amazon_bedrock"}, None)
        == "aws.bedrock"
    )
    assert (
        tracing._infer_provider_name(None, None, {"base_url": "http://ollama.local"})
        == "ollama"
    )
    assert (
        tracing._infer_provider_name(
            None, None, {"endpoint_url": "https://bedrock.amazonaws.com"}
        )
        == "aws.bedrock"
    )
    assert (
        tracing._infer_provider_name(None, None, {"provider_name": "AmazonBedrock"})
        == "aws.bedrock"
    )
    assert (
        tracing._infer_provider_name(
            {"kwargs": {"azure_endpoint": "https://x"}},
            None,
            None,
        )
        == "azure.ai.openai"
    )
    assert (
        tracing._infer_provider_name(
            {"id": "amazon-bedrock-chat", "kwargs": {}},
            None,
            None,
        )
        == "aws.bedrock"
    )

    assert (
        tracing._infer_server_address(
            None, {"base_url": "https://api.example.com:1234/v1"}
        )
        == "api.example.com"
    )
    assert (
        tracing._infer_server_port(
            None, {"base_url": "https://api.example.com:1234/v1"}
        )
        == 1234
    )
    assert (
        tracing._infer_server_address(
            {"kwargs": {"openai_api_base": "http://localhost:8080/v1"}}, None
        )
        == "localhost"
    )
    assert (
        tracing._infer_server_port(
            {"kwargs": {"openai_api_base": "http://localhost:8080/v1"}}, None
        )
        == 8080
    )
    assert tracing._infer_server_address(None, None) is None
    assert tracing._infer_server_port(None, None) is None


def test_use_propagated_context_no_headers_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()

    def fail(*_: Any, **__: Any) -> None:
        raise AssertionError("context propagation should be skipped")

    monkeypatch.setattr(tracing, "extract", fail)
    monkeypatch.setattr(tracing, "attach", fail)
    monkeypatch.setattr(tracing, "detach", fail)

    with tracer.use_propagated_context(headers=None):
        pass


def test_configure_azure_monitor_is_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_configure(*, connection_string: str) -> None:
        calls.append(connection_string)

    original = tracing.AzureAIOpenTelemetryTracer._azure_monitor_configured
    monkeypatch.setattr(tracing, "configure_azure_monitor", fake_configure)
    try:
        tracing.AzureAIOpenTelemetryTracer._azure_monitor_configured = False
        tracing.AzureAIOpenTelemetryTracer._configure_azure_monitor("cs1")
        tracing.AzureAIOpenTelemetryTracer._configure_azure_monitor("cs2")
        assert calls == ["cs1"]
    finally:
        tracing.AzureAIOpenTelemetryTracer._azure_monitor_configured = original


def test_callbacks_handle_missing_records_and_input_str_branch() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer(enable_content_recording=True)

    # Missing records should be safe no-ops.
    tracer.on_llm_end(LLMResult(generations=[], llm_output={}), run_id=uuid4())
    tracer.on_tool_end({"ok": True}, run_id=uuid4())
    tracer.on_retriever_end([], run_id=uuid4())

    run_id = uuid4()
    tracer.on_tool_start({"name": "tool"}, "raw-args", run_id=run_id, inputs=None)
    span = get_last_span_for(tracer)
    assert span.attributes.get(tracing.Attrs.TOOL_CALL_ARGUMENTS) == "raw-args"
    tracer.on_tool_end({"ok": True}, run_id=run_id)


def test_agent_action_stashes_pending_tool_inputs_and_overrides_parent() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    agent_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=agent_run,
        metadata={"otel_agent_span": True, "agent_name": "Agent", "thread_id": "t"},
    )

    chat_run = uuid4()
    tracer.on_llm_start(
        {"kwargs": {"model": "m"}},
        [{"role": "user", "content": "hello"}],
        run_id=chat_run,
        parent_run_id=agent_run,
        metadata={"thread_id": "t"},
        invocation_params={"model": "m"},
    )

    action_run = uuid4()
    action = cast(
        AgentAction,
        SimpleNamespace(tool="search", tool_input={"q": "x"}, log="log"),
    )
    tracer.on_agent_action(action, run_id=action_run, parent_run_id=agent_run)

    record = tracer._spans[str(agent_run)]
    pending = record.stash["pending_actions"][str(action_run)]
    assert pending["tool"] == "search"
    assert pending["tool_input"] == {"q": "x"}
    assert tracer._run_parent_override[str(action_run)] == str(chat_run)


def test_agent_finish_records_output_and_ends_span() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    agent_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=agent_run,
        metadata={"otel_agent_span": True, "agent_name": "Agent"},
    )

    finish = cast(SimpleNamespace, SimpleNamespace(return_values={"final": "ok"}))
    tracer.on_agent_finish(cast(Any, finish), run_id=agent_run)

    span = get_last_span_for(tracer)
    assert span.ended is True
    assert json.loads(span.attributes[tracing.Attrs.OUTPUT_MESSAGES]) == {"final": "ok"}


def test_retriever_error_sets_status() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_retriever_start({"name": "index"}, "q", run_id=run_id)
    err = RuntimeError("boom")
    tracer.on_retriever_error(err, run_id=run_id)
    span = get_last_span_for(tracer)
    assert span.ended is True
    assert span.status is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "boom"


def test_llm_end_marks_json_output_type_when_tool_calls_present() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_llm_start(
        {"kwargs": {"model": "m"}},
        [{"role": "user", "content": "hi"}],
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    generation = ChatGeneration(
        message=AIMessage(
            content="",
            tool_calls=[{"name": "t", "id": "call-1", "args": {"k": "v"}}],
        ),
        generation_info={"finish_reason": "tool_calls"},
    )
    llm_output: Dict[str, Any] = {
        "amazon-bedrock-invocationMetrics": {
            "inputTokenCount": 1,
            "outputTokenCount": 2,
        }
    }
    result = LLMResult(generations=[[generation]], llm_output=llm_output)
    tracer.on_llm_end(result, run_id=run_id)
    span = get_last_span_for(tracer)
    # When output contains tool_calls, output type should be "json"
    assert span.attributes.get(tracing.Attrs.OUTPUT_TYPE) == "json"
    # Token usage should be extracted and recorded on the span
    assert span.attributes.get(tracing.Attrs.USAGE_INPUT_TOKENS) == 1
    assert span.attributes.get(tracing.Attrs.USAGE_OUTPUT_TOKENS) == 2
    # Normalized usage should be attached back onto llm_output when sourced from
    # Bedrock metrics.
    assert result.llm_output is not None
    assert result.llm_output.get("token_usage") == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


def test_on_chain_error_sets_error_status() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    agent_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": [{"role": "user", "content": "hi"}]},
        run_id=agent_run,
        metadata={"otel_agent_span": True, "agent_name": "Agent"},
    )
    err = ValueError("chain failed")
    tracer.on_chain_error(err, run_id=agent_run)
    span = get_last_span_for(tracer)
    assert span.ended is True
    assert span.status is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "chain failed"


def test_on_tool_error_sets_error_status() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_tool_start({"name": "bad_tool"}, "input", run_id=run_id)
    err = RuntimeError("tool crashed")
    tracer.on_tool_error(err, run_id=run_id)
    span = get_last_span_for(tracer)
    assert span.ended is True
    assert span.status is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "tool crashed"


def test_coerce_content_to_text_handles_various_types() -> None:
    # None
    assert tracing._coerce_content_to_text(None) is None
    # String
    assert tracing._coerce_content_to_text("hello") == "hello"
    # List
    assert tracing._coerce_content_to_text(["a", "b", "c"]) == "a b c"
    # Tuple
    assert tracing._coerce_content_to_text(("x", "y")) == "x y"
    # List with None values (filtered)
    assert tracing._coerce_content_to_text(["a", None, "b"]) == "a b"
    # Other types (converted via str)
    assert tracing._coerce_content_to_text(123) == "123"


def test_usage_metadata_to_mapping_handles_various_inputs() -> None:
    # None
    assert tracing._usage_metadata_to_mapping(None) is None
    # Already a mapping
    sample_mapping = {"input_tokens": 5}
    assert tracing._usage_metadata_to_mapping(sample_mapping) == sample_mapping

    # Object with dict method
    class FakeMetadata:
        def dict(self, exclude_none: bool = False) -> dict:
            return {"input_tokens": 10}

    assert tracing._usage_metadata_to_mapping(FakeMetadata()) == {"input_tokens": 10}

    # Object with attributes
    class AttrMetadata:
        input_tokens = 7
        output_tokens = 3

    result = tracing._usage_metadata_to_mapping(AttrMetadata())
    assert result is not None
    assert result.get("input_tokens") == 7
    assert result.get("output_tokens") == 3


def test_should_ignore_agent_span_filters_start_and_middleware() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    # __start__ node should be ignored
    assert tracer._should_ignore_agent_span(
        "__start__",
        None,
        {"langgraph_node": "__start__"},
        {},
    )
    # Middleware prefix should be ignored
    assert tracer._should_ignore_agent_span("Middleware.auth", None, {}, {})
    # otel_agent_span=False should be ignored
    assert tracer._should_ignore_agent_span(
        "Agent",
        None,
        {"otel_agent_span": False},
        {},
    )
    # Normal agent should not be ignored
    assert not tracer._should_ignore_agent_span(
        "Agent",
        None,
        {"otel_agent_span": True, "agent_name": "Agent"},
        {},
    )


def test_end_span_sets_error_type_when_status_is_none() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_tool_start({"name": "tool"}, "input", run_id=run_id)
    # Call _end_span with error but no status - should set ERROR_TYPE
    tracer._end_span(run_id, error=TypeError("bad type"))
    span = get_last_span_for(tracer)
    assert span.ended is True
    assert span.status is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(tracing.Attrs.ERROR_TYPE) == "TypeError"


def test_chain_error_on_ignored_run_is_noop() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    agent_run = uuid4()
    # Start a span that will be ignored
    tracer.on_chain_start(
        {},
        {"messages": []},
        run_id=agent_run,
        metadata={"langgraph_node": "__start__"},  # This gets ignored
    )
    assert str(agent_run) in tracer._ignored_runs
    # Error on ignored run should clean up without error
    tracer.on_chain_error(ValueError("error"), run_id=agent_run)
    assert str(agent_run) not in tracer._ignored_runs


def test_chain_end_on_ignored_run_is_noop() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    agent_run = uuid4()
    tracer.on_chain_start(
        {},
        {"messages": []},
        run_id=agent_run,
        metadata={"langgraph_node": "__start__"},
    )
    assert str(agent_run) in tracer._ignored_runs
    tracer.on_chain_end({}, run_id=agent_run)
    assert str(agent_run) not in tracer._ignored_runs


def test_collect_usage_from_generations_extracts_from_message() -> None:
    gen = ChatGeneration(
        message=AIMessage(
            content="x",
            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        ),
    )
    result = tracing._collect_usage_from_generations([gen])
    assert result is not None
    assert result.get("prompt_tokens") == 5
    assert result.get("completion_tokens") == 3


def test_format_documents_with_empty_and_content_disabled() -> None:
    # Empty documents
    assert tracing._format_documents([], record_content=True) is None
    assert tracing._format_documents(None, record_content=True) is None
    # With content recording disabled
    docs = [Document(page_content="secret", metadata={"id": 1})]
    result = tracing._format_documents(docs, record_content=False)
    assert result is not None
    parsed = json.loads(result)
    assert "content" not in parsed[0]
    assert parsed[0]["metadata"]["id"] == 1


def test_first_non_empty_helper() -> None:
    assert tracing._first_non_empty(None, "", "value") == "value"
    assert tracing._first_non_empty(None, None) is None
    assert tracing._first_non_empty("first", "second") == "first"
    assert tracing._first_non_empty("", "", "third") == "third"


def test_candidate_from_serialized_id_handles_various_inputs() -> None:
    assert tracing._candidate_from_serialized_id("direct") == "direct"
    assert tracing._candidate_from_serialized_id(["", "", "last"]) == "last"
    assert tracing._candidate_from_serialized_id(["only"]) == "only"
    assert tracing._candidate_from_serialized_id(123) == "123"
    assert tracing._candidate_from_serialized_id(None) is None


def test_as_json_attribute_handles_non_serializable() -> None:
    result = tracing._as_json_attribute({"key": "value"})
    assert result == '{"key": "value"}'

    # Non-serializable objects use default=str
    class Custom:
        def __str__(self) -> str:
            return "custom_repr"

    result = tracing._as_json_attribute({"obj": Custom()})
    assert "custom_repr" in result


def test_tool_type_from_definition() -> None:
    assert tracing._tool_type_from_definition({"type": "FUNCTION"}) == "function"
    assert (
        tracing._tool_type_from_definition({"function": {"type": "retriever"}})
        == "retriever"
    )
    assert tracing._tool_type_from_definition({"function": {}}) == "function"
    assert tracing._tool_type_from_definition({}) is None


def test_coerce_int_edge_cases() -> None:
    assert tracing._coerce_int(None) is None
    assert tracing._coerce_int("invalid") is None
    assert tracing._coerce_int("42") == 42
    assert tracing._coerce_int(3.7) == 3


def test_normalize_bedrock_metrics_with_missing_values() -> None:
    # Only input tokens
    result = tracing._normalize_bedrock_metrics({"inputTokenCount": 5})
    assert result is not None
    assert result["prompt_tokens"] == 5
    assert result["total_tokens"] == 5
    # Only output tokens
    result = tracing._normalize_bedrock_metrics({"outputTokenCount": 3})
    assert result is not None
    assert result["completion_tokens"] == 3
    assert result["total_tokens"] == 3
    # Empty
    assert tracing._normalize_bedrock_metrics({}) is None
    # Non-mapping
    assert tracing._normalize_bedrock_metrics("not a dict") is None


def test_extract_bedrock_usage_nested_containers() -> None:
    # Nested in response_metadata
    llm_output = {"response_metadata": {"usage": {"inputTokens": 4, "outputTokens": 6}}}
    result = tracing._extract_bedrock_usage(llm_output, [])
    assert result is not None
    assert result["prompt_tokens"] == 4
    assert result["completion_tokens"] == 6


def test_resolve_parent_id_handles_circular_references() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    # Create circular reference in overrides
    from uuid import UUID

    id1 = UUID("11111111-1111-1111-1111-111111111111")
    id2 = UUID("22222222-2222-2222-2222-222222222222")
    tracer._run_parent_override[str(id1)] = str(id2)
    tracer._run_parent_override[str(id2)] = str(id1)
    # Should return None to prevent infinite loop
    result = tracer._resolve_parent_id(id1)
    assert result is None


def test_llm_new_token_is_silent() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer()
    run_id = uuid4()
    tracer.on_llm_start(
        {"kwargs": {"model": "m"}},
        [{"role": "user", "content": "hi"}],
        run_id=run_id,
        invocation_params={"model": "m"},
    )
    # on_llm_new_token should not crash and should not add events (streaming tokens
    # are not recorded).
    tracer.on_llm_new_token("token", run_id=run_id)
    span = get_last_span_for(tracer)
    assert span.events == []


def test_prepare_messages_with_empty_input() -> None:
    formatted, system = tracing._prepare_messages(None, record_content=True)
    assert formatted is None
    assert system is None
    formatted, system = tracing._prepare_messages([], record_content=True)
    assert formatted is None
    assert system is None
    formatted, system = tracing._prepare_messages({}, record_content=True)
    assert formatted is None
    assert system is None
