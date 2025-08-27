"""Azure OpenAI tracing callback (clean implementation).

Emits OpenTelemetry spans for LangChain / LangGraph events (LLM, chain, tool,
retriever, agent). Includes:
    * Attribute normalization (skip None, JSON encode complex values)
    * Optional content recording & redaction (env AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED)
    * Legacy compatibility keys (gen_ai.prompt/system/completion) if enabled
    * Async subclass delegating to sync implementation (no logic duplication)

File fully deduplicated (legacy multi-implementation blocks removed).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

try:  # pragma: no cover
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import Span, SpanKind, Status, StatusCode, set_span_in_context
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Install azure-monitor-opentelemetry and opentelemetry packages: pip install azure-monitor-opentelemetry"
    ) from e

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.WARNING)


class Attrs:
    PROVIDER_NAME = "gen_ai.provider.name"
    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_STOP = "gen_ai.request.stop_sequences"
    REQUEST_FREQ_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_PRES_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
    REQUEST_SEED = "gen_ai.request.seed"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"  # TODO
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_CALL_ARGS = "gen_ai.tool.call.arguments"
    TOOL_CALL_RESULT = "gen_ai.tool.call.result"
    DATA_SOURCE_ID = "gen_ai.data_source.id"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    CONVERSATION_ID = "gen_ai.conversation.id"
    SERVER_ADDRESS = "server.address"
    ERROR_TYPE = "error.type"
    LEGACY_SYSTEM = "gen_ai.system"
    LEGACY_PROMPT = "gen_ai.prompt"
    LEGACY_COMPLETION = "gen_ai.completion"
    LEGACY_KEYS_FLAG = "metadata.legacy_keys"
    METADATA_RUN_ID = "metadata.run_id"
    METADATA_PARENT_RUN_ID = "metadata.parent_run_id"
    METADATA_TAGS = "metadata.tags"
    METADATA_THREAD_PATH = "metadata.langgraph.path"
    METADATA_STEP = "metadata.langgraph.step"
    METADATA_NODE = "metadata.langgraph.node"
    METADATA_TRIGGERS = "metadata.langgraph.triggers"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:  # pragma: no cover
        return '"<unserializable>"'


def _msg_dict(msg: BaseMessage) -> Dict[str, Any]:
    d = {"type": msg.type, "content": msg.content}
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        d["tool_calls"] = msg.tool_calls
    if isinstance(msg, ToolMessage):
        d["tool_call_id"] = getattr(msg, "tool_call_id", None)
    return d


def _threads_json(threads: List[List[BaseMessage]]) -> str:  # type: ignore[override]
    return _safe_json([[ _msg_dict(m) for m in thread ] for thread in threads])


def _get_model(serialized: Dict[str, Any]) -> Optional[str]:
    if not serialized:
        return None
    kw = serialized.get("kwargs", {})
    return kw.get("deployment_name") or kw.get("model") or kw.get("name")


def _extract_params(serialized: Dict[str, Any]) -> Dict[str, Any]:
    if not serialized:
        return {}
    kw = serialized.get("kwargs", {})
    keep = ["max_tokens", "temperature", "top_p", "top_k", "stop", "frequency_penalty", "presence_penalty", "n", "seed"]
    return {k: kw[k] for k in keep if k in kw}


def _finish_reasons(gens: List[List[ChatGeneration]]) -> List[str]:
    reasons: List[str] = []
    for group in gens:
        for gen in group:
            info = gen.generation_info or {}
            if isinstance(info, dict):
                fr = info.get("finish_reason")
                if fr:
                    reasons.append(fr)
    return reasons


def _normalize(v: Any):  # returns json-safe primitive or None
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        if not v:
            return []
        if all(isinstance(x, (str, int, float, bool)) for x in v) and len({type(x) for x in v}) == 1:
            return list(v)
        return _safe_json(v)
    return _safe_json(v)


def _redact(messages_json: str) -> str:
    try:
        parsed = json.loads(messages_json)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                red = []
                for thread in parsed:
                    if isinstance(thread, list):
                        red.append([
                            {"role": m.get("role", "?"), "content": "[REDACTED]"}
                            for m in thread
                            if isinstance(m, dict)
                        ])
                    else:
                        red.append(thread)
            else:
                red = [
                    {"role": m.get("role", "?"), "content": "[REDACTED]"}
                    for m in parsed
                    if isinstance(m, dict)
                ]
            return _safe_json(red)
    except Exception:
        return '[{"role":"?","content":"[REDACTED]"}]'
    return messages_json


@dataclass
class _Run:
    span: Span
    operation: str
    model: Optional[str]


class _Core:
    def __init__(self, *, enable_content_recording: bool, redact: bool, include_legacy: bool, provider: str, tracer) -> None:
        self.enable_content_recording = enable_content_recording
        self.redact = redact
        self.include_legacy = include_legacy
        self.provider = provider
        self._tracer = tracer
        self._runs: Dict[UUID, _Run] = {}

    def start(self, *, run_id: UUID, name: str, kind: SpanKind, operation: str, parent_run_id: Optional[UUID], attrs: Dict[str, Any]) -> None:
        parent_ctx = None
        if parent_run_id and parent_run_id in self._runs:
            parent_ctx = set_span_in_context(self._runs[parent_run_id].span)
        span = self._tracer.start_span(name=name, kind=kind, context=parent_ctx)
        for k, v in attrs.items():
            nv = _normalize(v)
            if nv is not None:
                span.set_attribute(k, nv)
        self._runs[run_id] = _Run(span=span, operation=operation, model=attrs.get(Attrs.REQUEST_MODEL))

    def end(self, run_id: UUID, error: Optional[BaseException]) -> None:
        state = self._runs.pop(run_id, None)
        if not state:
            return
        if error:
            state.span.set_status(Status(StatusCode.ERROR, str(error)))
            state.span.set_attribute(Attrs.ERROR_TYPE, error.__class__.__name__)
            state.span.record_exception(error)
        state.span.end()

    def set(self, run_id: UUID, attrs: Dict[str, Any]) -> None:
        state = self._runs.get(run_id)
        if not state:
            return
        for k, v in attrs.items():
            nv = _normalize(v)
            if nv is not None:
                state.span.set_attribute(k, nv)

    def redact_messages(self, messages_json: str) -> str:
        if not self.enable_content_recording or self.redact:
            return _redact(messages_json)
        return messages_json

    def enrich_langgraph(self, attrs: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> None:
        if not metadata:
            return
        mapping = {
            "langgraph_step": Attrs.METADATA_STEP,
            "langgraph_node": Attrs.METADATA_NODE,
            "langgraph_triggers": Attrs.METADATA_TRIGGERS,
            "langgraph_path": Attrs.METADATA_THREAD_PATH,
            "thread_id": Attrs.CONVERSATION_ID,
            "session_id": Attrs.CONVERSATION_ID,
        }
        for src, dst in mapping.items():
            if src in metadata:
                nv = _normalize(metadata[src])
                if nv is not None:
                    attrs[dst] = nv

    def llm_start_attrs(self, *, serialized: Dict[str, Any], run_id: UUID, parent_run_id: Optional[UUID], tags: Optional[List[str]], metadata: Optional[Dict[str, Any]], messages_json: str, model: Optional[str], params: Dict[str, Any]) -> Dict[str, Any]:
        a: Dict[str, Any] = {
            Attrs.PROVIDER_NAME: self.provider,
            Attrs.OPERATION_NAME: "chat",
            Attrs.REQUEST_MODEL: model,
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
        }
        endpoint = serialized.get("kwargs", {}).get("azure_endpoint") if serialized else None
        if endpoint:
            a[Attrs.SERVER_ADDRESS] = endpoint
        if tags:
            a[Attrs.METADATA_TAGS] = tags
        self.enrich_langgraph(a, metadata)
        param_map = {"max_tokens": Attrs.REQUEST_MAX_TOKENS, "temperature": Attrs.REQUEST_TEMPERATURE, "top_p": Attrs.REQUEST_TOP_P, "top_k": Attrs.REQUEST_TOP_K, "stop": Attrs.REQUEST_STOP, "frequency_penalty": Attrs.REQUEST_FREQ_PENALTY, "presence_penalty": Attrs.REQUEST_PRES_PENALTY, "n": Attrs.REQUEST_CHOICE_COUNT, "seed": Attrs.REQUEST_SEED}
        for k, v in params.items():
            mapped = param_map.get(k)
            if mapped:
                a[mapped] = v
        a[Attrs.INPUT_MESSAGES] = self.redact_messages(messages_json)
        if self.include_legacy:
            a[Attrs.LEGACY_PROMPT] = a[Attrs.INPUT_MESSAGES]
            a[Attrs.LEGACY_SYSTEM] = self.provider
            a[Attrs.LEGACY_KEYS_FLAG] = True
        return a

    def llm_end_attrs(self, result: LLMResult) -> Dict[str, Any]:
        gens: List[List[ChatGeneration]] = getattr(result, "generations", [])
        finish = _finish_reasons(gens)
        outputs: List[Dict[str, Any]] = []
        for group in gens:
            for gen in group:
                outputs.append({"type": "ai", "content": gen.text})
        out: Dict[str, Any] = {Attrs.RESPONSE_FINISH_REASONS: finish or None, Attrs.OUTPUT_MESSAGES: self.redact_messages(_safe_json(outputs))}
        if self.include_legacy:
            out[Attrs.LEGACY_COMPLETION] = out[Attrs.OUTPUT_MESSAGES]
        llm_output = getattr(result, "llm_output", {}) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            out[Attrs.USAGE_INPUT_TOKENS] = usage.get("prompt_tokens") or usage.get("input_tokens")
            out[Attrs.USAGE_OUTPUT_TOKENS] = usage.get("completion_tokens") or usage.get("output_tokens")
            if usage.get("total_tokens") is not None:
                out[Attrs.USAGE_TOTAL_TOKENS] = usage.get("total_tokens")
        return out


class AzureOpenAITracingCallback(BaseCallbackHandler):
    def __init__(self, *, enable_content_recording: Optional[bool] = None, connection_string: Optional[str] = None, redact: bool = False, include_legacy_keys: bool = True, provider_name: str = "azure.ai.inference") -> None:
        super().__init__()
        if enable_content_recording is None:
            env_val = os.environ.get("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "false")
            enable_content_recording = env_val.lower() in {"1", "true", "yes"}
        if connection_string:
            configure_azure_monitor(connection_string=connection_string)
        elif os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
            configure_azure_monitor()
        tracer = otel_trace.get_tracer(__name__)
        self._core = _Core(enable_content_recording=enable_content_recording, redact=redact, include_legacy=include_legacy_keys, provider=provider_name, tracer=tracer)

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **_: Any) -> Any:
        model = _get_model(serialized)
        params = _extract_params(serialized)
        attrs = self._core.llm_start_attrs(serialized=serialized, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, messages_json=_threads_json(messages), model=model, params=params)
        self._core.start(run_id=run_id, name=f"chat {model}" if model else "chat", kind=SpanKind.CLIENT, operation="chat", parent_run_id=parent_run_id, attrs=attrs)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **_: Any) -> Any:
        model = _get_model(serialized)
        params = _extract_params(serialized)
        messages: List[List[BaseMessage]] = [[HumanMessage(content=p)] for p in prompts]
        attrs = self._core.llm_start_attrs(serialized=serialized, run_id=run_id, parent_run_id=parent_run_id, tags=tags, metadata=metadata, messages_json=_threads_json(messages), model=model, params=params)
        self._core.start(run_id=run_id, name=f"chat {model}" if model else "chat", kind=SpanKind.CLIENT, operation="chat", parent_run_id=parent_run_id, attrs=attrs)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **_: Any) -> Any:
        self._core.set(run_id, self._core.llm_end_attrs(response))
        self._core.end(run_id, None)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **_: Any) -> Any:
        self._core.end(run_id, error)

    def on_agent_action(self, action: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **__: Any) -> Any:
        name = getattr(action, "tool", None) or getattr(action, "log", "agent_action")
        attrs = {Attrs.PROVIDER_NAME: self._core.provider, Attrs.OPERATION_NAME: "invoke_agent", Attrs.METADATA_RUN_ID: str(run_id), Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None, Attrs.AGENT_NAME: getattr(action, "tool", None)}
        self._core.start(run_id=run_id, name=f"invoke_agent {name}" if name else "invoke_agent", kind=SpanKind.CLIENT, operation="invoke_agent", parent_run_id=parent_run_id, attrs=attrs)

    def on_agent_finish(self, finish: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **__: Any) -> Any:
        output = getattr(finish, "return_values", None) or getattr(finish, "log", None)
        if output is not None and not isinstance(output, list):
            out_list = [output]
        else:
            out_list = output or []
        attrs = {Attrs.AGENT_DESCRIPTION: getattr(finish, "log", None), Attrs.OUTPUT_MESSAGES: (self._core.redact_messages(_safe_json(out_list)) if out_list else None)}
        if self._core.include_legacy and attrs.get(Attrs.OUTPUT_MESSAGES):
            attrs[Attrs.LEGACY_COMPLETION] = attrs[Attrs.OUTPUT_MESSAGES]
        self._core.set(run_id, attrs)
        self._core.end(run_id, None)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **__: Any) -> Any:
        name = (serialized or {}).get("id") or (serialized or {}).get("name") or "chain" # TODO: clarify assumption
        attrs = {
            Attrs.PROVIDER_NAME: self._core.provider,
            Attrs.OPERATION_NAME: "invoke_agent",  # TODO: clarify assumption invoke_agent == chain start?
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
        }
        if tags:
            attrs[Attrs.METADATA_TAGS] = tags
        self._core.enrich_langgraph(attrs, metadata)
        if "messages" in inputs and isinstance(inputs["messages"], list):
            msgs = inputs["messages"]
            if msgs and isinstance(msgs[0], BaseMessage):
                msg_json = _safe_json([_msg_dict(m) for m in msgs])
            else:
                msg_json = _safe_json(msgs)
            attrs[Attrs.INPUT_MESSAGES] = self._core.redact_messages(msg_json)
        self._core.start(
            run_id=run_id,
            name=f"invoke_agent {name}", #TODO: clarify
            kind=SpanKind.INTERNAL,
            operation="invoke_agent", # TODO: clarify assumption chain == invoke_agent?
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **__: Any) -> Any:
        attrs: Dict[str, Any] = {}
        if "messages" in outputs and isinstance(outputs["messages"], list):
            attrs[Attrs.OUTPUT_MESSAGES] = self._core.redact_messages(_safe_json(outputs["messages"]))
        self._core.set(run_id, attrs)
        self._core.end(run_id, None)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **__: Any) -> Any:
        self._core.end(run_id, error)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, inputs: Optional[Dict[str, Any]] = None, **__: Any) -> Any:
        name = (serialized or {}).get("name") or "tool"
        args_val = inputs if inputs is not None else {"input_str": input_str}
        attrs = {Attrs.PROVIDER_NAME: self._core.provider, Attrs.OPERATION_NAME: "execute_tool", Attrs.TOOL_NAME: name, Attrs.METADATA_RUN_ID: str(run_id), Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None, Attrs.TOOL_CALL_ARGS: _safe_json(args_val)}
        self._core.start(run_id=run_id, name=f"execute_tool {name}", kind=SpanKind.INTERNAL, operation="execute_tool", parent_run_id=parent_run_id, attrs=attrs)

    def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **__: Any) -> Any:
        self._core.set(run_id, {Attrs.TOOL_CALL_RESULT: _safe_json(output)})
        self._core.end(run_id, None)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **__: Any) -> Any:
        self._core.end(run_id, error)

    def on_retriever_start(self, serialized: Dict[str, Any], query: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **__: Any) -> Any:
        name = (serialized or {}).get("id") or "retriever"
        attrs = {Attrs.PROVIDER_NAME: self._core.provider, Attrs.OPERATION_NAME: "retrieve", Attrs.DATA_SOURCE_ID: (serialized or {}).get("name"), Attrs.METADATA_RUN_ID: str(run_id), Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None, "retriever.query": query}
        self._core.start(run_id=run_id, name=f"retrieve {name}", kind=SpanKind.INTERNAL, operation="retrieve", parent_run_id=parent_run_id, attrs=attrs)

    def on_retriever_end(self, documents: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, **__: Any) -> Any:
        try:
            count = len(documents)
        except Exception:
            count = None
        self._core.set(run_id, {"retriever.documents.count": count})
        self._core.end(run_id, None)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **__: Any) -> Any:
        self._core.end(run_id, error)

    def on_text(self, text: str, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **__: Any) -> Any:
        state = self._core._runs.get(run_id)
        if not state:
            return
        try:
            state.span.add_event("gen_ai.text", {"text.length": len(text), "text.preview": text[:200]})
        except Exception:
            pass

    def on_retry(self, retry_state: Any, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **__: Any) -> Any:
        state = self._core._runs.get(run_id)
        if not state:
            return
        attempt = getattr(retry_state, "attempt_number", None)
        try:
            state.span.add_event("retry", {"retry.attempt": attempt})
        except Exception:
            pass

    def on_custom_event(self, name: str, data: Any, *, run_id: UUID, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **__: Any) -> Any:
        state = self._core._runs.get(run_id)
        if not state:
            return
        ev: Dict[str, Any] = {"data": _safe_json(data)}
        if tags:
            ev["event.tags"] = _safe_json(tags)
        if metadata:
            ev["event.metadata"] = _safe_json(metadata)
        try:
            state.span.add_event(name, ev)
        except Exception:
            pass

    def shutdown(self) -> None:  # pragma: no cover
        pass

    def force_flush(self) -> None:  # pragma: no cover
        pass


class AsyncAzureOpenAITracingCallback(AzureOpenAITracingCallback, AsyncCallbackHandler):
    async def on_chat_model_start(self, *a, **k):
        return AzureOpenAITracingCallback.on_chat_model_start(self, *a, **k)

    async def on_llm_start(self, *a, **k):
        return AzureOpenAITracingCallback.on_llm_start(self, *a, **k)

    async def on_llm_end(self, *a, **k):
        return AzureOpenAITracingCallback.on_llm_end(self, *a, **k)

    async def on_llm_error(self, *a, **k):
        return AzureOpenAITracingCallback.on_llm_error(self, *a, **k)

    async def on_agent_action(self, *a, **k):
        return AzureOpenAITracingCallback.on_agent_action(self, *a, **k)

    async def on_agent_finish(self, *a, **k):
        return AzureOpenAITracingCallback.on_agent_finish(self, *a, **k)

    async def on_chain_start(self, *a, **k):
        return AzureOpenAITracingCallback.on_chain_start(self, *a, **k)

    async def on_chain_end(self, *a, **k):
        return AzureOpenAITracingCallback.on_chain_end(self, *a, **k)

    async def on_chain_error(self, *a, **k):
        return AzureOpenAITracingCallback.on_chain_error(self, *a, **k)

    async def on_tool_start(self, *a, **k):
        return AzureOpenAITracingCallback.on_tool_start(self, *a, **k)

    async def on_tool_end(self, *a, **k):
        return AzureOpenAITracingCallback.on_tool_end(self, *a, **k)

    async def on_tool_error(self, *a, **k):
        return AzureOpenAITracingCallback.on_tool_error(self, *a, **k)

    async def on_retriever_start(self, *a, **k):
        return AzureOpenAITracingCallback.on_retriever_start(self, *a, **k)

    async def on_retriever_end(self, *a, **k):
        return AzureOpenAITracingCallback.on_retriever_end(self, *a, **k)

    async def on_retriever_error(self, *a, **k):
        return AzureOpenAITracingCallback.on_retriever_error(self, *a, **k)

    async def on_text(self, *a, **k):
        return AzureOpenAITracingCallback.on_text(self, *a, **k)

    async def on_retry(self, *a, **k):
        return AzureOpenAITracingCallback.on_retry(self, *a, **k)

    async def on_custom_event(self, *a, **k):
        return AzureOpenAITracingCallback.on_custom_event(self, *a, **k)


__all__ = ["AzureOpenAITracingCallback", "AsyncAzureOpenAITracingCallback"]

# End of file
