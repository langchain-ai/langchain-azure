"""Azure AI inference tracing callback (clean implementation).

Emits OpenTelemetry spans for LangChain / LangGraph events (LLM, chain, tool,
retriever, agent). Includes:
    * Attribute normalization (skip None, JSON encode complex values)
    * Optional content recording & redaction
      (env AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED)
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
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

try:  # pragma: no cover
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import (
        Span,
        SpanKind,
        Status,
        StatusCode,
        set_span_in_context,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Install azure-monitor-opentelemetry and opentelemetry packages: "
        "pip install azure-monitor-opentelemetry"
    ) from e

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(
    logging.WARNING
)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(
    logging.WARNING
)


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
    SERVER_PORT = "server.port"
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
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    OUTPUT_TYPE = "gen_ai.output.type"
    TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_TYPE = "gen_ai.tool.type"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    EMBEDDINGS_DIM_COUNT = "gen_ai.embeddings.dimension.count"
    AGENT_ID = "gen_ai.agent.id"
    REQUEST_MAX_INPUT_TOKENS = "gen_ai.request.max_input_tokens"
    REQUEST_MAX_OUTPUT_TOKENS = "gen_ai.request.max_output_tokens"
    OPENAI_REQUEST_SERVICE_TIER = "openai.request.service_tier"
    OPENAI_RESPONSE_SERVICE_TIER = "openai.response.service_tier"
    OPENAI_RESPONSE_SYSTEM_FINGERPRINT = "openai.response.system_fingerprint"
    AZURE_RESOURCE_NAMESPACE = "azure.resource_provider.namespace"


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:  # pragma: no cover
        return '"<unserializable>"'


def _msg_dict(msg: BaseMessage) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": msg.type, "content": msg.content}
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        d["tool_calls"] = _safe_json(getattr(msg, "tool_calls"))
    if isinstance(msg, ToolMessage):
        tcid = getattr(msg, "tool_call_id", None)
        if tcid is not None:
            d["tool_call_id"] = str(tcid)
    return d


def _threads_json(threads: List[List[BaseMessage]]) -> str:  # type: ignore[override]
    return _safe_json([[_msg_dict(m) for m in thread] for thread in threads])


def _extract_port(endpoint: Optional[str]) -> Optional[int]:
    """Extract a non-default port from an endpoint URL.

    Returns None if endpoint is None or if the port is default for the scheme
    (80 for http, 443 for https). If a non-default explicit port is present,
    returns that integer value.
    """
    if not endpoint:
        return None
    try:
        from urllib.parse import urlparse

        p = urlparse(endpoint)
        if not p.netloc:
            return None
        port = p.port
        if port is None:
            return None
        # Default ports: 80 for http, 443 for https
        if (p.scheme == "https" and port == 443) or (p.scheme == "http" and port == 80):
            return None
        return int(port)
    except Exception:  # pragma: no cover
        try:
            # Fallback parsing if urlparse failed; keep robust but conservative
            host_part = endpoint.split("//", 1)[-1].split("/", 1)[0]
            if ":" in host_part:
                cand = host_part.rsplit(":", 1)[-1]
                return int(cand) if cand.isdigit() else None
        except Exception:
            return None
        return None


def _get_model(serialized: Dict[str, Any]) -> Optional[str]:
    if not serialized:
        return None
    kw = serialized.get("kwargs", {})
    return kw.get("deployment_name") or kw.get("model") or kw.get("name")


def _extract_params(serialized: Dict[str, Any]) -> Dict[str, Any]:
    if not serialized:
        return {}
    kw = serialized.get("kwargs", {})
    keep = [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "seed",
    ]
    return {k: kw[k] for k in keep if k in kw}


def _finish_reasons(gens: List[List[ChatGeneration]]) -> List[str]:
    reasons: List[str] = []
    for group in gens:
        for gen in group:
            info = getattr(gen, "generation_info", None) or {}
            if not isinstance(info, dict):
                continue
            fr = (
                info.get("finish_reason")
                or info.get("finishReason")
                or info.get("finish_reasons")
                or info.get("reason")
            )
            if fr is None:
                continue
            if isinstance(fr, list):
                reasons.extend([str(r) for r in fr if r is not None])
            else:
                reasons.append(str(fr))
    dedup: List[str] = []
    seen = set()
    for r in reasons:
        if r not in seen:
            dedup.append(r)
            seen.add(r)
    return dedup


def _normalize(v: Any) -> Any:  # returns json-safe primitive or None
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        if not v:
            return []
        if (
            all(isinstance(x, (str, int, float, bool)) for x in v)
            and len({type(x) for x in v}) == 1
        ):
            return list(v)
        return _safe_json(v)
    return _safe_json(v)


def _redact(messages_json: str) -> str:
    try:
        parsed = json.loads(messages_json)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], list):
                red: Any = []
                for thread in parsed:
                    if isinstance(thread, list):
                        red.append(
                            [
                                {"role": m.get("role", "?"), "content": "[REDACTED]"}
                                for m in thread
                                if isinstance(m, dict)
                            ]
                        )
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
    response_id: Optional[str] = None


class _Core:
    def __init__(
        self,
        *,
        enable_content_recording: bool,
        redact: bool,
        include_legacy: bool,
        provider: str,
        tracer: Any,
        default_name: Optional[str] = None,
        default_id: Optional[str] = None,
    ) -> None:
        self.enable_content_recording = enable_content_recording
        self.redact = redact
        self.include_legacy = include_legacy
        self.provider = provider
        self._tracer = tracer
        self._runs: Dict[UUID, _Run] = {}
        # Optional defaults for generic attributes requested by callers
        self._default_name = default_name
        self._default_id = default_id

    def start(
        self,
        *,
        run_id: UUID,
        name: str,
        kind: SpanKind,
        operation: str,
        parent_run_id: Optional[UUID],
        attrs: Dict[str, Any],
    ) -> None:
        parent_ctx = None
        if parent_run_id and parent_run_id in self._runs:
            parent_ctx = set_span_in_context(self._runs[parent_run_id].span)
        span = self._tracer.start_span(name=name, kind=kind, context=parent_ctx)
        for k, v in attrs.items():
            nv = _normalize(v)
            if nv is not None:
                span.set_attribute(k, nv)
        # Ensure agent identity fields are present and mirrored
        try:
            # Determine values from attrs or defaults
            _nm = (
                attrs.get(Attrs.AGENT_NAME)
                or attrs.get(Attrs.TOOL_NAME)
                or self._default_name
            )
            _idv = (
                attrs.get(Attrs.AGENT_ID)
                or attrs.get(Attrs.TOOL_CALL_ID)
                or self._default_id
            )
            # Ensure gen_ai.agent.* are set when missing
            if _nm is not None and attrs.get(Attrs.AGENT_NAME) is None:
                span.set_attribute(Attrs.AGENT_NAME, _nm)
            if _idv is not None and attrs.get(Attrs.AGENT_ID) is None:
                span.set_attribute(Attrs.AGENT_ID, _idv)
            # Do not set generic convenience keys (name/id).
            # Rely on gen_ai.agent.*
        except Exception:
            pass
        self._runs[run_id] = _Run(
            span=span,
            operation=operation,
            model=attrs.get(Attrs.REQUEST_MODEL),
            response_id=None,
        )

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
        # Track response id if provided
        try:
            rid = attrs.get(Attrs.RESPONSE_ID)
            if rid is not None:
                state.response_id = str(rid)
        except Exception:
            pass

    def redact_messages(self, messages_json: str) -> Optional[str]:
        # Opt-in: if recording disabled, omit (return None)
        if not self.enable_content_recording:
            return None
        if self.redact:
            return _redact(messages_json)
        return messages_json

    def enrich_langgraph(
        self, attrs: Dict[str, Any], metadata: Optional[Dict[str, Any]]
    ) -> None:
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

    def llm_start_attrs(
        self,
        *,
        serialized: Dict[str, Any],
        run_id: UUID,
        parent_run_id: Optional[UUID],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        messages_json: str,
        model: Optional[str],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        a: Dict[str, Any] = {
            Attrs.PROVIDER_NAME: self.provider,
            Attrs.OPERATION_NAME: "chat",
            Attrs.REQUEST_MODEL: model,
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
        }
        endpoint = None
        if serialized:
            kw = serialized.get("kwargs", {}) or {}
            endpoint = (
                kw.get("azure_endpoint")
                or kw.get("base_url")
                or kw.get("api_base")
                or kw.get("endpoint")
            )
        if endpoint:
            a[Attrs.SERVER_ADDRESS] = endpoint
            port = _extract_port(endpoint)
            if port is not None:
                a[Attrs.SERVER_PORT] = port
        if tags:
            a[Attrs.METADATA_TAGS] = _safe_json(tags)
        self.enrich_langgraph(a, metadata)
        param_map = {
            "max_tokens": Attrs.REQUEST_MAX_TOKENS,
            "temperature": Attrs.REQUEST_TEMPERATURE,
            "top_p": Attrs.REQUEST_TOP_P,
            "top_k": Attrs.REQUEST_TOP_K,
            "stop": Attrs.REQUEST_STOP,
            "frequency_penalty": Attrs.REQUEST_FREQ_PENALTY,
            "presence_penalty": Attrs.REQUEST_PRES_PENALTY,
            "n": Attrs.REQUEST_CHOICE_COUNT,
            "seed": Attrs.REQUEST_SEED,
        }
        for k, v in params.items():
            if k == "n":
                # Only set choice.count when != 1 per spec
                if v is not None and v != 1:
                    a[Attrs.REQUEST_CHOICE_COUNT] = v
                continue
            mapped = param_map.get(k)
            if mapped:
                a[mapped] = v
            if k == "max_input_tokens":
                a[Attrs.REQUEST_MAX_INPUT_TOKENS] = v
            if k == "max_output_tokens":
                a[Attrs.REQUEST_MAX_OUTPUT_TOKENS] = v
        # output.type (conditionally) from response_format / output_format
        rf = (
            (serialized or {}).get("kwargs", {}).get("response_format")
            if serialized
            else None
        )
        if rf and isinstance(rf, dict):
            t = rf.get("type") or rf.get("format")
            if t:
                a[Attrs.OUTPUT_TYPE] = t
        of = (
            (serialized or {}).get("kwargs", {}).get("output_format")
            if serialized
            else None
        )
        if of and isinstance(of, str):
            a[Attrs.OUTPUT_TYPE] = of
        # system instructions (opt-in)
        sys_inst = None
        if serialized:
            kw = serialized.get("kwargs", {}) or {}
            sys_inst = (
                kw.get("system") or kw.get("system_message") or kw.get("instructions")
            )
        # Fallback: allow application to pass via metadata
        if sys_inst is None and metadata:
            sys_inst = metadata.get("system") or metadata.get("system_instructions")
        if sys_inst is not None and self.enable_content_recording:
            a[Attrs.SYSTEM_INSTRUCTIONS] = (
                self.redact_messages(_safe_json(sys_inst)) or None
            )
        # tool definitions (opt-in)
        if self.enable_content_recording and serialized:
            tools_def = (serialized.get("kwargs", {}) or {}).get("tools")
            if tools_def:
                a[Attrs.TOOL_DEFINITIONS] = _safe_json(tools_def)
        input_msgs = self.redact_messages(messages_json)
        if input_msgs is not None:
            a[Attrs.INPUT_MESSAGES] = input_msgs
        agent_name = None
        if metadata:
            agent_name = metadata.get("agent_name") or metadata.get("agent_type")
        if not agent_name and tags:
            for t in tags:
                if t.startswith("agent:"):
                    agent_name = t.split(":", 1)[1]
                    break
        if agent_name:
            a[Attrs.AGENT_NAME] = agent_name
        if self.include_legacy and a.get(Attrs.INPUT_MESSAGES) is not None:
            a[Attrs.LEGACY_PROMPT] = a[Attrs.INPUT_MESSAGES]
            a[Attrs.LEGACY_SYSTEM] = self.provider or "langgraph"
            a[Attrs.LEGACY_KEYS_FLAG] = True
        return a

    def llm_end_attrs(self, result: LLMResult) -> Dict[str, Any]:
        gens: List[List[ChatGeneration]] = getattr(result, "generations", [])
        finish = _finish_reasons(gens)
        outputs: List[Dict[str, Any]] = []
        for group in gens:
            for gen in group:
                content = getattr(gen, "text", None)
                if content is None and hasattr(gen, "message"):
                    content = getattr(gen.message, "content", None)
                outputs.append({"content": content})
        out: Dict[str, Any] = {Attrs.RESPONSE_FINISH_REASONS: finish or None}
        output_json = self.redact_messages(_safe_json(outputs)) if outputs else None
        if output_json is not None:
            out[Attrs.OUTPUT_MESSAGES] = output_json
            if self.include_legacy:
                out[Attrs.LEGACY_COMPLETION] = output_json
        llm_output = getattr(result, "llm_output", {}) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            in_tok = (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or usage.get("promptTokens")
            )
            out_tok = (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or usage.get("completionTokens")
            )
            if in_tok is not None:
                out[Attrs.USAGE_INPUT_TOKENS] = in_tok
            if out_tok is not None:
                out[Attrs.USAGE_OUTPUT_TOKENS] = out_tok
            # Note: gen_ai.usage.total_tokens is not in the current registry
            # spec; avoid emitting it to stay compliant.
        resp_model = llm_output.get("model") or llm_output.get("response_model")
        if resp_model:
            out[Attrs.RESPONSE_MODEL] = resp_model
        resp_id = llm_output.get("id") or llm_output.get("response_id")
        if resp_id:
            out[Attrs.RESPONSE_ID] = resp_id
        # OpenAI specific optional attributes if present
        if llm_output:
            service_tier_req = llm_output.get("service_tier") or llm_output.get(
                "request_service_tier"
            )
            if service_tier_req:
                out[Attrs.OPENAI_REQUEST_SERVICE_TIER] = service_tier_req
            service_tier_resp = llm_output.get("response_service_tier")
            if service_tier_resp:
                out[Attrs.OPENAI_RESPONSE_SERVICE_TIER] = service_tier_resp
            sys_fp = llm_output.get("system_fingerprint") or llm_output.get(
                "systemFingerprint"
            )
            if sys_fp:
                out[Attrs.OPENAI_RESPONSE_SYSTEM_FINGERPRINT] = sys_fp
        return out


class AzureAIOpenTelemetryTracer(BaseCallbackHandler):
    """Tracing callback that emits OpenTelemetry spans for GenAI activity.

    Supports LangChain and LangGraph callbacks for LLM/chat, chains, tools,
    retrievers, parsers, transformers, and embeddings. Offers optional
    content recording and redaction, plus legacy compatibility attributes.
    """

    def __init__(
        self,
        *,
        enable_content_recording: Optional[bool] = None,
        connection_string: Optional[str] = None,
        redact: bool = False,
        include_legacy_keys: bool = True,
        provider_name: str = "langchain-azure-ai",
        # Additional optional defaults for generic attributes on spans
        name: Optional[str] = None,
        id: Optional[str] = None,
    ) -> None:
        """Create a new tracing callback.

        - enable_content_recording: enable recording of prompts/outputs.
        - connection_string: Application Insights connection string (optional).
        - redact: if True, redact recorded content.
        - include_legacy_keys: include legacy gen_ai.* prompt/completion keys.
        - provider_name: value for gen_ai.provider.name.
        """
        super().__init__()
        if enable_content_recording is None:
            env_val = os.environ.get(
                "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "false"
            )
            enable_content_recording = env_val.lower() in {"1", "true", "yes"}
        if connection_string:
            configure_azure_monitor(connection_string=connection_string)
        elif os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
            configure_azure_monitor()
        tracer = otel_trace.get_tracer(__name__)
        self._core = _Core(
            enable_content_recording=enable_content_recording,
            redact=redact,
            include_legacy=include_legacy_keys,
            provider=provider_name,
            tracer=tracer,
            default_name=name,
            default_id=id,
        )
        # Cache for synthetic tool spans when on_tool_* callbacks are not fired.
        # Keyed by tool_call_id; value carries name, args, and an optional parent hint.
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        # Track which agents we've already emitted a create_agent span for
        self._created_agents: set[str] = set()

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Any:
        """Start a chat span for a chat model call."""
        # Emit synthetic tool spans from any ToolMessage results present in messages
        # (fallback when on_tool_* are not fired). Parent to the surrounding chain.
        try:
            for thread in messages or []:
                for m in thread or []:
                    if isinstance(m, ToolMessage):
                        tc_id = getattr(m, "tool_call_id", None)
                        name = getattr(m, "name", None)
                        if tc_id and tc_id in self._pending_tool_calls:
                            meta = self._pending_tool_calls.pop(tc_id, {})
                            run_tool_id = uuid4()
                            t_attrs = {
                                Attrs.PROVIDER_NAME: self._core.provider,
                                Attrs.OPERATION_NAME: "execute_tool",
                                Attrs.TOOL_NAME: name or meta.get("name"),
                                Attrs.TOOL_CALL_ID: tc_id,
                                Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",  # noqa: E501
                            }
                            if self._core.enable_content_recording:
                                if "args" in meta:
                                    t_attrs[Attrs.TOOL_CALL_ARGS] = _safe_json(
                                        meta["args"]
                                    )
                                # Record result content
                                t_attrs[Attrs.TOOL_CALL_RESULT] = _safe_json(
                                    getattr(m, "content", None)
                                )
                            self._core.start(
                                run_id=run_tool_id,
                                name=(
                                    f"execute_tool {t_attrs.get(Attrs.TOOL_NAME) or ''}"
                                ).strip(),
                                kind=SpanKind.INTERNAL,
                                operation="execute_tool",
                                parent_run_id=parent_run_id,
                                attrs=t_attrs,
                            )
                            self._core.end(run_id=run_tool_id, error=None)
        except Exception:
            # best-effort fallback; avoid interrupting main span creation
            pass
        model = _get_model(serialized)
        params = _extract_params(serialized)
        attrs = self._core.llm_start_attrs(
            serialized=serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            messages_json=_threads_json(messages),
            model=model,
            params=params,
        )
        self._core.start(
            run_id=run_id,
            name=f"chat {model}" if model else "chat",
            kind=SpanKind.CLIENT,
            operation="chat",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Any:
        """Start a chat span from a list of prompts."""
        model = _get_model(serialized)
        params = _extract_params(serialized)
        messages: List[List[BaseMessage]] = [[HumanMessage(content=p)] for p in prompts]
        attrs = self._core.llm_start_attrs(
            serialized=serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            messages_json=_threads_json(messages),
            model=model,
            params=params,
        )
        self._core.start(
            run_id=run_id,
            name=f"chat {model}" if model else "chat",
            kind=SpanKind.CLIENT,
            operation="chat",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **_: Any,
    ) -> Any:
        """Finish the chat span and attach response attributes."""
        # Derive end attributes via core helper (tool logic was erroneously here before)
        if response is None:
            self._core.end(run_id, None)
            return
        attrs = self._core.llm_end_attrs(result=response)
        if attrs:
            self._core.set(run_id, attrs)
        # Cache tool calls (if any) for synthetic tool spans
        try:
            gens: List[List[ChatGeneration]] = getattr(response, "generations", [])
            for group in gens:
                for gen in group:
                    msg = getattr(gen, "message", None)
                    tool_calls = getattr(msg, "tool_calls", None)
                    if not tool_calls:
                        continue
                    # tool_calls is a list of dicts like
                    # {id, function:{name, arguments}, type:'function'}
                    for tc in tool_calls:
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                        name = fn.get("name") if isinstance(fn, dict) else None
                        args = fn.get("arguments") if isinstance(fn, dict) else None
                        # Parse arguments if they look like JSON
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args)
                            except Exception:
                                parsed = args
                            args_val = parsed
                        else:
                            args_val = args
                        if tc_id:
                            self._pending_tool_calls[tc_id] = {
                                "name": name,
                                "args": args_val,
                                "llm_run_id": run_id,
                                "chain_parent_run_id": parent_run_id,
                            }
        except Exception:
            pass
        self._core.end(run_id, None)

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Any] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_: Any,
    ) -> Any:
        """Record a streaming token event on the active chat span."""
        state = self._core._runs.get(run_id)
        if not state:
            return
        try:
            state.span.add_event(
                "gen_ai.token",
                attributes={
                    "gen_ai.token.length": (len(token) if token is not None else 0),
                    "gen_ai.token.preview": (
                        token[:200] if isinstance(token, str) else ""
                    ),
                },
            )
        except Exception:
            pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **_: Any,
    ) -> Any:
        """Mark the chat span as errored and end it."""
        self._core.end(run_id, error)

    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **__: Any,
    ) -> Any:
        """Start an agent invocation span and emit create_agent when applicable."""
        name = (
            getattr(action, "tool", None)
            or getattr(action, "log", None)
            or getattr(action, "agent_name", None)
        )
        # Emit create_agent once per agent name (best-effort detection)
        try:
            if isinstance(name, str) and name and name not in self._created_agents:
                self._created_agents.add(name)
                ca_attrs = {
                    Attrs.PROVIDER_NAME: self._core.provider,
                    Attrs.OPERATION_NAME: "create_agent",
                    Attrs.AGENT_NAME: name,
                    Attrs.AGENT_ID: getattr(action, "agent_id", None),
                    Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
                }
                # If action carries instructions, record as system instructions
                # (opt-in)
                sys_inst = getattr(action, "system_instructions", None) or getattr(
                    action, "instructions", None
                )
                if sys_inst and self._core.enable_content_recording:
                    red = self._core.redact_messages(_safe_json(sys_inst))
                    if red is not None:
                        ca_attrs[Attrs.SYSTEM_INSTRUCTIONS] = red
                ca_run_id = uuid4()
                self._core.start(
                    run_id=ca_run_id,
                    name=f"create_agent {name}",
                    kind=SpanKind.CLIENT,
                    operation="create_agent",
                    parent_run_id=parent_run_id,
                    attrs=ca_attrs,
                )
                self._core.end(run_id=ca_run_id, error=None)
        except Exception:
            pass
        attrs = {
            Attrs.PROVIDER_NAME: self._core.provider,
            Attrs.OPERATION_NAME: "invoke_agent",
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            # Prefer explicit agent_name on the action; fall back to other hints
            Attrs.AGENT_NAME: (
                getattr(action, "agent_name", None)
                or getattr(action, "name", None)
                or getattr(action, "tool", None)
            ),
            Attrs.AGENT_ID: getattr(action, "agent_id", None),
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
        }
        # system instructions if present
        sys_inst = getattr(action, "system_instructions", None) or getattr(
            action, "instructions", None
        )
        if sys_inst and self._core.enable_content_recording:
            red = self._core.redact_messages(_safe_json(sys_inst))
            if red is not None:
                attrs[Attrs.SYSTEM_INSTRUCTIONS] = red
        self._core.start(
            run_id=run_id,
            name=f"invoke_agent {name}" if name else "invoke_agent",
            kind=SpanKind.CLIENT,
            operation="invoke_agent",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **__: Any,
    ) -> Any:
        """Finish the agent span and attach outputs (redacted if enabled)."""
        output = getattr(finish, "return_values", None) or getattr(finish, "log", None)
        if output is not None and not isinstance(output, list):
            out_list = [output]
        else:
            out_list = output or []
        attrs = {
            Attrs.AGENT_DESCRIPTION: getattr(finish, "log", None),
            Attrs.OUTPUT_MESSAGES: (
                self._core.redact_messages(_safe_json(out_list)) if out_list else None
            ),
        }
        if self._core.include_legacy and attrs.get(Attrs.OUTPUT_MESSAGES):
            attrs[Attrs.LEGACY_COMPLETION] = attrs[Attrs.OUTPUT_MESSAGES]
        self._core.set(run_id, attrs)
        self._core.end(run_id, None)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **__: Any,
    ) -> Any:
        """Start a chain span (treated as invoke_agent)."""
        _name = (
            (serialized or {}).get("id") or (serialized or {}).get("name") or "chain"
        )
        attrs = {
            Attrs.PROVIDER_NAME: self._core.provider,
            # invoke_agent for chain start
            Attrs.OPERATION_NAME: "invoke_agent",
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
        }
        if tags:
            attrs[Attrs.METADATA_TAGS] = _safe_json(tags)
        agent_name = None
        if metadata:
            agent_name = (
                metadata.get("agent_name")
                or metadata.get("agent_type")
                or metadata.get("langgraph_node")
            )
        if not agent_name and tags:
            for t in tags:
                if t.startswith("agent:"):
                    agent_name = t.split(":", 1)[1]
                    break
        if agent_name:
            attrs[Attrs.AGENT_NAME] = agent_name
        self._core.enrich_langgraph(attrs, metadata)
        if "messages" in inputs and isinstance(inputs["messages"], list):
            msgs = inputs["messages"]
            if msgs and isinstance(msgs[0], BaseMessage):
                msg_json = _safe_json([_msg_dict(m) for m in msgs])
            else:
                msg_json = _safe_json(msgs)
            msg_attr = self._core.redact_messages(msg_json)
            if msg_attr is not None:
                attrs[Attrs.INPUT_MESSAGES] = msg_attr
        self._core.start(
            run_id=run_id,
            # Follow spec: if agent name not available, use just "invoke_agent"
            name=(f"invoke_agent {agent_name}" if agent_name else "invoke_agent"),
            kind=SpanKind.INTERNAL,
            operation="invoke_agent",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Finish the chain span and attach outputs when relevant."""
        attrs: Dict[str, Any] = {}
        state = self._core._runs.get(run_id)
        op = getattr(state, "operation", None) if state else None
        if op in {"invoke_agent", "parse", "transform", "chat", "embeddings"}:
            if (
                isinstance(outputs, dict)
                and "messages" in outputs
                and isinstance(outputs["messages"], list)
            ):
                msg_attr = self._core.redact_messages(_safe_json(outputs["messages"]))
                if msg_attr is not None:
                    attrs[Attrs.OUTPUT_MESSAGES] = msg_attr
            elif (
                hasattr(outputs, "__class__")
                and outputs.__class__.__name__ == "Command"
            ):
                pass

            if attrs:
                self._core.set(run_id, attrs)

        try:
            pending_for_chain = [
                (tc_id, meta)
                for tc_id, meta in list(self._pending_tool_calls.items())
                if meta.get("chain_parent_run_id") == run_id
            ]
            for tc_id, meta in pending_for_chain:
                name = meta.get("name") or "tool"
                run_tool_id = uuid4()
                t_attrs = {
                    Attrs.PROVIDER_NAME: self._core.provider,
                    Attrs.OPERATION_NAME: "execute_tool",
                    Attrs.TOOL_NAME: name,
                    Attrs.TOOL_CALL_ID: tc_id,
                    Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
                }
                if self._core.enable_content_recording and ("args" in meta):
                    t_attrs[Attrs.TOOL_CALL_ARGS] = _safe_json(meta.get("args"))
                self._core.start(
                    run_id=run_tool_id,
                    name=f"execute_tool {name}",
                    kind=SpanKind.INTERNAL,
                    operation="execute_tool",
                    parent_run_id=run_id,
                    attrs=t_attrs,
                )
                self._core.end(run_tool_id, None)
                self._pending_tool_calls.pop(tc_id, None)
        except Exception:
            pass
        self._core.end(run_id, None)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Mark the chain span as errored and end it."""
        self._core.end(run_id, error)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **__: Any,
    ) -> Any:
        """Start a tool execution span and record arguments (opt-in)."""
        name = (serialized or {}).get("name") or "tool"
        args_val = inputs if inputs is not None else {"input_str": input_str}
        tool_call_id = None
        if inputs and isinstance(inputs, dict):
            tool_call_id = inputs.get("id") or inputs.get("tool_call_id")
        # Avoid duplicate synthetic spans: if we previously cached this tool_call_id
        # from an LLM tool_calls block, drop it now that we have a real on_tool_start.
        if tool_call_id:
            try:
                self._pending_tool_calls.pop(tool_call_id, None)
            except Exception:
                pass
        attrs = {
            Attrs.PROVIDER_NAME: self._core.provider,
            Attrs.OPERATION_NAME: "execute_tool",
            Attrs.TOOL_NAME: name,
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            # Respect opt-in content recording for tool arguments
            Attrs.TOOL_CALL_ARGS: (
                _safe_json(args_val) if self._core.enable_content_recording else None
            ),
            # Attempt optional fields per spec
            Attrs.TOOL_DESCRIPTION: (serialized or {}).get("description"),
            Attrs.TOOL_TYPE: (serialized or {}).get("type"),
            Attrs.TOOL_CALL_ID: tool_call_id,
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
        }
        self._core.start(
            run_id=run_id,
            name=f"execute_tool {name}",
            kind=SpanKind.INTERNAL,
            operation="execute_tool",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Finish the tool span and record result (opt-in)."""
        # Record tool call result only if content recording enabled (opt-in)
        if self._core.enable_content_recording:
            self._core.set(run_id, {Attrs.TOOL_CALL_RESULT: _safe_json(output)})
        self._core.end(run_id, None)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Mark the tool span as errored and end it."""
        self._core.end(run_id, error)

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **__: Any,
    ) -> Any:
        """Start a retriever span for a query."""
        name = (serialized or {}).get("id") or "retriever"
        attrs = {
            Attrs.PROVIDER_NAME: self._core.provider,
            Attrs.OPERATION_NAME: "retrieve",
            Attrs.DATA_SOURCE_ID: (serialized or {}).get("name"),
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            "retriever.query": query,
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
        }
        self._core.start(
            run_id=run_id,
            name=f"retrieve {name}",
            kind=SpanKind.INTERNAL,
            operation="retrieve",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Finish the retriever span and attach document count."""
        try:
            count = len(documents)
        except Exception:
            count = None
        self._core.set(run_id, {"retriever.documents.count": count})
        self._core.end(run_id, None)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **__: Any,
    ) -> Any:
        """Mark the retriever span as errored and end it."""
        self._core.end(run_id, error)

    # -------------- Parser callbacks (internal) --------------
    def on_parser_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **__: Any,
    ) -> Any:
        """Start a parser span and optionally record inputs/config."""
        name = (
            (serialized or {}).get("id") or (serialized or {}).get("name") or "parser"
        )
        kw = (serialized or {}).get("kwargs", {}) or {}
        ptype = (serialized or {}).get("type") or kw.get("_type") or kw.get("type")
        attrs: Dict[str, Any] = {
            Attrs.PROVIDER_NAME: self._core.provider,
            Attrs.OPERATION_NAME: "parse",
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            "parser.name": name,
            "parser.type": ptype,
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
        }
        if tags:
            attrs[Attrs.METADATA_TAGS] = _safe_json(tags)
        self._core.enrich_langgraph(attrs, metadata)
        if self._core.enable_content_recording and inputs is not None:
            try:
                attrs["parser.input"] = _safe_json(inputs)
            except Exception:
                pass
        if self._core.enable_content_recording and kw:
            try:
                attrs["parser.config"] = _safe_json(kw)
            except Exception:
                pass
        self._core.start(
            run_id=run_id,
            name=f"parse {name}",
            kind=SpanKind.INTERNAL,
            operation="parse",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_parser_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Finish the parser span and attach output metadata."""
        attrs: Dict[str, Any] = {}
        if self._core.enable_content_recording and outputs is not None:
            try:
                attrs["parser.output"] = _safe_json(outputs)
            except Exception:
                pass
        try:
            size = len(outputs) if hasattr(outputs, "__len__") else None
            attrs["parser.output.size"] = size
        except Exception:
            pass
        self._core.set(run_id, attrs)
        self._core.end(run_id, None)

    def on_parser_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Mark the parser span as errored and end it."""
        self._core.end(run_id, error)

    # -------------- Transformer callbacks (internal) --------------
    def on_transform_start(
        self,
        serialized: Dict[str, Any],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **__: Any,
    ) -> Any:
        """Start a transform span and optionally record inputs/config."""
        name = (
            (serialized or {}).get("id")
            or (serialized or {}).get("name")
            or "transform"
        )
        kw = (serialized or {}).get("kwargs", {}) or {}
        ttype = (serialized or {}).get("type") or kw.get("_type") or kw.get("type")
        attrs: Dict[str, Any] = {
            Attrs.PROVIDER_NAME: self._core.provider,
            Attrs.OPERATION_NAME: "transform",
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            "transform.name": name,
            "transform.type": ttype,
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
        }
        if tags:
            attrs[Attrs.METADATA_TAGS] = _safe_json(tags)
        self._core.enrich_langgraph(attrs, metadata)
        if self._core.enable_content_recording and inputs is not None:
            try:
                attrs["transform.input"] = _safe_json(inputs)
            except Exception:
                pass
        if self._core.enable_content_recording and kw:
            try:
                attrs["transform.config"] = _safe_json(kw)
            except Exception:
                pass
        # Simple metrics: input count if list-like
        try:
            attrs["transform.inputs.count"] = (
                len(inputs) if hasattr(inputs, "__len__") else None
            )
        except Exception:
            pass
        self._core.start(
            run_id=run_id,
            name=f"transform {name}",
            kind=SpanKind.INTERNAL,
            operation="transform",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_transform_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Finish the transform span and attach output metadata."""
        attrs: Dict[str, Any] = {}
        if self._core.enable_content_recording and outputs is not None:
            try:
                attrs["transform.output"] = _safe_json(outputs)
            except Exception:
                pass
        try:
            attrs["transform.outputs.count"] = (
                len(outputs) if hasattr(outputs, "__len__") else None
            )
        except Exception:
            pass
        self._core.set(run_id, attrs)
        self._core.end(run_id, None)

    def on_transform_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Mark the transform span as errored and end it."""
        self._core.end(run_id, error)

    # -------------- Events helpers --------------
    def emit_inference_details_event(
        self,
        *,
        run_id: UUID,
        details: Dict[str, Any],
    ) -> None:
        """Emit a detailed inference operation event on the given run span.

        Event name: "gen_ai.client.inference.operation.details".
        Attributes: pass a dict of additional details (normalized to
        primitives/JSON strings).
        """
        state = self._core._runs.get(run_id)
        if not state:
            return
        attrs: Dict[str, Any] = {}
        # Normalize values to be OTEL attribute safe
        for k, v in (details or {}).items():
            nv = _normalize(v)
            if nv is not None:
                attrs[k] = nv
        try:
            state.span.add_event(
                "gen_ai.client.inference.operation.details",
                attributes=attrs,
            )
        except Exception:
            pass

    def emit_evaluation_event(
        self,
        *,
        run_id: UUID,
        evaluation_name: str,
        score_value: float,
        score_label: str,
        explanation: Optional[str] = None,
        extra_attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a model evaluation result event on the given run span.

        Event name: "gen_ai.evaluation.result"
        Attributes include:
          - gen_ai.evaluation.name
          - gen_ai.evaluation.score.value
          - gen_ai.evaluation.score.label
          - gen_ai.evaluation.explanation (optional)
          - gen_ai.response.id (when available for this run)
          - any extra_attrs provided (normalized)
        """
        state = self._core._runs.get(run_id)
        if not state:
            return
        attrs: Dict[str, Any] = {
            "gen_ai.evaluation.name": evaluation_name,
            "gen_ai.evaluation.score.value": float(score_value),
            "gen_ai.evaluation.score.label": str(score_label),
        }
        if explanation is not None:
            attrs["gen_ai.evaluation.explanation"] = str(explanation)
        if getattr(state, "response_id", None):
            attrs[Attrs.RESPONSE_ID] = state.response_id
        for k, v in (extra_attrs or {}).items():
            nv = _normalize(v)
            if nv is not None:
                attrs[k] = nv
        try:
            state.span.add_event("gen_ai.evaluation.result", attributes=attrs)
        except Exception:
            pass

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **__: Any,
    ) -> Any:
        """Attach a free-form text event to the active span."""
        state = self._core._runs.get(run_id)
        if not state:
            return
        try:
            state.span.add_event(
                "gen_ai.text", {"text.length": len(text), "text.preview": text[:200]}
            )
        except Exception:
            pass

    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **__: Any,
    ) -> Any:
        """Record a retry event with the attempt count."""
        state = self._core._runs.get(run_id)
        if not state:
            return
        attempt = getattr(retry_state, "attempt_number", None)
        try:
            ev: Dict[str, Any] = {}
            if attempt is not None:
                try:
                    ev["retry.attempt"] = int(attempt)
                except Exception:
                    pass
            state.span.add_event("retry", ev)
        except Exception:
            pass

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **__: Any,
    ) -> Any:
        """Attach a custom event to the active span."""
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

    # Embeddings span compliance
    def on_embedding_start(
        self,
        serialized: Dict[str, Any],
        inputs: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **__: Any,
    ) -> Any:
        """Start an embeddings span and optionally record inputs."""
        model = _get_model(serialized)
        kw = (serialized or {}).get("kwargs", {}) or {}
        encoding_formats = kw.get("encoding_format") or kw.get("encoding_formats")
        if isinstance(encoding_formats, str):
            encoding_formats = [encoding_formats]
        dims = kw.get("dimensions") or kw.get("embedding_dimensions")
        endpoint = (
            kw.get("azure_endpoint")
            or kw.get("base_url")
            or kw.get("api_base")
            or kw.get("endpoint")
        )
        attrs = {
            Attrs.PROVIDER_NAME: self._core.provider,
            Attrs.OPERATION_NAME: "embeddings",
            Attrs.REQUEST_MODEL: model,
            Attrs.METADATA_RUN_ID: str(run_id),
            Attrs.METADATA_PARENT_RUN_ID: str(parent_run_id) if parent_run_id else None,
            Attrs.AZURE_RESOURCE_NAMESPACE: "Microsoft.CognitiveServices",
            Attrs.REQUEST_ENCODING_FORMATS: encoding_formats,
            Attrs.EMBEDDINGS_DIM_COUNT: dims,
        }
        if endpoint:
            attrs[Attrs.SERVER_ADDRESS] = endpoint
            port = _extract_port(endpoint)
            if port is not None:
                attrs[Attrs.SERVER_PORT] = port
        if tags:
            attrs[Attrs.METADATA_TAGS] = _safe_json(tags)
        self._core.enrich_langgraph(attrs, metadata)
        if self._core.enable_content_recording:
            inputs_json = self._core.redact_messages(
                _safe_json([{"content": i} for i in inputs])
            )
            if inputs_json is not None:
                attrs[Attrs.INPUT_MESSAGES] = inputs_json
        span_name = f"embeddings {model}" if model else "embeddings"
        self._core.start(
            run_id=run_id,
            name=span_name,
            kind=SpanKind.CLIENT,
            operation="embeddings",
            parent_run_id=parent_run_id,
            attrs=attrs,
        )

    def on_embedding_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **__: Any,
    ) -> Any:
        """Finish the embeddings span and attach usage when present."""
        usage: Dict[str, Any] = {}
        try:
            usage = getattr(response, "usage", {}) or {}
        except Exception:
            pass
        set_attrs = {}
        for key in ("input_tokens", "prompt_tokens"):
            if usage.get(key) is not None:
                set_attrs[Attrs.USAGE_INPUT_TOKENS] = usage[key]
                break
        self._core.set(run_id, set_attrs)
        self._core.end(run_id, None)

    def on_embedding_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **__: Any,
    ) -> Any:
        """Mark the embeddings span as errored and end it."""
        self._core.end(run_id, error)

    def shutdown(self) -> None:  # pragma: no cover
        """No-op shutdown hook for exporter lifecycles."""
        pass

    def force_flush(self) -> None:  # pragma: no cover
        """No-op force flush hook for exporter lifecycles."""
        pass


__all__ = ["AzureAIOpenTelemetryTracer"]

# End of file
