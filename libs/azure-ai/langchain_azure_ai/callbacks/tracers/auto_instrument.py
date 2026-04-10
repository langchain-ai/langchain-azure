"""Auto-instrumentation helpers for LangChain/LangGraph tracing."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any, Callable

from langchain_azure_ai._api.base import experimental

if TYPE_CHECKING:
    from langchain_azure_ai.callbacks.tracers.inference_tracing import (
        AzureAIOpenTelemetryTracer,
    )

try:
    from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
        BaseInstrumentor,
    )
    from opentelemetry.instrumentation.utils import unwrap
except ImportError as exc:
    _OTEL_INSTRUMENTATION_IMPORT_ERROR: Exception | None = exc

    class BaseInstrumentor:  # type: ignore[no-redef]
        """Fallback base class when opentelemetry instrumentation is unavailable."""

        def instrument(self, **kwargs: Any) -> None:
            """Raise a clear error when OTel instrumentation is not installed."""
            raise ImportError(
                "Azure auto tracing requires 'opentelemetry-instrumentation'. "
                "Install it via: pip install opentelemetry-instrumentation"
            ) from _OTEL_INSTRUMENTATION_IMPORT_ERROR

        def uninstrument(self, **kwargs: Any) -> None:
            """Raise a clear error when OTel instrumentation is not installed."""
            raise ImportError(
                "Azure auto tracing requires 'opentelemetry-instrumentation'. "
                "Install it via: pip install opentelemetry-instrumentation"
            ) from _OTEL_INSTRUMENTATION_IMPORT_ERROR

    def unwrap(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        """Raise a clear error when OTel instrumentation is not installed."""
        raise ImportError(
            "Azure auto tracing requires 'opentelemetry-instrumentation'. "
            "Install it via: pip install opentelemetry-instrumentation"
        ) from _OTEL_INSTRUMENTATION_IMPORT_ERROR

else:
    _OTEL_INSTRUMENTATION_IMPORT_ERROR = None

try:
    from wrapt import wrap_function_wrapper
except ImportError as exc:
    wrap_function_wrapper = None
    _WRAPT_IMPORT_ERROR: Exception | None = exc
else:
    _WRAPT_IMPORT_ERROR = None

_ENV_CONNECTION_STRING = "APPLICATION_INSIGHTS_CONNECTION_STRING"
_ENV_ENABLE_CONTENT_RECORDING = "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED"
_ENV_PROJECT_ENDPOINT = "AZURE_AI_PROJECT_ENDPOINT"
_ENV_PROVIDER_NAME = "AZURE_TRACING_PROVIDER_NAME"
_ENV_AGENT_ID = "AZURE_TRACING_AGENT_ID"
_ENV_TRACE_ALL_LANGGRAPH_NODES = "AZURE_TRACING_ALL_LANGGRAPH_NODES"
_ENV_MESSAGE_KEYS = "OTEL_MESSAGE_KEYS"
_ENV_MESSAGE_PATHS = "OTEL_MESSAGE_PATHS"
_ENV_AUTO_CONFIGURE_AZURE_MONITOR = "OTEL_AUTO_CONFIGURE_AZURE_MONITOR"

_BASE_CALLBACK_MANAGER_MODULE = "langchain_core.callbacks.base"
_BASE_CALLBACK_MANAGER_INIT = "BaseCallbackManager.__init__"
_BASE_CALLBACK_MANAGER_INIT_ATTR = "__init__"
_AUTO_TRACING_LOCK = threading.Lock()
_active_tracer: AzureAIOpenTelemetryTracer | None = None


class _BaseCallbackManagerInitWrapper:
    """Inject a tracer into inheritable handlers after callback manager init."""

    def __init__(self, tracer: AzureAIOpenTelemetryTracer) -> None:
        self._tracer = tracer
        self._tracer_type = type(tracer)

    def _has_existing_tracer(self, instance: Any) -> bool:
        handlers = (
            *getattr(instance, "handlers", ()),
            *getattr(instance, "inheritable_handlers", ()),
        )
        return any(
            handler is self._tracer or isinstance(handler, self._tracer_type)
            for handler in handlers
        )

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        result = wrapped(*args, **kwargs)
        if not self._has_existing_tracer(instance):
            instance.add_handler(self._tracer, True)
        return result


def _env_bool(key: str, default: bool) -> bool:
    """Read a boolean value from an environment variable."""
    val = os.getenv(key, "").strip().lower()
    if val in {"1", "true", "yes", "on"}:
        return True
    if val in {"0", "false", "no", "off"}:
        return False
    return default


def _ensure_wrapt_available() -> None:
    """Ensure wrapt is installed before patching callbacks."""
    if wrap_function_wrapper is None:
        raise ImportError(
            "Azure auto tracing requires 'wrapt'. Install it via: pip install wrapt"
        ) from _WRAPT_IMPORT_ERROR


def _ensure_otel_instrumentation_available() -> None:
    """Ensure OpenTelemetry instrumentation package is installed."""
    if _OTEL_INSTRUMENTATION_IMPORT_ERROR is not None:
        raise ImportError(
            "Azure auto tracing requires 'opentelemetry-instrumentation'. "
            "Install it via: pip install opentelemetry-instrumentation"
        ) from _OTEL_INSTRUMENTATION_IMPORT_ERROR


def _load_tracer_class() -> type[AzureAIOpenTelemetryTracer]:
    """Load Azure tracer class lazily with a clear dependency error."""
    try:
        from langchain_azure_ai.callbacks.tracers.inference_tracing import (
            AzureAIOpenTelemetryTracer,
        )
    except ImportError as exc:
        raise ImportError(
            "Azure auto tracing requires the tracing dependencies from "
            "'langchain-azure-ai[opentelemetry]'. Install them via:\n"
            "    pip install 'langchain-azure-ai[opentelemetry]'"
        ) from exc
    return AzureAIOpenTelemetryTracer


def _load_callback_manager_class() -> type[Any]:
    """Load the BaseCallbackManager class for precise unpatching."""
    from langchain_core.callbacks.base import BaseCallbackManager

    return BaseCallbackManager


@experimental()
def enable_auto_tracing(
    *,
    connection_string: str | None = None,
    enable_content_recording: bool | None = None,
    project_endpoint: str | None = None,
    credential: Any | None = None,
    provider_name: str | None = None,
    agent_id: str | None = None,
    trace_all_langgraph_nodes: bool | None = None,
    message_keys: list[str] | tuple[str, ...] | None = None,
    message_paths: list[str] | tuple[str, ...] | None = None,
    auto_configure_azure_monitor: bool | None = None,
    trace_state: bool | None = None,
    max_state_size: int | None = None,
    tracer: AzureAIOpenTelemetryTracer | None = None,
) -> None:
    """Enable auto-injection of Azure tracer into callback managers.

    When called, every new ``BaseCallbackManager`` instance created by
    LangChain/LangGraph will automatically include an
    ``AzureAIOpenTelemetryTracer`` in its inheritable handlers.

    Each parameter falls back to an environment variable when not supplied:

    * *connection_string* ← ``APPLICATION_INSIGHTS_CONNECTION_STRING``
    * *enable_content_recording* ← ``AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED``
    * *project_endpoint* ← ``AZURE_AI_PROJECT_ENDPOINT``
    * *provider_name* ← ``AZURE_TRACING_PROVIDER_NAME``
    * *agent_id* ← ``AZURE_TRACING_AGENT_ID``
    * *trace_all_langgraph_nodes* ← ``AZURE_TRACING_ALL_LANGGRAPH_NODES``
    * *message_keys* ← ``OTEL_MESSAGE_KEYS`` (comma-separated)
    * *message_paths* ← ``OTEL_MESSAGE_PATHS`` (comma-separated)
    * *auto_configure_azure_monitor* ← ``OTEL_AUTO_CONFIGURE_AZURE_MONITOR``
    * *trace_state* ← ``OTEL_TRACE_LANGGRAPH_STATE``
    * *max_state_size* ← ``OTEL_MAX_STATE_SIZE``

    Args:
        connection_string: Application Insights connection string.
        enable_content_recording: Whether to capture message/content payloads.
        project_endpoint: Azure AI project endpoint for connection string
            resolution.
        credential: Azure credential used with project endpoint resolution.
        provider_name: Default provider name for emitted GenAI spans.
        agent_id: Default agent identifier for emitted spans.
        trace_all_langgraph_nodes: Whether to trace all LangGraph nodes
            (default ``True``).
        message_keys: State keys that hold messages (e.g. ``["messages"]``).
        message_paths: Dotted paths for nested message locations.
        auto_configure_azure_monitor: Set to ``False`` to skip automatic
            Azure Monitor configuration.
        trace_state: Whether to capture the full LangGraph state on each
            agent node span (default ``False``).
        max_state_size: Maximum character length for serialized state
            (default ``32768``).
        tracer: Pre-built tracer to inject directly.  When supplied, all
            other configuration arguments are ignored.
    """
    global _active_tracer

    with _AUTO_TRACING_LOCK:
        if _active_tracer is not None:
            return

        _ensure_otel_instrumentation_available()
        _ensure_wrapt_available()
        assert wrap_function_wrapper is not None

        if tracer is None:
            tracer_class = _load_tracer_class()
            resolved_connection_string = connection_string or os.getenv(
                _ENV_CONNECTION_STRING
            )
            resolved_enable_content_recording = (
                enable_content_recording
                if enable_content_recording is not None
                else _env_bool(_ENV_ENABLE_CONTENT_RECORDING, True)
            )
            resolved_project_endpoint = project_endpoint or os.getenv(
                _ENV_PROJECT_ENDPOINT
            )
            resolved_provider_name = provider_name or os.getenv(_ENV_PROVIDER_NAME)
            resolved_agent_id = agent_id or os.getenv(_ENV_AGENT_ID)
            resolved_trace_all_langgraph_nodes = (
                trace_all_langgraph_nodes
                if trace_all_langgraph_nodes is not None
                else _env_bool(_ENV_TRACE_ALL_LANGGRAPH_NODES, True)
            )
            resolved_auto_configure = (
                auto_configure_azure_monitor
                if auto_configure_azure_monitor is not None
                else _env_bool(_ENV_AUTO_CONFIGURE_AZURE_MONITOR, True)
            )

            # Derive tracer name from agent_id or OTEL_SERVICE_NAME so that
            # _resolve_agent_name treats it as a known generic marker and
            # falls through to per-node langgraph_node names.
            resolved_name = resolved_agent_id or os.getenv("OTEL_SERVICE_NAME")

            tracer_kwargs: dict[str, Any] = {
                "connection_string": resolved_connection_string,
                "enable_content_recording": resolved_enable_content_recording,
                "project_endpoint": resolved_project_endpoint,
                "credential": credential,
                "provider_name": resolved_provider_name,
                "agent_id": resolved_agent_id,
                "trace_all_langgraph_nodes": resolved_trace_all_langgraph_nodes,
                "auto_configure_azure_monitor": resolved_auto_configure,
            }
            if resolved_name:
                tracer_kwargs["name"] = resolved_name
            if message_keys is not None:
                tracer_kwargs["message_keys"] = message_keys
            if message_paths is not None:
                tracer_kwargs["message_paths"] = message_paths
            if trace_state is not None:
                tracer_kwargs["trace_state"] = trace_state
            if max_state_size is not None:
                tracer_kwargs["max_state_size"] = max_state_size

            tracer = tracer_class(**tracer_kwargs)

        wrap_function_wrapper(
            _BASE_CALLBACK_MANAGER_MODULE,
            _BASE_CALLBACK_MANAGER_INIT,
            _BaseCallbackManagerInitWrapper(tracer),
        )
        _active_tracer = tracer


@experimental()
def disable_auto_tracing() -> None:
    """Disable callback manager auto-tracing and restore original behavior."""
    global _active_tracer

    with _AUTO_TRACING_LOCK:
        if _active_tracer is None:
            return

        _ensure_otel_instrumentation_available()
        unwrap(_load_callback_manager_class(), _BASE_CALLBACK_MANAGER_INIT_ATTR)
        _active_tracer = None


def is_auto_tracing_enabled() -> bool:
    """Return whether auto-tracing monkey patch is currently enabled."""
    return _active_tracer is not None


@experimental()
class AzureAIOpenTelemetryInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor implementation for LangChain auto-tracing."""

    def instrumentation_dependencies(self) -> tuple[str, ...]:
        """Return package dependency constraints for this instrumentor."""
        return ("langchain-core > 0.1.0",)

    def _instrument(self, **kwargs: Any) -> None:
        """Enable LangChain callback auto-tracing."""
        enable_auto_tracing(**kwargs)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable LangChain callback auto-tracing."""
        del kwargs
        disable_auto_tracing()


__all__ = [
    "AzureAIOpenTelemetryInstrumentor",
    "disable_auto_tracing",
    "enable_auto_tracing",
    "is_auto_tracing_enabled",
]
