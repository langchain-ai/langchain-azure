import importlib
import threading
import warnings
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from langchain_core.callbacks import BaseCallbackManager

from langchain_azure_ai._api.base import ExperimentalWarning

pytestmark = pytest.mark.filterwarnings(
    "ignore::langchain_azure_ai._api.base.ExperimentalWarning"
)

# Skip tests cleanly if required deps or the target module are not present.
pytest.importorskip("azure.monitor.opentelemetry")
pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.instrumentation")
tracing = importlib.import_module(
    "langchain_azure_ai.callbacks.tracers.inference_tracing"
)

auto_instrument = pytest.importorskip(
    "langchain_azure_ai.callbacks.tracers.auto_instrument"
)


class MockSpan:
    def __init__(self, name: str, attributes: dict[str, object] | None = None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})
        self.events: list[tuple[str, dict[str, object]]] = []
        self.ended = False
        self.status: object = None
        self.exceptions: list[Exception] = []
        self._context = SimpleNamespace(is_valid=True)

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, object] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: object) -> None:
        self.status = status

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)

    def end(self) -> None:
        self.ended = True

    def get_span_context(self) -> object:
        return self._context

    def update_name(self, name: str) -> None:
        self.name = name


class MockTracer:
    def __init__(self) -> None:
        self.spans: list[MockSpan] = []

    def start_span(
        self,
        name: str,
        kind: object = None,
        context: object = None,
        attributes: dict[str, object] | None = None,
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
    monkeypatch.setattr(tracing, "configure_azure_monitor", lambda **kwargs: None)


@pytest.fixture(autouse=True)
def cleanup_auto_tracing() -> Iterator[None]:
    auto_instrument.disable_auto_tracing()
    yield
    auto_instrument.disable_auto_tracing()


def _get_inheritable_tracers(
    manager: BaseCallbackManager,
) -> list[Any]:
    return [
        handler
        for handler in manager.inheritable_handlers
        if isinstance(handler, tracing.AzureAIOpenTelemetryTracer)
    ]


def _get_all_tracers(
    manager: BaseCallbackManager,
) -> list[Any]:
    return [
        handler
        for handler in [*manager.handlers, *manager.inheritable_handlers]
        if isinstance(handler, tracing.AzureAIOpenTelemetryTracer)
    ]


def test_enable_auto_tracing_patches_callback_manager() -> None:
    auto_instrument.enable_auto_tracing()

    manager = BaseCallbackManager(handlers=[])

    inheritable_tracers = _get_inheritable_tracers(manager)
    assert len(inheritable_tracers) == 1


def test_disable_auto_tracing_restores_original() -> None:
    auto_instrument.enable_auto_tracing()
    auto_instrument.disable_auto_tracing()

    manager = BaseCallbackManager(handlers=[])

    assert _get_inheritable_tracers(manager) == []


def test_disable_auto_tracing_emits_experimental_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ExperimentalWarning)
        auto_instrument.disable_auto_tracing()

    experimental_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, ExperimentalWarning)
    ]
    assert experimental_warnings
    assert (
        "disable_auto_tracing is currently in preview and is subject to change"
        in str(experimental_warnings[0].message)
    )


def test_deduplication_no_double_injection() -> None:
    auto_instrument.enable_auto_tracing()

    first_manager = BaseCallbackManager(handlers=[])
    second_manager = BaseCallbackManager(handlers=[])

    assert len(_get_inheritable_tracers(first_manager)) == 1
    assert len(_get_inheritable_tracers(second_manager)) == 1


def test_manual_and_auto_coexist() -> None:
    auto_instrument.enable_auto_tracing()
    manual_tracer = tracing.AzureAIOpenTelemetryTracer()

    # When manual tracer is added as inheritable, auto should not double-inject
    manager = BaseCallbackManager(handlers=[], inheritable_handlers=[manual_tracer])

    inheritable_tracers = _get_inheritable_tracers(manager)
    assert len(inheritable_tracers) == 1
    assert inheritable_tracers[0] is manual_tracer


def test_manual_non_inheritable_tracer_is_not_duplicated() -> None:
    auto_instrument.enable_auto_tracing()
    manual_tracer = tracing.AzureAIOpenTelemetryTracer()

    manager = BaseCallbackManager(handlers=[manual_tracer])

    all_tracers = _get_all_tracers(manager)
    assert len(all_tracers) == 1
    assert all_tracers[0] is manual_tracer


def test_is_auto_tracing_enabled() -> None:
    assert auto_instrument.is_auto_tracing_enabled() is False

    auto_instrument.enable_auto_tracing()
    assert auto_instrument.is_auto_tracing_enabled() is True

    auto_instrument.disable_auto_tracing()
    assert auto_instrument.is_auto_tracing_enabled() is False


def test_env_var_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED", "false")
    monkeypatch.setenv("AZURE_TRACING_PROVIDER_NAME", "test-provider")
    monkeypatch.setenv("AZURE_TRACING_AGENT_ID", "test-agent")

    auto_instrument.enable_auto_tracing()
    manager = BaseCallbackManager(handlers=[])

    tracer = _get_inheritable_tracers(manager)[0]
    assert tracer._content_recording is False  # type: ignore[attr-defined]
    assert tracer._default_provider_name == "test-provider"  # type: ignore[attr-defined]
    assert tracer._default_agent_id == "test-agent"  # type: ignore[attr-defined]


def test_env_bool_accepts_on_off_and_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_BOOL_TRUE", " On ")
    monkeypatch.setenv("AUTO_BOOL_FALSE", " off ")
    monkeypatch.setenv("AUTO_BOOL_DEFAULT", " maybe ")

    assert auto_instrument._env_bool("AUTO_BOOL_TRUE", False) is True
    assert auto_instrument._env_bool("AUTO_BOOL_FALSE", True) is False
    assert auto_instrument._env_bool("AUTO_BOOL_DEFAULT", True) is True


def test_instrumentor_instrument_and_uninstrument() -> None:
    instrumentor = auto_instrument.AzureAIOpenTelemetryInstrumentor()

    instrumentor.instrument()
    assert auto_instrument.is_auto_tracing_enabled() is True

    instrumentor.uninstrument()
    assert auto_instrument.is_auto_tracing_enabled() is False


def test_custom_tracer_instance() -> None:
    custom_tracer = tracing.AzureAIOpenTelemetryTracer(provider_name="custom-provider")

    auto_instrument.enable_auto_tracing(tracer=custom_tracer)
    manager = BaseCallbackManager(handlers=[])

    assert custom_tracer in manager.inheritable_handlers


def test_enable_auto_tracing_serializes_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTracer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    wrap_started = threading.Event()
    allow_wrap_finish = threading.Event()
    wrap_calls = 0

    def fake_wrap_function_wrapper(module: str, name: str, wrapper: object) -> None:
        del wrapper
        nonlocal wrap_calls
        assert module == "langchain_core.callbacks.base"
        assert name == "BaseCallbackManager.__init__"
        wrap_started.set()
        assert allow_wrap_finish.wait(timeout=2)
        wrap_calls += 1

    monkeypatch.setattr(auto_instrument, "_active_tracer", None)
    monkeypatch.setattr(
        auto_instrument, "_ensure_otel_instrumentation_available", lambda: None
    )
    monkeypatch.setattr(auto_instrument, "_ensure_wrapt_available", lambda: None)
    monkeypatch.setattr(
        auto_instrument, "wrap_function_wrapper", fake_wrap_function_wrapper
    )
    monkeypatch.setattr(auto_instrument, "_load_tracer_class", lambda: DummyTracer)

    first_thread = threading.Thread(target=auto_instrument.enable_auto_tracing)
    second_thread = threading.Thread(target=auto_instrument.enable_auto_tracing)

    first_thread.start()
    assert wrap_started.wait(timeout=2)
    second_thread.start()
    allow_wrap_finish.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert wrap_calls == 1
    assert auto_instrument.is_auto_tracing_enabled() is True
    monkeypatch.setattr(auto_instrument, "_active_tracer", None)


def test_disable_auto_tracing_uses_matching_unwrap_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyCallbackManager:
        pass

    captured: dict[str, object] = {}

    def fake_unwrap(module: object, name: str) -> None:
        captured["module"] = module
        captured["name"] = name

    monkeypatch.setattr(auto_instrument, "_active_tracer", object())
    monkeypatch.setattr(
        auto_instrument, "_ensure_otel_instrumentation_available", lambda: None
    )
    monkeypatch.setattr(
        auto_instrument,
        "_load_callback_manager_class",
        lambda: DummyCallbackManager,
    )
    monkeypatch.setattr(auto_instrument, "unwrap", fake_unwrap)

    auto_instrument.disable_auto_tracing()

    assert captured == {
        "module": DummyCallbackManager,
        "name": "__init__",
    }
    assert auto_instrument.is_auto_tracing_enabled() is False
