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


def test_enable_auto_tracing_defaults_for_hosted_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_var in (
        "APPLICATION_INSIGHTS_CONNECTION_STRING",
        "AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED",
        "AZURE_TRACING_PROVIDER_NAME",
        "AZURE_TRACING_AGENT_ID",
        "AZURE_TRACING_ALL_LANGGRAPH_NODES",
        "OTEL_MESSAGE_KEYS",
        "OTEL_AUTO_CONFIGURE_AZURE_MONITOR",
    ):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv(
        "APPLICATION_INSIGHTS_CONNECTION_STRING",
        "InstrumentationKey=00000000-0000-0000-0000-000000000000",
    )

    configure_calls: list[str] = []

    def fake_configure(cls: type[Any], connection_string: str) -> None:
        del cls
        configure_calls.append(connection_string)

    monkeypatch.setattr(
        tracing.AzureAIOpenTelemetryTracer,
        "_configure_azure_monitor",
        classmethod(fake_configure),
    )

    auto_instrument.enable_auto_tracing()
    manager = BaseCallbackManager(handlers=[])

    tracer = _get_inheritable_tracers(manager)[0]
    assert tracer._content_recording is False  # type: ignore[attr-defined]
    assert tracer._default_provider_name == "azure.ai.openai"  # type: ignore[attr-defined]
    assert tracer._message_keys == ("messages",)  # type: ignore[attr-defined]
    assert tracer._trace_all_langgraph_nodes is True  # type: ignore[attr-defined]
    assert configure_calls == []


def test_enable_auto_tracing_normalizes_azure_provider_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_TRACING_PROVIDER_NAME", " azure_openai ")

    auto_instrument.enable_auto_tracing()
    manager = BaseCallbackManager(handlers=[])

    tracer = _get_inheritable_tracers(manager)[0]
    assert tracer._default_provider_name == "azure.ai.openai"  # type: ignore[attr-defined]


def test_callback_manager_factory_wrapper_injects_tracer() -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer(provider_name="test-provider")
    wrapper = auto_instrument._CallbackManagerFactoryWrapper(tracer)
    manager = BaseCallbackManager(handlers=[])

    returned = wrapper(lambda *args, **kwargs: manager, None, (), {})

    assert returned is manager
    assert _get_inheritable_tracers(manager) == [tracer]


def test_inject_tracer_fallback_on_add_handler_rejection() -> None:
    """When add_handler rejects the tracer (TypeError), fall back to list append."""
    tracer = tracing.AzureAIOpenTelemetryTracer(provider_name="test-provider")
    injector = auto_instrument._CallbackManagerInjector(tracer)

    class StrictManager:
        def __init__(self) -> None:
            self.handlers: list[Any] = []
            self.inheritable_handlers: list[Any] = []

        def add_handler(self, handler: Any, inherit: bool = False) -> None:
            raise TypeError("handler must be a GraphCallbackHandler")

    manager = StrictManager()
    injector._inject_tracer(manager)
    assert tracer in manager.inheritable_handlers


def test_inject_tracer_reraises_unrelated_type_error() -> None:
    """TypeError not related to handler rejection should propagate."""
    tracer = tracing.AzureAIOpenTelemetryTracer(provider_name="test-provider")
    injector = auto_instrument._CallbackManagerInjector(tracer)

    class BrokenManager:
        def __init__(self) -> None:
            self.handlers: list[Any] = []
            self.inheritable_handlers: list[Any] = []

        def add_handler(self, handler: Any, inherit: bool = False) -> None:
            raise TypeError("unexpected argument 'foo'")

    manager = BrokenManager()
    with pytest.raises(TypeError, match="unexpected argument"):
        injector._inject_tracer(manager)


def test_tracer_is_instance_of_base_callback_handler() -> None:
    """AzureAIOpenTelemetryTracer must inherit from BaseCallbackHandler."""
    from langchain_core.callbacks.base import BaseCallbackHandler

    tracer = tracing.AzureAIOpenTelemetryTracer()
    assert isinstance(tracer, BaseCallbackHandler)


def test_patch_langgraph_callback_manager_helpers_wraps_async_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer(provider_name="test-provider")
    wrap_calls: list[tuple[str, str]] = []

    targets = (
        ("langgraph.fake_config", "get_callback_manager_for_config"),
        ("langgraph.fake_config", "get_async_callback_manager_for_config"),
        ("langgraph.fake_runnable", "get_callback_manager_for_config"),
        ("langgraph.fake_runnable", "get_async_callback_manager_for_config"),
    )

    def fake_load_optional_module(module_name: str) -> object:
        return SimpleNamespace(
            get_callback_manager_for_config=object(),
            get_async_callback_manager_for_config=object(),
        )

    def fake_wrap_function_wrapper(module: str, name: str, wrapper: object) -> None:
        del wrapper
        wrap_calls.append((module, name))

    monkeypatch.setattr(auto_instrument, "_LANGGRAPH_CALLBACK_MANAGER_TARGETS", targets)
    monkeypatch.setattr(
        auto_instrument, "_load_optional_module", fake_load_optional_module
    )
    monkeypatch.setattr(
        auto_instrument, "wrap_function_wrapper", fake_wrap_function_wrapper
    )
    auto_instrument._patched_langgraph_targets.clear()

    auto_instrument._patch_langgraph_callback_manager_helpers(tracer)

    assert set(wrap_calls) == set(targets)
    assert auto_instrument._patched_langgraph_targets == list(targets)
    auto_instrument._patched_langgraph_targets.clear()


def test_patch_langgraph_callback_manager_helpers_skips_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = tracing.AzureAIOpenTelemetryTracer(provider_name="test-provider")
    wrap_calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        auto_instrument,
        "_LANGGRAPH_CALLBACK_MANAGER_TARGETS",
        (
            ("langgraph.good", "get_async_callback_manager_for_config"),
            ("langgraph.bad", "get_async_callback_manager_for_config"),
        ),
    )

    def fake_load_optional_module(module_name: str) -> object:
        if module_name == "langgraph.bad":
            raise ImportError("missing optional langgraph dependency")
        return SimpleNamespace(get_async_callback_manager_for_config=object())

    def fake_wrap_function_wrapper(module: str, name: str, wrapper: object) -> None:
        del wrapper
        wrap_calls.append((module, name))

    monkeypatch.setattr(
        auto_instrument, "_load_optional_module", fake_load_optional_module
    )
    monkeypatch.setattr(
        auto_instrument, "wrap_function_wrapper", fake_wrap_function_wrapper
    )
    auto_instrument._patched_langgraph_targets.clear()

    auto_instrument._patch_langgraph_callback_manager_helpers(tracer)

    assert wrap_calls == [("langgraph.good", "get_async_callback_manager_for_config")]
    assert auto_instrument._patched_langgraph_targets == [
        ("langgraph.good", "get_async_callback_manager_for_config")
    ]
    auto_instrument._patched_langgraph_targets.clear()


def test_unpatch_langgraph_callback_manager_helpers_skips_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unwrap_calls: list[tuple[object, str]] = []
    good_module = SimpleNamespace(get_async_callback_manager_for_config=object())

    auto_instrument._patched_langgraph_targets[:] = [
        ("langgraph.good", "get_async_callback_manager_for_config"),
        ("langgraph.bad", "get_async_callback_manager_for_config"),
    ]

    def fake_load_optional_module(module_name: str) -> object:
        if module_name == "langgraph.bad":
            raise ImportError("missing optional langgraph dependency")
        return good_module

    def fake_unwrap(module: object, name: str) -> None:
        unwrap_calls.append((module, name))

    monkeypatch.setattr(
        auto_instrument, "_load_optional_module", fake_load_optional_module
    )
    monkeypatch.setattr(auto_instrument, "unwrap", fake_unwrap)

    auto_instrument._unpatch_langgraph_callback_manager_helpers()

    assert unwrap_calls == [(good_module, "get_async_callback_manager_for_config")]
    assert auto_instrument._patched_langgraph_targets == []


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

    patch_started = threading.Event()
    allow_patch_finish = threading.Event()
    patch_calls = {"base": 0, "langgraph": 0}

    def fake_patch_base_callback_manager(tracer: object) -> None:
        del tracer
        patch_started.set()
        assert allow_patch_finish.wait(timeout=2)
        patch_calls["base"] += 1

    def fake_patch_langgraph_callback_manager_helpers(tracer: object) -> None:
        del tracer
        patch_calls["langgraph"] += 1

    monkeypatch.setattr(auto_instrument, "_active_tracer", None)
    monkeypatch.setattr(
        auto_instrument, "_ensure_otel_instrumentation_available", lambda: None
    )
    monkeypatch.setattr(auto_instrument, "_ensure_wrapt_available", lambda: None)
    monkeypatch.setattr(
        auto_instrument,
        "_patch_base_callback_manager",
        fake_patch_base_callback_manager,
    )
    monkeypatch.setattr(
        auto_instrument,
        "_patch_langgraph_callback_manager_helpers",
        fake_patch_langgraph_callback_manager_helpers,
    )
    monkeypatch.setattr(auto_instrument, "_load_tracer_class", lambda: DummyTracer)

    first_thread = threading.Thread(target=auto_instrument.enable_auto_tracing)
    second_thread = threading.Thread(target=auto_instrument.enable_auto_tracing)

    first_thread.start()
    assert patch_started.wait(timeout=2)
    second_thread.start()
    allow_patch_finish.set()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert patch_calls == {"base": 1, "langgraph": 1}
    assert auto_instrument.is_auto_tracing_enabled() is True
    monkeypatch.setattr(auto_instrument, "_active_tracer", None)


def test_enable_auto_tracing_rolls_back_patches_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyTracer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    cleanup_calls: list[str] = []

    monkeypatch.setattr(auto_instrument, "_active_tracer", None)
    monkeypatch.setattr(
        auto_instrument, "_ensure_otel_instrumentation_available", lambda: None
    )
    monkeypatch.setattr(auto_instrument, "_ensure_wrapt_available", lambda: None)
    monkeypatch.setattr(auto_instrument, "_load_tracer_class", lambda: DummyTracer)
    monkeypatch.setattr(
        auto_instrument, "_patch_base_callback_manager", lambda tracer: None
    )

    def fail_patch_langgraph_callback_manager_helpers(tracer: object) -> None:
        del tracer
        raise RuntimeError("langgraph patch failed")

    monkeypatch.setattr(
        auto_instrument,
        "_patch_langgraph_callback_manager_helpers",
        fail_patch_langgraph_callback_manager_helpers,
    )
    monkeypatch.setattr(
        auto_instrument,
        "_unpatch_base_callback_manager",
        lambda: cleanup_calls.append("base"),
    )
    monkeypatch.setattr(
        auto_instrument,
        "_unpatch_langgraph_callback_manager_helpers",
        lambda: cleanup_calls.append("langgraph"),
    )

    with pytest.raises(RuntimeError, match="langgraph patch failed"):
        auto_instrument.enable_auto_tracing()

    assert cleanup_calls == ["langgraph", "base"]
    assert auto_instrument.is_auto_tracing_enabled() is False


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
