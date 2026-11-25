"""Agent factories load OpenTelemetry only when tracing is requested."""

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

FACTORY_MODULES = [
    "langchain_azure_ai.agents._v1.agent_service",
    "langchain_azure_ai.agents",
]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module_name", FACTORY_MODULES, ids=["v1", "v2"])
@pytest.mark.parametrize("trace", [False, True])
def test_factory_without_tracing_dependencies(module_name: str, trace: bool) -> None:
    if "._v1." in module_name:
        pytest.importorskip("azure.ai.agents")

    source = r"""
        import importlib
        import sys
        from importlib.abc import MetaPathFinder
        from unittest.mock import patch

        class BlockTracingImports(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "opentelemetry" or fullname.startswith(
                    ("opentelemetry.", "azure.monitor")
                ):
                    raise ModuleNotFoundError(fullname, name=fullname)

        sys.meta_path.insert(0, BlockTracingImports())

        import pytest
        from langchain_core.messages import AIMessage, HumanMessage
        from langchain_core.runnables import RunnableLambda

        factory_class = importlib.import_module(sys.argv[1]).AgentServiceFactory
        trace = sys.argv[2] == "True"
        factory = factory_class(project_endpoint="https://test.endpoint.com")
        node = RunnableLambda(lambda state: {"messages": [AIMessage(content="hello")]})
        node.agent_id = "test-agent:1"
        node.get_metadata = lambda: {"agent_id": node.agent_id}

        with patch.object(
            factory_class, "create_prompt_agent_node", return_value=node
        ) as create_node:
            if trace:
                with pytest.raises(
                    ImportError, match=r"langchain-azure-ai\[opentelemetry\]"
                ):
                    factory.create_prompt_agent(
                        name="test-agent", model="test-model", trace=True
                    )
                create_node.assert_not_called()
            else:
                graph = factory.create_prompt_agent(
                    name="test-agent", model="test-model", trace=False
                )
                result = graph.invoke({"messages": [HumanMessage(content="hi")]})
                assert result["messages"][-1].content == "hello"

        tracer_module = "langchain_azure_ai.callbacks.tracers.inference_tracing"
        assert tracer_module not in sys.modules
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), module_name, str(trace)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=PACKAGE_ROOT,
        env={**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module_name", FACTORY_MODULES, ids=["v1", "v2"])
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_factory_tracing_records_spans(
    module_name: str, use_async: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if "._v1." in module_name:
        pytest.importorskip("azure.ai.agents")
    pytest.importorskip("azure.monitor.opentelemetry")
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    monkeypatch.setenv("OTEL_AUTO_CONFIGURE_AZURE_MONITOR", "false")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel_trace, "get_tracer", provider.get_tracer)
    factory_class = importlib.import_module(module_name).AgentServiceFactory
    factory = factory_class(project_endpoint="https://test.endpoint.com")
    node: RunnableLambda = RunnableLambda(
        lambda state: {"messages": [AIMessage(content="hello")]}
    )
    setattr(node, "agent_id", "test-agent:1")
    setattr(node, "get_metadata", lambda: {"agent_id": "test-agent:1"})

    try:
        with patch.object(factory_class, "create_prompt_agent_node", return_value=node):
            graph = factory.create_prompt_agent(
                name="test-agent", model="test-model", trace=True
            )
        state = {"messages": [HumanMessage(content="hi")]}
        result = await graph.ainvoke(state) if use_async else graph.invoke(state)
        assert result["messages"][-1].content == "hello"
        spans = exporter.get_finished_spans()
        assert spans
        assert any("hello" in str(span.attributes) for span in spans)
    finally:
        provider.shutdown()
