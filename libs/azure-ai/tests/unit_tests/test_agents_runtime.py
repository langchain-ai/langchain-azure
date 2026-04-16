"""Unit tests for langchain_azure_ai.agents.runtime."""

import asyncio
import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ---------------------------------------------------------------------------
# Stub out azure.ai.agentserver.invocations and starlette before any import
# of the runtime module so the optional-dependency guard never fires.
# ---------------------------------------------------------------------------


def _make_agentserver_stub() -> None:
    """Register minimal stubs for azure.ai.agentserver.invocations."""

    # Build the namespace hierarchy: azure -> azure.ai -> azure.ai.agentserver
    # -> azure.ai.agentserver.invocations
    for mod_name in (
        "azure.ai.agentserver",
        "azure.ai.agentserver.invocations",
    ):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    # InvocationAgentServerHost stub
    class _FakeHost:
        def __init__(self) -> None:
            self._handler: Any = None

        def invoke_handler(self, func: Any) -> Any:
            self._handler = func
            return func

        def run(self, **kwargs: Any) -> None:
            pass

    sys.modules["azure.ai.agentserver.invocations"].InvocationAgentServerHost = (  # type: ignore[attr-defined]
        _FakeHost
    )


def _make_starlette_stub() -> None:
    """Register minimal stubs for starlette.requests and starlette.responses."""
    for mod_name in ("starlette", "starlette.requests", "starlette.responses"):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    class _FakeRequest:
        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        async def json(self) -> dict[str, Any]:
            return self._body

    class _FakeJSONResponse:
        def __init__(self, content: Any) -> None:
            self.content = content

    sys.modules["starlette.requests"].Request = _FakeRequest  # type: ignore[attr-defined]
    sys.modules["starlette.responses"].Response = object  # type: ignore[attr-defined]
    sys.modules["starlette.responses"].JSONResponse = _FakeJSONResponse  # type: ignore[attr-defined]


_make_agentserver_stub()
_make_starlette_stub()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(return_value: dict[str, Any]) -> MagicMock:
    """Return a mock graph whose ainvoke returns the given value."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=return_value)
    return graph


# ---------------------------------------------------------------------------
# Tests: default_request_parser
# ---------------------------------------------------------------------------


class TestDefaultRequestParser:
    """Tests for the default_request_parser helper."""

    def test_message_key(self) -> None:
        from langchain_azure_ai.agents.runtime import default_request_parser

        result = default_request_parser({"message": "hello"})
        assert result["messages"] == [HumanMessage(content="hello")]

    def test_input_key(self) -> None:
        from langchain_azure_ai.agents.runtime import default_request_parser

        result = default_request_parser({"input": "world"})
        assert result["messages"] == [HumanMessage(content="world")]

    def test_query_key(self) -> None:
        from langchain_azure_ai.agents.runtime import default_request_parser

        result = default_request_parser({"query": "foo"})
        assert result["messages"] == [HumanMessage(content="foo")]

    def test_message_takes_priority_over_input(self) -> None:
        from langchain_azure_ai.agents.runtime import default_request_parser

        result = default_request_parser({"message": "first", "input": "second"})
        assert result["messages"] == [HumanMessage(content="first")]

    def test_empty_body_gives_empty_string(self) -> None:
        from langchain_azure_ai.agents.runtime import default_request_parser

        result = default_request_parser({})
        assert result["messages"] == [HumanMessage(content="")]


# ---------------------------------------------------------------------------
# Tests: default_response_formatter
# ---------------------------------------------------------------------------


class TestDefaultResponseFormatter:
    """Tests for the default_response_formatter helper."""

    def test_last_ai_message_returned(self) -> None:
        from langchain_azure_ai.agents.runtime import default_response_formatter

        result = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="first"),
                AIMessage(content="final"),
            ]
        }
        assert default_response_formatter(result) == "final"

    def test_dict_assistant_message(self) -> None:
        from langchain_azure_ai.agents.runtime import default_response_formatter

        result = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "response"},
            ]
        }
        assert default_response_formatter(result) == "response"

    def test_output_key_fallback(self) -> None:
        from langchain_azure_ai.agents.runtime import default_response_formatter

        assert default_response_formatter({"output": "from output"}) == "from output"

    def test_response_key_fallback(self) -> None:
        from langchain_azure_ai.agents.runtime import default_response_formatter

        assert (
            default_response_formatter({"response": "from response"}) == "from response"
        )

    def test_final_answer_key_fallback(self) -> None:
        from langchain_azure_ai.agents.runtime import default_response_formatter

        assert default_response_formatter({"final_answer": "plan"}) == "plan"

    def test_output_key_has_lower_priority_than_messages(self) -> None:
        from langchain_azure_ai.agents.runtime import default_response_formatter

        result = {
            "messages": [AIMessage(content="from messages")],
            "output": "from output",
        }
        assert default_response_formatter(result) == "from messages"

    def test_empty_result_returns_empty_string(self) -> None:
        from langchain_azure_ai.agents.runtime import default_response_formatter

        assert default_response_formatter({}) == ""


# ---------------------------------------------------------------------------
# Tests: AzureAIAgentServerRuntime
# ---------------------------------------------------------------------------


class TestAzureAIAgentServerRuntime:
    """Tests for AzureAIAgentServerRuntime."""

    def test_construction_creates_host(self) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph = _make_graph({"messages": [AIMessage(content="ok")]})
        runtime = AzureAIAgentServerRuntime(graph=graph)
        assert runtime._host is not None

    def test_handler_is_registered(self) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph = _make_graph({"messages": [AIMessage(content="ok")]})
        runtime = AzureAIAgentServerRuntime(graph=graph)
        # The stub host stores the registered handler
        assert runtime._host._handler is not None

    def test_invoke_handler_calls_graph_and_returns_response(self) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph = _make_graph({"messages": [AIMessage(content="travel plan")]})
        runtime = AzureAIAgentServerRuntime(graph=graph)

        # Simulate a request
        from starlette.requests import Request  # type: ignore[import]

        fake_request = Request({"message": "Plan a trip to Tokyo"})
        response = asyncio.run(runtime._host._handler(fake_request))

        graph.ainvoke.assert_awaited_once()
        call_args = graph.ainvoke.await_args[0][0]
        assert call_args == {"messages": [HumanMessage(content="Plan a trip to Tokyo")]}
        assert response.content == {"response": "travel plan"}

    def test_custom_request_parser_is_used(self) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph = _make_graph({"output": "custom"})

        def my_input(data: dict) -> dict:  # type: ignore[type-arg]
            return {"messages": [HumanMessage(content=data["custom_key"])]}

        runtime = AzureAIAgentServerRuntime(graph=graph, request_parser=my_input)

        from starlette.requests import Request  # type: ignore[import]

        fake_request = Request({"custom_key": "hello"})
        response = asyncio.run(runtime._host._handler(fake_request))

        call_args = graph.ainvoke.await_args[0][0]
        assert call_args == {"messages": [HumanMessage(content="hello")]}
        assert response.content == {"response": "custom"}

    def test_custom_response_formatter_is_used(self) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph = _make_graph({"my_result": "special"})

        def my_output(result: dict) -> str:  # type: ignore[type-arg]
            return result["my_result"]

        runtime = AzureAIAgentServerRuntime(graph=graph, response_formatter=my_output)

        from starlette.requests import Request  # type: ignore[import]

        fake_request = Request({"message": "hi"})
        response = asyncio.run(runtime._host._handler(fake_request))

        assert response.content == {"response": "special"}

    def test_config_is_forwarded_to_graph(self) -> None:
        from langchain_core.runnables import RunnableConfig

        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        config: RunnableConfig = {"configurable": {"thread_id": "abc"}}
        graph = _make_graph({"messages": [AIMessage(content="ok")]})
        runtime = AzureAIAgentServerRuntime(graph=graph, config=config)

        from starlette.requests import Request  # type: ignore[import]

        fake_request = Request({"message": "hi"})
        asyncio.run(runtime._host._handler(fake_request))

        graph.ainvoke.assert_awaited_once_with(
            {"messages": [HumanMessage(content="hi")]},
            config=config,
        )

    def test_run_delegates_to_host(self) -> None:
        from langchain_azure_ai.agents.runtime import AzureAIAgentServerRuntime

        graph = _make_graph({})
        runtime = AzureAIAgentServerRuntime(graph=graph)
        runtime._host.run = MagicMock()
        runtime.run(host="0.0.0.0", port=8080)
        runtime._host.run.assert_called_once_with(host="0.0.0.0", port=8080)

    def test_import_error_without_package(self) -> None:
        """Removing the stub should raise ImportError with an install hint."""

        fake_graph = _make_graph({})

        # Temporarily hide the agentserver module
        invocations_mod = sys.modules.pop("azure.ai.agentserver.invocations", None)
        try:
            with pytest.raises(ImportError, match="azure-ai-agentserver-invocations"):
                # Force re-import of _host to trigger the guard
                import importlib

                import langchain_azure_ai.agents.runtime._host as _host_mod

                importlib.reload(_host_mod)
                _host_mod.AzureAIAgentServerRuntime(graph=fake_graph)
        finally:
            if invocations_mod is not None:
                sys.modules["azure.ai.agentserver.invocations"] = invocations_mod
