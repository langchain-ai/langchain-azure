# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for Azure AI invoke runtime helpers."""

import json
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request

from langchain_azure_ai.agents.runtime._invoke_host import JSONValue


def _make_invoke_request(payload: Any = None) -> Request:
    """Create a Starlette request with a JSON body for invoke-host tests."""
    body = b""
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/invocations",
        "headers": [(b"content-type", b"application/json")],
        "query_string": b"",
    }
    request = Request(scope, receive)
    request.state.session_id = "session-123"
    request.state.invocation_id = "invocation-123"
    return request


def _make_invoke_host(graph: Any = None, **kwargs: Any) -> Any:
    """Instantiate AzureAIInvokeAgentHost with SDK imports mocked out."""
    if graph is None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"ok": True})

    mock_host_cls = MagicMock()
    mock_host_instance = MagicMock()
    mock_host_cls.return_value = mock_host_instance

    with patch(
        "langchain_azure_ai.agents.runtime._invoke_host.InvocationAgentServerHost",
        mock_host_cls,
    ):
        from langchain_azure_ai.agents.runtime._invoke_host import (
            AzureAIInvokeAgentHost,
        )

        host = AzureAIInvokeAgentHost(graph=graph, **kwargs)
        host._app = mock_host_instance
        return host


class TestInvokeInputParser:
    """Tests for the default invocation input parser."""

    async def test_accepts_json_object(self) -> None:
        from langchain_azure_ai.agents.runtime._invoke_host import (
            GraphInvocationInput,
            invoke_input_parser,
        )

        request = _make_invoke_request({"message": "hello"})
        result: GraphInvocationInput[Any, Any] = await invoke_input_parser(request)
        assert result == GraphInvocationInput(
            input={"message": "hello"},
            context=None,
            config={"configurable": {"thread_id": "session-123"}},
        )

    async def test_logs_parsed_payload(self, caplog: pytest.LogCaptureFixture) -> None:
        from langchain_azure_ai.agents.runtime._invoke_host import invoke_input_parser

        caplog.set_level(logging.DEBUG, logger="langchain_azure_ai.agents.runtime")
        await invoke_input_parser(_make_invoke_request({"message": "hello"}))

        assert "Parsing invoke request" in caplog.text
        assert "Parsed invoke request payload" in caplog.text

    async def test_rejects_non_object_payload(self) -> None:
        from langchain_azure_ai.agents.runtime._invoke_host import invoke_input_parser

        request = _make_invoke_request(["hello"])
        with pytest.raises(ValueError, match="JSON object"):
            await invoke_input_parser(request)


class TestInvokeOutputParser:
    """Tests for the default invocation output parser."""

    def test_passes_through_jsonable_dict(self) -> None:
        from langchain_azure_ai.agents.runtime._invoke_host import (
            invoke_output_parser,
        )

        payload = cast(dict[str, JSONValue], {"answer": "ok", "count": 2})
        result = invoke_output_parser(payload, _make_invoke_request())
        assert result == {"answer": "ok", "count": 2}

    def test_extracts_last_message_content(self) -> None:
        from langchain_azure_ai.agents.runtime._invoke_host import (
            invoke_output_parser,
        )

        payload = cast(dict[str, JSONValue], {"messages": ["hi", "done"]})
        result = invoke_output_parser(payload, _make_invoke_request())
        assert result == {"messages": ["hi", "done"]}

    def test_logs_output_parser_branch(self, caplog: pytest.LogCaptureFixture) -> None:
        from langchain_azure_ai.agents.runtime._invoke_host import (
            invoke_output_parser,
        )

        caplog.set_level(logging.DEBUG, logger="langchain_azure_ai.agents.runtime")
        invoke_output_parser(
            cast(dict[str, JSONValue], {"answer": "done"}),
            _make_invoke_request(),
        )

        assert (
            "Output parser returning JSON-serializable non-message result"
            in caplog.text
        )


class TestAzureAIInvokeAgentHost:
    """Tests for AzureAIInvokeAgentHost."""

    async def test_handle_invoke_uses_session_id_as_thread_id(self) -> None:
        received_configs: list[Any] = []

        async def _ainvoke(
            *, input: Any, config: Any = None, context: Any = None
        ) -> Any:
            del input, context
            received_configs.append(config)
            return {"ok": True}

        graph = MagicMock()
        graph.ainvoke = _ainvoke
        host = _make_invoke_host(graph=graph)

        response = await host._handle_invoke(_make_invoke_request({"hello": "world"}))
        assert response.status_code == 200
        assert received_configs[0]["configurable"]["thread_id"] == "session-123"

    async def test_handle_invoke_returns_bad_request_for_invalid_payload(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"ok": True})
        host = _make_invoke_host(graph=graph)

        response = await host._handle_invoke(_make_invoke_request(["nope"]))
        assert response.status_code == 500
        assert json.loads(response.body) == {
            "error": "input_parser_error",
            "message": (
                "The configured input_parser 'invoke_input_parser' raised "
                "ValueError: Request body must be a JSON object."
            ),
            "hook": "input_parser",
            "parser": "invoke_input_parser",
            "exception_type": "ValueError",
        }
        graph.ainvoke.assert_not_called()

    async def test_handle_invoke_reports_custom_input_parser_error(self) -> None:
        async def my_input_parser(request: Any) -> Any:
            del request
            raise ValueError("missing required field 'question'")

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"ok": True})
        host = _make_invoke_host(graph=graph, input_parser=my_input_parser)

        response = await host._handle_invoke(_make_invoke_request({"ignored": True}))

        assert response.status_code == 500
        assert json.loads(response.body) == {
            "error": "input_parser_error",
            "message": (
                "The configured input_parser 'my_input_parser' raised "
                "ValueError: missing required field 'question'"
            ),
            "hook": "input_parser",
            "parser": "my_input_parser",
            "exception_type": "ValueError",
        }
        graph.ainvoke.assert_not_called()

    async def test_handle_invoke_returns_json_response(self) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"answer": 42})
        host = _make_invoke_host(graph=graph)

        response = await host._handle_invoke(_make_invoke_request({"question": "x"}))
        assert response.status_code == 200
        assert json.loads(response.body) == {"answer": 42}

    async def test_handle_invoke_reports_output_parser_error(self) -> None:
        def my_output_parser(result: Any, request: Any) -> Any:
            del result
            raise RuntimeError("could not serialize final state")

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"answer": 42})
        host = _make_invoke_host(graph=graph, output_parser=my_output_parser)

        response = await host._handle_invoke(_make_invoke_request({"question": "x"}))

        assert response.status_code == 500
        assert json.loads(response.body) == {
            "error": "output_parser_error",
            "message": (
                "The configured output_parser 'my_output_parser' raised "
                "RuntimeError: could not serialize final state"
            ),
            "hook": "output_parser",
            "parser": "my_output_parser",
            "exception_type": "RuntimeError",
        }

    async def test_custom_input_and_output_parsers_are_used(self) -> None:
        seen_requests: list[Any] = []

        from langchain_azure_ai.agents.runtime._invoke_host import GraphInvocationInput

        async def my_input_parser(
            request: Any,
        ) -> GraphInvocationInput[dict[str, int], dict[str, str]]:
            seen_requests.append(request)
            return GraphInvocationInput(
                input={"value": 7},
                context={"user_name": "Ada"},
                config={"tags": ["invoke-host"]},
            )

        def my_output_parser(result: Any, request: Any) -> Any:
            return {"wrapped": result["result"]}

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"result": 14})
        host = _make_invoke_host(
            graph=graph,
            input_parser=my_input_parser,
            output_parser=my_output_parser,
        )

        response = await host._handle_invoke(_make_invoke_request({"ignored": True}))

        assert response.status_code == 200
        assert len(seen_requests) == 1
        graph.ainvoke.assert_awaited_once_with(
            input={"value": 7},
            config={"tags": ["invoke-host"]},
            context={"user_name": "Ada"},
        )
        assert json.loads(response.body) == {"wrapped": 14}

    async def test_parser_config_preserves_explicit_thread_id(self) -> None:
        from langchain_azure_ai.agents.runtime._invoke_host import GraphInvocationInput

        async def my_input_parser(
            request: Any,
        ) -> GraphInvocationInput[dict[str, int], None]:
            del request
            return GraphInvocationInput(
                input={"value": 7},
                config={"configurable": {"thread_id": "parser-thread"}},
            )

        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"answer": 14})
        host = _make_invoke_host(graph=graph, input_parser=my_input_parser)

        response = await host._handle_invoke(_make_invoke_request({"ignored": True}))

        assert response.status_code == 200
        graph.ainvoke.assert_awaited_once_with(
            input={"value": 7},
            config={"configurable": {"thread_id": "parser-thread"}},
            context=None,
        )
        assert json.loads(response.body) == {"answer": 14}

    async def test_handle_invoke_logs_request_lifecycle(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = MagicMock()
        graph.ainvoke = AsyncMock(return_value={"answer": 42})
        host = _make_invoke_host(graph=graph)

        caplog.set_level(logging.DEBUG, logger="langchain_azure_ai.agents.runtime")
        response = await host._handle_invoke(_make_invoke_request({"question": "x"}))

        assert response.status_code == 200
        assert "Handling invoke request" in caplog.text
        assert "Invoking graph" in caplog.text
        assert "Returning invoke response" in caplog.text


