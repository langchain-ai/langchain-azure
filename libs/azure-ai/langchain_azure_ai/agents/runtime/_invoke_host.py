# Copyright (c) Microsoft. All rights reserved.

"""Hosting adapter for running a LangGraph graph behind Foundry's invocation API.

This module bridges the generic invocation protocol with a compiled LangGraph
graph. Unlike the Responses host, it accepts a trivial JSON request payload,
invokes the graph once, and returns a trivial JSON response payload.

Request state provided by ``InvocationAgentServerHost`` is surfaced to custom
parsers via the Starlette ``Request`` object. For graphs compiled with a
checkpointer, the default input parser sets ``configurable.thread_id`` from
``request.state.session_id``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar, cast

from azure.ai.agentserver.invocations import InvocationAgentServerHost
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from langchain_azure_ai._api.base import experimental

logger = logging.getLogger(__package__)

GraphInputT = TypeVar("GraphInputT")
ContextT = TypeVar("ContextT")
GraphOutputT = TypeVar("GraphOutputT")

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]


@experimental()
@dataclass(slots=True)
class GraphInvocationInput(Generic[GraphInputT, ContextT]):
    """Structured invocation payload returned by custom input parsers.

    Args:
        input: Input passed to ``graph.invoke()`` / ``graph.ainvoke()``.
        context: Optional static runtime context passed via the graph's
            ``context=...`` argument.
        config: Optional ``RunnableConfig`` passed to the graph unchanged.
    """

    input: GraphInputT
    context: ContextT | None = None
    config: RunnableConfig | None = None


InvokeInputParser: TypeAlias = Callable[
    [Request], Awaitable[GraphInvocationInput[GraphInputT, ContextT]]
]
InvokeOutputParser: TypeAlias = Callable[[GraphOutputT, Request], JSONValue]


def _parser_name(parser: object) -> str:
    """Return a stable human-readable name for a parser callable."""
    return cast(str, getattr(parser, "__name__", type(parser).__name__))


def _parser_error_response(
    *,
    hook_name: str,
    parser: object,
    exc: Exception,
) -> JSONResponse:
    """Build a diagnostic response for parser failures."""
    parser_name = _parser_name(parser)
    exception_type = type(exc).__name__
    detail = str(exc) or repr(exc)
    logger.exception(
        "Configured %s failed",
        hook_name,
        extra={
            "hook_name": hook_name,
            "parser_name": parser_name,
            "exception_type": exception_type,
        },
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": f"{hook_name}_error",
            "message": (
                f"The configured {hook_name} '{parser_name}' raised "
                f"{exception_type}: {detail}"
            ),
            "hook": hook_name,
            "parser": parser_name,
            "exception_type": exception_type,
        },
    )


def _coerce_message_output(message: BaseMessage) -> JSONValue:
    """Convert a LangChain message into a JSON-serializable payload."""
    content = message.content
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content

    if isinstance(content, dict):
        return cast(JSONValue, content)

    if isinstance(content, list):
        parts: list[JSONValue] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, (str, int, float, bool)) or text is None:
                    parts.append(text)
                else:
                    parts.append(cast(JSONValue, part))
            elif isinstance(part, (int, float, bool)) or part is None:
                parts.append(part)
            else:
                parts.append(str(part))
        return parts

    return str(content)


def _ensure_jsonable(value: object) -> JSONValue:
    """Raise when *value* is not JSON-serializable."""
    try:
        json.dumps(value)
    except TypeError as exc:  # pragma: no cover - exact message is irrelevant
        raise TypeError(
            "Graph output must be JSON-serializable, a LangChain message, "
            'or a dict containing a "messages" list.'
        ) from exc
    return cast(JSONValue, value)


@experimental()
async def invoke_input_parser(
    request: Request,
) -> GraphInvocationInput[GraphInputT, ContextT]:
    """Default invocation input parser.

    Expects the request body to be a JSON object and passes it verbatim to the
    graph as an ``GraphInvocationInput`` with explicit ``input``, ``context``, and
    ``config`` fields. By default, ``config`` includes
    ``{"configurable": {"thread_id": request.state.session_id}}``. Use a
    custom parser when the graph expects a different input shape or additional
    runtime context.
    """
    logger.debug(
        "Parsing invoke request",
        extra={
            "path": request.url.path,
            "session_id": request.state.session_id,
        },
    )

    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        logger.debug("Invoke request body is not valid JSON")
        raise ValueError("Request body must be a valid JSON object.") from exc

    if not isinstance(payload, dict):
        logger.debug(
            "Invoke request JSON is not an object of the expected type",
            extra={"payload_type": type(payload).__name__},
        )
        raise ValueError("Request body must be a JSON object.")

    input = cast(GraphInputT, payload)

    logger.debug(
        "Parsed invoke request payload",
        extra={"payload_keys": sorted(payload.keys())},
    )

    return GraphInvocationInput(
        input=input,
        context=None,
        config={
            "configurable": {
                "thread_id": request.state.session_id,
            }
        },
    )


@experimental()
def invoke_output_parser(output: GraphOutputT, request: Request) -> JSONValue:
    """Default invocation output parser.

    Returns JSON-serializable values unchanged. When the graph returns a
    LangChain message or a state dict containing a ``messages`` list, the last
    message content is normalized into ``{"output": ...}``.

    Args:
        output: The raw result returned by the graph after invocation.
        request: The original Starlette request object, in case additional context is
            needed for output parsing.

    Returns:
        A JSON-serializable value to return in the HTTP response.
    """
    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if isinstance(last_message, BaseMessage):
                logger.debug(
                    "Output parser extracted final message content",
                    extra={
                        "messages_count": len(messages),
                        "message_type": type(last_message).__name__,
                    },
                )
                return {"output": _coerce_message_output(last_message)}

        logger.debug(
            "Output parser returning JSON object result",
            extra={"result_keys": sorted(output.keys())},
        )
        return _ensure_jsonable(output)

    if isinstance(output, BaseMessage):
        logger.debug(
            "Output parser coercing BaseMessage result",
            extra={"message_type": type(output).__name__},
        )
        return {"output": _coerce_message_output(output)}

    logger.debug(
        "Output parser returning JSON-serializable non-message result",
        extra={"result_type": type(output).__name__},
    )
    return _ensure_jsonable(output)


@experimental()
class AzureAIInvokeAgentHost(Generic[GraphInputT, ContextT, GraphOutputT]):
    """Host a compiled LangGraph graph behind Azure AI Foundry's invocation API.

    The host registers an ``invoke_handler`` on an ``InvocationAgentServerHost``
    that:

    1. Parses the request body into graph input plus optional runtime context
         and config.
    2. Invokes the LangGraph graph once via ``graph.ainvoke()``.
    3. Normalizes the graph result into JSON and returns a ``JSONResponse``.

    By default the incoming JSON object is passed through unchanged and the
    outgoing value is returned unchanged when already JSON-serializable. For
    ``MessagesState``-style graphs, the default output parser maps the final
    LangChain message content to ``{"output": ...}``.

    Args:
        graph: A compiled LangGraph graph or any LangChain ``Runnable`` with
            ``ainvoke`` support.
        openapi_spec: Optional OpenAPI spec served by the underlying
            ``InvocationAgentServerHost`` at
            ``GET /invocations/docs/openapi.json``.
        input_parser: Async callable that returns a ``GraphInvocationInput``
            containing graph ``input``, optional static runtime ``context``,
            and optional ``RunnableConfig``.
        output_parser: Callable that converts a graph result into a
            JSON-serializable value for the HTTP response.

    .. code-block:: python
        from langgraph.graph import StateGraph, MessagesState, START, END
        from langchain_azure_ai.agents.runtime import AzureAIInvokeAgentHost

        builder = StateGraph(MessagesState)
        builder.add_node("agent", my_agent_node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        graph = builder.compile()

        def my_input_parser(request: Request) -> GraphInvocationInput[MessagesState, None]:
            # Example of a trivial custom input parser that reuses the default logic
            payload = await request.json()
            
            return GraphInvocationInput(input=cast(MessagesState, payload))

        def my_output_parser(output: GraphOutputT, request: Request) -> JSONValue:
            # Example custom output parser that wraps the graph output in a "result" field
            return {"result": output}

        host = AzureAIInvokeAgentHost(
            graph=graph,
            input_parser=my_input_parser,  # Optional custom input parser
            output_parser=my_output_parser,  # Optional custom output parser
        )

        if __name__ == "__main__":
            host.run()
    """

    def __init__(
        self,
        graph: Runnable[GraphInputT, GraphOutputT],
        *,
        openapi_spec: dict[str, JSONValue] | None = None,
        input_parser: InvokeInputParser[GraphInputT, ContextT] | None = None,
        output_parser: InvokeOutputParser[GraphOutputT] | None = None,
    ) -> None:
        self._graph: Runnable[GraphInputT, GraphOutputT] = graph
        self._input_parser = (
            input_parser
            if input_parser is not None
            else cast(InvokeInputParser[GraphInputT, ContextT], invoke_input_parser)
        )
        self._output_parser = (
            output_parser
            if output_parser is not None
            else cast(InvokeOutputParser[GraphOutputT], invoke_output_parser)
        )

        self._app: InvocationAgentServerHost = InvocationAgentServerHost(  # type: ignore[misc]
            openapi_spec=openapi_spec
        )
        self._app.invoke_handler(self._handle_invoke)

    async def _handle_invoke(self, request: Request) -> Response:
        """Handle a single invocation request from Foundry."""
        logger.debug(
            "Handling invoke request",
            extra={
                "path": request.url.path,
                "session_id": request.state.session_id,
                "invocation_id": getattr(request.state, "invocation_id", None),
            },
        )

        try:
            invocation = await self._input_parser(request)
        except ValueError as exc:
            return _parser_error_response(
                hook_name="input_parser",
                parser=self._input_parser,
                exc=exc,
            )

        logger.debug(
            "Invoking graph",
            extra={
                "has_input": invocation.input is not None,
                "has_context": invocation.context is not None,
                "has_config": invocation.config is not None,
                "config_keys": sorted(invocation.config.keys())
                if invocation.config
                else [],
            },
        )
        result = await self._graph.ainvoke(  # type: ignore[union-attr]
            input=invocation.input,
            context=invocation.context,
            config=invocation.config,
        )
        try:
            payload = self._output_parser(result, request)
        except Exception as exc:
            return _parser_error_response(
                hook_name="output_parser",
                parser=self._output_parser,
                exc=exc,
            )
        logger.debug(
            "Returning invoke response",
            extra={"payload_type": type(payload).__name__},
        )
        return JSONResponse(content=payload)

    def run(self, host: str = "0.0.0.0", port: int | None = None) -> None:
        """Start the invocation host."""
        self._app.run(host=host, port=port)
