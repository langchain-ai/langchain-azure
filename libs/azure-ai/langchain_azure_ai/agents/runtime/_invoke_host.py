# Copyright (c) Microsoft. All rights reserved.

"""Hosting adapter for running a LangGraph graph behind Foundry's invocation API.

This module bridges the generic invocation protocol with a compiled LangGraph
graph. Unlike the Responses host, it accepts a trivial JSON request payload,
invokes the graph once, and returns a trivial JSON response payload.

Request state provided by ``InvocationAgentServerHost`` is surfaced to custom
parsers via the Starlette ``Request`` object. For graphs compiled with a
checkpointer, the default input parser sets ``configurable.thread_id`` from
``request.state.session_id``.

Error handling follows a plain HTTP model. The host adds diagnostic logging at
the request parse, graph invocation, and response stages. Parser failures that
this module handles are turned into JSON responses naming the failing hook and
parser callable:

* ``input_parser`` failures are wrapped only when the parser raises
    ``ValueError``.
* ``output_parser`` failures are wrapped for any exception.

Exceptions from ``graph.ainvoke()`` and ``input_parser`` exceptions outside the
``ValueError`` path are intentionally left to the underlying invocation server.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias, TypeVar, cast

from langchain_core.runnables import Runnable, RunnableConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

try:
    from azure.ai.agentserver.invocations import InvocationAgentServerHost
except ImportError as exc:
    raise ImportError(
        "The azure-ai-agentserver-invocations package is required to use "
        "AzureAIInvokeAgentHost. Please install it via "
        "`pip install azure-ai-agentserver-invocations` or "
        "`pip install langchain-azure-ai[runtime]`."
    ) from exc

from langchain_azure_ai._api.base import experimental

logger = logging.getLogger(__package__)

GraphInputT = TypeVar("GraphInputT")
ContextT = TypeVar("ContextT")
GraphOutputT = TypeVar("GraphOutputT")

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
InvokeOutputResponse: TypeAlias = JSONValue


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


InvokeInputRequest: TypeAlias = Request
InvokeInputParser: TypeAlias = Callable[
    [InvokeInputRequest], Awaitable[GraphInvocationInput[GraphInputT, ContextT]]
]
InvokeOutputParser: TypeAlias = Callable[[GraphOutputT, InvokeInputRequest], JSONValue]


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
def invoke_output_parser(
    output: GraphOutputT, request: InvokeInputRequest
) -> InvokeOutputResponse:
    """Default invocation output parser.

    Returns JSON-serializable values unchanged from the graph output.

    Args:
        output: The raw result returned by the graph after invocation.
        request: The original Starlette request object, in case additional context is
            needed for output parsing.

    Returns:
        A JSON-serializable value to return in the HTTP response.
    """
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

    Developer guidance:

    * By default the incoming JSON object is passed through unchanged to the graph
        input and the outgoing value is returned unchanged as JSON-serializable.
    * `config` is populated with a `thread_id` by default, but otherwise unused by
            the host; use it as needed for your graph or custom parsers.
    * Provide a custom ``input_parser`` when the graph expects more than the
        request JSON-object payload, or when you need to supply ``context`` or a
        custom ``RunnableConfig``.
    * If you want parser-specific request diagnostics from a custom
        ``input_parser``, raise ``ValueError`` for request-shape problems.
    * If a custom ``output_parser`` raises, the host returns a JSON error
        payload naming the failing parser and exception.
    * Enable debug logging for ``langchain_azure_ai.agents.runtime`` to inspect
        parser, invocation, and response flow while integrating a new host.

    Args:
        graph: A compiled LangGraph graph or any LangChain ``Runnable`` with
            ``ainvoke`` support.
        openapi_spec: Optional OpenAPI spec served by the underlying
            ``InvocationAgentServerHost`` at
            ``GET /invocations/docs/openapi.json``.
        input_parser: Async callable that returns a ``GraphInvocationInput``
            containing graph ``input``, optional static runtime ``context``,
            and optional ``RunnableConfig``. If not provided, the default parser
            expects the request body to be a JSON object and passes it verbatim
            as ``input`` with no context and a ``RunnableConfig`` containing the
            session ID as ``configurable.thread_id``.
        output_parser: Callable that converts a graph result into a
            JSON-serializable value for the HTTP response. If not provided, the
            default parser returns JSON-serializable values unchanged and raises
            a ``TypeError`` for unsupported types.

    Example:
            
    ```python
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langchain_azure_ai.agents.runtime import (
        AzureAIInvokeAgentHost,
        InvokeInputRequest,
        InvokeOutputResponse,
        GraphInvocationInput,
    )

    builder = StateGraph(MessagesState)
    builder.add_node("agent", my_agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    async def my_input_parser(
        request: InvokeInputRequest,
    ) -> GraphInvocationInput[MessagesState, None]:
        # Example of a trivial custom input parser that reuses the default logic
        payload = await request.json()

        return GraphInvocationInput(
            input=cast(MessagesState, payload),
            context=None,
            config={
                "configurable": {
                    "thread_id": request.state.session_id,
                }
            },
        )

    async def my_output_parser(
        output: MessagesState, request: InvokeInputRequest
    ) -> InvokeOutputResponse:
        # Example custom output parser that wraps the graph output
        # in a "result" field.
        return {"result": output["messages"][-1].content}

    host = AzureAIInvokeAgentHost(
        graph=graph,
        input_parser=my_input_parser,  # Optional custom input parser
        output_parser=my_output_parser,  # Optional custom output parser
    )

    if __name__ == "__main__":
        host.run()
    ```
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

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike[str] = "langgraph.json",
        *,
        graph_name: str | None = None,
        openapi_spec: dict[str, JSONValue] | None = None,
        input_parser: InvokeInputParser[Any, Any] | None = None,
        output_parser: InvokeOutputParser[Any] | None = None,
    ) -> "AzureAIInvokeAgentHost[Any, Any, Any]":
        """Create an instance by loading the graph from a ``langgraph.json`` file.

        Reads the ``graphs`` section of *path* to locate and import the graph
        module, then constructs an :class:`AzureAIInvokeAgentHost` with the
        loaded graph and any additional arguments supplied.

        Args:
            path: Path to the ``langgraph.json`` configuration file.
                Defaults to ``"langgraph.json"`` in the current working
                directory.
            graph_name: Key of the graph to load from the ``"graphs"`` dict.
                May be omitted when the file defines exactly one graph;
                required when multiple graphs are present.
            openapi_spec: Optional OpenAPI spec forwarded to the underlying
                ``InvocationAgentServerHost``.
            input_parser: Optional custom async input parser callable.
            output_parser: Optional custom output parser callable.

        Returns:
            A fully configured :class:`AzureAIInvokeAgentHost` instance.

        Example::

            # Single-graph config — graph_name is optional
            host = AzureAIInvokeAgentHost.from_config()

            # Multi-graph config — graph_name is required
            host = AzureAIInvokeAgentHost.from_config(graph_name="agent")

            # Custom config file path
            host = AzureAIInvokeAgentHost.from_config(path="/app/langgraph.json")
        """
        from langchain_azure_ai.agents.runtime._config import (
            load_graph_from_langgraph_config,
        )

        graph = load_graph_from_langgraph_config(path, graph_name=graph_name)
        return cls(
            graph=graph,
            openapi_spec=openapi_spec,
            input_parser=input_parser,
            output_parser=output_parser,
        )
