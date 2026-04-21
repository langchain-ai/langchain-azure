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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar, cast

from azure.ai.agentserver.invocations import InvocationAgentServerHost
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from langchain_azure_ai._api.base import experimental

GraphInputT = TypeVar("GraphInputT")
ContextT = TypeVar("ContextT")
GraphResultT = TypeVar("GraphResultT")

JSONPrimitive: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONPrimitive | list["JSONValue"] | dict[str, "JSONValue"]
InvokeInput: TypeAlias = dict[str, JSONValue]
InvokeResult: TypeAlias = (
    BaseMessage | JSONValue | dict[str, JSONValue | list[BaseMessage]]
)


@experimental()
@dataclass(slots=True)
class InvokeInvocation(Generic[GraphInputT, ContextT]):
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
    [Request], Awaitable[InvokeInvocation[GraphInputT, ContextT]]
]
InvokeOutputParser: TypeAlias = Callable[[GraphResultT], JSONValue]


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
) -> InvokeInvocation[InvokeInput, object]:
    """Default invocation input parser.

    Expects the request body to be a JSON object and passes it verbatim to the
    graph as an ``InvokeInvocation`` with explicit ``input``, ``context``, and
    ``config`` fields. By default, ``config`` includes
    ``{"configurable": {"thread_id": request.state.session_id}}``. Use a
    custom parser when the graph expects a different input shape or additional
    runtime context.
    """
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be a valid JSON object.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    return InvokeInvocation(
        input=cast(InvokeInput, payload),
        context=None,
        config={
            "configurable": {
                "thread_id": request.state.session_id,
            }
        },
    )


@experimental()
def invoke_output_parser(result: InvokeResult) -> JSONValue:
    """Default invocation output parser.

    Returns JSON-serializable values unchanged. When the graph returns a
    LangChain message or a state dict containing a ``messages`` list, the last
    message content is normalized into ``{"output": ...}``.
    """
    if isinstance(result, dict):
        messages = result.get("messages")
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            if isinstance(last_message, BaseMessage):
                return {"output": _coerce_message_output(last_message)}
        return _ensure_jsonable(result)

    if isinstance(result, BaseMessage):
        return {"output": _coerce_message_output(result)}

    return _ensure_jsonable(result)


@experimental()
class AzureAIInvokeAgentHost(Generic[GraphInputT, ContextT, GraphResultT]):
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
        input_parser: Async callable that returns an ``InvokeInvocation``
            containing graph ``input``, optional static runtime ``context``,
            and optional ``RunnableConfig``.
        output_parser: Callable that converts a graph result into a
            JSON-serializable value for the HTTP response. The result passed to
            the parser is one of: a ``BaseMessage``, a JSON value, or a state
            dict whose values are JSON values except for an optional
            ``messages`` entry containing ``list[BaseMessage]``.
    """

    def __init__(
        self,
        graph: Runnable[GraphInputT, GraphResultT],
        *,
        openapi_spec: dict[str, JSONValue] | None = None,
        input_parser: InvokeInputParser[GraphInputT, ContextT] | None = None,
        output_parser: InvokeOutputParser[GraphResultT] | None = None,
    ) -> None:
        self._graph: Runnable[GraphInputT, GraphResultT] = graph
        self._input_parser = (
            input_parser
            if input_parser is not None
            else cast(InvokeInputParser[GraphInputT, ContextT], invoke_input_parser)
        )
        self._output_parser = (
            output_parser
            if output_parser is not None
            else cast(InvokeOutputParser[GraphResultT], invoke_output_parser)
        )

        self._app: InvocationAgentServerHost = InvocationAgentServerHost(  # type: ignore[misc]
            openapi_spec=openapi_spec
        )
        self._app.invoke_handler(self._handle_invoke)

    async def _handle_invoke(self, request: Request) -> Response:
        """Handle a single invocation request from Foundry."""
        try:
            invocation = await self._input_parser(request)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid_request", "message": str(exc)},
            )

        config = cast(RunnableConfig, invocation.config or {})

        if invocation.context is None:
            result = await self._graph.ainvoke(  # type: ignore[union-attr]
                invocation.input,
                config=config,
            )
        else:
            result = await self._graph.ainvoke(  # type: ignore[union-attr]
                invocation.input,
                config=config,
                context=invocation.context,
            )
        payload = self._output_parser(result)
        return JSONResponse(content=payload)

    def run(self, host: str = "0.0.0.0", port: int | None = None) -> None:
        """Start the invocation host."""
        self._app.run(host=host, port=port)
