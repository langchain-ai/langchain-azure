# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host a LangChain ``Runnable`` as the Azure AI Invocations API.

Quick start::

    import os

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver

    from langchain_azure_ai.agents.hosting import InvocationsHostServer

    model = ChatOpenAI(
        model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o"),
    )
    graph = create_agent(model, tools=[], checkpointer=MemorySaver())

    if __name__ == "__main__":
        InvocationsHostServer(graph).run(port=int(os.environ.get("PORT", "8088")))

Then call the local server::

    curl -i -X POST http://127.0.0.1:8088/invocations \
        -H 'Content-Type: application/json' \
        -d '{"message":"My name is Alice."}'

    curl -X POST 'http://127.0.0.1:8088/invocations?agent_session_id=<id>' \
        -H 'Content-Type: application/json' \
        -d '{"message":"What is my name?"}'
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Generic, Optional, TypeVar, cast

try:
    from azure.ai.agentserver.invocations import InvocationAgentServerHost
except ImportError as exc:
    raise ImportError(
        "The azure-ai-agentserver-invocations package is required to use "
        "InvocationsHostServer. Please install it via "
        "`pip install azure-ai-agentserver-invocations` or "
        "`pip install langchain-azure-ai[hosting]`."
    ) from exc

from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable, RunnableConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from langchain_azure_ai._api.base import experimental

from ._converters import (
    build_messages_input_from_text,
    extract_text,
    is_messages_state_schema,
    last_ai_message_text,
)

logger = logging.getLogger(__name__)

GraphInputT = TypeVar("GraphInputT")
GraphOutputT = TypeVar("GraphOutputT")
InvocationOutputParser = Callable[[GraphOutputT], str]


@experimental()
class InvocationsHostServer(Generic[GraphInputT, GraphOutputT]):
    """Host a LangChain ``Runnable`` as the Invocations API.

    Example:
        Create an agent graph with a checkpointer and host it on
        ``POST /invocations``::

            import os

            from langchain.agents import create_agent
            from langchain_openai import ChatOpenAI
            from langgraph.checkpoint.memory import MemorySaver

            from langchain_azure_ai.agents.hosting import InvocationsHostServer

            model = ChatOpenAI(
                model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o"),
            )
            graph = create_agent(model, tools=[], checkpointer=MemorySaver())

            InvocationsHostServer(graph).run(port=8088)

        The host forwards ``agent_session_id`` to the graph as
        ``RunnableConfig.configurable.thread_id`` so follow-up turns can
        continue the same checkpointed conversation.

    .. code-block:: json

        { "message": "Hello!", "stream": false }

    Where:

    - ``message`` (required) — user message text.
    - ``stream`` (optional, default ``false``) — when ``true`` returns SSE
      with token deltas; when ``false`` returns a single JSON response.

    Multi-turn continuation uses the ``agent_session_id`` query param /
    ``x-agent-session-id`` header populated by
    :class:`InvocationAgentServerHost`. The session id is forwarded to the
    graph as ``RunnableConfig.configurable.thread_id``, so LangGraph graphs
    compiled with a checkpointer continue automatically.

    Args:
        graph: The runnable to host. The default converters expect a
            LangGraph-style messages state input and output. Pass
            ``output_parser`` or subclass the request/output hooks for custom
            runnable shapes.

    Keyword Args:
        output_parser: Optional callable that converts the runnable result
            into response text for non-streaming requests. When omitted, the
            default parser reads the last AI message from a ``messages`` state.
        app: Optional existing :class:`InvocationAgentServerHost` to
            attach to (e.g. a multi-protocol mixin). In this mode the
            host-level kwargs are ignored — the caller is expected to
            have configured them on ``app`` itself.
        applicationinsights_connection_string: Forwarded to
            :class:`AgentServerHost`.
        graceful_shutdown_timeout: Forwarded to :class:`AgentServerHost`.

    Raises:
        ValueError: If the graph's state schema does not declare a
            ``messages`` field. Override this class to host custom-state
            graphs.
    """

    def __init__(
        self,
        graph: Runnable[GraphInputT, GraphOutputT],
        *,
        output_parser: Optional[InvocationOutputParser[GraphOutputT]] = None,
        app: Optional[InvocationAgentServerHost] = None,
        applicationinsights_connection_string: Optional[str] = None,
        graceful_shutdown_timeout: Optional[int] = None,
    ) -> None:
        self._validate_graph_schema(graph)
        self._graph = graph
        self._output_parser = output_parser

        if app is not None:
            # Attach to an existing host (e.g. a multi-protocol mixin).
            # In this mode the host-level kwargs are ignored — the caller
            # is expected to have configured them on ``app`` itself.
            self._app = app
        else:
            host_kwargs: dict[str, Any] = {}
            if applicationinsights_connection_string is not None:
                host_kwargs["applicationinsights_connection_string"] = (
                    applicationinsights_connection_string
                )
            if graceful_shutdown_timeout is not None:
                host_kwargs["graceful_shutdown_timeout"] = graceful_shutdown_timeout
            self._app = InvocationAgentServerHost(**host_kwargs)

        self._app.invoke_handler(self._handle_invoke)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def app(self) -> InvocationAgentServerHost:
        """The underlying :class:`InvocationAgentServerHost`."""
        return self._app

    @property
    def graph(self) -> Runnable[GraphInputT, GraphOutputT]:
        """The hosted runnable."""
        return self._graph

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        """Start the server synchronously.

        Once running, the host exposes ``POST /invocations``. The default
        request body is:

        .. code-block:: json

            {"message": "Hello!", "stream": false}

        ``message`` is the required user text. ``stream`` is optional and
        defaults to ``false``. Non-streaming requests return JSON:

        .. code-block:: json

            {"response": "Assistant text"}

        Streaming requests return ``text/event-stream`` with token payloads:

        .. code-block:: text

            data: {"token": "..."}

            event: done
            data: {}

        Multi-turn callers should reuse the ``x-agent-session-id`` response
        header as the next request's ``agent_session_id`` query parameter.

        Args:
            host: Network interface to bind. Defaults to ``"0.0.0.0"``.
            port: Port to bind. Defaults to ``PORT`` env var or 8088.
        """
        self._app.run(host=host, port=port)

    async def run_async(
        self, host: str = "0.0.0.0", port: Optional[int] = None
    ) -> None:
        """Start the server asynchronously.

        Exposes the same ``POST /invocations`` contract as :meth:`run`.
        The default request body is:

        .. code-block:: json

            {"message": "Hello!", "stream": false}

        Non-streaming requests return ``{"response": "Assistant text"}``.
        Streaming requests return ``text/event-stream`` with
        ``data: {"token": "..."}`` payloads followed by ``event: done``.
        Multi-turn callers should reuse the ``x-agent-session-id`` response
        header as the next request's ``agent_session_id`` query parameter.

        Args:
            host: Network interface to bind.
            port: Port to bind.
        """
        await self._app.run_async(host=host, port=port)

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    async def parse_request(self, request: Request) -> tuple[str, bool]:
        """Parse the invocation request body.

        Default implementation reads ``{"message": str, "stream": bool}``.
        Override to support a different body schema.

        Args:
            request: The Starlette request.

        Returns:
            ``(message, stream)`` tuple.

        Raises:
            ValueError: If the body cannot be parsed or is missing the
                ``message`` field.
        """
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")

        message = data.get("message")
        if not isinstance(message, str) or not message:
            raise ValueError("Request body must include a non-empty 'message' string.")

        stream = bool(data.get("stream", False))
        return message, stream

    def build_runnable_config(self, request: Request) -> RunnableConfig:
        """Build a ``RunnableConfig`` for the invocation.

        Sets ``configurable.thread_id`` from ``request.state.session_id``
        so LangGraph graphs compiled with a checkpointer naturally continue
        the right conversation across turns of the same session.

        Args:
            request: The Starlette request.

        Returns:
            A ``RunnableConfig`` dict.
        """
        session_id = getattr(request.state, "session_id", None)
        return {"configurable": {"thread_id": session_id or "default"}}

    def build_input(self, message: str) -> GraphInputT:
        """Build the runnable input from the parsed message.

        Default implementation produces
        ``{"messages": [HumanMessage(...)]}``. Override to support
        custom-state graphs.

        Args:
            message: The user message text.

        Returns:
            A runnable input value.
        """
        return cast(GraphInputT, build_messages_input_from_text(message))

    def parse_output(self, output: GraphOutputT) -> str:
        """Translate a non-streaming runnable result into response text.

        Args:
            output: The value returned by ``graph.ainvoke``.

        Returns:
            Text for the ``response`` field.
        """
        if self._output_parser is not None:
            return self._output_parser(output)
        return last_ai_message_text(_messages_from_state(output))

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    async def _handle_invoke(self, request: Request) -> Response:
        try:
            message, stream = await self.parse_request(request)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        graph_input = self.build_input(message)
        config = self.build_runnable_config(request)

        if stream:
            return StreamingResponse(
                self._stream_tokens(graph_input, config),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            output = await self._graph.ainvoke(graph_input, config=config)
        except Exception:  # noqa: BLE001
            logger.exception("LangGraph invocation failed")
            return JSONResponse({"error": "Internal server error."}, status_code=500)

        text = self.parse_output(output)
        return JSONResponse({"response": text})

    async def _stream_tokens(
        self,
        graph_input: GraphInputT,
        config: RunnableConfig,
    ) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._graph.astream(
                graph_input, config=config, stream_mode="messages"
            ):
                message_chunk = _extract_message_chunk(chunk)
                if message_chunk is None:
                    continue
                text = extract_text(message_chunk.content)
                if not text:
                    continue
                payload = json.dumps({"token": text}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.exception("LangGraph streaming invocation failed")
            payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return

        yield b"event: done\ndata: {}\n\n"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_graph_schema(graph: Runnable[Any, Any]) -> None:
        builder = getattr(graph, "builder", None)
        state_schema = (
            getattr(builder, "state_schema", None) if builder is not None else None
        )
        if state_schema is None:
            return
        if is_messages_state_schema(state_schema):
            return
        raise ValueError(
            "InvocationsHostServer's default request converter only "
            "supports graphs whose state schema declares a 'messages' field. "
            "Subclass InvocationsHostServer and override `build_input` "
            "(and optionally `parse_request`) to host custom-state graphs."
        )


def _messages_from_state(state: Any) -> list[Any]:
    if isinstance(state, dict):
        return list(state.get("messages") or [])
    return list(getattr(state, "messages", None) or [])


def _extract_message_chunk(chunk: Any) -> Optional[AIMessageChunk]:
    if isinstance(chunk, AIMessageChunk):
        return chunk
    if isinstance(chunk, tuple) and chunk:
        candidate = chunk[0]
        if isinstance(candidate, AIMessageChunk):
            return candidate
    return None
