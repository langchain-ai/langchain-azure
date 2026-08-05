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

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any, Generic, Optional, TypeVar, cast

try:
    from azure.ai.agentserver.core.streaming import (
        EventStream,
        EventStreamClosedError,
        EventStreamNotFoundError,
        streams,
    )
    from azure.ai.agentserver.core.tasks import (
        LastInputIdPreconditionFailed,
        MultiTurnTask,
        SteeringQueueFull,
        TaskCancelled,
        TaskConflictError,
        TaskContext,
        TaskRun,
        multi_turn_task,
    )
    from azure.ai.agentserver.invocations import InvocationAgentServerHost
    from azure.ai.agentserver.responses import (
        ResponsesAgentServerHost,
        ResponsesServerOptions,
    )
except ImportError as exc:
    raise ImportError(
        "The azure-ai-agentserver-invocations and "
        "azure-ai-agentserver-responses packages are required to use "
        "InvocationsHostServer. Please install them via "
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

_INVOCATION_TASK_NAME = "langchain_invocations"
_METADATA_INPUT_ID = "invocation_input_id"
_METADATA_RESPONSE = "invocation_response"
_METADATA_CHECKPOINT_ID = "langgraph_checkpoint_id"
_METADATA_CHECKPOINT_THREAD_ID = "langgraph_thread_id"
_REPLAY_EVENT_TTL_SECONDS = 600.0


def _uses_langgraph_checkpointer(graph: Runnable[Any, Any]) -> bool:
    return getattr(graph, "checkpointer", None) is not None


def _invocation_event_cursor(event: Any) -> int:
    sequence_number = (
        event.get("sequence_number")
        if isinstance(event, dict)
        else getattr(event, "sequence_number", None)
    )
    if not isinstance(sequence_number, int):
        raise ValueError(
            "Agent Server stream events must contain an integer sequence_number."
        )
    return sequence_number


def _serialize_stream_event(event: Any) -> bytes:
    if hasattr(event, "as_dict") and callable(event.as_dict):
        data = event.as_dict()
    elif isinstance(event, dict):
        data = event
    else:
        data = dict(event)
    return json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")


def _deserialize_stream_event(payload: bytes) -> Any:
    return json.loads(payload.decode("utf-8"))


def _classify_graph_stream_chunk(chunk: Any) -> tuple[Optional[str], Any]:
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], str):
        return chunk[0], chunk[1]
    return None, chunk


def _accumulate_stream_output(current: Any, chunk: Any) -> Any:
    if current is None:
        return chunk
    try:
        return current + chunk
    except (TypeError, ValueError):
        return chunk


def _checkpoint_from_stream_payload(payload: Any) -> tuple[str, str] | None:
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    checkpoint_id = configurable.get("checkpoint_id")
    if not isinstance(thread_id, str) or not thread_id:
        return None
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        return None
    return thread_id, checkpoint_id


async def _next_async_item(iterator: AsyncIterator[Any]) -> Any:
    return await anext(iterator)


async def _close_async_iterator(iterator: AsyncIterator[Any]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


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
        - ``background`` (optional, default ``false``) — when ``true`` starts a
            durable invocation and returns ``202``. Requires
            ``options.resilient_background=True`` and a LangGraph checkpointer.
        - ``previous_invocation_id`` (optional) — linear-chain precondition for
            a continued ``agent_session_id``.

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
        options: Optional :class:`ResponsesServerOptions`. The
            ``resilient_background`` and ``steerable_conversations`` values
            configure durable invocation recovery and in-flight conversation
            steering respectively.
        app: Optional existing :class:`InvocationAgentServerHost` to
            attach to (e.g. a multi-protocol mixin). In this mode the
            host-level kwargs are ignored — the caller is expected to
            have configured them on ``app`` itself.
        applicationinsights_connection_string: Forwarded to
            :class:`AgentServerHost`.
        graceful_shutdown_timeout: Forwarded to :class:`AgentServerHost`.

    Raises:
        ValueError: If the graph's state schema does not declare a
            ``messages`` field, or if ``resilient_background=True`` is
            configured without a LangGraph checkpointer. Override this class
            to host custom-state graphs.
    """

    def __init__(
        self,
        graph: Runnable[GraphInputT, GraphOutputT],
        *,
        output_parser: Optional[InvocationOutputParser[GraphOutputT]] = None,
        options: Optional[ResponsesServerOptions] = None,
        app: Optional[InvocationAgentServerHost] = None,
        applicationinsights_connection_string: Optional[str] = None,
        graceful_shutdown_timeout: Optional[int] = None,
    ) -> None:
        self._validate_graph_schema(graph)
        self._graph = graph
        self._supports_langgraph_stream_modes = (
            getattr(graph, "builder", None) is not None
        )
        self._graph_has_checkpointer = _uses_langgraph_checkpointer(graph)
        if (
            options is not None
            and options.resilient_background
            and not self._graph_has_checkpointer
        ):
            raise ValueError(
                "InvocationsHostServer requires a LangGraph checkpointer when "
                "resilient_background=True."
            )
        self._output_parser = output_parser
        self._options = options or ResponsesServerOptions()
        self._invocation_task: Optional[
            MultiTurnTask[dict[str, Any], dict[str, Any]]
        ] = None
        self._active_runs: dict[str, TaskRun[dict[str, Any]]] = {}
        self._run_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cancel_requests: dict[str, asyncio.Event] = {}

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

        if self._options.resilient_background or self._options.steerable_conversations:
            if self._options.resilient_background:
                streams.use_file_backed_replay(
                    cursor_fn=_invocation_event_cursor,
                    ttl_seconds=_REPLAY_EVENT_TTL_SECONDS,
                    serializer=_serialize_stream_event,
                    deserializer=_deserialize_stream_event,
                )
            elif app is None or not isinstance(app, ResponsesAgentServerHost):
                streams.use_in_memory_replay(
                    cursor_fn=_invocation_event_cursor,
                    ttl_seconds=_REPLAY_EVENT_TTL_SECONDS,
                )
            self._invocation_task = self._register_invocation_task()
            self._app.get_invocation_handler(self._handle_get_invocation)
            self._app.cancel_invocation_handler(self._handle_cancel_invocation)

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

        With ``options.resilient_background=True``, a request containing
        ``{"message": "Hello!", "background": true}`` returns ``202``.
        Retrieve or cancel it through ``GET /invocations/{invocation_id}``
        and ``POST /invocations/{invocation_id}/cancel`` respectively.

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
        Resilient background requests return ``202`` and are observed through
        the invocation GET and cancel endpoints.
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

        stream = data.get("stream", False)
        if not isinstance(stream, bool):
            raise ValueError("Request body field 'stream' must be a boolean.")

        return message, stream

    async def parse_execution_options(
        self,
        request: Request,
        *,
        stream: bool,
    ) -> tuple[bool, Optional[str]]:
        """Parse task-backed execution options from the request body.

        Override this hook alongside :meth:`parse_request` when a custom body
        schema carries background or continuation controls elsewhere.

        Returns:
            ``(background, previous_invocation_id)`` tuple.
        """
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")

        background = data.get("background", False)
        if not isinstance(background, bool):
            raise ValueError("Request body field 'background' must be a boolean.")
        if background and stream:
            raise ValueError("Background invocations do not support streaming.")

        previous_invocation_id = data.get("previous_invocation_id")
        if previous_invocation_id is not None and (
            not isinstance(previous_invocation_id, str) or not previous_invocation_id
        ):
            raise ValueError(
                "Request body field 'previous_invocation_id' must be a "
                "non-empty string when set."
            )
        return background, previous_invocation_id

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

    def build_task_runnable_config(
        self,
        session_id: str,
        context: TaskContext[dict[str, Any]],
    ) -> RunnableConfig:
        """Build config for a task-backed invocation attempt.

        The task context is transport-scoped rather than checkpointed graph
        state. Nodes can use it to inspect recovery, cancellation, steering,
        and shutdown signals for the current attempt.
        """
        configurable: dict[str, Any] = {
            "thread_id": session_id,
            "invocation_context": context,
            "invocation_cancellation_signal": context.cancel,
        }
        if (
            context.entry_mode == "recovered"
            and context.metadata.get(_METADATA_INPUT_ID) == context.input_id
        ):
            checkpoint_thread_id = context.metadata.get(_METADATA_CHECKPOINT_THREAD_ID)
            checkpoint_id = context.metadata.get(_METADATA_CHECKPOINT_ID)
            if (
                isinstance(checkpoint_thread_id, str)
                and checkpoint_thread_id
                and isinstance(checkpoint_id, str)
                and checkpoint_id
            ):
                configurable.update(
                    thread_id=checkpoint_thread_id,
                    checkpoint_id=checkpoint_id,
                    checkpoint_ns="",
                )
        return cast(RunnableConfig, {"configurable": configurable})

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
            background, previous_invocation_id = await self.parse_execution_options(
                request, stream=stream
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        if background:
            return await self._start_background_invocation(
                request,
                message,
                previous_invocation_id=previous_invocation_id,
            )

        if self._invocation_task is not None:
            return await self._start_task_backed_invocation(
                request,
                message,
                stream=stream,
                previous_invocation_id=previous_invocation_id,
            )

        if previous_invocation_id is not None:
            return JSONResponse(
                {
                    "error": "previous_invocation_id requires resilient background "
                    "or steerable conversations to be enabled."
                },
                status_code=400,
            )

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

    async def _start_background_invocation(
        self,
        request: Request,
        message: str,
        *,
        previous_invocation_id: Optional[str],
    ) -> Response:
        if not self._options.resilient_background or self._invocation_task is None:
            return JSONResponse(
                {
                    "error": "Background invocations require "
                    "options.resilient_background=True."
                },
                status_code=400,
            )

        try:
            (
                task_run,
                event_stream,
                invocation_id,
                session_id,
            ) = await self._start_task_invocation(
                request,
                message,
                stream=False,
                previous_invocation_id=previous_invocation_id,
            )
        except (TaskConflictError, LastInputIdPreconditionFailed) as exc:
            invocation_id = str(request.state.invocation_id)
            session_id = str(request.state.session_id)
            event_stream = await streams.get_or_create(invocation_id)
            code = (
                "conversation_precondition_failed"
                if isinstance(exc, LastInputIdPreconditionFailed)
                else "conversation_locked"
            )
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="failed",
                error={"code": code, "message": str(exc)},
                close=True,
            )
            return JSONResponse(
                {
                    "error": (
                        "previous_invocation_id does not match the latest turn."
                        if isinstance(exc, LastInputIdPreconditionFailed)
                        else "Conversation already has an active invocation."
                    )
                },
                status_code=409,
            )
        except SteeringQueueFull as exc:
            invocation_id = str(request.state.invocation_id)
            session_id = str(request.state.session_id)
            event_stream = await streams.get_or_create(invocation_id)
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="failed",
                error={"code": "steering_queue_full", "message": str(exc)},
                close=True,
            )
            return JSONResponse({"error": str(exc)}, status_code=429)

        status = "queued" if task_run.is_queued else "in_progress"
        return JSONResponse(
            {
                "id": invocation_id,
                "status": status,
                "agent_session_id": session_id,
            },
            status_code=202,
        )

    async def _start_task_backed_invocation(
        self,
        request: Request,
        message: str,
        *,
        stream: bool,
        previous_invocation_id: Optional[str],
    ) -> Response:
        try:
            task_run, event_stream, _, _ = await self._start_task_invocation(
                request,
                message,
                stream=stream,
                previous_invocation_id=previous_invocation_id,
            )
        except (TaskConflictError, LastInputIdPreconditionFailed) as exc:
            invocation_id = str(request.state.invocation_id)
            session_id = str(request.state.session_id)
            event_stream = await streams.get_or_create(invocation_id)
            is_precondition_failure = isinstance(exc, LastInputIdPreconditionFailed)
            message_text = (
                "previous_invocation_id does not match the latest turn."
                if is_precondition_failure
                else "Conversation already has an active invocation."
            )
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="failed",
                error={
                    "code": (
                        "conversation_precondition_failed"
                        if is_precondition_failure
                        else "conversation_locked"
                    ),
                    "message": message_text,
                },
                close=True,
            )
            return JSONResponse(
                {"error": message_text},
                status_code=409,
            )
        except SteeringQueueFull as exc:
            invocation_id = str(request.state.invocation_id)
            session_id = str(request.state.session_id)
            event_stream = await streams.get_or_create(invocation_id)
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="failed",
                error={"code": "steering_queue_full", "message": str(exc)},
                close=True,
            )
            return JSONResponse({"error": str(exc)}, status_code=429)

        if stream:
            return StreamingResponse(
                self._stream_task_events(event_stream),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        try:
            result = await task_run.result()
        except TaskCancelled:
            return JSONResponse(
                {"error": "Invocation was cancelled."},
                status_code=409,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Task-backed LangGraph invocation failed")
            return JSONResponse({"error": "Internal server error."}, status_code=500)

        if result.get("status") == "completed":
            return JSONResponse({"response": result.get("response", "")})
        error = result.get("error") or {}
        return JSONResponse(
            {"error": error.get("message", "Invocation was cancelled.")},
            status_code=409,
        )

    async def _start_task_invocation(
        self,
        request: Request,
        message: str,
        *,
        stream: bool,
        previous_invocation_id: Optional[str],
    ) -> tuple[TaskRun[dict[str, Any]], EventStream, str, str]:
        if self._invocation_task is None:
            raise RuntimeError("Task-backed invocation hosting is not configured.")

        invocation_id = str(request.state.invocation_id)
        session_id = str(request.state.session_id)
        event_stream = await streams.get_or_create(invocation_id)
        await self._emit_invocation_status(
            event_stream,
            invocation_id=invocation_id,
            session_id=session_id,
            status="queued",
        )

        start_kwargs: dict[str, Any] = {
            "task_id": session_id,
            "input_id": invocation_id,
            "input": {
                "invocation_id": invocation_id,
                "session_id": session_id,
                "message": message,
                "stream": stream,
            },
        }
        if previous_invocation_id is not None:
            start_kwargs["if_last_input_id"] = previous_invocation_id

        task_run = await self._invocation_task.start(**start_kwargs)
        self._active_runs[invocation_id] = task_run
        cleanup_task = asyncio.create_task(
            self._remove_active_run_when_done(invocation_id, task_run)
        )
        self._run_cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._run_cleanup_tasks.discard)
        return task_run, event_stream, invocation_id, session_id

    async def _remove_active_run_when_done(
        self,
        invocation_id: str,
        task_run: TaskRun[dict[str, Any]],
    ) -> None:
        try:
            await task_run.result()
        except (Exception, asyncio.CancelledError):
            pass
        finally:
            if self._active_runs.get(invocation_id) is task_run:
                self._active_runs.pop(invocation_id, None)

    def _register_invocation_task(
        self,
    ) -> MultiTurnTask[dict[str, Any], dict[str, Any]]:
        server = self

        @multi_turn_task(
            name=_INVOCATION_TASK_NAME,
            steerable=self._options.steerable_conversations,
        )
        async def _run_invocation(
            ctx: TaskContext[dict[str, Any]],
        ) -> dict[str, Any]:
            return await server._execute_task_invocation(ctx)

        return _run_invocation

    async def _execute_task_invocation(
        self,
        context: TaskContext[dict[str, Any]],
    ) -> dict[str, Any]:
        task_input = context.input
        invocation_id = str(task_input["invocation_id"])
        session_id = str(task_input["session_id"])
        message = str(task_input["message"])
        stream_tokens = bool(task_input.get("stream", False))
        event_stream = await streams.get_or_create(invocation_id)
        cancel_request = self._cancel_requests.setdefault(
            invocation_id,
            asyncio.Event(),
        )

        if context.shutdown.is_set():
            return await context.exit_for_recovery()

        if context.metadata.get(_METADATA_INPUT_ID) != context.input_id:
            context.metadata[_METADATA_INPUT_ID] = context.input_id
            for key in (
                _METADATA_RESPONSE,
                _METADATA_CHECKPOINT_THREAD_ID,
                _METADATA_CHECKPOINT_ID,
            ):
                if key in context.metadata:
                    del context.metadata[key]
            await context.metadata.flush()

        if context.entry_mode == "recovered":
            terminal_event = await self._latest_invocation_event(invocation_id)
            if terminal_event is not None and terminal_event.get("status") in {
                "completed",
                "failed",
                "cancelled",
            }:
                self._cancel_requests.pop(invocation_id, None)
                return self._public_invocation_event(terminal_event)
            event_stream = await streams.get_or_create(invocation_id)

        if (
            context.entry_mode == "recovered"
            and context.metadata.get(_METADATA_INPUT_ID) == context.input_id
            and isinstance(context.metadata.get(_METADATA_RESPONSE), str)
        ):
            response_text = context.metadata[_METADATA_RESPONSE]
            result = self._invocation_envelope(
                invocation_id,
                session_id,
                "completed",
                response=response_text,
            )
            try:
                await self._emit_invocation_status(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    status="completed",
                    response=response_text,
                    close=True,
                )
            except (EventStreamClosedError, EventStreamNotFoundError):
                logger.debug(
                    "Invocation %s already has a terminal replay stream",
                    invocation_id,
                )
            self._cancel_requests.pop(invocation_id, None)
            return result

        if cancel_request.is_set():
            result = await self._complete_cancelled_invocation(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                steered=False,
            )
            self._cancel_requests.pop(invocation_id, None)
            return result

        try:
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="in_progress",
            )
        except (EventStreamClosedError, EventStreamNotFoundError):
            terminal_result = await self._terminal_invocation_result(invocation_id)
            if terminal_result is not None:
                self._cancel_requests.pop(invocation_id, None)
                return terminal_result
            if cancel_request.is_set():
                result = await self._complete_cancelled_invocation(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    steered=False,
                )
                self._cancel_requests.pop(invocation_id, None)
                return result
            raise

        config = self.build_task_runnable_config(session_id, context)
        resume_from_checkpoint = (
            context.entry_mode == "recovered"
            and context.metadata.get(_METADATA_INPUT_ID) == context.input_id
            and isinstance(context.metadata.get(_METADATA_CHECKPOINT_ID), str)
        )
        graph_input = None if resume_from_checkpoint else self.build_input(message)
        latest_state: Any = None

        if self._supports_langgraph_stream_modes:
            stream_modes = ["values"]
            if stream_tokens:
                stream_modes.append("messages")
            graph_stream_kwargs: dict[str, Any] = {}
            if self._graph_has_checkpointer:
                stream_modes.append("checkpoints")
                graph_stream_kwargs["durability"] = "sync"
            graph_stream = self._graph.astream(
                cast(GraphInputT, graph_input),
                config=config,
                stream_mode=stream_modes,
                **graph_stream_kwargs,
            )
        else:
            graph_stream = self._graph.astream(
                cast(GraphInputT, graph_input),
                config=config,
            )
        graph_iterator = aiter(cast(AsyncIterator[Any], graph_stream))

        try:
            while True:
                next_chunk = asyncio.create_task(_next_async_item(graph_iterator))
                cancel_waiter = asyncio.create_task(context.cancel.wait())
                request_cancel_waiter = asyncio.create_task(cancel_request.wait())
                shutdown_waiter = asyncio.create_task(context.shutdown.wait())
                done, pending = await asyncio.wait(
                    {
                        next_chunk,
                        cancel_waiter,
                        request_cancel_waiter,
                        shutdown_waiter,
                    },
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for waiter in pending:
                    waiter.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

                stream_finished = False
                chunk_error: Optional[Exception] = None
                if next_chunk in done:
                    try:
                        chunk = next_chunk.result()
                    except StopAsyncIteration:
                        stream_finished = True
                    except Exception as exc:  # noqa: BLE001
                        chunk_error = exc
                    else:
                        mode, payload = _classify_graph_stream_chunk(chunk)
                        if mode == "values":
                            latest_state = payload
                        elif mode is None:
                            latest_state = _accumulate_stream_output(
                                latest_state,
                                payload,
                            )
                            if stream_tokens:
                                message_chunk = _extract_message_chunk(payload)
                                if message_chunk is not None:
                                    token = extract_text(message_chunk.content)
                                    if token:
                                        await self._emit_invocation_status(
                                            event_stream,
                                            invocation_id=invocation_id,
                                            session_id=session_id,
                                            status="in_progress",
                                            token=token,
                                        )
                        elif mode == "messages" and stream_tokens:
                            message_chunk = _extract_message_chunk(payload)
                            if message_chunk is not None:
                                token = extract_text(message_chunk.content)
                                if token:
                                    await self._emit_invocation_status(
                                        event_stream,
                                        invocation_id=invocation_id,
                                        session_id=session_id,
                                        status="in_progress",
                                        token=token,
                                    )
                        elif mode == "checkpoints":
                            checkpoint = _checkpoint_from_stream_payload(payload)
                            if checkpoint is not None:
                                thread_id, checkpoint_id = checkpoint
                                context.metadata[_METADATA_INPUT_ID] = context.input_id
                                context.metadata[_METADATA_CHECKPOINT_THREAD_ID] = (
                                    thread_id
                                )
                                context.metadata[_METADATA_CHECKPOINT_ID] = (
                                    checkpoint_id
                                )
                                await context.metadata.flush()

                if context.shutdown.is_set():
                    next_chunk.cancel()
                    await asyncio.gather(next_chunk, return_exceptions=True)
                    await _close_async_iterator(graph_iterator)
                    return await context.exit_for_recovery()

                if cancel_request.is_set() or context.cancel.is_set():
                    next_chunk.cancel()
                    await asyncio.gather(next_chunk, return_exceptions=True)
                    await _close_async_iterator(graph_iterator)
                    result = await self._complete_cancelled_invocation(
                        event_stream,
                        invocation_id=invocation_id,
                        session_id=session_id,
                        steered=(
                            not cancel_request.is_set() and not context.cancel_requested
                        ),
                    )
                    self._cancel_requests.pop(invocation_id, None)
                    return result

                if chunk_error is not None:
                    raise chunk_error

                if stream_finished:
                    break

            if latest_state is None:
                raise RuntimeError("LangGraph invocation produced no output state.")
            response_text = self.parse_output(cast(GraphOutputT, latest_state))
            context.metadata[_METADATA_INPUT_ID] = context.input_id
            context.metadata[_METADATA_RESPONSE] = response_text
            await context.metadata.flush()
        except (EventStreamClosedError, EventStreamNotFoundError):
            await _close_async_iterator(graph_iterator)
            terminal_result = await self._terminal_invocation_result(invocation_id)
            if terminal_result is not None:
                self._cancel_requests.pop(invocation_id, None)
                return terminal_result
            if cancel_request.is_set():
                result = await self._complete_cancelled_invocation(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    steered=False,
                )
                self._cancel_requests.pop(invocation_id, None)
                return result
            self._cancel_requests.pop(invocation_id, None)
            raise
        except Exception as exc:
            logger.exception("Task-backed LangGraph invocation failed")
            try:
                await self._emit_invocation_status(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    status="failed",
                    error={"code": "internal_error", "message": str(exc)},
                    close=True,
                )
            except (EventStreamClosedError, EventStreamNotFoundError):
                logger.debug(
                    "Invocation %s already has a terminal replay stream",
                    invocation_id,
                )
            finally:
                self._cancel_requests.pop(invocation_id, None)
            raise

        result = self._invocation_envelope(
            invocation_id,
            session_id,
            "completed",
            response=response_text,
        )
        try:
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="completed",
                response=response_text,
                close=True,
            )
        except (EventStreamClosedError, EventStreamNotFoundError):
            terminal_result = await self._terminal_invocation_result(invocation_id)
            if terminal_result is not None:
                result = terminal_result
            elif cancel_request.is_set():
                result = self._invocation_envelope(
                    invocation_id,
                    session_id,
                    "cancelled",
                    error={
                        "code": "cancelled",
                        "message": "Invocation was cancelled.",
                    },
                )
            else:
                raise
        finally:
            self._cancel_requests.pop(invocation_id, None)
        return result

    async def _complete_cancelled_invocation(
        self,
        event_stream: EventStream,
        *,
        invocation_id: str,
        session_id: str,
        steered: bool,
    ) -> dict[str, Any]:
        code = "steered" if steered else "cancelled"
        message = (
            "Invocation was superseded by a steered turn."
            if steered
            else "Invocation was cancelled."
        )
        error = {"code": code, "message": message}
        result = self._invocation_envelope(
            invocation_id,
            session_id,
            "cancelled",
            error=error,
        )
        try:
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="cancelled",
                error=error,
                close=True,
            )
        except (EventStreamClosedError, EventStreamNotFoundError):
            logger.debug(
                "Invocation %s already has a terminal replay stream",
                invocation_id,
            )
            terminal_result = await self._terminal_invocation_result(invocation_id)
            if terminal_result is not None:
                return terminal_result
        return result

    async def _handle_get_invocation(self, request: Request) -> Response:
        invocation_id = str(request.path_params["invocation_id"])
        event = await self._latest_invocation_event(invocation_id)
        if event is None:
            return JSONResponse(
                {"error": "Invocation not found."},
                status_code=404,
            )
        return JSONResponse(self._public_invocation_event(event))

    async def _terminal_invocation_result(
        self,
        invocation_id: str,
    ) -> Optional[dict[str, Any]]:
        event = await self._latest_invocation_event(invocation_id)
        if event is None or event.get("status") not in {
            "completed",
            "failed",
            "cancelled",
        }:
            return None
        return self._public_invocation_event(event)

    async def _handle_cancel_invocation(self, request: Request) -> Response:
        invocation_id = str(request.path_params["invocation_id"])
        event = await self._latest_invocation_event(invocation_id)
        if event is None:
            return JSONResponse(
                {"error": "Invocation not found."},
                status_code=404,
            )
        if event.get("status") in {"completed", "failed", "cancelled"}:
            return JSONResponse(self._public_invocation_event(event))

        session_id = event.get("agent_session_id")
        active_run: Optional[TaskRun[dict[str, Any]]] = None
        if (
            self._invocation_task is not None
            and isinstance(session_id, str)
            and session_id
        ):
            active_run = await self._invocation_task.get_active_run(
                session_id,
                invocation_id,
            )
        task_run = active_run or self._active_runs.get(invocation_id)
        if task_run is None:
            latest_event = await self._latest_invocation_event(invocation_id)
            if latest_event is not None and latest_event.get("status") in {
                "completed",
                "failed",
                "cancelled",
            }:
                return JSONResponse(self._public_invocation_event(latest_event))
            return JSONResponse(
                {"error": "Invocation is not active in this process."},
                status_code=409,
            )

        cancel_request = self._cancel_requests.setdefault(
            invocation_id,
            asyncio.Event(),
        )
        cancel_request.set()
        await task_run.cancel()
        session_id = str(session_id)
        status = "cancelling"
        if active_run is None and task_run.is_queued:
            try:
                await task_run.result()
            except TaskCancelled:
                status = "cancelled"
                event_stream = await streams.get_or_create(invocation_id)
                await self._complete_cancelled_invocation(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    steered=False,
                )
                asyncio.get_running_loop().call_later(
                    _REPLAY_EVENT_TTL_SECONDS,
                    self._expire_cancel_request,
                    invocation_id,
                    cancel_request,
                )

        latest_event = await self._latest_invocation_event(invocation_id)
        if latest_event is not None and latest_event.get("status") in {
            "completed",
            "failed",
            "cancelled",
        }:
            return JSONResponse(self._public_invocation_event(latest_event))
        return JSONResponse(
            self._invocation_envelope(invocation_id, session_id, status)
        )

    def _expire_cancel_request(
        self,
        invocation_id: str,
        cancel_request: asyncio.Event,
    ) -> None:
        if self._cancel_requests.get(invocation_id) is cancel_request:
            self._cancel_requests.pop(invocation_id, None)

    async def _latest_invocation_event(
        self,
        invocation_id: str,
    ) -> Optional[dict[str, Any]]:
        try:
            event_stream = await streams.get_or_create(invocation_id)
            cursor = await event_stream.last_cursor()
        except EventStreamNotFoundError:
            await streams.delete(invocation_id)
            return None
        if cursor is None:
            await streams.delete(invocation_id)
            return None

        subscriber: Optional[AsyncIterator[Any]] = None
        try:
            subscriber = event_stream.subscribe(after=cursor - 1)
            event = await anext(subscriber)
        except (EventStreamNotFoundError, StopAsyncIteration):
            return None
        finally:
            if subscriber is not None:
                close = getattr(subscriber, "aclose", None)
                if close is not None:
                    await close()
        return event if isinstance(event, dict) else None

    async def _emit_invocation_status(
        self,
        event_stream: EventStream,
        *,
        invocation_id: str,
        session_id: str,
        status: str,
        response: Optional[str] = None,
        error: Optional[dict[str, str]] = None,
        token: Optional[str] = None,
        close: bool = False,
    ) -> None:
        cursor = await event_stream.last_cursor()
        event = self._invocation_envelope(
            invocation_id,
            session_id,
            status,
            response=response,
            error=error,
        )
        if token is not None:
            event["token"] = token
        event["sequence_number"] = 0 if cursor is None else cursor + 1
        await event_stream.emit(event, close=close)

    @staticmethod
    def _invocation_envelope(
        invocation_id: str,
        session_id: str,
        status: str,
        *,
        response: Optional[str] = None,
        error: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "id": invocation_id,
            "status": status,
            "agent_session_id": session_id,
        }
        if response is not None:
            event["response"] = response
        if error is not None:
            event["error"] = error
        return event

    @staticmethod
    def _public_invocation_event(event: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in event.items()
            if key not in {"sequence_number", "token"}
        }

    async def _stream_task_events(
        self,
        event_stream: EventStream,
    ) -> AsyncIterator[bytes]:
        async for event in event_stream.subscribe():
            if not isinstance(event, dict):
                continue
            token = event.get("token")
            if isinstance(token, str) and token:
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")
            status = event.get("status")
            if status == "completed":
                yield b"event: done\ndata: {}\n\n"
                return
            if status in {"failed", "cancelled"}:
                payload = json.dumps(event.get("error") or {}, ensure_ascii=False)
                yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
                return

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
