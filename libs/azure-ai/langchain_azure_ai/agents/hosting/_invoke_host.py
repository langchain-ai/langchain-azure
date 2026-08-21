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
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, cast

try:
    from azure.ai.agentserver.core import (
        FoundryAgentRequestContext,
        reset_request_context,
        resolve_state_subdir,
        set_request_context,
    )
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
        resilient_tasks_enabled,
        set_resilient_tasks_enabled,
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
from langgraph.types import Command
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.hosting import (
    HostingFeature,
    _add_process_hosting_features,
    _hosting_feature_scope,
)

from ._converters import (
    build_messages_input_from_text,
    detect_approval_rejection,
    detect_pending_interrupts,
    extract_text,
    interrupt_output_items,
    is_messages_state_schema,
    last_ai_message_text,
    parse_resume_command,
    track_pending_interrupts,
)
from ._invocation_store import (
    InvocationStateStore,
    create_invocation_state_store,
)

logger = logging.getLogger(__name__)

GraphInputT = TypeVar("GraphInputT")
GraphOutputT = TypeVar("GraphOutputT")
InvocationOutputParser = Callable[[GraphOutputT], str]
InvocationInput = str | list[dict[str, Any]]

_INVOCATION_TASK_NAME = "langchain_invocations"
_METADATA_RESPONSE = "invocation_response"
_METADATA_OUTPUT = "invocation_output"
_METADATA_CHECKPOINT_ID = "langgraph_checkpoint_id"
_METADATA_CHECKPOINT_THREAD_ID = "langgraph_thread_id"
_REPLAY_EVENT_TTL_SECONDS = 600.0
_USER_ID_HEADER = "x-agent-user-id"
_FOUNDRY_CALL_ID_HEADER = "x-agent-foundry-call-id"


class _HITLRequestError(ValueError):
    def __init__(self, message: str, *, code: str = "invalid_hitl_input") -> None:
        super().__init__(message)
        self.code = code


@contextmanager
def _platform_request_context(
    *,
    user_id: Optional[str],
    call_id: Optional[str],
    session_id: Optional[str],
) -> Iterator[None]:
    """Bind platform identity for the current execution scope."""
    context_token = set_request_context(
        FoundryAgentRequestContext(
            call_id=call_id,
            user_id=user_id,
            session_id=session_id,
        )
    )
    try:
        yield
    finally:
        reset_request_context(context_token)


@contextmanager
def _invocation_request_context(request: Request) -> Iterator[None]:
    """Bind platform identity for Invocations GET and cancel handlers."""
    user_id = request.headers.get(_USER_ID_HEADER)
    call_id = request.headers.get(_FOUNDRY_CALL_ID_HEADER)
    raw_session_id = getattr(request.state, "session_id", None)
    session_id = raw_session_id if isinstance(raw_session_id, str) else None
    request.state.user_id = user_id or ""
    request.state.call_id = call_id or ""
    with _platform_request_context(
        call_id=call_id or None,
        user_id=user_id or None,
        session_id=session_id or None,
    ):
        yield


def _internal_session_id(session_id: str, user_id: object) -> str:
    """Return a bounded task/thread ID scoped by user and public session."""
    normalized_user_id = user_id if isinstance(user_id, str) else ""
    digest = hashlib.sha256()
    digest.update(normalized_user_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(session_id.encode("utf-8"))
    return f"session-{digest.hexdigest()}"


def _validate_hitl_input_items(items: list[dict[str, Any]]) -> None:
    for index, item in enumerate(items):
        item_type = item.get("type")
        if item_type == "function_call_output":
            if not isinstance(item.get("call_id"), str) or not item["call_id"]:
                raise ValueError(
                    f"Structured HITL item {index} must include a non-empty "
                    "'call_id' string."
                )
            if "output" not in item or not isinstance(
                item["output"], (str, list, dict)
            ):
                raise ValueError(
                    f"Structured HITL item {index} must include 'output' as "
                    "a string, list, or object."
                )
            continue
        if item_type == "mcp_approval_response":
            approval_id = item.get("approval_request_id")
            if not isinstance(approval_id, str) or not approval_id:
                raise ValueError(
                    f"Structured HITL item {index} must include a non-empty "
                    "'approval_request_id' string."
                )
            if not isinstance(item.get("approve"), bool):
                raise ValueError(
                    f"Structured HITL item {index} must include a boolean "
                    "'approve' field."
                )
            reason = item.get("reason")
            if reason is not None and not isinstance(reason, str):
                raise ValueError(
                    f"Structured HITL item {index} field 'reason' must be a string."
                )
            continue
        raise ValueError(
            f"Structured HITL item {index} must have type "
            "'function_call_output' or 'mcp_approval_response'."
        )


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


def _checkpoint_from_recovery_state(
    state: dict[str, Any],
) -> tuple[str, str] | None:
    thread_id = state.get(_METADATA_CHECKPOINT_THREAD_ID)
    checkpoint_id = state.get(_METADATA_CHECKPOINT_ID)
    if not isinstance(thread_id, str) or not thread_id:
        return None
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        return None
    return thread_id, checkpoint_id


def _stale_recovery_stream_lock_path(exc: RuntimeError) -> Optional[Path]:
    """Return an SDK lock file that a recovered task owns and can reclaim."""
    cause = exc.__cause__
    if not isinstance(cause, FileExistsError) or not isinstance(cause.filename, str):
        return None

    lock_path = Path(cause.filename)
    stream_path = lock_path.with_suffix("")
    if lock_path.suffix != ".lock" or stream_path.suffix != ".jsonl":
        return None
    try:
        stream_dir = Path(resolve_state_subdir("streams")).resolve()
        if lock_path.parent.resolve() != stream_dir or not stream_path.is_file():
            return None
    except OSError:
        return None
    return lock_path


async def _get_or_create_invocation_event_stream(
    invocation_id: str,
    *,
    reclaim_stale_lock: bool = False,
) -> EventStream:
    try:
        return await streams.get_or_create(invocation_id)
    except RuntimeError as exc:
        lock_path = (
            _stale_recovery_stream_lock_path(exc) if reclaim_stale_lock else None
        )
        if lock_path is None:
            raise
        # Recovery begins only after the durable task manager has reclaimed
        # ownership. Fresh invocations must never remove another writer's lock.
        try:
            lock_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            raise exc from cleanup_exc
        logger.warning(
            "Recovered invocation %s reclaimed stale replay-stream lock %s",
            invocation_id,
            lock_path,
        )
        return await streams.get_or_create(invocation_id)


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

        The host maps ``agent_session_id`` to a user-partitioned internal
        ``RunnableConfig.configurable.thread_id`` so follow-up turns continue
        the same checkpointed conversation without colliding across users.

    .. code-block:: json

        { "message": "Hello!", "stream": false }

    Where:

        - ``message`` (required) — user message text, or a non-empty list containing
            a Responses-style ``function_call_output`` / ``mcp_approval_response``
            item that answers a pending LangGraph interrupt.
    - ``stream`` (optional, default ``false``) — when ``true`` returns SSE
      with token deltas; when ``false`` returns a single JSON response.
        - ``background`` (optional, default ``false``) — when ``true`` starts a
            durable invocation and returns ``202``. Requires
            ``options.resilient_background=True`` and a LangGraph checkpointer.
        - ``previous_invocation_id`` (optional) — linear-chain precondition for a
            continued ``agent_session_id``.

        Pending LangGraph interrupts are exposed beside ``response`` as the same
        paired ``function_call`` and ``mcp_approval_request`` output items used by
        :class:`ResponsesHostServer`. Streaming requests emit each item as an
        ``output_item`` SSE event.

    Multi-turn continuation uses the ``agent_session_id`` query param /
    ``x-agent-session-id`` header populated by
    :class:`InvocationAgentServerHost`. The public session id remains in the
    API envelope, while task and LangGraph state use an internal id derived
    from the user partition and session id.

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
        self._hosting_features = HostingFeature.INVOCATIONS
        if options is not None and options.resilient_background:
            self._hosting_features |= HostingFeature.RESILIENT_BACKGROUND
        if options is not None and options.steerable_conversations:
            self._hosting_features |= HostingFeature.STEERABLE_CONVERSATIONS
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
        _add_process_hosting_features(self._hosting_features)
        self._output_parser = output_parser
        self._options = options or ResponsesServerOptions()
        self._invocation_task: Optional[
            MultiTurnTask[dict[str, Any], dict[str, Any]]
        ] = None
        self._active_runs: dict[str, TaskRun[dict[str, Any]]] = {}
        self._run_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._cancel_requests: dict[str, asyncio.Event] = {}
        self._invocation_state_store: Optional[InvocationStateStore] = None

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
            if not resilient_tasks_enabled():
                set_resilient_tasks_enabled(True)
            self._invocation_state_store = create_invocation_state_store(
                hosted=bool(getattr(self._app.config, "is_hosted", False))
            )
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

        ``message`` is either user text or a list containing a structured HITL
        response item. ``stream`` is optional and defaults to ``false``.
        Non-streaming requests return JSON:

        .. code-block:: json

            {"response": "Assistant text"}

        A pending LangGraph interrupt adds Responses-style ``function_call``
        and ``mcp_approval_request`` items under ``output``. Resume it by
        sending the matching ``function_call_output`` or
        ``mcp_approval_response`` as the next request's ``message`` list.

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
        ``data: {"token": "..."}`` payloads, any pending HITL items as
        ``event: output_item``, and finally ``event: done``.
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

    async def parse_request(self, request: Request) -> tuple[InvocationInput, bool]:
        """Parse the invocation request body.

        Default implementation reads ``message`` as either non-empty text or a
        non-empty list of structured HITL response items, plus an optional
        boolean ``stream``. Override to support a different body schema.

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
        if isinstance(message, str):
            if not message:
                raise ValueError(
                    "Request body must include a non-empty 'message' string."
                )
        elif not (
            isinstance(message, list)
            and message
            and all(isinstance(item, dict) for item in message)
        ):
            raise ValueError(
                "Request body must include a non-empty 'message' string or "
                "a non-empty list of structured HITL items."
            )
        else:
            _validate_hitl_input_items(message)

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

        Sets ``configurable.thread_id`` from a user-scoped internal form of
        ``request.state.session_id`` so checkpointed conversations cannot
        collide across users that choose the same public session ID.

        Args:
            request: The Starlette request.

        Returns:
            A ``RunnableConfig`` dict.
        """
        session_id = getattr(request.state, "session_id", None)
        public_session_id = session_id if isinstance(session_id, str) else "default"
        return {
            "configurable": {
                "thread_id": _internal_session_id(
                    public_session_id,
                    getattr(request.state, "user_id", None),
                )
            }
        }

    def build_task_runnable_config(
        self,
        session_id: str,
        context: TaskContext[dict[str, Any]],
    ) -> RunnableConfig:
        """Build config for a task-backed invocation attempt.

        The graph thread uses ``context.task_id``, the user-partitioned
        internal ID for the public session. The public ``session_id`` remains
        available in the invocation envelope and task input only.

        The task context is transport-scoped rather than checkpointed graph
        state. Nodes can use it to inspect recovery, cancellation, steering,
        and shutdown signals for the current attempt.
        """
        del session_id  # retained for subclass signature compatibility
        configurable: dict[str, Any] = {
            "thread_id": context.task_id,
            "invocation_context": context,
            "invocation_cancellation_signal": context.cancel,
        }
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

    async def _prepare_graph_input(
        self,
        message: InvocationInput,
        config: RunnableConfig,
    ) -> tuple[Optional[GraphInputT], list[dict[str, Any]]]:
        pending = await detect_pending_interrupts(self._graph, config)
        if not pending:
            if not isinstance(message, str):
                raise _HITLRequestError(
                    "Structured HITL items require a pending LangGraph interrupt."
                )
            return self.build_input(message), []

        pending_items = interrupt_output_items(pending)
        if isinstance(message, str):
            graph_input = self.build_input(message)
            if isinstance(graph_input, Command):
                return graph_input, []
            return None, pending_items

        rejection = detect_approval_rejection(message, pending)
        if rejection is not None:
            raise _HITLRequestError(rejection, code="interrupt_rejected")
        resume_command, _ = parse_resume_command(message, pending)
        if resume_command is not None:
            return cast(GraphInputT, resume_command), []
        return None, pending_items

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

    async def _invoke_graph(
        self,
        graph_input: GraphInputT,
        config: RunnableConfig,
    ) -> tuple[GraphOutputT, list[Any]]:
        if not (self._supports_langgraph_stream_modes and self._graph_has_checkpointer):
            output = await self._graph.ainvoke(graph_input, config=config)
            pending = await detect_pending_interrupts(self._graph, config)
            return output, list(pending)

        latest_state: Any = None
        active_interrupts: list[Any] = []
        graph_stream = track_pending_interrupts(
            self._graph.astream(
                graph_input,
                config=config,
                stream_mode=["values", "updates"],
            ),
            active_interrupts,
        )
        async for chunk in graph_stream:
            mode, payload = _classify_graph_stream_chunk(chunk)
            if mode == "values":
                latest_state = payload
            elif mode is None:
                latest_state = _accumulate_stream_output(latest_state, payload)
        if latest_state is None:
            raise RuntimeError("LangGraph invocation produced no output state.")
        return cast(GraphOutputT, latest_state), active_interrupts

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    async def _handle_invoke(self, request: Request) -> Response:
        with _hosting_feature_scope(self._hosting_features):
            return await self._handle_invoke_with_features(request)

    async def _handle_invoke_with_features(self, request: Request) -> Response:
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

        config = self.build_runnable_config(request)
        try:
            graph_input, pending_items = await self._prepare_graph_input(
                message, config
            )
        except _HITLRequestError as exc:
            status_code = 409 if exc.code == "interrupt_rejected" else 400
            return JSONResponse({"error": str(exc)}, status_code=status_code)

        if graph_input is None:
            if stream:
                return StreamingResponse(
                    self._stream_pending_items(pending_items),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
            return JSONResponse({"response": "", "output": pending_items})

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
            output, active_interrupts = await self._invoke_graph(graph_input, config)
        except Exception:  # noqa: BLE001
            logger.exception("LangGraph invocation failed")
            return JSONResponse({"error": "Internal server error."}, status_code=500)

        text = self.parse_output(output)
        body: dict[str, Any] = {"response": text}
        pending_items = interrupt_output_items(active_interrupts)
        if pending_items:
            body["output"] = pending_items
        return JSONResponse(body)

    async def _start_background_invocation(
        self,
        request: Request,
        message: InvocationInput,
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
        message: InvocationInput,
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
            body: dict[str, Any] = {"response": result.get("response", "")}
            output_items = result.get("output")
            if isinstance(output_items, list) and output_items:
                body["output"] = output_items
            return JSONResponse(body)
        error = result.get("error") or {}
        status_code = 400 if error.get("code") == "invalid_hitl_input" else 409
        return JSONResponse(
            {"error": error.get("message", "Invocation was cancelled.")},
            status_code=status_code,
        )

    async def _start_task_invocation(
        self,
        request: Request,
        message: InvocationInput,
        *,
        stream: bool,
        previous_invocation_id: Optional[str],
    ) -> tuple[TaskRun[dict[str, Any]], EventStream, str, str]:
        if self._invocation_task is None:
            raise RuntimeError("Task-backed invocation hosting is not configured.")

        invocation_id = str(request.state.invocation_id)
        session_id = str(request.state.session_id)
        user_id = getattr(request.state, "user_id", None)
        call_id = getattr(request.state, "call_id", None)
        task_id = _internal_session_id(session_id, user_id)
        event_stream = await streams.get_or_create(invocation_id)
        await self._emit_invocation_status(
            event_stream,
            invocation_id=invocation_id,
            session_id=session_id,
            status="queued",
        )

        start_kwargs: dict[str, Any] = {
            "task_id": task_id,
            "input_id": invocation_id,
            "input": {
                "invocation_id": invocation_id,
                "session_id": session_id,
                "user_id": user_id,
                "call_id": call_id,
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

    async def _get_task_recovery_state(
        self,
        invocation_id: str,
    ) -> dict[str, Any]:
        if self._invocation_state_store is None:
            raise RuntimeError("Task-backed invocation state is not configured.")
        return (
            await self._invocation_state_store.get_recovery_state(invocation_id) or {}
        )

    async def _set_task_recovery_state(
        self,
        invocation_id: str,
        state: dict[str, Any],
    ) -> None:
        if self._invocation_state_store is None:
            raise RuntimeError("Task-backed invocation state is not configured.")
        await self._invocation_state_store.set_recovery_state(invocation_id, state)

    async def _execute_task_invocation(
        self,
        context: TaskContext[dict[str, Any]],
    ) -> dict[str, Any]:
        raw_user_id = context.input.get("user_id")
        user_id = raw_user_id if isinstance(raw_user_id, str) and raw_user_id else None
        raw_call_id = context.input.get("call_id")
        call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id else None
        with _platform_request_context(
            call_id=call_id,
            user_id=user_id,
            session_id=str(context.input["session_id"]),
        ):
            return await self._execute_task_invocation_with_context(context)

    async def _execute_task_invocation_with_context(
        self,
        context: TaskContext[dict[str, Any]],
    ) -> dict[str, Any]:
        task_input = context.input
        invocation_id = str(task_input["invocation_id"])
        session_id = str(task_input["session_id"])
        message = cast(InvocationInput, task_input["message"])
        stream_tokens = bool(task_input.get("stream", False))
        event_stream = await _get_or_create_invocation_event_stream(
            invocation_id,
            reclaim_stale_lock=context.entry_mode == "recovered",
        )
        cancel_request = self._cancel_requests.setdefault(
            invocation_id,
            asyncio.Event(),
        )

        if context.shutdown.is_set():
            return await context.exit_for_recovery()

        recovery_state: dict[str, Any] = {}
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
            recovery_state = await self._get_task_recovery_state(invocation_id)

        if context.entry_mode == "recovered" and isinstance(
            recovery_state.get(_METADATA_RESPONSE), str
        ):
            response_text = recovery_state[_METADATA_RESPONSE]
            stored_output = recovery_state.get(_METADATA_OUTPUT)
            output_items = stored_output if isinstance(stored_output, list) else []
            result = self._invocation_envelope(
                invocation_id,
                session_id,
                "completed",
                response=response_text,
                output=output_items,
            )
            try:
                await self._emit_invocation_status(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    status="completed",
                    response=response_text,
                    output=output_items,
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
        recovery_checkpoint = (
            _checkpoint_from_recovery_state(recovery_state)
            if context.entry_mode == "recovered"
            else None
        )
        if recovery_checkpoint is not None:
            checkpoint_thread_id, checkpoint_id = recovery_checkpoint
            configurable = cast(
                dict[str, Any],
                config.setdefault("configurable", {}),
            )
            configurable.update(
                thread_id=checkpoint_thread_id,
                checkpoint_id=checkpoint_id,
                checkpoint_ns="",
            )
        resume_from_checkpoint = recovery_checkpoint is not None
        pending_items: list[dict[str, Any]] = []
        if resume_from_checkpoint:
            graph_input = None
        else:
            try:
                graph_input, pending_items = await self._prepare_graph_input(
                    message, config
                )
            except _HITLRequestError as exc:
                return await self._complete_failed_task_invocation(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    error={"code": exc.code, "message": str(exc)},
                )
            if graph_input is None:
                response_text = ""
                recovery_state[_METADATA_RESPONSE] = response_text
                recovery_state[_METADATA_OUTPUT] = pending_items
                await self._set_task_recovery_state(
                    invocation_id,
                    recovery_state,
                )
                return await self._complete_task_invocation(
                    event_stream,
                    invocation_id=invocation_id,
                    session_id=session_id,
                    response_text=response_text,
                    output_items=pending_items,
                    cancel_request=cancel_request,
                )
        latest_state: Any = None
        active_interrupts: list[Any] = []

        if self._supports_langgraph_stream_modes:
            stream_modes = ["values", "updates"]
            if stream_tokens:
                stream_modes.append("messages")
            graph_stream_kwargs: dict[str, Any] = {}
            if self._graph_has_checkpointer:
                stream_modes.append("checkpoints")
                graph_stream_kwargs["durability"] = "sync"
            raw_graph_stream = self._graph.astream(
                cast(GraphInputT, graph_input),
                config=config,
                stream_mode=stream_modes,
                **graph_stream_kwargs,
            )
            graph_stream = track_pending_interrupts(
                raw_graph_stream,
                active_interrupts,
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
                                recovery_state[_METADATA_CHECKPOINT_THREAD_ID] = (
                                    thread_id
                                )
                                recovery_state[_METADATA_CHECKPOINT_ID] = checkpoint_id
                                await self._set_task_recovery_state(
                                    invocation_id,
                                    recovery_state,
                                )

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
            pending_items = interrupt_output_items(active_interrupts)
            recovery_state[_METADATA_RESPONSE] = response_text
            recovery_state[_METADATA_OUTPUT] = pending_items
            await self._set_task_recovery_state(
                invocation_id,
                recovery_state,
            )
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

        return await self._complete_task_invocation(
            event_stream,
            invocation_id=invocation_id,
            session_id=session_id,
            response_text=response_text,
            output_items=pending_items,
            cancel_request=cancel_request,
        )

    async def _complete_task_invocation(
        self,
        event_stream: EventStream,
        *,
        invocation_id: str,
        session_id: str,
        response_text: str,
        output_items: list[dict[str, Any]],
        cancel_request: asyncio.Event,
    ) -> dict[str, Any]:
        result = self._invocation_envelope(
            invocation_id,
            session_id,
            "completed",
            response=response_text,
            output=output_items,
        )
        try:
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="completed",
                response=response_text,
                output=output_items,
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

    async def _complete_failed_task_invocation(
        self,
        event_stream: EventStream,
        *,
        invocation_id: str,
        session_id: str,
        error: dict[str, str],
    ) -> dict[str, Any]:
        result = self._invocation_envelope(
            invocation_id,
            session_id,
            "failed",
            error=error,
        )
        try:
            await self._emit_invocation_status(
                event_stream,
                invocation_id=invocation_id,
                session_id=session_id,
                status="failed",
                error=error,
                close=True,
            )
        except (EventStreamClosedError, EventStreamNotFoundError):
            terminal_result = await self._terminal_invocation_result(invocation_id)
            if terminal_result is not None:
                result = terminal_result
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
        with _invocation_request_context(request):
            return await self._handle_get_invocation_with_context(request)

    async def _handle_get_invocation_with_context(self, request: Request) -> Response:
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
        with _invocation_request_context(request):
            return await self._handle_cancel_invocation_with_context(request)

    async def _handle_cancel_invocation_with_context(
        self,
        request: Request,
    ) -> Response:
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
            task_id = _internal_session_id(
                session_id,
                getattr(request.state, "user_id", None),
            )
            active_run = await self._invocation_task.get_active_run(
                task_id,
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
        if self._invocation_state_store is not None:
            persisted = await self._invocation_state_store.get(invocation_id)
            return persisted

        # Hosts without task-backed execution have no durable state provider.
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
        output: Optional[list[dict[str, Any]]] = None,
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
            output=output,
            error=error,
        )
        if token is not None:
            event["token"] = token
        event["sequence_number"] = 0 if cursor is None else cursor + 1
        if self._invocation_state_store is not None and token is None:
            await self._invocation_state_store.set(event)
        await event_stream.emit(event, close=close)

    @staticmethod
    def _invocation_envelope(
        invocation_id: str,
        session_id: str,
        status: str,
        *,
        response: Optional[str] = None,
        output: Optional[list[dict[str, Any]]] = None,
        error: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "id": invocation_id,
            "status": status,
            "agent_session_id": session_id,
        }
        if response is not None:
            event["response"] = response
        if output:
            event["output"] = output
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
                for item in event.get("output") or []:
                    payload = json.dumps(item, ensure_ascii=False)
                    yield f"event: output_item\ndata: {payload}\n\n".encode("utf-8")
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
        with _hosting_feature_scope(self._hosting_features):
            active_interrupts: list[Any] = []
            try:
                if (
                    self._supports_langgraph_stream_modes
                    and self._graph_has_checkpointer
                ):
                    graph_stream = track_pending_interrupts(
                        self._graph.astream(
                            graph_input,
                            config=config,
                            stream_mode=["messages", "updates"],
                        ),
                        active_interrupts,
                    )
                else:
                    graph_stream = self._graph.astream(
                        graph_input,
                        config=config,
                        stream_mode="messages",
                    )
                async for chunk in graph_stream:
                    mode, payload = _classify_graph_stream_chunk(chunk)
                    message_chunk = _extract_message_chunk(
                        payload if mode == "messages" else chunk
                    )
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

            if not (
                self._supports_langgraph_stream_modes and self._graph_has_checkpointer
            ):
                active_interrupts.extend(
                    await detect_pending_interrupts(self._graph, config)
                )
            for item in interrupt_output_items(active_interrupts):
                payload = json.dumps(item, ensure_ascii=False)
                yield f"event: output_item\ndata: {payload}\n\n".encode("utf-8")
            yield b"event: done\ndata: {}\n\n"

    @staticmethod
    async def _stream_pending_items(
        pending_items: list[dict[str, Any]],
    ) -> AsyncIterator[bytes]:
        for item in pending_items:
            payload = json.dumps(item, ensure_ascii=False)
            yield f"event: output_item\ndata: {payload}\n\n".encode("utf-8")
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
