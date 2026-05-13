# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host a LangGraph ``CompiledStateGraph`` as the Azure AI Responses API.

Modeled after Microsoft Agent Framework's ``ResponsesHostServer``: pass a
graph, get a server.

Quick start::

    from langchain_azure_ai.agents.hosting import LangGraphResponsesAgentHost

    LangGraphResponsesAgentHost(my_compiled_graph).run()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Optional

try:
    from azure.ai.agentserver.responses import (
        CreateResponse,
        ResponseContext,
        ResponseEventStream,
        ResponseProviderProtocol,
        ResponsesAgentServerHost,
        ResponsesServerOptions,
    )
    from azure.ai.agentserver.responses.models._helpers import to_item
except ImportError as exc:
    raise ImportError(
        "The azure-ai-agentserver-responses package is required to use "
        "LangGraphResponsesAgentHost. Please install it via "
        "`pip install azure-ai-agentserver-responses` or "
        "`pip install langchain-azure-ai[hosting]`."
    ) from exc

from ._converters import (
    build_messages_input,
    detect_pending_interrupts,
    emit_interrupts,
    is_messages_state_schema,
    parse_resume_command,
    state_to_events,
    stream_graph_to_events,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Command, Interrupt

logger = logging.getLogger(__name__)


class LangGraphResponsesAgentHost:
    """Host a LangGraph ``CompiledStateGraph`` as the Azure AI Responses API.

    The host owns an internal :class:`ResponsesAgentServerHost` and
    registers a default request → graph → events conversion pipeline
    against it. For advanced scenarios (custom routes, multi-protocol
    composition, custom converter), users may either:

    - subclass and override :meth:`handle_create`, or
    - drop down to :class:`ResponsesAgentServerHost` directly and write
      their own ``@response_handler``.

    Args:
        graph: The compiled LangGraph state graph to host. By default the
            state schema must declare a ``messages`` field. Non-
            ``MessagesState`` graphs require subclassing and overriding
            :meth:`build_input` / :meth:`handle_create` (typically by
            reusing this class as a starting point).

    Keyword Args:
        app: Optional existing :class:`ResponsesAgentServerHost` to attach
            to (e.g. a multi-protocol mixin). In this mode the host-level
            kwargs are ignored — the caller is expected to have
            configured them on ``app`` itself.
        options: Optional :class:`ResponsesServerOptions` forwarded to
            :class:`ResponsesAgentServerHost`.
        store: Optional :class:`ResponseProviderProtocol`. When ``None``,
            the responses package defaults apply (in-memory provider, or
            ``FoundryStorageProvider`` when running on Foundry).
        prefix: URL prefix for response routes (e.g. ``"/v1"``).
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
        graph: "CompiledStateGraph",
        *,
        app: Optional[ResponsesAgentServerHost] = None,
        options: Optional[ResponsesServerOptions] = None,
        store: Optional[ResponseProviderProtocol] = None,
        prefix: str = "",
        applicationinsights_connection_string: Optional[str] = None,
        graceful_shutdown_timeout: Optional[int] = None,
    ) -> None:
        self._validate_graph_schema(graph)
        self._graph = graph

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

            self._app = ResponsesAgentServerHost(
                prefix=prefix,
                options=options,
                store=store,
                **host_kwargs,
            )

        # Wire the create handler.
        self._app.response_handler(self._handle_create_async_gen)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def app(self) -> ResponsesAgentServerHost:
        """The underlying :class:`ResponsesAgentServerHost`."""
        return self._app

    @property
    def graph(self) -> "CompiledStateGraph":
        """The hosted compiled state graph."""
        return self._graph

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self, host: str = "0.0.0.0", port: Optional[int] = None) -> None:
        """Start the server synchronously.

        Args:
            host: Network interface to bind. Defaults to ``"0.0.0.0"``.
            port: Port to bind. Defaults to ``PORT`` env var or 8088.
        """
        self._app.run(host=host, port=port)

    async def run_async(
        self, host: str = "0.0.0.0", port: Optional[int] = None
    ) -> None:
        """Start the server asynchronously.

        Args:
            host: Network interface to bind.
            port: Port to bind.
        """
        await self._app.run_async(host=host, port=port)

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    async def build_input(
        self,
        request: CreateResponse,
        context: ResponseContext,
        *,
        skip_call_ids: Optional[frozenset[str]] = None,
    ) -> dict[str, Any]:
        """Translate the request into LangGraph input.

        Default implementation builds a ``{"messages": [...]}`` payload by
        prepending ``request.instructions``, then any conversation history
        resolved from ``previous_response_id`` / ``conversation`` via the
        configured :class:`ResponseProviderProtocol`, then the current
        request's input items. Override in a subclass to support
        custom-state graphs.

        Args:
            request: The parsed create-response request.
            context: The response context for the request.
            skip_call_ids: ``function_call_output`` items whose
                ``call_id`` is in this set are excluded from the messages
                payload. Populated by the HITL resume path so the resume
                item isn't double-fed to the graph.

        Returns:
            A LangGraph input value (typically a state dict).
        """
        history_output_items = await context.get_history()
        history_items = [
            it
            for output_item in history_output_items
            if (it := to_item(output_item)) is not None
        ]
        current_items = list(await context.get_input_items())
        all_items = history_items + current_items
        instructions = getattr(request, "instructions", None)
        return build_messages_input(
            all_items,
            instructions=instructions if isinstance(instructions, str) else None,
            skip_call_ids=skip_call_ids or frozenset(),
        )

    async def build_resume_command(
        self,
        request: CreateResponse,
        context: ResponseContext,
        pending: Sequence["Interrupt"],
    ) -> tuple[Optional["Command"], frozenset[str]]:
        """Build a resume :class:`Command` from the request's input items.

        Default implementation scans the current request for a
        ``function_call_output`` whose ``call_id`` matches one of the
        pending interrupts and decodes its ``output`` JSON into a
        :class:`Command`. Override to plug in custom resume protocols.

        Args:
            request: The parsed create-response request.
            context: The response context for the request.
            pending: Interrupts currently pending on the checkpointed
                thread.

        Returns:
            A ``(command, consumed_call_ids)`` pair. ``command`` is
            ``None`` when no matching resume item was found.
        """
        del request  # unused in the default implementation
        items = await context.get_input_items()
        return parse_resume_command(items, pending)

    def build_runnable_config(
        self,
        request: CreateResponse,
        context: ResponseContext,
    ) -> dict[str, Any]:
        """Build a LangGraph ``RunnableConfig`` for the request.

        Sets ``configurable.thread_id`` so graphs compiled with a
        checkpointer naturally continue the right conversation (preferring
        ``conversation_id``, then ``previous_response_id``, then a
        per-response synthetic key).

        The synthetic ``previous_response_id`` -> ``thread_id`` mapping is
        symmetric: the initial turn's ``thread_id`` is derived from
        ``context.response_id``, and a subsequent turn carrying
        ``previous_response_id=<that id>`` reuses the same ``thread_id``
        so the checkpointed state is found.

        Args:
            request: The parsed create-response request.
            context: The response context for the request.

        Returns:
            A ``RunnableConfig`` dict.
        """
        previous_response_id = getattr(request, "previous_response_id", None)
        if context.conversation_id:
            thread_id = context.conversation_id
        elif isinstance(previous_response_id, str) and previous_response_id:
            thread_id = f"resp-{previous_response_id}"
        else:
            thread_id = f"resp-{context.response_id}"
        return {"configurable": {"thread_id": thread_id}}

    async def handle_create(
        self,
        request: CreateResponse,
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
    ) -> AsyncIterator[dict[str, Any]]:
        """Drive the graph and yield Responses API events.

        Override this when wholesale customisation is needed. By default
        the method:

        1. emits ``response.created`` / ``response.in_progress``,
        2. checks for pending ``interrupt()`` pauses on the checkpointed
           thread and, when the request includes a matching resume
           ``function_call_output``, resumes the graph with a
           :class:`langgraph.types.Command` instead of fresh messages,
        3. drives the graph via :meth:`CompiledStateGraph.astream` or
           :meth:`CompiledStateGraph.ainvoke` depending on
           ``request.stream``,
        4. emits the resulting output items, surfacing any new pending
           interrupts as ``function_call`` items named
           ``__hosted_agent_adapter_interrupt__``, and
        5. emits ``response.completed`` (or ``response.failed`` /
           ``response.cancelled`` on error).

        Args:
            request: The parsed create-response request.
            context: The response context for the request.
            cancellation_signal: Set when the request is cancelled.

        Yields:
            Responses API event payload dicts.
        """
        stream = ResponseEventStream(response_id=context.response_id, request=request)
        yield stream.emit_created()
        yield stream.emit_in_progress()

        try:
            config = self.build_runnable_config(request, context)

            # Detect a pause from a previous turn and try to resume it.
            pending = await detect_pending_interrupts(self._graph, config)
            resume_command: Optional["Command"] = None
            consumed_call_ids: frozenset[str] = frozenset()
            if pending:
                resume_command, consumed_call_ids = await self.build_resume_command(
                    request, context, pending
                )

            if pending and resume_command is None:
                # Graph is paused but the client did not supply a matching
                # ``function_call_output``. Re-emitting the pending
                # interrupts (instead of driving the graph with fresh
                # input) keeps the conversation in a recoverable state and
                # avoids sending an unbalanced message list — with a
                # dangling ``AskHuman``-style tool_call — to the model.
                async for event in emit_interrupts(pending, stream):
                    yield event
                yield stream.emit_completed()
                return

            if resume_command is not None:
                graph_input: Any = resume_command
            else:
                graph_input = await self.build_input(
                    request, context, skip_call_ids=consumed_call_ids
                )

            if context.mode_flags.stream:
                graph_stream = self._graph.astream(
                    graph_input,
                    config=config,
                    stream_mode=["updates", "messages"],
                )
                async for event in stream_graph_to_events(
                    graph_stream, stream, cancellation_signal=cancellation_signal
                ):
                    yield event
                if cancellation_signal.is_set():
                    yield stream.emit_failed(
                        code="cancelled",
                        message="Request was cancelled.",
                    )
                    return
            else:
                state = await self._graph.ainvoke(graph_input, config=config)
                async for event in state_to_events(state, stream):
                    yield event

            # Surface any interrupts the graph paused on during this turn.
            new_pending = await detect_pending_interrupts(self._graph, config)
            if new_pending:
                async for event in emit_interrupts(new_pending, stream):
                    yield event

            yield stream.emit_completed()
        except Exception as exc:  # noqa: BLE001
            logger.exception("LangGraph response handler failed")
            yield stream.emit_failed(code="internal_error", message=str(exc))

    # ------------------------------------------------------------------
    # Internal — registered as the @response_handler. Wraps handle_create
    # so subclasses only need to override the async method.
    # ------------------------------------------------------------------

    async def _handle_create_async_gen(
        self,
        request: CreateResponse,
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
    ) -> AsyncIterator[dict[str, Any]]:
        async for event in self.handle_create(request, context, cancellation_signal):
            yield event

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_graph_schema(graph: "CompiledStateGraph") -> None:
        builder = getattr(graph, "builder", None)
        state_schema = (
            getattr(builder, "state_schema", None) if builder is not None else None
        )
        if state_schema is None:
            # Older graphs may not expose a builder; trust the user.
            return
        if is_messages_state_schema(state_schema):
            return
        raise ValueError(
            "LangGraphResponsesAgentHost's default request converter only "
            "supports graphs whose state schema declares a 'messages' field. "
            "Subclass LangGraphResponsesAgentHost and override `build_input` "
            "(and optionally `handle_create`) to host custom-state graphs."
        )
