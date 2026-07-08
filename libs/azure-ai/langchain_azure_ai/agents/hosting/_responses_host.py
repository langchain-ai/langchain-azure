# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host a LangGraph ``CompiledStateGraph`` as the Azure AI Responses API.

Quick start::

    import os

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI

    from langchain_azure_ai.agents.hosting import ResponsesHostServer

    model = ChatOpenAI(
        model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o"),
    )
    graph = create_agent(model, tools=[])

    if __name__ == "__main__":
        ResponsesHostServer(graph).run(port=int(os.environ.get("PORT", "8088")))

Then call the local server::

    curl -N -X POST http://127.0.0.1:8088/responses \
        -H 'Content-Type: application/json' \
        -d '{"input":"Hello!","stream":true}'
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

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
        "ResponsesHostServer. Please install it via "
        "`pip install azure-ai-agentserver-responses` or "
        "`pip install langchain-azure-ai[hosting]`."
    ) from exc

from langchain_core.runnables import RunnableConfig

from langchain_azure_ai._api.base import experimental

from ._converters import (
    build_messages_input,
    detect_approval_rejection,
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

StreamMode = Literal[
    "values",
    "updates",
    "checkpoints",
    "tasks",
    "debug",
    "messages",
    "custom",
]

ResolvedConversationManagementMode = Literal[
    "responses_history",
    "langgraph_checkpoint",
]


def _uses_langgraph_checkpointer(graph: "CompiledStateGraph") -> bool:
    return getattr(graph, "checkpointer", None) is not None


def _is_in_memory_checkpointer(checkpointer: Any) -> bool:
    if checkpointer is None:
        return False
    cls = type(checkpointer)
    return cls.__name__ in {"MemorySaver", "InMemorySaver"} or "memory" in (
        cls.__module__.lower()
    )


def _message_count(
    items: Sequence[Any],
    *,
    skip_call_ids: frozenset[str] = frozenset(),
) -> int:
    return len(build_messages_input(items, skip_call_ids=skip_call_ids)["messages"])


@experimental()
class ResponsesHostServer:
    """Host a LangGraph ``CompiledStateGraph`` as the Azure AI Responses API.

    Example:
        Create a LangChain agent graph and host it on ``POST /responses``::

            import os

            from langchain.agents import create_agent
            from langchain_openai import ChatOpenAI

            from langchain_azure_ai.agents.hosting import ResponsesHostServer

            model = ChatOpenAI(
                model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4o"),
            )
            graph = create_agent(model, tools=[])

            ResponsesHostServer(graph).run(port=8088)

        A minimal non-streaming request is ``{"input": "Hello!"}``.
        Streaming requests use ``{"input": "Hello!", "stream": true}``.

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
        self._graph_has_checkpointer = _uses_langgraph_checkpointer(graph)

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

        self._log_conversation_management_selection()

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

        Once running, the host exposes ``POST /responses`` using the OpenAI
        Responses API create contract. See the `OpenAI Responses API create
        docs <https://platform.openai.com/docs/api-reference/responses/create>`_
        for the full request and response schema. A minimal request is:

        .. code-block:: json

            {"input": "Hello!", "stream": false}

        Common request fields include ``input``, ``instructions``,
        ``previous_response_id``, ``conversation``, and ``stream``.
        Non-streaming requests return a Responses API response object whose
        ``output`` array contains items such as ``message``, ``function_call``,
        ``function_call_output``, and ``mcp_approval_request``. Streaming
        requests return ``text/event-stream`` with lifecycle events such as
        ``response.created``, ``response.output_text.delta``, and
        ``response.completed``. Continue a conversation with
        ``previous_response_id`` or ``conversation.id``.

        Args:
            host: Network interface to bind. Defaults to ``"0.0.0.0"``.
            port: Port to bind. Defaults to ``PORT`` env var or 8088.
        """
        self._app.run(host=host, port=port)

    async def run_async(
        self, host: str = "0.0.0.0", port: Optional[int] = None
    ) -> None:
        """Start the server asynchronously.

        Exposes the same ``POST /responses`` contract as :meth:`run`, using
        the OpenAI Responses API create request and response schema. See the
        `OpenAI Responses API create docs
        <https://platform.openai.com/docs/api-reference/responses/create>`_
        for the complete contract. A minimal request is:

        .. code-block:: json

            {"input": "Hello!", "stream": false}

        Non-streaming requests return a Responses API response object whose
        ``output`` array contains items such as ``message``, ``function_call``,
        ``function_call_output``, and ``mcp_approval_request``. Streaming
        requests return ``text/event-stream`` with events such as
        ``response.created``, ``response.output_text.delta``, and
        ``response.completed``. Continue a conversation with
        ``previous_response_id`` or ``conversation.id``.

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

                Default implementation builds a ``{"messages": [...]}`` payload from
                exactly one conversation-state source:

                - ``responses_history`` mode prepends conversation history resolved
                    from ``previous_response_id`` / ``conversation`` via the configured
                    :class:`ResponseProviderProtocol`.
                - ``langgraph_checkpoint`` mode passes only the current request input;
                    prior state is restored by LangGraph through ``thread_id``.

                Override in a subclass to support custom-state graphs.

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
        mode = self._resolve_conversation_management()
        current_items = list(await context.get_input_items())
        instructions = getattr(request, "instructions", None)

        if mode == "langgraph_checkpoint":
            graph_input = build_messages_input(
                current_items,
                instructions=instructions if isinstance(instructions, str) else None,
                skip_call_ids=skip_call_ids or frozenset(),
            )
            self._log_conversation_input_built(
                mode=mode,
                request=request,
                context=context,
                history_item_count=0,
                history_message_count=0,
                current_item_count=len(current_items),
                current_message_count=_message_count(
                    current_items,
                    skip_call_ids=skip_call_ids or frozenset(),
                ),
            )
            return graph_input

        history_output_items = await context.get_history()
        history_items = [
            it
            for output_item in history_output_items
            if (it := to_item(output_item)) is not None
        ]
        graph_input = build_messages_input(
            history_items + current_items,
            instructions=instructions if isinstance(instructions, str) else None,
            skip_call_ids=skip_call_ids or frozenset(),
        )
        self._log_conversation_input_built(
            mode=mode,
            request=request,
            context=context,
            history_item_count=len(history_items),
            history_message_count=_message_count(history_items),
            current_item_count=len(current_items),
            current_message_count=_message_count(
                current_items,
                skip_call_ids=skip_call_ids or frozenset(),
            ),
        )
        return graph_input

    async def build_resume_command(
        self,
        request: CreateResponse,
        context: ResponseContext,
        pending: Sequence["Interrupt"],
    ) -> tuple[Optional["Command"], frozenset[str]]:
        """Build a resume :class:`Command` from the request's input items.

        Default implementation scans the current request for either a
        ``function_call_output`` whose ``call_id`` matches one of the
        pending interrupts, or an ``mcp_approval_response`` whose
        ``approval_request_id`` matches. The former decodes its
        ``output`` JSON into a :class:`Command`; the latter resumes with
        the interrupt's own value when ``approve=True``. Rejections
        (``approve=False``) are surfaced via :meth:`detect_rejection`
        instead. Override to plug in custom resume protocols.

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

    async def detect_rejection(
        self,
        request: CreateResponse,
        context: ResponseContext,
        pending: Sequence["Interrupt"],
    ) -> Optional[str]:
        """Detect a client-issued rejection of a pending interrupt.

        Default implementation scans the request for an
        ``mcp_approval_response`` item whose ``approval_request_id``
        matches a pending interrupt and whose ``approve`` is ``False``.
        When found, :meth:`handle_create` short-circuits the turn into
        ``response.failed(code="interrupt_rejected", …)`` instead of
        driving the graph.

        Override to plug in custom rejection protocols (e.g. recognising
        a sentinel ``function_call_output`` payload as a rejection).

        Args:
            request: The parsed create-response request.
            context: The response context for the request.
            pending: Interrupts currently pending on the checkpointed
                thread.

        Returns:
            A human-readable rejection message, or ``None`` when no
            rejection was found.
        """
        del request  # unused in the default implementation
        items = await context.get_input_items()
        return detect_approval_rejection(items, pending)

    async def build_runnable_config(
        self,
        request: CreateResponse,
        context: ResponseContext,
    ) -> RunnableConfig:
        """Build a LangGraph ``RunnableConfig`` for the request.

        Sets ``configurable.thread_id`` so graphs compiled with a
        checkpointer naturally continue the framework-derived conversation
        chain. Also exposes the full :class:`ResponseContext` under
        ``configurable.responses_context`` so nodes can read per-attempt
        transport facts (for example ``responses_context.is_recovery``) that
        are deliberately not part of the checkpointed graph state.

        Args:
            request: The parsed create-response request.
            context: The response context for the request.

        Returns:
            A ``RunnableConfig`` dict.
        """
        return cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": context.conversation_chain_id,
                    "responses_context": context,
                },
            },
        )

    def build_runnable_config_sync(
        self,
        request: CreateResponse,
        context: ResponseContext,
    ) -> RunnableConfig:
        """Build a sync best-effort runnable config.

        Prefer :meth:`build_runnable_config` in async request handling. This
        sync helper uses the same spec-guaranteed conversation chain id and
        exposes the same ``configurable.responses_context``.
        """
        return cast(
            RunnableConfig,
            {
                "configurable": {
                    "thread_id": context.conversation_chain_id,
                    "responses_context": context,
                },
            },
        )

    def _resolve_conversation_management(self) -> ResolvedConversationManagementMode:
        return (
            "langgraph_checkpoint"
            if self._graph_has_checkpointer
            else "responses_history"
        )

    def _log_conversation_management_selection(self) -> None:
        mode = self._resolve_conversation_management()
        logger.debug(
            "Responses conversation management selected: mode=%s "
            "graph_has_checkpointer=%s",
            mode,
            self._graph_has_checkpointer,
        )

        if mode == "langgraph_checkpoint" and _is_in_memory_checkpointer(
            getattr(self._graph, "checkpointer", None)
        ):
            config = getattr(self._app, "config", None)
            if bool(getattr(config, "is_hosted", False)):
                logger.warning(
                    "ResponsesHostServer selected LangGraph checkpoint "
                    "conversation mode with an in-memory checkpointer in "
                    "hosted mode; conversation state may be lost between "
                    "restarts. Use a Foundry-backed checkpointer."
                )

    def _log_conversation_input_built(
        self,
        *,
        mode: ResolvedConversationManagementMode,
        request: CreateResponse,
        context: ResponseContext,
        history_item_count: int,
        history_message_count: int,
        current_item_count: int,
        current_message_count: int,
    ) -> None:
        logger.debug(
            "Responses conversation input built: mode=%s response_id=%s "
            "conversation_id=%s previous_response_id=%s history_items=%d "
            "history_messages=%d current_items=%d current_messages=%d",
            mode,
            context.response_id,
            context.conversation_id,
            getattr(request, "previous_response_id", None),
            history_item_count,
            history_message_count,
            current_item_count,
            current_message_count,
        )

    async def handle_create(
        self,
        request: CreateResponse,
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
    ) -> AsyncIterator[Any]:
        """Drive the graph and yield Responses API events.

        Override this when wholesale customisation is needed. By default
        the method:

        1. emits ``response.created`` / ``response.in_progress``,
        2. checks for pending ``interrupt()`` pauses on the checkpointed
           thread and:

           - if the request contains an ``mcp_approval_response`` with
             ``approve=false`` for a pending interrupt, emits
             ``response.failed(code="interrupt_rejected", …)`` and
             stops;
           - otherwise tries to resume from a matching
             ``function_call_output`` (rich) or
             ``mcp_approval_response{approve:true}`` (echo the
             interrupt value back),
        3. drives the graph via :meth:`CompiledStateGraph.astream` or
           :meth:`CompiledStateGraph.ainvoke` depending on
           ``request.stream``,
        4. emits the resulting output items, surfacing any new pending
           interrupts as a pair of ``function_call`` +
           ``mcp_approval_request`` items both keyed by the LangGraph
           interrupt id, and
        5. emits ``response.completed`` (or ``response.failed`` /
           ``response.cancelled`` on error).

        Args:
            request: The parsed create-response request.
            context: The response context for the request.
            cancellation_signal: Set when the request is cancelled.

        Yields:
            Responses API event payload dicts.
        """
        stream = self._new_stream(request, context)
        yield stream.emit_created()
        yield stream.emit_in_progress()

        recovering = bool(context.is_recovery)

        try:
            config = await self.build_runnable_config(request, context)

            # Record the LangGraph thread id (and any progress breadcrumb) onto
            # the streaming context so a recovered attempt — and any inspecting
            # operator — can see which checkpointed thread this response drives.
            await self._record_thread_progress(stream, context, config)

            resume_command: Optional["Command"] = None
            consumed_call_ids: frozenset[str] = frozenset()

            if recovering:
                # Crash-recovered re-entry. The graph's own persistent
                # checkpointer holds the mid-turn state, so resume it (input
                # ``None``) rather than re-injecting the original input. If the
                # thread has no checkpoint yet (crash before the first node
                # committed), fall back to a fresh run.
                graph_input: Any = await self._resume_graph_input(request, context, config)
            else:
                # Detect a pause from a previous turn and try to resume it.
                pending = await detect_pending_interrupts(self._graph, config)
                if pending:
                    # HITL:
                    # Rejection short-circuits the turn into ``response.failed``
                    # so a client-issued ``mcp_approval_response{approve:false}``
                    # is not silently dropped.
                    rejection_message = await self.detect_rejection(
                        request, context, pending
                    )
                    if rejection_message is not None:
                        yield stream.emit_failed(
                            code="interrupt_rejected",
                            message=rejection_message,
                        )
                        return
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
                    graph_input = resume_command
                else:
                    graph_input = await self.build_input(
                        request, context, skip_call_ids=consumed_call_ids
                    )

            if context.mode_flags.stream:
                stream_modes: list[StreamMode] = ["updates", "messages"]
                graph_stream = self._graph.astream(
                    graph_input,
                    config=config,
                    stream_mode=stream_modes,
                    durability="sync",
                )
                async for event in stream_graph_to_events(
                    graph_stream,
                    stream,
                    cancellation_signal=cancellation_signal,
                    checkpoint_each_phase=True,
                ):
                    yield event
                if cancellation_signal.is_set():
                    yield stream.emit_failed(
                        code="cancelled",
                        message="Request was cancelled.",
                    )
                    return
            else:
                state = await self._graph.ainvoke(
                    graph_input,
                    config=config,
                    durability="sync",
                )
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
    # Recovery helpers (resilient background responses)
    # ------------------------------------------------------------------

    def _new_stream(
        self, request: CreateResponse, context: ResponseContext
    ) -> ResponseEventStream:
        """Create the event stream, seeding it on a recovered entry.

        On a crash-recovered entry the framework hands us the last
        resiliently-persisted snapshot as ``context.persisted_response``. We
        seed the stream from it so the phases already checkpointed before the
        crash are present in ``stream.response.output`` (and the output-index
        allocator continues past them). The subsequent ``response.in_progress``
        then carries those items as the client-visible reset point, and the
        graph — resumed from its own checkpoint — only appends the remaining
        phases. On a fresh entry we build an empty envelope from the request.
        """
        if context.is_recovery and context.persisted_response is not None:
            return ResponseEventStream(
                response_id=context.response_id,
                response=context.persisted_response,
            )
        return ResponseEventStream(response_id=context.response_id, request=request)

    async def _resume_graph_input(
        self,
        request: CreateResponse,
        context: ResponseContext,
        config: RunnableConfig,
    ) -> Any:
        """Choose the graph input for a recovered entry.

        When the LangGraph thread already has a persisted checkpoint we return
        ``None`` so ``astream`` resumes the graph from that checkpoint (running
        only the pending / remaining nodes) instead of re-injecting the
        original input and re-running the whole turn. When no checkpoint exists
        yet — a crash before the first node committed — we fall back to the
        fresh input so the turn still produces a correct response.
        """
        snapshot = await self._graph.aget_state(config)
        has_state = bool(snapshot.created_at)
        if has_state:
            logger.debug("Recovery: resuming graph from persisted checkpoint")
            return None
        logger.debug("Recovery: no checkpoint found, re-running from fresh input")
        return await self.build_input(request, context)

    async def _record_thread_progress(
        self,
        stream: ResponseEventStream,
        context: ResponseContext,
        config: RunnableConfig,
    ) -> None:
        """Persist the LangGraph thread id and progress onto the stream context.

        Two facilities are written:

        - ``stream.internal_metadata`` — a single-turn, per-response map that is
          persisted with each ``stream.checkpoint()`` snapshot (and stripped
          before client-facing payloads). Recording the thread id here means a
          recovered attempt can read it back from ``context.persisted_response``.
        - ``context.conversation_chain_metadata`` — the cross-turn resilient
          watermark store. We stamp the thread id (a small reference, not bulk
          checkpoint data — the graph state itself lives in the LangGraph
          checkpointer) and flush so it survives a crash.
        """
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            return

        stream.internal_metadata["langgraph_thread_id"] = thread_id
        context.conversation_chain_metadata["langgraph_thread_id"] = thread_id
        await context.conversation_chain_metadata.flush()

    # ------------------------------------------------------------------
    # Internal — registered as the @response_handler. Wraps handle_create
    # so subclasses only need to override the async method.
    # ------------------------------------------------------------------

    async def _handle_create_async_gen(
        self,
        request: CreateResponse,
        context: ResponseContext,
        cancellation_signal: asyncio.Event,
    ) -> AsyncIterator[Any]:
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
            "ResponsesHostServer's default request converter only "
            "supports graphs whose state schema declares a 'messages' field. "
            "Subclass ResponsesHostServer and override `build_input` "
            "(and optionally `handle_create`) to host custom-state graphs."
        )
