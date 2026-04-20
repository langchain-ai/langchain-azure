# Copyright (c) Microsoft. All rights reserved.

"""Hosting adapter for running a LangGraph graph in Azure AI Foundry's Agent Service.

This module bridges the **server side** of the Responses API protocol with a compiled
LangGraph graph.  Foundry invokes the agent; the graph *is* the agent logic.

Unlike :mod:`langchain_azure_ai.agents._v2.prebuilt.factory` (which is a *client*
that calls Foundry-hosted agents), this module makes *your graph* the agent that
Foundry calls into.

Conversation state is managed by the platform via ``previous_response_id`` and
``context.get_history()``. No application-side session storage is required —
Foundry maintains the conversation chain; the graph receives the full history
as LangChain messages on every invocation.  For graphs compiled with a
checkpointer, ``previous_response_id`` is automatically used as the
``thread_id`` in the LangGraph ``RunnableConfig``.

Required extras::

    pip install langchain-azure-ai[runtime]

Required environment variables:
    FOUNDRY_PROJECT_ENDPOINT  Foundry project endpoint (auto-injected by the platform).

Two streaming paths are supported:

``MessagesState``-compatible graphs (default)
    ``graph.astream(stream_mode="messages")`` yields ``AIMessageChunk`` objects
    whose ``content`` is piped directly into a ``TextResponse``.

Non-``MessagesState`` graphs (when ``output_extractor`` is provided)
    ``graph.astream(stream_mode="values")`` yields full state dicts.
    A user-supplied ``output_extractor`` extracts the text payload from each
    state snapshot, which is then emitted on a ``ResponseEventStream``.

Cancellation is wired in both paths: when the ``cancellation_signal``
``asyncio.Event`` fires, the running graph ``asyncio.Task`` is cancelled and
the response stream is closed cleanly.
"""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.types import Command

from langchain_azure_ai._api.base import experimental

if TYPE_CHECKING:
    from azure.ai.agentserver.responses import (
        CreateResponse,
        ResponseContext,
        ResponseEventStream,
        ResponsesAgentServerHost,
        ResponsesServerOptions,
        TextResponse,
    )
    from azure.ai.agentserver.responses.models import (
        MessageContentInputTextContent,
        MessageContentOutputTextContent,
    )

logger = logging.getLogger(__package__)

_AGENTSERVER_IMPORT_ERROR: Optional[ImportError] = None

try:
    from azure.ai.agentserver.responses import (  # type: ignore[import-not-found]
        CreateResponse,
        ResponseContext,
        ResponseEventStream,
        ResponsesAgentServerHost,
        ResponsesServerOptions,
        TextResponse,
    )
    from azure.ai.agentserver.responses.models import (  # type: ignore[import-not-found]
        MessageContentInputTextContent,
        MessageContentOutputTextContent,
    )
except ImportError as _err:
    _AGENTSERVER_IMPORT_ERROR = ImportError(
        "The 'runtime' extras are required to use AzureAIResponsesAgentHost. "
        "Install them with:  pip install langchain-azure-ai[runtime]"
    )

    class _NeverMatch:  # type: ignore[no-redef]
        """Sentinel — never an instance of a real SDK type."""

    # Define placeholder sentinels so isinstance checks in _history_to_messages
    # use a class that nothing will ever be an instance of, and so that
    # unittest.mock.patch() can replace these names in tests.
    MessageContentInputTextContent = _NeverMatch  # type: ignore[assignment,misc]
    MessageContentOutputTextContent = _NeverMatch  # type: ignore[assignment,misc]
    ResponsesAgentServerHost = _NeverMatch  # type: ignore[assignment,misc]
    ResponsesServerOptions = _NeverMatch  # type: ignore[assignment,misc]
    TextResponse = _NeverMatch  # type: ignore[assignment,misc]
    ResponseEventStream = _NeverMatch  # type: ignore[assignment,misc]
    CreateResponse = _NeverMatch  # type: ignore[assignment,misc]
    ResponseContext = _NeverMatch  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Message translation helpers
# ---------------------------------------------------------------------------


def _history_to_messages(
    history: list,
) -> list[BaseMessage]:
    """Convert Foundry conversation history to a list of LangChain messages.

    Args:
        history: History items returned by ``context.get_history()``.  Each
            item is expected to have a ``content`` attribute holding a list of
            content blocks (``MessageContentInputTextContent`` for user turns
            and ``MessageContentOutputTextContent`` for assistant turns).

    Returns:
        List of ``BaseMessage`` objects ready to be passed to a LangGraph graph.
    """
    messages: list[BaseMessage] = []
    for item in history:
        if not hasattr(item, "content") or not item.content:
            continue
        for content in item.content:
            if isinstance(content, MessageContentInputTextContent) and content.text:
                messages.append(HumanMessage(content=content.text))
            elif isinstance(content, MessageContentOutputTextContent) and content.text:
                messages.append(AIMessage(content=content.text))

    return messages


# ---------------------------------------------------------------------------
# Cancellation-aware stream helpers
# ---------------------------------------------------------------------------


async def _stream_messages(
    graph: Runnable,
    graph_input: "dict[str, Any] | Command",
    config: RunnableConfig,
    cancellation_signal: "asyncio.Event",
    stream_mode: str = "messages",
) -> AsyncGenerator[str, None]:
    """Yield text chunks from a ``MessagesState``-compatible graph.

    Runs ``graph.astream(stream_mode=stream_mode)`` in a background
    ``asyncio.Task`` and forwards ``AIMessageChunk`` content strings to the
    caller.  When *cancellation_signal* fires, the task is cancelled and the
    generator returns cleanly.

    Args:
        graph: The compiled LangGraph graph to stream from.
        graph_input: Passed verbatim as the first argument to
            ``graph.astream()``.  Use ``{"messages": [...]}`` for a normal
            turn or ``Command(resume=...)`` to resume an interrupted graph.
        config: ``RunnableConfig`` carrying the ``thread_id`` and other
            LangChain / LangGraph runtime configuration.
        cancellation_signal: ``asyncio.Event`` that, when set, cancels the
            background producer task and stops the generator.
        stream_mode: LangGraph stream mode forwarded to ``graph.astream()``.

    Note:
        The default ``stream_mode="messages"`` yields ``(chunk, metadata)``
        tuples from LangGraph and expects ``AIMessageChunk`` objects.
        Changing this value requires that the graph emits a compatible format.
    """
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def _producer() -> None:
        try:
            async for chunk, _ in graph.astream(  # type: ignore[union-attr]
                graph_input,
                config,
                stream_mode=stream_mode,
            ):
                if not isinstance(chunk, AIMessageChunk):
                    continue
                content = chunk.content
                if isinstance(content, str):
                    if content:
                        await queue.put(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, str) and part:
                            await queue.put(part)
                        elif isinstance(part, dict):
                            text = part.get("text") or part.get("content") or ""
                            if text:
                                await queue.put(str(text))
        except asyncio.CancelledError:
            pass
        finally:
            await queue.put(None)  # sentinel — signals end of stream

    task = asyncio.create_task(_producer())

    async def _watch_cancel() -> None:
        await cancellation_signal.wait()
        task.cancel()

    watcher = asyncio.create_task(_watch_cancel())

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()


async def _emit_events(
    graph: Runnable,
    graph_input: "dict[str, Any] | Command",
    config: RunnableConfig,
    cancellation_signal: "asyncio.Event",
    stream: Any,
    output_extractor: Callable[[dict], str],
    stream_mode: str = "values",
) -> None:
    """Emit graph state snapshots onto a ``ResponseEventStream``.

    Used for non-``MessagesState`` graphs.  Each state update is passed to
    *output_extractor*; non-empty results are emitted on *stream*.  When
    *cancellation_signal* fires, the task is cancelled and the stream is
    closed.

    Args:
        graph: The compiled LangGraph graph to stream from.
        graph_input: Passed verbatim as the first argument to
            ``graph.astream()``.  Use ``{"messages": [...]}`` for a normal
            turn or ``Command(resume=...)`` to resume an interrupted graph.
        config: ``RunnableConfig`` carrying the ``thread_id`` and other
            LangChain / LangGraph runtime configuration.
        cancellation_signal: ``asyncio.Event`` that, when set, cancels the
            background producer task and closes the stream.
        stream: ``ResponseEventStream`` instance to emit text events onto.
        output_extractor: Callable that maps a graph state snapshot to a
            string for emission.  Returning an empty string suppresses the
            event.
        stream_mode: LangGraph stream mode forwarded to ``graph.astream()``.
    """

    async def _producer() -> None:
        try:
            async for state in graph.astream(  # type: ignore[union-attr]
                graph_input,
                config,
                stream_mode=stream_mode,
            ):
                try:
                    text = output_extractor(state)
                    if text:
                        await stream.emit(text)
                except Exception:
                    logger.debug(
                        "output_extractor raised on state update; skipping",
                        exc_info=True,
                    )
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    task = asyncio.create_task(_producer())

    async def _watch_cancel() -> None:
        await cancellation_signal.wait()
        task.cancel()

    watcher = asyncio.create_task(_watch_cancel())
    try:
        await task
    finally:
        watcher.cancel()


# ---------------------------------------------------------------------------
# Interrupt / MCP-approval helpers
# ---------------------------------------------------------------------------


async def _pending_interrupts(
    graph: Runnable,
    config: RunnableConfig,
) -> list[Any]:
    """Return any pending ``Interrupt`` objects for the current thread.

    Uses ``graph.aget_state()`` to inspect checkpointed state.  Returns an
    empty list when the graph has no checkpointer, when ``aget_state`` is
    unavailable, or when no interrupt is pending.

    Args:
        graph: The compiled LangGraph graph.
        config: ``RunnableConfig`` carrying the ``thread_id``.

    Returns:
        List of ``Interrupt`` objects from the most recent checkpoint tasks.
    """
    try:
        state = await graph.aget_state(config)  # type: ignore[union-attr,attr-defined]
    except Exception:
        logger.debug("graph.aget_state unavailable or failed; assuming no interrupt")
        return []
    return [
        interrupt
        for task in getattr(state, "tasks", ())
        for interrupt in getattr(task, "interrupts", ())
    ]


def _extract_mcp_resume_value(
    request: "CreateResponse",
) -> Optional[dict[str, Any]]:
    """Look for an MCP approval response in the request input items.

    Scans ``request.input`` for an item whose class name contains
    ``"McpApproval"`` (case-insensitive, underscore-insensitive) and returns
    the approval decision as a structured dict.

    Args:
        request: The incoming ``CreateResponse`` request from Foundry.

    Returns:
        ``{"approved": bool, "approval_request_id": str | None}`` when an
        MCP approval response item is present, or ``None`` otherwise.
    """
    for item in getattr(request, "input", None) or []:
        if "mcpapproval" in type(item).__name__.lower().replace("_", ""):
            return {
                "approved": bool(getattr(item, "approve", False)),
                "approval_request_id": getattr(item, "approval_request_id", None),
            }
    return None


# ---------------------------------------------------------------------------
# Public host class
# ---------------------------------------------------------------------------


@experimental()
class AzureAIResponsesAgentHost:
    """Host a compiled LangGraph graph as an agent inside Azure AI Foundry.

    This class is the *server/host* side of the Foundry Agent Service integration.
    It registers a ``response_handler`` on a ``ResponsesAgentServerHost`` that:

    1. Fetches the conversation history from Foundry via ``context.get_history()``.
    2. Translates history items into LangChain ``BaseMessage`` objects.
    3. Appends the new user turn as a ``HumanMessage``.
    4. Invokes the LangGraph graph asynchronously.
    5. Streams the output back to Foundry — either via ``TextResponse``
       (``MessagesState``-compatible graphs) or via ``ResponseEventStream``
       (custom state schemas with an ``output_extractor``).
    6. Wires the platform-supplied ``cancellation_signal`` to cancel the
       running graph task on early termination.

    Example::

        from langgraph.graph import StateGraph, MessagesState, START, END
        from langchain_azure_ai.agents.runtime import AzureAIResponsesAgentHost

        builder = StateGraph(MessagesState)
        builder.add_node("agent", my_agent_node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        graph = builder.compile()

        host = AzureAIResponsesAgentHost(
            graph=graph,
        )

        if __name__ == "__main__":
            host.run()

    Args:
        graph: A compiled LangGraph graph.  For the default ``TextResponse``
            streaming path the graph must accept ``{"messages": list[BaseMessage]}``
            as input and emit ``AIMessageChunk`` objects (``MessagesState`` or
            any compatible subclass).
        options: Optional ``ResponsesServerOptions`` forwarded verbatim to the
            underlying ``ResponsesAgentServerHost`` (e.g.
            ``default_fetch_history_count``).
        stream_mode: LangGraph ``stream_mode`` passed to ``graph.astream()``.
            Default is ``"messages"``, which works for ``MessagesState``-compatible
            graphs on the ``TextResponse`` path.  When *output_extractor* is
            provided, this must be set to ``"values"`` (or another mode that
            yields full state snapshots); passing ``"messages"`` with
            *output_extractor* raises ``ValueError``.
        output_extractor: Callable ``(state: dict) -> str`` that extracts a
            text payload from a graph state snapshot.  When provided, the host
            switches to the ``ResponseEventStream`` path (``stream_mode="values"``)
            and calls ``emit()`` for each state update.  Use this for graphs
            that do *not* use ``MessagesState``.  Default: reads
            ``state["messages"][-1].content``.

    Raises:
        ImportError: If the ``runtime`` extras are not installed.
    """

    def __init__(
        self,
        graph: Runnable,
        *,
        options: Optional["ResponsesServerOptions"] = None,
        stream_mode: str = "messages",
        output_extractor: Optional[Callable[[dict], str]] = None,
    ) -> None:
        if _AGENTSERVER_IMPORT_ERROR is not None:
            raise _AGENTSERVER_IMPORT_ERROR

        self._graph = graph
        self._output_extractor = output_extractor
        if output_extractor is not None and stream_mode == "messages":
            raise ValueError(
                "stream_mode='messages' is incompatible with output_extractor. "
                "Use stream_mode='values' (or another mode that yields full state "
                "snapshots) when providing an output_extractor."
            )
        self._stream_mode: str = stream_mode

        self._app: "ResponsesAgentServerHost" = ResponsesAgentServerHost(  # type: ignore[misc]
            options=options
        )
        # Register the response handler (equivalent to @self._app.response_handler)
        self._app.response_handler(self._handle_create)

    async def _handle_create(
        self,
        request: "CreateResponse",
        context: "ResponseContext",
        cancellation_signal: asyncio.Event,
    ) -> Any:
        """Handle a single Responses API request from Foundry.

        Translates the Foundry conversation history + current user input into
        LangChain messages, invokes the LangGraph graph, and returns a
        streaming response object to the server host.  When a pending
        ``interrupt()`` is detected the graph is resumed with
        ``Command(resume=...)`` instead of replaying history.
        """
        thread_id: str = getattr(request, "previous_response_id", None) or str(
            uuid.uuid4()
        )
        config = RunnableConfig(configurable={"thread_id": thread_id})

        interrupts = await _pending_interrupts(self._graph, config)
        if interrupts:
            mcp_decision = _extract_mcp_resume_value(request)
            resume_value: Any = (
                mcp_decision
                if mcp_decision is not None
                else ((await context.get_input_text()) or "")
            )
            graph_input: "dict[str, Any] | Command" = Command(resume=resume_value)
            logger.debug("Resuming interrupted graph (thread_id=%s)", thread_id)
        else:
            history = await context.get_history()
            user_input = (await context.get_input_text()) or ""
            messages = _history_to_messages(history) + [
                HumanMessage(content=user_input)
            ]
            graph_input = {"messages": messages}

        if self._output_extractor is not None:
            # Non-MessagesState path: ResponseEventStream + emit()
            stream = ResponseEventStream(context, request)  # type: ignore[misc]
            asyncio.create_task(
                _emit_events(
                    self._graph,
                    graph_input,
                    config,
                    cancellation_signal,
                    stream,
                    self._output_extractor,
                    self._stream_mode,
                )
            )
            return stream

        # MessagesState path: TextResponse with async text generator
        return TextResponse(  # type: ignore[misc]
            context,
            request,
            text=_stream_messages(
                self._graph, graph_input, config, cancellation_signal, self._stream_mode
            ),
        )

    def run(self, **kwargs: Any) -> None:
        """Start the agent server.

        Accepts the same keyword arguments as
        ``ResponsesAgentServerHost.run()``.
        """
        self._app.run(**kwargs)
