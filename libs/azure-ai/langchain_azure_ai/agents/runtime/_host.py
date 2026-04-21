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
    ``graph.astream(stream_mode="values")`` yields full state dicts; or use
    ``stream_mode="messages"`` to receive ``AIMessageChunk`` objects per token.
    A user-supplied ``output_extractor`` extracts the text payload from each
    item (chunk or state snapshot), which is then emitted on a
    ``ResponseEventStream``.

Cancellation is wired in both paths: when the ``cancellation_signal``
``asyncio.Event`` fires, the running graph ``asyncio.Task`` is cancelled and
the response stream is closed cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

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
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.types import Command

from langchain_azure_ai._api.base import experimental

logger = logging.getLogger(__package__)


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
    graph_input: dict[str, Any] | Command,
    config: RunnableConfig,
    cancellation_signal: asyncio.Event,
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
    graph_input: dict[str, Any] | Command,
    config: RunnableConfig,
    cancellation_signal: asyncio.Event,
    stream: Any,
    output_extractor: Callable[[AIMessageChunk | dict[str, Any]], str],
    stream_mode: str = "values",
) -> None:
    """Emit graph output onto a ``ResponseEventStream`` via *output_extractor*.

    Supports both streaming modes:

    * ``stream_mode="messages"`` — LangGraph yields ``(AIMessageChunk, metadata)``
      tuples; the chunk (first element) is passed to *output_extractor*, enabling
      token-by-token streaming identical to :func:`_stream_messages`.
    * Other modes (e.g. ``"values"``) — each item is a full state snapshot dict
      passed directly to *output_extractor*.

    In both cases *output_extractor* is called for every item; returning an empty
    string suppresses the ``emit()`` call for that item.  When
    *cancellation_signal* fires, the task is cancelled and the stream is closed.

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
        output_extractor: Callable that maps an ``AIMessageChunk``
            (``stream_mode="messages"``) or a full state snapshot
            ``dict[str, Any]`` (other modes) to a string for emission.
            Returning an empty string suppresses the event.
        stream_mode: LangGraph stream mode forwarded to ``graph.astream()``.
    """

    async def _producer() -> None:
        try:
            async for item in graph.astream(  # type: ignore[union-attr]
                graph_input,
                config,
                stream_mode=stream_mode,
            ):
                extractor_input = item[0] if stream_mode == "messages" else item
                try:
                    text = output_extractor(extractor_input)
                    if text:
                        await stream.emit(text)
                except Exception:
                    logger.debug(
                        "output_extractor raised on item; skipping",
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
    request: CreateResponse,
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
async def messages_input_parser(
    request: CreateResponse,  # noqa: ARG001
    context: ResponseContext,
) -> dict[str, Any]:
    """Default input parser: fetch conversation history and current user text.

    Builds a ``{"messages": [...]}`` dict compatible with LangGraph's
    ``MessagesState``.  Conversation history is translated via
    :func:`_history_to_messages` and the current user turn is appended as a
    ``HumanMessage``.

    Args:
        request: The incoming ``CreateResponse`` request (unused by the
            default implementation; present so custom parsers can inspect it).
        context: The ``ResponseContext`` used to fetch history and user input.

    Returns:
        ``{"messages": list[BaseMessage]}`` ready to be passed to
        ``graph.astream()``.
    """
    history = await context.get_history()
    user_input = (await context.get_input_text()) or ""
    return {
        "messages": _history_to_messages(history) + [HumanMessage(content=user_input)]
    }


@experimental()
class AzureAIResponsesAgentHost:
    """Host a compiled LangGraph graph as an agent inside Azure AI Foundry.

    This class is the *server/host* side of the Foundry Agent Service integration.
    It registers a ``response_handler`` on a ``ResponsesAgentServerHost`` that:

    1. Checks for a pending ``interrupt()`` (e.g. MCP tool-call approval).  If
       one is found, resumes the graph via ``Command(resume=...)``.  Otherwise
       calls *input_parser* to build the graph input for the new turn (default:
       history + user text as ``{"messages": [...]}``).
    2. Invokes the LangGraph graph asynchronously with the prepared input.
    3. Streams the output back to Foundry — either via ``TextResponse``
       (``MessagesState``-compatible graphs) or via ``ResponseEventStream``
       (custom state schemas with an ``output_extractor``).
    4. Wires the platform-supplied ``cancellation_signal`` to cancel the
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
            provided, ``"messages"`` mode passes each ``AIMessageChunk`` to
            the extractor for token-by-token streaming; ``"values"`` (or any
            other mode) passes full state snapshots instead.
        output_extractor: Callable ``(item: AIMessageChunk | dict[str, Any]) -> str``
            called for every item yielded by ``graph.astream()``.  When
            provided, the host switches to the ``ResponseEventStream`` path
            and calls ``emit()`` for each non-empty result.  Use this for
            graphs that do *not* use ``MessagesState`` or when you need
            custom chunk extraction.

            The item type depends on ``stream_mode``:

            * ``"messages"`` (default) — item is an ``AIMessageChunk``;
              the extractor is called per token, enabling true streaming.
            * ``"values"`` — item is a ``dict[str, Any]`` full state
              snapshot emitted after *each node* completes.  The extractor
              may be called multiple times with intermediate states; return
              an empty string to suppress emission for a given snapshot.
            * ``"updates"`` — item is a ``dict[str, Any]`` containing only
              the keys changed by each node; similar to ``"values"`` but
              sparser.

            There is no ``ainvoke``-style "run once, emit once" path.  To
            approximate it with ``stream_mode="values"``, only return a
            non-empty string when the final output key is present::

                def my_extractor(item: AIMessageChunk | dict[str, Any]) -> str:
                    if isinstance(item, dict) and "answer" in item:
                        return item["answer"]
                    return ""  # suppress intermediate state snapshots
        input_parser: Async callable
            ``async (request, context) -> dict[str, Any]`` that builds the
            graph input for a new (non-interrupt) turn.  Receives the raw
            ``CreateResponse`` and ``ResponseContext`` objects so it can read
            ``request.input``, call ``context.get_history()``, access
            ``context.response_id``, etc.  The returned dict is passed
            verbatim to ``graph.astream()``.  Defaults to
            :func:`messages_input_parser` which wraps conversation history
            and the current user text in ``{"messages": [...]}``.  Override
            this when your graph state schema has additional keys (e.g.
            ``"constraints"``, ``"metadata"``) or when you need full control
            over message formatting.

    Raises:
        ImportError: If the ``runtime`` extras are not installed.
    """

    def __init__(
        self,
        graph: Runnable,
        *,
        options: Optional[ResponsesServerOptions] = None,
        stream_mode: str = "messages",
        input_parser: Optional[
            Callable[[CreateResponse, ResponseContext], Awaitable[dict[str, Any]]]
        ] = None,
        output_extractor: Optional[
            Callable[[AIMessageChunk | dict[str, Any]], str]
        ] = None,
    ) -> None:
        self._graph = graph
        self._output_extractor = output_extractor
        self._input_parser = (
            input_parser if input_parser is not None else messages_input_parser
        )
        self._stream_mode: str = stream_mode

        self._app: ResponsesAgentServerHost = ResponsesAgentServerHost(  # type: ignore[misc]
            options=options
        )
        # Register the response handler (equivalent to @self._app.response_handler)
        self._app.response_handler(self._handle_create)

    async def _handle_create(
        self,
        request: CreateResponse,
        context: ResponseContext,
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
            graph_input: dict[str, Any] | Command = Command(resume=resume_value)
            logger.debug("Resuming interrupted graph (thread_id=%s)", thread_id)
        else:
            graph_input = await self._input_parser(request, context)

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
