# Copyright (c) Microsoft. All rights reserved.

"""Hosting adapter for running a LangGraph graph in Azure AI Foundry's Agent Service.

This module bridges the **server side** of the Responses API protocol with a compiled
LangGraph graph.  Foundry invokes the agent; the graph *is* the agent logic.

Conversation state is managed by the platform via ``previous_response_id``. No
application-side session storage is required — Foundry maintains the conversation chain;
the graph receives the full history as LangChain messages on every invocation.

For graphs compiled with a checkpointer, ``previous_response_id`` is automatically used
as the ``thread_id`` in the LangGraph ``RunnableConfig``.

Required extras::

    pip install langchain-azure-ai[runtime]

Two streaming paths are supported:

``MessagesState``-compatible graphs (default)
    ``graph.astream(stream_mode="messages")`` yields ``AIMessageChunk`` objects
    whose ``content`` is piped directly into a ``TextResponse``.

Non-``MessagesState`` graphs (when ``output_parser`` is provided)
    ``graph.astream(stream_mode="values")`` yields full state dicts; or use
    ``stream_mode="messages"`` to receive ``AIMessageChunk`` objects per token.
    A user-supplied ``output_parser`` extracts the text payload from each
    item (chunk or state snapshot), which is then emitted on a
    ``ResponseEventStream``.

Cancellation is wired in both paths: when the ``cancellation_signal``
``asyncio.Event`` fires, the running graph ``asyncio.Task`` is cancelled and
the response stream is closed cleanly.

Error handling follows the Responses API streaming model instead of a plain
HTTP response model. This module distinguishes between parser-hook failures and
general runtime failures:

* Custom ``input_parser`` failures are logged and converted into a short
    ``response.created`` -> ``response.in_progress`` -> ``response.failed``
    sequence before graph execution starts.
* Custom ``output_parser`` failures are logged, emitted as
    ``response.failed`` on the active stream, and then terminate that stream.
* The default parser path, graph/runtime failures outside custom parser hooks,
    and SDK-level validation continue through the underlying Responses host.
* Interrupt inspection is intentionally best-effort: if ``graph.aget_state()``
    fails, the host logs at debug level and continues as if no interrupt were
    pending.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generic,
    Literal,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)

from azure.ai.agentserver.responses import (
    CreateResponse,
    ResponseContext,
    ResponseEventStream,
    ResponsesAgentServerHost,
    ResponsesServerOptions,
)
from azure.ai.agentserver.responses.models import (
    ItemMessage,
    ItemOutputMessage,
    MessageContentInputFileContent,
    MessageContentInputImageContent,
    MessageContentInputTextContent,
    MessageContentOutputTextContent,
    MessageContentReasoningTextContent,
    MessageContentRefusalContent,
    OutputItemMessage,
    OutputItemOutputMessage,
    OutputMessageContentOutputTextContent,
    OutputMessageContentRefusalContent,
    ResponseStreamEvent,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.types import Command

from langchain_azure_ai._api.base import experimental

logger = logging.getLogger(__package__)

ResponsesGraphStateT = TypeVar("ResponsesGraphStateT")

ResponsesInputRequest: TypeAlias = CreateResponse
ResponsesInputContext: TypeAlias = ResponseContext
ResponsesInputParser: TypeAlias = Callable[
    [ResponsesInputRequest, ResponsesInputContext],
    Awaitable[ResponsesGraphStateT],
]
ResponsesOutputItem: TypeAlias = AIMessageChunk | dict[str, Any]
ResponsesOutputParser: TypeAlias = Callable[[ResponsesOutputItem], str]


def _parser_name(parser: object) -> str:
    """Return a stable human-readable name for a parser callable."""
    name = getattr(parser, "__name__", None)
    return name if isinstance(name, str) else type(parser).__name__


def _parser_failure_message(*, hook_name: str, parser: object, exc: Exception) -> str:
    """Build a developer-facing parser failure message."""
    detail = str(exc) or repr(exc)
    return (
        f"The configured {hook_name} '{_parser_name(parser)}' raised "
        f"{type(exc).__name__}: {detail}"
    )


async def _emit_stream_failure(stream: Any, *, message: str) -> None:
    """Emit a terminal failed event on a response stream when supported."""
    emit_failed = getattr(stream, "emit_failed", None)
    if callable(emit_failed):
        maybe_result = emit_failed(code="server_error", message=message)
        if inspect.isawaitable(maybe_result):
            await maybe_result


async def _failed_response_events(
    *,
    request: CreateResponse,
    context: ResponseContext,
    hook_name: str,
    parser: object,
    exc: Exception,
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """Return failed response lifecycle events for a parser failure.

    This is used for custom ``input_parser`` failures that happen before the
    host has returned a live stream object to the Responses SDK.
    """
    message = _parser_failure_message(hook_name=hook_name, parser=parser, exc=exc)
    logger.exception(
        "Configured %s failed",
        hook_name,
        extra={
            "hook_name": hook_name,
            "parser_name": _parser_name(parser),
            "exception_type": type(exc).__name__,
        },
    )

    stream = ResponseEventStream(
        response_id=context.response_id,
        request=request,
    )
    yield stream.emit_created()
    yield stream.emit_in_progress()
    yield stream.emit_failed(code="server_error", message=message)


# ---------------------------------------------------------------------------
# Message translation helpers
# ---------------------------------------------------------------------------


_MessageContentBlock: TypeAlias = dict[str, Any]
_MessageContent: TypeAlias = str | list[str | dict[str, Any]]
_FoundryContentPart: TypeAlias = (
    str
    | MessageContentInputTextContent
    | MessageContentInputImageContent
    | MessageContentInputFileContent
    | MessageContentOutputTextContent
    | OutputMessageContentOutputTextContent
    | MessageContentReasoningTextContent
    | MessageContentRefusalContent
    | OutputMessageContentRefusalContent
)
_FoundryMessageItem: TypeAlias = (
    ItemMessage | OutputItemMessage | ItemOutputMessage | OutputItemOutputMessage
)
_FoundryRole: TypeAlias = Literal["assistant", "system", "developer", "user"]


def _infer_message_role(item: _FoundryMessageItem) -> _FoundryRole:
    """Infer the LangChain message role for a Foundry history/input item."""
    role = item.role
    if role in {"assistant", "system", "developer", "user"}:
        return role

    if isinstance(item, (ItemOutputMessage, OutputItemOutputMessage)):
        return "assistant"

    if isinstance(item.content, str):
        return "user"

    for part in item.content:
        if isinstance(
            part,
            (
                MessageContentOutputTextContent,
                OutputMessageContentOutputTextContent,
                MessageContentReasoningTextContent,
                MessageContentRefusalContent,
                OutputMessageContentRefusalContent,
            ),
        ):
            return "assistant"
        if isinstance(
            part,
            (
                MessageContentInputTextContent,
                MessageContentInputImageContent,
                MessageContentInputFileContent,
            ),
        ):
            return "user"

    return "user"


def _content_part_to_message_content(
    part: _FoundryContentPart,
    *,
    wrap_text: bool,
) -> Optional[str | dict[str, Any]]:
    """Translate a Foundry content part into LangChain message content."""
    if isinstance(part, str):
        if not part:
            return None
        if wrap_text:
            return {"type": "text", "text": part}
        return part

    if isinstance(
        part,
        (
            MessageContentInputTextContent,
            MessageContentOutputTextContent,
            OutputMessageContentOutputTextContent,
            MessageContentReasoningTextContent,
        ),
    ):
        if not part.text:
            return None
        if wrap_text:
            return {"type": "text", "text": part.text}
        return part.text

    if isinstance(
        part,
        (MessageContentRefusalContent, OutputMessageContentRefusalContent),
    ):
        if not part.refusal:
            return None
        if wrap_text:
            return {"type": "text", "text": part.refusal}
        return part.refusal

    if isinstance(part, MessageContentInputImageContent):
        if part.image_url:
            image_block: dict[str, str | dict[str, str]] = {
                "type": "image_url",
                "image_url": {"url": part.image_url},
            }
            if part.detail:
                image_url = image_block["image_url"]
                if isinstance(image_url, dict):
                    image_url["detail"] = str(part.detail)
            return image_block

        if part.file_id:
            file_image_block: dict[str, str] = {
                "type": "image",
                "source_type": "file",
                "file_id": part.file_id,
            }
            if part.detail:
                file_image_block["detail"] = str(part.detail)
            return file_image_block

        return None

    if isinstance(part, MessageContentInputFileContent):
        file_block: dict[str, str] = {"type": "file"}
        if part.file_id:
            file_block["file_id"] = part.file_id
        if part.filename:
            file_block["filename"] = part.filename
        if part.file_url:
            file_block["file_url"] = part.file_url
        if part.file_data:
            file_block["data"] = part.file_data
        return file_block if len(file_block) > 1 else None

    return None


def _item_to_message(item: _FoundryMessageItem) -> Optional[BaseMessage]:
    """Convert a Foundry item with role/content fields into a LangChain message."""
    if not item.content:
        return None

    raw_content = item.content
    if isinstance(raw_content, str):
        message_content: _MessageContent = raw_content
    else:
        supported_parts = [
            part
            for part in raw_content
            if _content_part_to_message_content(part, wrap_text=False) is not None
        ]
        if not supported_parts:
            return None

        wrap_text = len(supported_parts) > 1
        parts = [
            converted
            for converted in (
                _content_part_to_message_content(part, wrap_text=wrap_text)
                for part in supported_parts
            )
            if converted is not None
        ]
        if len(parts) == 1 and isinstance(parts[0], str):
            message_content = parts[0]
        else:
            message_content = parts

    role = _infer_message_role(item)
    if role == "assistant":
        return AIMessage(content=message_content)
    if role in {"system", "developer"}:
        return SystemMessage(content=message_content)
    return HumanMessage(content=message_content)


def _history_to_messages(
    history: Sequence[_FoundryMessageItem],
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
        message = _item_to_message(item)
        if message is not None:
            messages.append(message)

    return messages


def _graph_has_messages_input(graph: Runnable) -> bool:  # type: ignore[type-arg]
    """Return True if the graph's input schema has a ``messages`` field.

    Inspects the graph's ``input_schema`` Pydantic model (v1 and v2) for a
    ``messages`` field.  Returns ``True`` when the schema cannot be inspected
    so that the check fails open rather than incorrectly rejecting valid graphs.

    Args:
        graph: The compiled LangGraph graph to inspect.

    Returns:
        ``True`` if ``messages`` is present in the input schema or if the
        schema could not be determined; ``False`` only when the schema is
        conclusively known to lack a ``messages`` field.
    """
    try:
        schema = getattr(graph, "input_schema", None)
        if schema is None:
            return True  # cannot determine — fail open
        # Pydantic v2
        fields = getattr(schema, "model_fields", None)
        if fields is not None:
            return "messages" in fields
        # Pydantic v1
        fields = getattr(schema, "__fields__", None)
        if fields is not None:
            return "messages" in fields
    except Exception:  # noqa: BLE001
        pass
    return True  # fail open


# ---------------------------------------------------------------------------
# Cancellation-aware stream helpers
# ---------------------------------------------------------------------------


async def _stream_messages(
    graph: Runnable[ResponsesGraphStateT | Command, Any],
    graph_input: ResponsesGraphStateT | Command,
    config: RunnableConfig,
    cancellation_signal: asyncio.Event,
    stream_mode: str = "messages",
) -> AsyncGenerator[str, None]:
    """Yield text chunks from a ``MessagesState``-compatible graph.

    Runs ``graph.astream(stream_mode=stream_mode)`` in a background
    ``asyncio.Task`` and forwards LangChain message content strings to the
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
        tuples from LangGraph and expects LangChain ``BaseMessage`` objects.
        Changing this value requires that the graph emits a compatible format.
    """
    queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

    async def _enqueue_message_content(message: BaseMessage) -> None:
        content = message.content
        if isinstance(content, str):
            if content:
                await queue.put(content)
            return

        for part in content:
            if isinstance(part, str) and part:
                await queue.put(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content") or ""
                if text:
                    await queue.put(str(text))

    async def _producer() -> None:
        try:
            async for chunk, _ in graph.astream(  # type: ignore[union-attr]
                graph_input,
                config,
                stream_mode=stream_mode,
            ):
                if not isinstance(chunk, BaseMessage):
                    continue
                await _enqueue_message_content(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            await queue.put(None)  # sentinel — signals end of stream

    task = asyncio.create_task(_producer())

    async def _watch_cancel() -> None:
        await cancellation_signal.wait()
        task.cancel()
        await queue.put(None)

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


def _message_content_to_output_part(
    part: str | dict[str, Any],
) -> dict[str, Any] | None:
    """Convert LangChain message content into a Foundry output content part."""
    if isinstance(part, str):
        if not part:
            return None
        return {
            "type": "output_text",
            "text": part,
            "annotations": [],
            "logprobs": [],
        }

    part_type = part.get("type")
    if part_type == "output_text":
        output_text_part = dict(part)
        output_text_part.setdefault("annotations", [])
        output_text_part.setdefault("logprobs", [])
        return output_text_part

    text = part.get("text") or part.get("content")
    if isinstance(text, str) and text:
        return {
            "type": "output_text",
            "text": text,
            "annotations": list(part.get("annotations", [])),
            "logprobs": list(part.get("logprobs", [])),
        }

    return dict(part)


async def _stream_message_events(
    graph: Runnable[ResponsesGraphStateT | Command, Any],
    graph_input: ResponsesGraphStateT | Command,
    config: RunnableConfig,
    cancellation_signal: asyncio.Event,
    request: CreateResponse,
    context: ResponseContext,
    stream_mode: str = "messages",
) -> AsyncGenerator[ResponseStreamEvent, None]:
    """Stream response events while preserving final non-text message content."""
    queue: asyncio.Queue[ResponseStreamEvent | None] = asyncio.Queue()

    async def _producer() -> None:
        stream = ResponseEventStream(
            response_id=context.response_id,
            request=request,
        )
        message = stream.add_output_item_message()
        final_content: list[dict[str, Any]] = []
        text_builder = None
        text_fragments: list[str] = []

        async def _flush_text_builder() -> None:
            nonlocal text_builder, text_fragments
            if text_builder is None:
                return

            final_text = "".join(text_fragments)
            await queue.put(text_builder.emit_text_done(final_text))
            await queue.put(text_builder.emit_done())
            final_content.append(
                {
                    "type": "output_text",
                    "text": final_text,
                    "annotations": [],
                    "logprobs": [],
                }
            )
            text_builder = None
            text_fragments = []

        try:
            await queue.put(stream.emit_created())
            await queue.put(stream.emit_in_progress())
            await queue.put(message.emit_added())

            async for chunk, _ in graph.astream(  # type: ignore[union-attr]
                graph_input,
                config,
                stream_mode=stream_mode,
            ):
                if not isinstance(chunk, BaseMessage):
                    continue

                content_parts = (
                    [chunk.content]
                    if isinstance(chunk.content, str)
                    else list(chunk.content)
                )

                for raw_part in content_parts:
                    if not isinstance(raw_part, (str, dict)):
                        continue

                    output_part = _message_content_to_output_part(raw_part)
                    if output_part is None:
                        continue

                    if output_part.get("type") == "output_text":
                        text = output_part.get("text", "")
                        if not isinstance(text, str) or not text:
                            continue
                        if text_builder is None:
                            text_builder = message.add_text_content()
                            await queue.put(text_builder.emit_added())
                        text_fragments.append(text)
                        await queue.put(text_builder.emit_delta(text))
                        continue

                    await _flush_text_builder()
                    final_content.append(output_part)

            await _flush_text_builder()

            if not final_content:
                text_builder = message.add_text_content()
                await queue.put(text_builder.emit_added())
                await queue.put(text_builder.emit_text_done(""))
                await queue.put(text_builder.emit_done())
                final_content.append(
                    {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                        "logprobs": [],
                    }
                )

            completed_message = OutputItemMessage(
                {
                    "type": "message",
                    "id": message.item_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": final_content,
                }
            )
            await queue.put(message._emit_done(completed_message.as_dict()))  # type: ignore[attr-defined]
            await queue.put(stream.emit_completed())
        except asyncio.CancelledError:
            pass
        finally:
            await queue.put(None)

    task = asyncio.create_task(_producer())

    async def _watch_cancel() -> None:
        await cancellation_signal.wait()
        task.cancel()

    watcher = asyncio.create_task(_watch_cancel())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        watcher.cancel()
        if not task.done():
            task.cancel()


async def _emit_events(
    graph: Runnable[ResponsesGraphStateT | Command, Any],
    graph_input: ResponsesGraphStateT | Command,
    config: RunnableConfig,
    cancellation_signal: asyncio.Event,
    stream: Any,
    output_parser: ResponsesOutputParser,
    stream_mode: str = "values",
) -> None:
    """Emit graph output onto a ``ResponseEventStream`` via *output_parser*.

    Supports both streaming modes:

    * ``stream_mode="messages"`` — LangGraph yields ``(AIMessageChunk, metadata)``
      tuples; the chunk (first element) is passed to *output_parser*, enabling
      token-by-token streaming identical to :func:`_stream_messages`.
    * Other modes (e.g. ``"values"``) — each item is a full state snapshot dict
      passed directly to *output_parser*.

    In both cases *output_parser* is called for every item; returning an empty
    string suppresses the ``emit()`` call for that item.  When
    *cancellation_signal* fires, the task is cancelled and the stream is closed.
    If a custom ``output_parser`` raises, the error is logged, a terminal
    ``response.failed`` event is emitted when the stream supports it, and the
    producer stops without attempting further emissions.

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
        output_parser: Callable that maps an ``AIMessageChunk``
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
                    text = output_parser(extractor_input)
                    if text:
                        await stream.emit(text)
                except Exception as exc:
                    message = _parser_failure_message(
                        hook_name="output_parser",
                        parser=output_parser,
                        exc=exc,
                    )
                    logger.exception(
                        "Configured output_parser failed",
                        extra={
                            "hook_name": "output_parser",
                            "parser_name": _parser_name(output_parser),
                            "exception_type": type(exc).__name__,
                        },
                    )
                    await _emit_stream_failure(stream, message=message)
                    return
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
async def default_input_parser(
    request: ResponsesInputRequest,  # noqa: ARG001
    context: ResponsesInputContext,
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
    history_items = await context.get_history()
    current_items = await context.get_input_items()
    history_messages = _history_to_messages(list(history_items))
    current_messages = _history_to_messages(list(current_items))
    if not current_messages:
        user_input = (await context.get_input_text()) or ""
        if user_input:
            current_messages = [HumanMessage(content=user_input)]

    return {"messages": history_messages + current_messages}


@experimental()
class AzureAIResponsesAgentHost(Generic[ResponsesGraphStateT]):
    """Host a compiled LangGraph graph as an agent inside Azure AI Foundry.

    This class is the *server/host* side of the Foundry Agent Service integration.
    It registers a ``response_handler`` on a ``ResponsesAgentServerHost`` that:

    1. Checks for a pending ``interrupt()`` (e.g. MCP tool-call approval).  If
       one is found, resumes the graph via ``Command(resume=...)``.  Otherwise
       calls *input_parser* to build the graph input for the new turn (default:
       history + user text as ``{"messages": [...]}``).
    2. Invokes the LangGraph graph asynchronously with the prepared input.
    3. Streams the output back to Foundry via ``ResponseEventStream``.
    4. Wires the platform-supplied ``cancellation_signal`` to cancel the
       running graph task on early termination.

    Developer guidance:

    * When the graph uses an messages-based state (e.g. via ``MessagesState``),
        the default input parser and ``stream_mode="messages"`` work out of the box,
        yielding token-level streaming if automatically.
    * The default input parser pass the entire conversation to the graph so you
        don't need to maintain session state.  Adjust the *responses_history_count*
        parameter based on your graph's typical conversation turns.
    * If the graph uses a custom state schema, provide a custom ``input_parser`` to
        shape the Foundry input into the graph's expected format. Use
        `context.get_history()` and `context.get_input_items()` to access conversation
        history and the current user input.
    * When streaming_mode is not "messages", the graph can emit any serializable state
        dict; use a custom ``output_parser`` to extract the text to emit on the
        response stream.
    * If a custom ``input_parser`` raises, the client receives a
        ``response.failed`` event sequence naming the failing parser.
    * If a custom ``output_parser`` raises while streaming, the stream ends with
        ``response.failed`` naming the failing parser.
    * Only developer-supplied parser hooks use that parser-specific diagnostic
        path; the built-in parser continues to rely on the SDK's normal request
        handling.
    * Enable debug logging for ``langchain_azure_ai.agents.runtime`` when
        integrating custom parsers or interrupt/resume flows.

    Example:

    .. code-block:: python

        from langgraph.graph import StateGraph, MessagesState, START, END
        from langchain_azure_ai.agents.runtime import AzureAIResponsesAgentHost

        builder = StateGraph(MessagesState)
        builder.add_node("agent", my_agent_node)
        builder.add_edge(START, "agent")
        builder.add_edge("agent", END)
        graph = builder.compile()

        host = AzureAIResponsesAgentHost[MessagesState](
            graph=graph,
        )

        if __name__ == "__main__":
            host.run()

    Args:
        graph: A compiled LangGraph graph whose non-interrupt input type
            matches ``ResponsesGraphInputT``. Unless a custom ``input_parser``
            is provided, that is typically ``dict[str, Any]`` shaped like
            ``{"messages": list[BaseMessage]}``.
        responses_history_count: Number of past conversation turns to fetch from
            Foundry for each turn. Passed as ``default_fetch_history_count`` to
            the underlying ``ResponsesAgentServerHost``.  Adjust this based on your
            graph's typical context window and the average length of conversation
            turns.  Default is 100.
        stream_mode: LangGraph ``stream_mode`` passed to ``graph.astream()``.
            Default is ``"messages"``, which works for ``MessagesState``-compatible
            graphs. When *output_parser* is provided, ``"messages"`` mode passes
            each ``AIMessageChunk`` to the extractor for token-by-token streaming;
            ``"values"`` (or any other mode) passes full state snapshots instead.
        input_parser: Async callable matching
            ``ResponsesInputParser[ResponsesGraphInputT]`` that builds the graph
            input for a new (non-interrupt) turn. Receives the raw
            ``ResponsesInputRequest`` and ``ResponsesInputContext`` objects so
            it can read ``context.response_id``, etc. The returned value is
            passed verbatim to ``graph.astream()``. Defaults to
            :func:`default_input_parser` which wraps conversation history
            and the current user text in ``{"messages": [...]}``.  Override
            this when your graph state schema has additional keys (e.g.
            ``"constraints"``, ``"metadata"``) or when you need full control
            over message formatting.
        output_parser: Callable matching ``ResponsesOutputParser`` called for
            every item yielded by ``graph.astream()``. When provided, the host
            switches to the ``ResponseEventStream`` path and calls ``emit()``
            for each non-empty result. Use this for graphs that do *not* use
            ``MessagesState`` or when you need custom chunk extraction.

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
    """

    def __init__(
        self,
        graph: Runnable[ResponsesGraphStateT | Command, Any],
        *,
        responses_history_count: int = 100,
        stream_mode: str = "messages",
        input_parser: ResponsesInputParser[ResponsesGraphStateT] | None = None,
        output_parser: ResponsesOutputParser | None = None,
        **kwargs: Any,
    ) -> None:
        if input_parser is None and not _graph_has_messages_input(graph):
            raise ValueError(
                "The graph's input schema does not have a 'messages' key, so the "
                "default input parser cannot be used. Provide a custom 'input_parser' "
                "that builds the graph's expected input from the Foundry request and "
                "conversation context."
            )

        self._graph = graph
        self._output_parser = output_parser
        self._uses_default_input_parser = input_parser is None
        self._input_parser = (
            input_parser
            if input_parser is not None
            else cast(
                ResponsesInputParser[ResponsesGraphStateT],
                default_input_parser,
            )
        )
        self._stream_mode: str = stream_mode

        self._app: ResponsesAgentServerHost = ResponsesAgentServerHost(  # type: ignore[misc]
            options=ResponsesServerOptions(
                default_fetch_history_count=responses_history_count,
            ),
            **kwargs,
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
            graph_input: ResponsesGraphStateT | Command = Command(resume=resume_value)
            logger.debug("Resuming interrupted graph (thread_id=%s)", thread_id)
        else:
            try:
                graph_input = await self._input_parser(request, context)
            except Exception as exc:
                if self._uses_default_input_parser:
                    raise
                return _failed_response_events(
                    request=request,
                    context=context,
                    hook_name="input_parser",
                    parser=self._input_parser,
                    exc=exc,
                )

        if self._output_parser is not None:
            # Non-MessagesState path: ResponseEventStream + emit()
            stream = ResponseEventStream(context, request)  # type: ignore[misc]
            asyncio.create_task(
                _emit_events(
                    self._graph,
                    graph_input,
                    config,
                    cancellation_signal,
                    stream,
                    self._output_parser,
                    self._stream_mode,
                )
            )
            return stream

        # MessagesState path: stream response events directly so non-text
        # content can be preserved in the final assistant message.
        return _stream_message_events(
            self._graph,
            graph_input,
            config,
            cancellation_signal,
            request,
            context,
            self._stream_mode,
        )

    def run(self, **kwargs: Any) -> None:
        """Start the agent server.

        Accepts the same keyword arguments as
        ``ResponsesAgentServerHost.run()``.
        """
        self._app.run(**kwargs)

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike[str] = "langgraph.json",
        *,
        graph_name: str | None = None,
        responses_history_count: int = 100,
        stream_mode: str = "messages",
        input_parser: ResponsesInputParser[ResponsesGraphStateT] | None = None,
        output_parser: ResponsesOutputParser | None = None,
    ) -> "AzureAIResponsesAgentHost[ResponsesGraphStateT]":
        """Create an instance by loading the graph from a ``langgraph.json`` file.

        Reads the ``graphs`` section of *path* to locate and import the graph
        module, then constructs an :class:`AzureAIResponsesAgentHost` with the
        loaded graph and any additional arguments supplied.

        Args:
            path: Path to the ``langgraph.json`` configuration file.
                Defaults to ``"langgraph.json"`` in the current working
                directory.
            graph_name: Key of the graph to load from the ``"graphs"`` dict.
                May be omitted when the file defines exactly one graph;
                required when multiple graphs are present.
            responses_history_count: Number of past responses to retain in history.
                Defaults to 100.
            stream_mode: LangGraph stream mode passed to ``graph.astream()``.
                Defaults to ``"messages"``.
            input_parser: Optional custom async input parser callable.
            output_parser: Optional custom output parser callable.

        Returns:
            A fully configured :class:`AzureAIResponsesAgentHost` instance.

        Example::

            # Single-graph config — graph_name is optional
            host = AzureAIResponsesAgentHost.from_config()

            # Multi-graph config — graph_name is required
            host = AzureAIResponsesAgentHost.from_config(graph_name="agent")

            # Custom config file path
            host = AzureAIResponsesAgentHost.from_config(path="/app/langgraph.json")
        """
        from langchain_azure_ai.agents.runtime._config import (
            load_graph_from_langgraph_config,
        )

        graph = load_graph_from_langgraph_config(path, graph_name=graph_name)
        return cls(
            graph=graph,
            responses_history_count=responses_history_count,
            stream_mode=stream_mode,
            input_parser=input_parser,  # type: ignore[arg-type]
            output_parser=output_parser,
        )
