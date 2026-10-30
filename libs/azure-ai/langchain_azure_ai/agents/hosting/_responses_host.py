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
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, closing
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

try:
    from azure.ai.agentserver.core import resolve_state_subdir
    from azure.ai.agentserver.core.streaming import EventStream, streams
    from azure.ai.agentserver.responses import (
        ResponseContext,
        ResponseEventStream,
        ResponseProviderProtocol,
        ResponsesAgentServerHost,
        ResponsesServerOptions,
    )
    from azure.ai.agentserver.responses.models._helpers import (
        get_conversation_id,
        to_item,
    )
except ImportError as exc:
    raise ImportError(
        "The azure-ai-agentserver-responses package is required to use "
        "ResponsesHostServer. Please install it via "
        "`pip install azure-ai-agentserver-responses` or "
        "`pip install langchain-azure-ai[hosting]`."
    ) from exc

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from langchain_azure_ai._api.base import experimental
from langchain_azure_ai.agents.hosting import (
    HostingFeature,
    _add_process_hosting_features,
    _add_request_hosting_features,
    _hosting_feature_scope,
    _request_hosting_features,
)

from ._checkpoint._checkpoint_reuse import (
    _checkpoint_reads,
    _CheckpointReadContext,
    _RequestCheckpointSaver,
)
from ._converters import (
    UsageAccumulator,
    build_messages_input,
    detect_pending_interrupts,
    emit_interrupts,
    is_messages_state_schema,
    parse_resume_command,
    stream_graph_to_events,
    track_pending_interrupts,
    validate_approval_responses,
)
from ._instructions import _ResponseInstructions, with_response_instructions
from ._responses import (
    CONVERSATION_CHECKPOINT_KEY,
    METADATA_LANGGRAPH_ROOT_RESPONSE_ID,
    CheckpointRef,
    ConversationChainStoreProtocol,
    FoundryConversationChainStore,
    HostingRunnableConfig,
    TaskStorageManager,
)
from ._storage import state_store_client_scope
from ._storage.state_store_factory import (
    _CleanupTimeoutError,
    _close_resource,
    _finish_cleanup,
)

if TYPE_CHECKING:
    from azure.ai.agentserver.responses.models import CreateResponse
    from langgraph.types import Interrupt


@asynccontextmanager
async def _closing_stream(
    stream: AsyncIterator[Any],
) -> AsyncIterator[AsyncIterator[Any]]:
    """Close before final writes; active errors and control signals take priority."""
    close = getattr(stream, "aclose", None)
    if close is None:
        yield stream
    else:
        async with _finish_cleanup(_close_resource(close, "graph_iterator")):
            yield stream


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

METADATA_STEERABLE_CONVERSATION = "foundry.agent.steerable_conversation"
_RECOVERY_STREAM_LOCK_WORKAROUND = "_langchain_azure_recovery_stream_lock"
_USER_NAMESPACE_PREFIX = "user-"
_USER_NAMESPACE_SEPARATOR = ":"
_USER_NAMESPACE_HASH = hashlib.sha256


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


async def _get_or_create_response_event_stream(
    response_id: str,
    *,
    reclaim_stale_lock: bool = False,
) -> EventStream:
    try:
        return await streams.get_or_create(response_id)
    except RuntimeError as exc:
        lock_path = (
            _stale_recovery_stream_lock_path(exc) if reclaim_stale_lock else None
        )
        if lock_path is None:
            raise
        # Recovery begins only after the durable task manager has reclaimed
        # ownership. Fresh responses must never remove another writer's lock.
        try:
            lock_path.unlink(missing_ok=True)
        except OSError as cleanup_exc:
            raise exc from cleanup_exc
        logger.warning(
            "Recovered response %s reclaimed stale replay-stream lock %s",
            response_id,
            lock_path,
        )
        return await streams.get_or_create(response_id)


def _install_recovery_stream_lock_workaround(
    app: ResponsesAgentServerHost,
) -> None:
    """Retry a recovered SDK stream body after reclaiming its stale lock."""
    orchestrator = getattr(app, "_orchestrator", None)
    run_stream_body = getattr(orchestrator, "_run_resilient_stream_body", None)
    if not callable(run_stream_body) or getattr(
        run_stream_body,
        _RECOVERY_STREAM_LOCK_WORKAROUND,
        False,
    ):
        return

    @wraps(run_stream_body)
    async def run_with_recovery_lock_retry(*args: Any, **kwargs: Any) -> Any:
        context = kwargs.get("context")
        response_id = kwargs.get("response_id")
        if bool(getattr(context, "is_recovery", False)) and isinstance(
            response_id,
            str,
        ):
            await _get_or_create_response_event_stream(
                response_id,
                reclaim_stale_lock=True,
            )
        return await run_stream_body(*args, **kwargs)

    setattr(
        run_with_recovery_lock_retry,
        _RECOVERY_STREAM_LOCK_WORKAROUND,
        True,
    )
    setattr(orchestrator, "_run_resilient_stream_body", run_with_recovery_lock_retry)


def _validate_response_branching_checkpointer(checkpointer: object) -> None:
    """Reject missing async implementations on the original checkpoint saver."""
    while isinstance(checkpointer, _RequestCheckpointSaver):
        checkpointer = checkpointer.__wrapped__
    unsupported = []
    for name in ("aget_tuple", "aput", "aput_writes"):
        method = getattr(checkpointer, name, None)
        if not callable(method) or getattr(method, "__func__", method) is getattr(
            BaseCheckpointSaver, name
        ):
            unsupported.append(name)
    if unsupported:
        raise ValueError(
            "Response branching requires a checkpointer with asynchronous "
            f"implementations of {', '.join(unsupported)}."
        )


def _uses_langgraph_checkpointer(graph: "CompiledStateGraph") -> bool:
    return getattr(graph, "checkpointer", None) is not None


def _is_in_memory_checkpointer(checkpointer: Any) -> bool:
    if checkpointer is None:
        return False
    # Proxies expose the wrapped saver's class through __class__.
    cls = checkpointer.__class__
    return cls.__name__ in {"MemorySaver", "InMemorySaver"} or "memory" in (
        cls.__module__.lower()
    )


def _message_count(
    items: Sequence[Any],
    *,
    skip_call_ids: frozenset[str] = frozenset(),
) -> int:
    return len(build_messages_input(items, skip_call_ids=skip_call_ids)["messages"])


def _scope_thread_id(thread_id: str, context: ResponseContext) -> str:
    """Namespace a LangGraph thread by the platform-provided user key."""
    platform_context = getattr(context, "platform_context", None)
    user_id_key = getattr(platform_context, "user_id_key", None)
    if not isinstance(user_id_key, str) or not user_id_key:
        return thread_id
    user_namespace = _USER_NAMESPACE_HASH(user_id_key.encode()).hexdigest()
    return (
        f"{_USER_NAMESPACE_PREFIX}{user_namespace}"
        f"{_USER_NAMESPACE_SEPARATOR}{thread_id}"
    )


def _thread_namespace(thread_id: str) -> str:
    """Return a hosted thread's user namespace, or an empty string."""
    prefix, separator, _ = thread_id.partition(_USER_NAMESPACE_SEPARATOR)
    digest = prefix.removeprefix(_USER_NAMESPACE_PREFIX)
    if (
        separator
        and prefix.startswith(_USER_NAMESPACE_PREFIX)
        and len(digest) == _USER_NAMESPACE_HASH().digest_size * 2
        and all(char in "0123456789abcdef" for char in digest)
    ):
        return prefix + separator
    return ""


def _response_field(response: Any, name: str) -> str | None:
    value = getattr(response, name, None)
    if value is None and hasattr(response, "get"):
        value = response.get(name)
    return value if isinstance(value, str) and value else None


def _response_conversation_id(response: Any) -> str | None:
    conversation = getattr(response, "conversation", None)
    if conversation is None and hasattr(response, "get"):
        conversation = response.get("conversation")
    if isinstance(conversation, dict):
        value = conversation.get("id")
    else:
        value = getattr(conversation, "id", None)
    return value if isinstance(value, str) and value else None


def _response_root_id(response: Any) -> str | None:
    """Read a cached root from JSON or legacy dictionary metadata."""
    metadata = getattr(response, "metadata", None)
    if metadata is None and hasattr(response, "get"):
        metadata = response.get("metadata")
    if not isinstance(metadata, dict):
        return None
    internal = metadata.get("_internal_metadata")
    if isinstance(internal, str):
        try:
            internal = json.loads(internal)
        except ValueError:
            return None
    if not isinstance(internal, dict):
        return None
    value = internal.get(METADATA_LANGGRAPH_ROOT_RESPONSE_ID)
    return value if isinstance(value, str) and value else None


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
            configured them on ``app`` itself. Response branching requires
            ``app=None``.
        options: Optional :class:`ResponsesServerOptions` forwarded to
            :class:`ResponsesAgentServerHost`.
        store: Optional :class:`ResponseProviderProtocol`. This provider
            owns Responses protocol data, including response objects, input and
            output items, status, and history used by ``previous_response_id``.
            When omitted, Agent Server uses :class:`FileResponseStore` locally
            and :class:`FoundryStorageProvider` when hosted on Foundry.
        conversation_chain_store: Optional :class:`ConversationChainStoreProtocol`
            for dictionaries shared across turns. Implementations choose their
            serialization and durability model and must support concurrent calls,
            atomic replacement, and explicit failures. Defaults to
            :class:`FoundryConversationChainStore`, which uses one
            :class:`FoundryStateStore` namespace per conversation chain.
        enable_response_branching: Start ``previous_response_id`` requests from
            the specified completed response's checkpoint instead of the
            conversation head. Defaults to ``False``. Requires a
            history-preserving checkpointer with asynchronous ``aget_tuple``,
            ``aput``, and ``aput_writes`` implementations, steering disabled,
            and ``app=None``.
            Branching from paused graphs is not supported.
        prefix: URL prefix for response routes (e.g. ``"/v1"``).
        applicationinsights_connection_string: Forwarded to
            :class:`AgentServerHost`.
        graceful_shutdown_timeout: Forwarded to :class:`AgentServerHost`.

    Raises:
        ValueError: If the graph's state schema does not declare a
            ``messages`` field, or if ``resilient_background=True`` is
            configured without a LangGraph checkpointer. Override this class
            to host custom-state graphs. Also raised if response branching is
            enabled without the required async checkpoint implementations,
            with steering enabled, or with an external ``app``.
    """

    def __init__(
        self,
        graph: "CompiledStateGraph",
        *,
        app: Optional[ResponsesAgentServerHost] = None,
        options: Optional[ResponsesServerOptions] = None,
        store: Optional[ResponseProviderProtocol] = None,
        conversation_chain_store: Optional[ConversationChainStoreProtocol] = None,
        enable_response_branching: bool = False,
        prefix: str = "",
        applicationinsights_connection_string: Optional[str] = None,
        graceful_shutdown_timeout: Optional[int] = None,
    ) -> None:
        self._validate_graph_schema(graph)
        saver = getattr(graph, "checkpointer", None)
        if enable_response_branching:
            _validate_response_branching_checkpointer(saver)
        if isinstance(saver, BaseCheckpointSaver) and not isinstance(
            saver, _RequestCheckpointSaver
        ):
            graph_updates: dict[str, Any] = {
                "checkpointer": _RequestCheckpointSaver(saver)
            }
            if isinstance(graph, CompiledStateGraph):
                graph_updates["channels"] = dict(graph.channels)
            graph = graph.copy(graph_updates)
        if isinstance(graph, CompiledStateGraph):
            graph = with_response_instructions(graph)
        self._graph = graph
        self._graph_has_checkpointer = _uses_langgraph_checkpointer(graph)
        self._enable_response_branching = enable_response_branching
        if enable_response_branching and (
            not self._graph_has_checkpointer
            or app is not None
            or (options is not None and options.steerable_conversations)
        ):
            raise ValueError(
                "Response branching requires a history-preserving checkpointer, "
                "steerable_conversations=False, and app=None."
            )
        self._conversation_chain_store = (
            conversation_chain_store
            if conversation_chain_store is not None
            else FoundryConversationChainStore()
        )
        if (
            app is None
            and options is not None
            and options.resilient_background
            and not self._graph_has_checkpointer
        ):
            raise ValueError(
                "ResponsesHostServer requires a LangGraph checkpointer when "
                "resilient_background=True."
            )
        self._stream_modes: list[StreamMode] = ["updates", "messages"]
        self._durability: Literal["sync"] | None = None
        if self._graph_has_checkpointer:
            self._stream_modes.append("checkpoints")
            self._durability = "sync"
        self._steerable_conversations = bool(
            options is not None and options.steerable_conversations
        )
        self._hosting_features = HostingFeature.RESPONSES
        if options is not None and options.resilient_background:
            self._hosting_features |= HostingFeature.RESILIENT_BACKGROUND
        if self._steerable_conversations:
            self._hosting_features |= HostingFeature.STEERABLE_CONVERSATIONS
        _add_process_hosting_features(self._hosting_features)

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
        _install_recovery_stream_lock_workaround(self._app)

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
        """The hosted graph, copied to decorate a configured checkpoint saver."""
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

                Top-level ``instructions`` is a temporary developer message for
                this response. The saver excludes it from durable graph history;
                explicit system/developer input messages are retained.

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
        instructions = _ResponseInstructions.create(
            request.get("instructions"), context.response_id
        )

        if mode == "langgraph_checkpoint":
            graph_input = build_messages_input(
                current_items,
                skip_call_ids=skip_call_ids or frozenset(),
            )
            graph_input["messages"] = instructions.prepend(graph_input["messages"])
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
            skip_call_ids=skip_call_ids or frozenset(),
        )
        graph_input["messages"] = instructions.prepend(graph_input["messages"])
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
        ``output`` JSON into a :class:`Command`; the latter maps recognized
        LangChain ``HumanInTheLoopMiddleware`` requests to ordered
        approve/reject decisions.
        Other approvals echo the interrupt's value. Override to plug in
        custom resume protocols.

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
        ``mcp_approval_response`` item whose decision cannot be translated.
        This includes custom interrupt rejections and
        ``HumanInTheLoopMiddleware`` decisions not allowed for every action.
        When found, :meth:`handle_create` short-circuits the turn into
        ``response.failed(code="interrupt_rejected", …)``.

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
        return validate_approval_responses(items, pending)

    async def build_runnable_config(
        self,
        request: CreateResponse,
        context: ResponseContext,
    ) -> RunnableConfig:
        """Build a LangGraph ``RunnableConfig`` for the request.

        The logical thread is namespaced by the platform-provided user key.
        For ``previous_response_id`` chains, a cached root response ID is reused
        when available; otherwise the root response is resolved through the
        Responses provider when possible. Checkpointed graphs
        additionally restore the exact ``(thread_id, checkpoint_id)`` saved
        for this user and conversation chain. With any checkpoint saver,
        that reference fixes the request's input. Latest-checkpoint queries
        still reach storage to preserve historical-branch protection.
        When response branching is enabled, new ``previous_response_id``
        requests instead select that completed response's saved checkpoint.
        The full :class:`ResponseContext`
        is exposed under ``configurable.response_context`` for transient
        execution facts such as recovery state.

        Configurable keys prefixed with ``__langchain_azure_`` are reserved for
        host implementation details. Subclasses can preserve root caching by
        calling ``await super().build_runnable_config(request, context)`` and
        retaining the returned hosting-owned entries.

        Args:
            request: The parsed create-response request.
            context: The response context for the request.

        Returns:
            A ``RunnableConfig`` dict.
        """
        if self._is_response_branch(request, context):
            return await self._response_branch_config(request, context)
        hosting_config = await self._resolve_thread_config(request, context)
        if not self._graph_has_checkpointer:
            return hosting_config.runnable_config

        previous_response_id = request.get("previous_response_id")
        conversation_id = get_conversation_id(request) or context.conversation_id
        if (
            not context.is_recovery
            and not (isinstance(previous_response_id, str) and previous_response_id)
            and not (isinstance(conversation_id, str) and conversation_id)
        ):
            return hosting_config.runnable_config

        conversation_chain_id = _scope_thread_id(
            context.conversation_chain_id,
            context,
        )
        checkpoint_ref = CheckpointRef.from_dict(
            await self._conversation_chain_store.get(
                conversation_chain_id,
                CONVERSATION_CHECKPOINT_KEY,
            )
        )
        if checkpoint_ref is not None:
            hosting_config = hosting_config.with_checkpoint_ref(checkpoint_ref)
        return hosting_config.runnable_config

    def _is_response_branch(
        self, request: CreateResponse, context: ResponseContext
    ) -> bool:
        # shortcut: `conversation` is not checked here. A request that sets both
        # previous_response_id and conversation branches from the parent and ignores
        # the conversation, instead of returning the HTTP 400 required by the design.
        return (
            self._enable_response_branching
            and bool(request.get("previous_response_id"))
            # shortcut: Branches assume no recovery; revisit when branch
            # recovery is added.
            and not context.is_recovery
        )

    async def _response_branch_config(
        self, request: CreateResponse, context: ResponseContext
    ) -> RunnableConfig:
        parent_id = request.get("previous_response_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError("previous_response_id must be a nonempty string.")
        provider = getattr(context, "_provider", None)
        if provider is None:
            raise RuntimeError("Response branching requires a response provider.")
        parent = await provider.get_response(
            parent_id, context=context.platform_context
        )
        # shortcut: Ineligible parents (not completed, paused, or owned by another
        # user) are rejected after response.created, so clients receive
        # response.failed with server_error rather than a 4xx before execution.
        if parent.get("id") != parent_id or parent.get("status") != "completed":
            raise ValueError("The previous response must be completed.")
        reference = TaskStorageManager.from_stream(
            ResponseEventStream(response_id=parent_id, response=parent)
        ).checkpoint_ref
        if reference is None:
            raise ValueError("The previous response has no saved graph checkpoint.")
        namespace = _scope_thread_id("", context)
        if _thread_namespace(reference.thread_id) != namespace:
            raise ValueError("The previous response belongs to a different user.")
        return HostingRunnableConfig.create_from_checkpoint(
            reference, context
        ).runnable_config

    async def _resolve_thread_config(
        self,
        request: CreateResponse,
        context: ResponseContext,
    ) -> HostingRunnableConfig:
        previous_response_id = request.get("previous_response_id")
        conversation_id = get_conversation_id(request) or context.conversation_id
        thread_id: str
        if isinstance(conversation_id, str) and conversation_id:
            thread_id = conversation_id
        elif isinstance(previous_response_id, str) and previous_response_id:
            resolved_config = await self._thread_config_from_response_chain(
                previous_response_id,
                context,
            )
            if resolved_config is not None:
                return resolved_config
            thread_id = previous_response_id
        else:
            return HostingRunnableConfig.create(
                _scope_thread_id(context.response_id, context), context
            ).with_root_response_id(context.response_id)
        return HostingRunnableConfig.create(
            _scope_thread_id(thread_id, context), context
        )

    async def _thread_config_from_response_chain(
        self,
        previous_response_id: str,
        context: ResponseContext,
    ) -> HostingRunnableConfig | None:
        """Resolve a user-scoped thread and cache only a verified chain root."""
        provider = getattr(context, "_provider", None)
        if provider is None:
            return None

        response_id: str | None = previous_response_id
        seen: set[str] = set()
        last_response_id = previous_response_id
        root_response_id: str | None = None
        while response_id and response_id not in seen:
            seen.add(response_id)
            try:
                response = await provider.get_response(
                    response_id,
                    context=context.platform_context,
                )
            except Exception:
                logger.debug(
                    "Failed to resolve response chain for thread mapping",
                    exc_info=True,
                )
                return None

            conversation_id = _response_conversation_id(response)
            if conversation_id:
                return HostingRunnableConfig.create(
                    _scope_thread_id(conversation_id, context), context
                )

            cached_root_id = _response_root_id(response)
            if cached_root_id is not None:
                root_response_id = cached_root_id
                break

            last_response_id = response_id
            response_id = _response_field(response, "previous_response_id")
            if response_id is None:
                root_response_id = last_response_id
                break

        hosting_config = HostingRunnableConfig.create(
            _scope_thread_id(root_response_id or last_response_id, context), context
        )
        if root_response_id is not None:
            hosting_config = hosting_config.with_root_response_id(root_response_id)
        return hosting_config

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
            request.get("previous_response_id"),
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

             1. emits ``response.created`` / ``response.in_progress`` and durably
                 checkpoints handler admission,
                    2. resolves the LangGraph thread and checks for pending
                     ``interrupt()`` pauses on the checkpointed
           thread and:

           - if an ``mcp_approval_response`` cannot be mapped to the pending
             interrupt, emits ``response.failed(code="interrupt_rejected",
             …)`` and stops;
           - otherwise tries to resume from a matching
             ``function_call_output`` (rich) or
             ``mcp_approval_response`` (``HumanInTheLoopMiddleware`` decision
             conversion or the existing custom-interrupt approval behavior),
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
        recovering = bool(context.is_recovery)
        stream = self._new_stream(request, context)
        task_storage = TaskStorageManager.from_stream(stream)
        conversation_chain_id = _scope_thread_id(
            context.conversation_chain_id,
            context,
        )
        yield stream.emit_created()

        # Shutdown and request cancellation are independent. Check shutdown
        # first so resilient background work is deferred to the next lifetime.
        if context.shutdown.is_set():
            await context.exit_for_recovery()

        if cancellation_signal.is_set():
            yield stream.emit_in_progress()
            if context.client_cancelled:
                yield stream.emit_failed(
                    code="cancelled",
                    message="Request was cancelled.",
                )
            else:
                yield stream.emit_completed()
            return

        yield stream.emit_in_progress()

        if not recovering:
            # Persist handler admission before any provider or graph I/O.
            yield stream.checkpoint()

        usage = UsageAccumulator()
        try:
            config = await self.build_runnable_config(request, context)
            root_response_id = HostingRunnableConfig(config).root_response_id
            if root_response_id is not None:
                stream.internal_metadata[METADATA_LANGGRAPH_ROOT_RESPONSE_ID] = (
                    root_response_id
                )
            # Attach transient execution state after the overridable config
            # hook. The public Responses handler contract supplies this exact
            # Event as its third argument; keeping it out of build_runnable_config's
            # signature also preserves existing subclass overrides.
            config = (
                HostingRunnableConfig(config)
                .with_cancellation_signal(cancellation_signal)
                .runnable_config
            )
            instructions = _ResponseInstructions.create(
                request.get("instructions"), context.response_id
            )
            config = instructions.with_config(config)

            resume_command: Optional["Command"] = None
            consumed_call_ids: frozenset[str] = frozenset()
            graph_input: dict[str, Any] | Command | None
            checkpoint_ref = task_storage.checkpoint_ref if recovering else None
            if checkpoint_ref is not None:
                config = (
                    HostingRunnableConfig(config)
                    .with_checkpoint_ref(checkpoint_ref)
                    .runnable_config
                )
            checkpoint_reads = _checkpoint_reads.get()
            if checkpoint_reads is not None:
                checkpoint_reads.pin(getattr(self._graph, "checkpointer", None), config)

            if self._is_response_branch(request, context):
                state = await self._graph.aget_state(config)
                if (
                    HostingRunnableConfig(state.config).checkpoint_ref
                    != HostingRunnableConfig(config).checkpoint_ref
                    or (state.config.get("configurable") or {}).get("checkpoint_ns", "")
                    != ""
                    or state.created_at is None
                ):
                    raise ValueError(
                        "The previous response's graph checkpoint is missing."
                    )
                # shortcut: Assume no unanswered tool calls; validate parent outputs
                # in follow-up.
                if state.next or state.tasks or state.interrupts:
                    raise ValueError("Branching requires a finished parent graph.")

            if checkpoint_ref is not None:
                # Crash-recovered re-entry. The graph's own persistent
                # checkpointer holds the mid-turn state. An empty resume map
                # preserves completed tasks at an explicitly addressed checkpoint.
                # If the response has no checkpoint yet, replay the request through
                # the normal input / HITL parsing path below.
                logger.debug("Recovery: resuming graph from persisted checkpoint")
                graph_input = Command(resume={})

            else:
                if recovering:
                    logger.debug("Recovery: replaying request input")
                # Detect a pause from a previous turn and try to resume it.
                pending = await detect_pending_interrupts(self._graph, config)
                if pending:
                    _add_request_hosting_features(HostingFeature.HITL)
                    # HITL:
                    # Rejection short-circuits the turn into ``response.failed``
                    # so a client-issued ``mcp_approval_response{approve:false}``
                    # is not silently dropped.
                    approval_error = await self.detect_rejection(
                        request, context, pending
                    )
                    if approval_error is not None:
                        yield stream.emit_failed(
                            code="interrupt_rejected",
                            message=approval_error,
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
                    ref = HostingRunnableConfig(config).checkpoint_ref
                    if ref is not None:
                        task_storage.store_checkpoint_ref(ref)
                        await self._conversation_chain_store.set(
                            conversation_chain_id,
                            CONVERSATION_CHECKPOINT_KEY,
                            ref.to_dict(),
                        )
                    yield stream.emit_completed()
                    return

                if resume_command is not None:
                    graph_input = resume_command
                else:
                    graph_input = await self.build_input(
                        request, context, skip_call_ids=consumed_call_ids
                    )
            active_interrupts: list["Interrupt"] = []
            async with _closing_stream(
                instructions.wrap_stream(
                    self._graph.astream(
                        graph_input,
                        config=config,
                        stream_mode=self._stream_modes,
                        durability=self._durability,
                    )
                )
            ) as graph_stream:
                graph_stream = track_pending_interrupts(graph_stream, active_interrupts)
                async for event in stream_graph_to_events(
                    graph_stream,
                    stream,
                    cancellation_signal=cancellation_signal,
                    shutdown_signal=context.shutdown,
                    usage=usage,
                ):
                    yield event
            checkpoint_ref = task_storage.checkpoint_ref
            if checkpoint_ref is not None and not (
                self._enable_response_branching
                and bool(request.get("previous_response_id"))
            ):
                await self._conversation_chain_store.set(
                    conversation_chain_id,
                    CONVERSATION_CHECKPOINT_KEY,
                    checkpoint_ref.to_dict(),
                )
            if context.shutdown.is_set():
                await context.exit_for_recovery()
            if cancellation_signal.is_set():
                if context.client_cancelled:
                    yield stream.emit_failed(
                        code="cancelled",
                        message="Request was cancelled.",
                        usage=usage.response_usage,
                    )
                else:
                    # Steering supersedes this turn without cancelling it.
                    # Preserve its checkpointed partial output and let the
                    # queued turn begin from this response as its parent.
                    yield stream.emit_completed(usage=usage.response_usage)
                return

            # The updates stream carries the exact active set for this run;
            # checkpoint task history may retain answered parallel siblings.
            new_pending = (
                tuple(active_interrupts) if self._graph_has_checkpointer else ()
            )
            if new_pending:
                _add_request_hosting_features(HostingFeature.HITL)
                async for event in emit_interrupts(new_pending, stream):
                    yield event

            yield stream.emit_completed(usage=usage.response_usage)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LangGraph response handler failed")
            code, message = "server_error", "An internal error occurred."
            if (
                isinstance(exc, _CleanupTimeoutError)
                and exc.resource == "graph_iterator"
            ):
                # Unsettled writes cannot advance the chain or complete steering.
                if context.shutdown.is_set():
                    await context.exit_for_recovery()
                if cancellation_signal.is_set() and context.client_cancelled:
                    code, message = "cancelled", "Request was cancelled."
            yield stream.emit_failed(
                code=code,
                message=message,
                usage=usage.response_usage,
            )

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
            stream = ResponseEventStream(
                response_id=context.response_id,
                response=context.persisted_response,
            )
        else:
            stream = ResponseEventStream(
                response_id=context.response_id,
                request=request,
            )

        stream.response["instructions"] = request.get("instructions")
        metadata = dict(stream.response.get("metadata") or {})
        metadata[METADATA_STEERABLE_CONVERSATION] = str(
            self._steerable_conversations
        ).lower()
        stream.response["metadata"] = metadata
        return stream

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
        features = self._hosting_features
        with closing(_CheckpointReadContext()) as checkpoint_reads:
            async with state_store_client_scope(bind=False) as factory:
                events = self.handle_create(request, context, cancellation_signal)

                async def close() -> None:
                    close = getattr(events, "aclose", None)
                    if close is not None:
                        with (
                            factory.bind(),
                            checkpoint_reads.bind(),
                            _hosting_feature_scope(features),
                        ):
                            await _close_resource(close, "handler_iterator", factory)

                async with _finish_cleanup(close()):
                    while True:
                        # AgentServer can resume this iterator in a different Task.
                        with (
                            factory.bind(),
                            checkpoint_reads.bind(),
                            _hosting_feature_scope(features),
                        ):
                            try:
                                event = await anext(events)
                            except StopAsyncIteration:
                                return
                            finally:
                                features = _request_hosting_features.get()
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
