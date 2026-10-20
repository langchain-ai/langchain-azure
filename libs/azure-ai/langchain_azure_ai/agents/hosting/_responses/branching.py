"""Exact checkpoint selection for completed, non-paused response branches."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from copy import copy
from inspect import signature
from threading import Lock
from typing import TYPE_CHECKING, Any

from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseEventStream,
    ResponseProviderProtocol,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .checkpoint_ref import CheckpointRef
from .conversation_chain_store import ConversationChainStoreProtocol
from .hosting_runnable_config import HostingRunnableConfig
from .task_storage_manager import TaskStorageManager

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

BRANCH_MODE_HEADER = "x-client-langchain-response-branching"
BRANCH_MODE = "checkpoint-v1"
BRANCH_ORIGIN_KEY = "langgraph_branch_origin_v1"
BRANCH_BOUNDARY_KEY = "langgraph_response_boundary_v1"
BRANCH_MODE_METADATA = "langgraph_response_branching"


class _ResponseCheckpointMetadata(CheckpointMetadata):
    langgraph_response_execution: str


class BranchingError(ValueError):
    """A branch failure with an internal code and a client-safe message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class BranchingAdmissionMiddleware:
    """Stamp trusted execution mode and validate opt-in response linkage.

    Args:
        app: The next ASGI application.
        enabled: Whether fresh requests may use response branching. Client mode
            headers are removed even when branching is disabled.
    """

    def __init__(self, app: ASGIApp, *, enabled: bool) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate fresh HTTP requests before SDK admission."""
        if scope["type"] == "http":
            headers = [
                (name, value)
                for name, value in scope.get("headers", [])
                if name.lower() != BRANCH_MODE_HEADER.encode("ascii")
            ]
            if self.enabled:
                headers.append(
                    (BRANCH_MODE_HEADER.encode("ascii"), BRANCH_MODE.encode("ascii"))
                )
            scope = {**scope, "headers": headers}
            if (
                self.enabled
                and scope["method"] == "POST"
                and scope["path"].rstrip("/").endswith("/responses")
            ):
                body = await Request(scope, receive).body()
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    payload = None
                if isinstance(payload, dict):
                    invalid = self._invalid_linkage(payload)
                    if invalid is not None:
                        parameter, message = invalid
                        await JSONResponse(
                            {
                                "error": {
                                    "type": "invalid_request_error",
                                    "message": message,
                                    "param": parameter,
                                    "code": None,
                                }
                            },
                            status_code=400,
                        )(scope, receive, send)
                        return
                original_receive = receive
                body_sent = False

                async def replay_body() -> Any:
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {
                            "type": "http.request",
                            "body": body,
                            "more_body": False,
                        }
                    return await original_receive()

                receive = replay_body
        await self.app(scope, receive, send)

    @staticmethod
    def _invalid_linkage(payload: dict[str, Any]) -> tuple[str, str] | None:
        if "response_id" in payload:
            return "response_id", "Unknown parameter: response_id."
        parent = payload.get("previous_response_id")
        if parent is not None:
            if not isinstance(parent, str) or not parent.strip():
                return (
                    "previous_response_id",
                    "previous_response_id must be a non-empty string or null.",
                )
            if payload.get("conversation") is not None:
                return (
                    "previous_response_id",
                    "previous_response_id and conversation are mutually exclusive.",
                )
        return None


class ResponseBranchStore:
    """Keep confirmed origins separate from completed response boundaries.

    Args:
        store: The existing conversation-chain store for checkpoint records.
        provider: The configured response provider for authorized parent lookup.
    """

    def __init__(
        self,
        store: ConversationChainStoreProtocol,
        provider: ResponseProviderProtocol | None = None,
    ) -> None:
        self._store = store
        self._provider = provider

    @staticmethod
    def _record(ref: CheckpointRef, *, paused: bool) -> dict[str, str]:
        return {
            "version": "1",
            "checkpoint_ns": "",
            "paused": str(paused).lower(),
            **ref.to_dict(),
        }

    @staticmethod
    def _reference(record: Any) -> CheckpointRef:
        if (
            not isinstance(record, dict)
            or record.get("version") != "1"
            or record.get("checkpoint_ns") != ""
            or record.get("paused") not in {"true", "false"}
        ):
            raise BranchingError(
                "invalid_branch_state", "The response checkpoint record is invalid."
            )
        ref = CheckpointRef.from_dict(record)
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable", "The response checkpoint is unavailable."
            )
        return ref

    @staticmethod
    def _require_completed_tools(parent: Mapping[str, Any]) -> None:
        output = parent.get("output") or ()
        calls = {
            item.get("call_id")
            for item in output
            if item.get("type") == "function_call"
        }
        results = {
            item.get("call_id")
            for item in output
            if item.get("type") == "function_call_output"
        }
        if calls - results or any(
            item.get("type") == "mcp_approval_request" for item in output
        ):
            raise BranchingError(
                "unsupported_branch_parent",
                "Response branching does not support parents with pending "
                "approvals or tool results.",
            )

    async def prepare(
        self,
        *,
        response_key: str,
        parent_key: str,
        parent_id: str,
        context: ResponseContext,
        graph: CompiledStateGraph | None = None,
        config: RunnableConfig | None = None,
        select_only: bool = False,
    ) -> CheckpointRef:
        """Select a completed parent, then confirm it after config hooks."""
        existing = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
        if context.is_recovery:
            if await self._store.get(response_key, BRANCH_BOUNDARY_KEY) is not None:
                raise BranchingError(
                    "invalid_branch_state",
                    "A response with a published boundary cannot be resumed.",
                )
            ref = self._reference(existing)
            if (
                existing is None
                or existing.get("mode") != BRANCH_MODE
                or existing.get("parent_response_id") != parent_id
                or existing.get("paused") != "false"
            ):
                raise BranchingError(
                    "invalid_branch_state", "The confirmed branch origin is invalid."
                )
            return ref

        if self._provider is None:
            raise BranchingError(
                "invalid_branch_state", "The Responses provider is unavailable."
            )
        parent = await self._provider.get_response(
            parent_id, context=context.platform_context
        )
        if parent is None or parent.get("status") != "completed":
            raise BranchingError(
                "invalid_branch_state",
                "The parent response must be stored and completed.",
            )
        metadata = (parent.get("metadata") or {}).get("_internal_metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise BranchingError(
                    "invalid_branch_state", "The parent checkpoint metadata is invalid."
                ) from exc
        if not isinstance(metadata, Mapping):
            raise BranchingError(
                "invalid_branch_state", "The parent checkpoint metadata is invalid."
            )
        boundary = metadata.get(BRANCH_BOUNDARY_KEY)
        if boundary is None:
            raise BranchingError(
                "invalid_branch_state",
                "The parent response has no completed response-branching boundary.",
            )
        if (
            isinstance(boundary, dict)
            and "thread_id" not in boundary
            and "checkpoint_id" not in boundary
        ):
            captured_ref = TaskStorageManager(dict(metadata)).checkpoint_ref
            if captured_ref is None:
                raise BranchingError(
                    "checkpoint_unavailable", "The parent has no completed checkpoint."
                )
            boundary = {**boundary, **captured_ref.to_dict()}
        ref = self._reference(boundary)
        if boundary["paused"] == "true":
            raise BranchingError(
                "unsupported_branch_parent",
                "Response branching does not support paused parent graphs.",
            )
        self._require_completed_tools(parent)
        if await self._store.get(parent_key, BRANCH_BOUNDARY_KEY) != boundary:
            raise BranchingError(
                "invalid_branch_state", "The parent checkpoint records do not agree."
            )
        origin = {
            **boundary,
            "mode": BRANCH_MODE,
            "parent_response_id": parent_id,
        }
        if existing is not None and existing != origin:
            raise BranchingError(
                "invalid_branch_state", "The confirmed branch origin is invalid."
            )
        if select_only:
            return ref
        if graph is None or not isinstance(graph.checkpointer, BaseCheckpointSaver):
            raise BranchingError(
                "checkpoint_unavailable", "The graph checkpoint saver is unavailable."
            )
        selected = copy(graph)
        selected.checkpointer = ResponseCheckpointSaver(
            graph.checkpointer, branching=True
        )
        selected_config = (
            HostingRunnableConfig(config or {}).with_checkpoint_ref(ref).runnable_config
        )
        snapshot = await selected.aget_state(selected_config)
        if snapshot.next or any(task.interrupts for task in snapshot.tasks):
            raise BranchingError(
                "unsupported_branch_parent",
                "Response branching does not support paused parent graphs.",
            )
        if existing is None:
            await self._store.set(response_key, BRANCH_ORIGIN_KEY, origin)
        return ref

    async def publish(
        self,
        response_key: str,
        stream: ResponseEventStream,
        ref: CheckpointRef | None,
        *,
        paused: bool,
    ) -> None:
        """Index a boundary before the SDK commits the terminal response."""
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable",
                "The response produced no checkpoint boundary.",
            )
        record = self._record(ref, paused=paused)
        existing = await self._store.get(response_key, BRANCH_BOUNDARY_KEY)
        if existing is not None and existing != record:
            raise BranchingError(
                "invalid_branch_state", "The response boundary cannot be replaced."
            )
        if TaskStorageManager.from_stream(stream).checkpoint_ref != ref:
            raise BranchingError(
                "invalid_branch_state",
                "The completed boundary does not match the execution checkpoint.",
            )
        stream.internal_metadata[BRANCH_BOUNDARY_KEY] = {
            key: value
            for key, value in record.items()
            if key not in {"thread_id", "checkpoint_id"}
        }
        if existing is None:
            await self._store.set(response_key, BRANCH_BOUNDARY_KEY, record)


class ResponseCheckpointSaver(BaseCheckpointSaver[Any]):
    """Validate exact reads and record request-owned recovery checkpoints.

    The graph-owned saver retains storage and lifecycle ownership. Only root
    recovery reads use a request-local head; private subgraph state selection
    and checkpoint history remain application-owned.

    Args:
        saver: The graph-owned checkpoint saver.
        branching: Whether explicit checkpoint reads must match exactly.
        response_key: Current response's user-scoped key, or ``None`` to leave
            checkpoint metadata unchanged.
        origin: Confirmed parent checkpoint, or ``None`` for a new root.
    """

    def __init__(
        self,
        saver: BaseCheckpointSaver[Any],
        *,
        branching: bool,
        response_key: str | None = None,
        origin: CheckpointRef | None = None,
    ) -> None:
        super().__init__(serde=saver.serde)
        self._saver = saver
        self._branching = branching
        self._put_writes_accepts_task_path = (
            "task_path" in signature(saver.put_writes).parameters
        )
        self._aput_writes_accepts_task_path = (
            "task_path" in signature(saver.aput_writes).parameters
        )
        self._execution = (
            json.dumps(
                [response_key, origin.thread_id, origin.checkpoint_id]
                if origin is not None
                else [response_key, None, None]
            )
            if branching and response_key is not None
            else None
        )
        self._resume_head: CheckpointRef | None = None
        self._checkpoint_writes: dict[CheckpointRef, Future[None]] = {}
        self._write_error: BaseException | None = None
        self._write_lock = Lock()

    @property
    def config_specs(self) -> Any:
        """Preserve the wrapped saver's configurable fields."""
        return self._saver.config_specs

    async def wait_for_checkpoint(self, ref: CheckpointRef) -> None:
        """Wait for persistence before publishing a recovery reference."""
        with self._write_lock:
            future = self._checkpoint_writes.get(ref)
            if future is None:
                future = self._checkpoint_writes[ref] = Future()
                if self._write_error is not None:
                    future.set_exception(self._write_error)
        await asyncio.wrap_future(future)

    def _record_checkpoint(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> None:
        configurable = config.get("configurable") or {}
        if configurable.get("checkpoint_ns", "") != "":
            return
        ref = CheckpointRef(configurable["thread_id"], checkpoint["id"])
        with self._write_lock:
            future = self._checkpoint_writes.setdefault(ref, Future())
            if not future.done():
                future.set_result(None)

    @contextmanager
    def _track_writes(self) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:
            with self._write_lock:
                self._write_error = exc
                for future in self._checkpoint_writes.values():
                    if not future.done():
                        future.set_exception(exc)
            raise

    async def validate_progress(self, config: RunnableConfig) -> None:
        """Require recovery progress owned by this response and parent."""
        saved = await self.aget_tuple(config)
        if (
            self._execution is None
            or saved is None
            or saved.metadata.get("langgraph_response_execution") != self._execution
        ):
            raise BranchingError(
                "invalid_branch_state",
                "The recovery checkpoint's response ownership could not be verified.",
            )

    def _metadata(
        self, config: RunnableConfig, metadata: CheckpointMetadata
    ) -> CheckpointMetadata:
        if (
            self._execution is None
            or (config.get("configurable") or {}).get("checkpoint_ns", "") != ""
        ):
            return metadata
        stamped: _ResponseCheckpointMetadata = {
            **metadata,
            "langgraph_response_execution": self._execution,
        }
        return stamped

    def resume_config(self, config: RunnableConfig) -> RunnableConfig:
        """Use native recovery without replaying already completed tasks."""
        ref = HostingRunnableConfig(config).checkpoint_ref
        configurable = dict(config.get("configurable") or {})
        if (
            not self._branching
            or ref is None
            or configurable.get("checkpoint_ns", "") != ""
        ):
            raise BranchingError(
                "invalid_branch_state", "The branch recovery checkpoint is invalid."
            )
        self._resume_head = ref
        del configurable["checkpoint_id"]
        return {**config, "configurable": configurable}

    def _resolve(self, config: RunnableConfig) -> RunnableConfig:
        head = self._resume_head
        requested = config.get("configurable") or {}
        if head is None or requested.get("checkpoint_ns", "") != "":
            return config
        if requested.get("thread_id") != head.thread_id:
            raise BranchingError(
                "invalid_branch_state", "The branch recovery thread does not match."
            )
        if requested.get("checkpoint_id"):
            return config
        return HostingRunnableConfig(config).with_checkpoint_ref(head).runnable_config

    def _advance_head(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        saved_config: RunnableConfig,
    ) -> RunnableConfig:
        head = self._resume_head
        if head is not None and not (config.get("configurable") or {}).get(
            "checkpoint_ns"
        ):
            ref = HostingRunnableConfig(saved_config).checkpoint_ref
            if (
                ref is None
                or ref.thread_id != head.thread_id
                or ref.checkpoint_id != checkpoint["id"]
                or (saved_config.get("configurable") or {}).get("checkpoint_ns", "")
                != ""
            ):
                raise BranchingError(
                    "checkpoint_unavailable",
                    "The branch recovery checkpoint was not saved.",
                )
            self._resume_head = ref
        self._record_checkpoint(config, checkpoint)
        return saved_config

    def _validate(
        self, config: RunnableConfig, saved: CheckpointTuple | None
    ) -> CheckpointTuple | None:
        if not self._branching:
            return saved
        requested = config.get("configurable") or {}
        checkpoint_id = requested.get("checkpoint_id")
        if saved is None:
            if checkpoint_id:
                raise BranchingError(
                    "checkpoint_unavailable", "The required checkpoint is unavailable."
                )
            return None
        actual = saved.config.get("configurable") or {}
        if checkpoint_id and (
            actual.get("thread_id") != requested.get("thread_id")
            or actual.get("checkpoint_ns", "") != requested.get("checkpoint_ns", "")
            or actual.get("checkpoint_id") != checkpoint_id
            or saved.checkpoint.get("id") != checkpoint_id
        ):
            raise BranchingError(
                "checkpoint_unavailable",
                "The saver did not return the required checkpoint.",
            )
        execution_options = {
            key: value
            for key, value in requested.items()
            if key not in {"thread_id", "checkpoint_ns", "checkpoint_id"}
        }
        execution_metadata: RunnableConfig = (
            {"metadata": config["metadata"]} if "metadata" in config else {}
        )
        return saved._replace(
            config={
                **saved.config,
                **execution_metadata,
                "configurable": {**execution_options, **actual},
            },
            parent_config=(
                {
                    **saved.parent_config,
                    **execution_metadata,
                    "configurable": {
                        **execution_options,
                        **(saved.parent_config.get("configurable") or {}),
                    },
                }
                if saved.parent_config is not None
                else None
            ),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read and validate an explicit checkpoint selection."""
        config = self._resolve(config)
        saved = self._validate(config, self._saver.get_tuple(config))
        if saved is not None:
            self._record_checkpoint(config, saved.checkpoint)
        return saved

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read and validate an asynchronous checkpoint selection."""
        config = self._resolve(config)
        saved = self._validate(config, await self._saver.aget_tuple(config))
        if saved is not None:
            self._record_checkpoint(config, saved.checkpoint)
        return saved

    def list(self, *args: Any, **kwargs: Any) -> Iterator[CheckpointTuple]:
        """Delegate checkpoint history without changing retention."""
        return self._saver.list(*args, **kwargs)

    async def alist(self, *args: Any, **kwargs: Any) -> AsyncIterator[CheckpointTuple]:
        """Delegate asynchronous checkpoint history."""
        async for saved in self._saver.alist(*args, **kwargs):
            yield saved

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write a checkpoint using the graph-owned saver."""
        with self._track_writes():
            config = self._resolve(config)
            saved_config = self._saver.put(
                config, checkpoint, self._metadata(config, metadata), new_versions
            )
            return self._advance_head(config, checkpoint, saved_config)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write an asynchronous checkpoint using the graph-owned saver."""
        with self._track_writes():
            config = self._resolve(config)
            saved_config = await self._saver.aput(
                config, checkpoint, self._metadata(config, metadata), new_versions
            )
            return self._advance_head(config, checkpoint, saved_config)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve pending writes and optional task paths."""
        with self._track_writes():
            config = self._resolve(config)
            if self._put_writes_accepts_task_path:
                self._saver.put_writes(config, writes, task_id, task_path)
            else:
                self._saver.put_writes(config, writes, task_id)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve asynchronous pending writes and optional task paths."""
        with self._track_writes():
            config = self._resolve(config)
            if self._aput_writes_accepts_task_path:
                await self._saver.aput_writes(config, writes, task_id, task_path)
            else:
                await self._saver.aput_writes(config, writes, task_id)

    def delete_thread(self, thread_id: str) -> None:
        """Delegate thread deletion to the graph-owned saver."""
        self._saver.delete_thread(thread_id)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delegate asynchronous thread deletion to the graph-owned saver."""
        await self._saver.adelete_thread(thread_id)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        """Allocate versions with the graph-owned saver's version scheme."""
        return self._saver.get_next_version(current, channel)
