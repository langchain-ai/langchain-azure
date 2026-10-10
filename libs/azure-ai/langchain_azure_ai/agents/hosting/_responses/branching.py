"""Internal state and strict checkpoint access for response-ID branches."""

from __future__ import annotations

import json
from collections.abc import (
    AsyncIterator,
    Iterator,
    Mapping,
    Sequence,
)
from typing import Any

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
    copy_checkpoint,
)
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .checkpoint_ref import CheckpointRef
from .conversation_chain_store import ConversationChainStoreProtocol
from .task_storage_manager import TaskStorageManager

BRANCH_MODE_HEADER = "x-client-langchain-response-branching"
BRANCH_MODE = "checkpoint-v1"
BRANCH_ORIGIN_KEY = "langgraph_branch_origin_v1"
BRANCH_BOUNDARY_KEY = "langgraph_response_boundary_v1"
BRANCH_MODE_METADATA = "langgraph_response_branching"


class BranchingAdmissionMiddleware:
    """Validate opt-in linkage and stamp trusted mode into persisted headers.

    Args:
        app: The next ASGI application.
        enabled: Whether fresh requests may use response branching. Incoming
            client mode values are removed even when the feature is disabled,
            but disabled requests retain the SDK's existing validation.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool,
    ) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject invalid opt-in requests before SDK admission or execution."""
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
                request = Request(scope, receive)
                body = await request.body()
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
                        return {"type": "http.request", "body": body}
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


class BranchingError(ValueError):
    """A branch failure with an internal reason code and client-safe message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


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

    async def prepare(
        self,
        *,
        response_key: str,
        parent_key: str,
        parent_id: str,
        context: ResponseContext,
        saver: BaseCheckpointSaver[Any] | None = None,
    ) -> CheckpointRef:
        """Confirm the origin, isolating paused pending writes with the saver."""
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
            ):
                raise BranchingError(
                    "invalid_branch_state", "The confirmed branch origin is invalid."
                )
            return ref

        provider = self._provider
        if provider is None:
            raise BranchingError(
                "invalid_branch_state", "The Responses provider is unavailable."
            )
        parent = await provider.get_response(
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
        if boundary is not None:
            if (
                isinstance(boundary, dict)
                and "thread_id" not in boundary
                and "checkpoint_id" not in boundary
            ):
                captured_ref = TaskStorageManager(dict(metadata)).checkpoint_ref
                if captured_ref is None:
                    raise BranchingError(
                        "checkpoint_unavailable",
                        "The parent has no completed checkpoint.",
                    )
                boundary = {**boundary, **captured_ref.to_dict()}
            ref = self._reference(boundary)
            indexed = await self._store.get(parent_key, BRANCH_BOUNDARY_KEY)
            if indexed != boundary:
                raise BranchingError(
                    "invalid_branch_state",
                    "The parent checkpoint records do not agree.",
                )
        else:
            if BRANCH_MODE_METADATA in metadata:
                raise BranchingError(
                    "checkpoint_unavailable", "The parent has no completed checkpoint."
                )
            raise BranchingError(
                "invalid_branch_state",
                "The parent response was not created with response branching.",
            )

        if boundary["paused"] == "true":
            if saver is None:
                raise BranchingError(
                    "checkpoint_unavailable",
                    "The graph checkpoint saver is unavailable.",
                )
            saved = await ResponseCheckpointSaver(saver, branching=True).aget_tuple(
                {"configurable": {**ref.to_dict(), "checkpoint_ns": ""}}
            )
            if saved is None:
                raise BranchingError(
                    "checkpoint_unavailable", "The parent checkpoint is unavailable."
                )
            copied_config = await saver.aput(
                {"configurable": {"thread_id": response_key, "checkpoint_ns": ""}},
                copy_checkpoint(saved.checkpoint),
                {**saved.metadata, "source": "fork"},
                saved.checkpoint["channel_versions"],
            )
            writes_by_task: dict[str, list[tuple[str, Any]]] = {}
            for task_id, channel, value in saved.pending_writes or []:
                writes_by_task.setdefault(task_id, []).append((channel, value))
            for task_id, writes in writes_by_task.items():
                await saver.aput_writes(copied_config, writes, task_id)
            copied_ref = CheckpointRef.from_dict(copied_config["configurable"])
            if copied_ref is None:
                raise BranchingError(
                    "checkpoint_unavailable", "The copied checkpoint is unavailable."
                )
            ref = copied_ref
            boundary = {**boundary, **ref.to_dict()}

        origin = {
            **boundary,
            "mode": BRANCH_MODE,
            "parent_response_id": parent_id,
        }
        if existing is not None and existing != origin:
            raise BranchingError(
                "invalid_branch_state", "The confirmed branch origin cannot be changed."
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
        continue_pause: bool = False,
    ) -> None:
        """Index this run's checkpoint before the SDK commits its terminal event."""
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable",
                "The response produced no checkpoint boundary.",
            )
        if continue_pause:
            origin = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
            origin_ref = self._reference(origin)
            if origin is None or origin_ref != ref or origin["paused"] != "true":
                raise BranchingError(
                    "invalid_branch_state", "The pending pause origin is invalid."
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
    """Enforce exact checkpoint reads without changing graph state or metadata.

    Checkpoint writes, pending writes, and history are delegated unchanged to
    the graph-owned saver.

    Args:
        saver: The graph-owned saver. Ownership and lifecycle stay with its
            caller; this request-scoped adapter does not open or close it.
        branching: Whether this request uses response branching. Branching
            rejects unavailable or mismatched explicit checkpoints; otherwise
            reads preserve the wrapped saver's behavior. Both modes propagate
            backend errors.

    """

    def __init__(self, saver: BaseCheckpointSaver[Any], *, branching: bool) -> None:
        super().__init__(serde=saver.serde)
        self._saver = saver
        self._branching = branching

    @property
    def config_specs(self) -> Any:
        """Preserve the wrapped saver's configurable fields."""
        return self._saver.config_specs

    def _validate(
        self, config: RunnableConfig, saved: CheckpointTuple | None
    ) -> CheckpointTuple | None:
        requested = config.get("configurable") or {}
        checkpoint_id = requested.get("checkpoint_id")
        if not self._branching or not checkpoint_id:
            return saved
        if saved is None:
            raise BranchingError(
                "checkpoint_unavailable", "The required checkpoint is unavailable."
            )
        actual = saved.config.get("configurable") or {}
        if (
            actual.get("thread_id") != requested.get("thread_id")
            or actual.get("checkpoint_ns", "") != requested.get("checkpoint_ns", "")
            or actual.get("checkpoint_id") != checkpoint_id
            or saved.checkpoint.get("id") != checkpoint_id
        ):
            raise BranchingError(
                "checkpoint_unavailable",
                "The saver did not return the required checkpoint.",
            )
        return saved

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read a checkpoint and validate an explicit branch selection."""
        return self._validate(config, self._saver.get_tuple(config))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read an async checkpoint and validate an explicit branch selection."""
        return self._validate(config, await self._saver.aget_tuple(config))

    def list(self, *args: Any, **kwargs: Any) -> Iterator[CheckpointTuple]:
        """Delegate checkpoint history without changing retention."""
        return self._saver.list(*args, **kwargs)

    async def alist(self, *args: Any, **kwargs: Any) -> AsyncIterator[CheckpointTuple]:
        """Delegate asynchronous checkpoint history without changing retention."""
        async for saved in self._saver.alist(*args, **kwargs):
            yield saved

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write a checkpoint using the original saver."""
        return self._saver.put(config, checkpoint, metadata, new_versions)

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Write an asynchronous checkpoint using the original saver."""
        return await self._saver.aput(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve the saver's pending-write semantics."""
        self._saver.put_writes(config, writes, task_id, task_path)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve the saver's asynchronous pending-write semantics."""
        await self._saver.aput_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        """Delete a thread's checkpoints and writes using the original saver.

        Args:
            thread_id: Thread whose checkpoints and pending writes are deleted.
        """
        self._saver.delete_thread(thread_id)

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete a thread's checkpoints and writes using the async saver method.

        Args:
            thread_id: Thread whose checkpoints and pending writes are deleted.
        """
        await self._saver.adelete_thread(thread_id)

    def get_next_version(self, current: Any, channel: Any) -> Any:
        """Allocate versions with the original saver's version scheme."""
        return self._saver.get_next_version(current, channel)
