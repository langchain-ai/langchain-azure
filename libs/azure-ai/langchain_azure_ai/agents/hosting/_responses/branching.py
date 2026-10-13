"""Internal state and strict checkpoint access for response-ID branches."""

from __future__ import annotations

import json
from collections.abc import (
    AsyncIterator,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import replace
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
    PendingWrite,
    copy_checkpoint,
    create_checkpoint,
)
from langgraph.types import Command, Interrupt, PregelTask
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
BRANCH_RESUME_IDS_CONFIG = "__langgraph_response_resume_ids"


def remap_pending_interrupts(
    pending: Sequence[Interrupt], resume_ids: Mapping[str, str]
) -> tuple[Interrupt, ...]:
    """Expose parent interrupt IDs while parsing replies to a paused fork."""
    parent_ids = {fork_id: parent_id for parent_id, fork_id in resume_ids.items()}
    return tuple(
        replace(interrupt, id=parent_ids.get(interrupt.id, interrupt.id))
        for interrupt in pending
    )


def remap_resume_command(command: Command, resume_ids: Mapping[str, str]) -> Command:
    """Translate an ID-keyed resume command to the fork's interrupt IDs."""
    if isinstance(command.resume, dict) and all(
        interrupt_id in resume_ids for interrupt_id in command.resume
    ):
        return replace(
            command,
            resume={
                resume_ids[interrupt_id]: value
                for interrupt_id, value in command.resume.items()
            },
        )
    return command


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


def _remap_pause_writes(
    parent_tasks: Sequence[PregelTask],
    fork_tasks: Sequence[PregelTask],
    pending_writes: Sequence[PendingWrite],
) -> tuple[dict[str, str], dict[str, list[tuple[str, Any]]]]:
    """Rekey paused writes without modifying the parent tasks or values."""
    tasks = {(task.name, task.path): task for task in fork_tasks}
    if len(tasks) != len(fork_tasks) or set(tasks) != {
        (task.name, task.path) for task in parent_tasks
    }:
        raise BranchingError(
            "invalid_branch_state", "The fork checkpoint tasks do not match."
        )
    task_ids = {task.id: tasks[task.name, task.path] for task in parent_tasks}
    resume_ids: dict[str, str] = {}
    writes_by_task: dict[str, list[tuple[str, Any]]] = {}
    for task_id, channel, value in pending_writes:
        task = task_ids.get(task_id)
        if channel == "__interrupt__":
            if (
                task is None
                or not isinstance(value, (list, tuple))
                or not all(isinstance(item, Interrupt) for item in value)
            ):
                raise BranchingError(
                    "invalid_branch_state", "The paused checkpoint is invalid."
                )
            fork_interrupts = []
            for interrupt in value:
                fork_id = Interrupt.from_ns(
                    interrupt.value, f"{task.name}:{task.id}"
                ).id
                resume_ids[interrupt.id] = fork_id
                fork_interrupts.append(replace(interrupt, id=fork_id))
            value = tuple(fork_interrupts)
        writes_by_task.setdefault(task.id if task else task_id, []).append(
            (channel, value)
        )
    if not resume_ids:
        raise BranchingError(
            "invalid_branch_state", "The paused checkpoint has no interrupts."
        )
    return resume_ids, writes_by_task


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
            or record.get("version") not in {"1", "2"}
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

    @classmethod
    def _origin(cls, record: Any) -> tuple[CheckpointRef, dict[str, str]]:
        ref = cls._reference(record)
        if record["version"] == "1":
            return ref, {}
        try:
            resume_ids = json.loads(record.get("resume_ids", ""))
        except (TypeError, ValueError) as exc:
            raise BranchingError(
                "invalid_branch_state", "The response resume mapping is invalid."
            ) from exc
        if (
            not isinstance(resume_ids, dict)
            or not all(
                isinstance(parent_id, str)
                and parent_id
                and isinstance(fork_id, str)
                and fork_id
                for parent_id, fork_id in resume_ids.items()
            )
            or len(set(resume_ids.values())) != len(resume_ids)
            or (record["paused"] == "true" and not resume_ids)
        ):
            raise BranchingError(
                "invalid_branch_state", "The response resume mapping is invalid."
            )
        return ref, resume_ids

    @staticmethod
    async def _fork_pause(
        graph: CompiledStateGraph, ref: CheckpointRef, context: ResponseContext
    ) -> tuple[CheckpointRef, dict[str, str]]:
        checkpointer = graph.checkpointer
        if not isinstance(checkpointer, BaseCheckpointSaver):
            raise BranchingError(
                "checkpoint_unavailable", "The graph checkpoint saver is unavailable."
            )
        saver = ResponseCheckpointSaver(checkpointer, branching=True)
        guarded_graph = graph.copy({"checkpointer": saver})
        config = HostingRunnableConfig.create_from_checkpoint(
            ref, context
        ).runnable_config
        saved = await saver.aget_tuple(config)
        if saved is None:
            raise BranchingError(
                "checkpoint_unavailable", "The parent checkpoint is unavailable."
            )
        parent = await guarded_graph.aget_state(config)
        if any(task.state is not None for task in parent.tasks):
            raise BranchingError(
                "invalid_branch_state", "Paused subgraph branching is not supported."
            )
        checkpoint = create_checkpoint(
            copy_checkpoint(saved.checkpoint), None, saved.metadata.get("step", -1)
        )
        fork_config = await saver.aput(
            config,
            checkpoint,
            {**saved.metadata, "source": "fork"},
            checkpoint["channel_versions"],
        )
        fork_ref = CheckpointRef.from_dict(fork_config.get("configurable"))
        if (
            fork_ref is None
            or fork_ref.thread_id != ref.thread_id
            or fork_ref.checkpoint_id != checkpoint["id"]
            or fork_ref.checkpoint_id == ref.checkpoint_id
            or fork_config["configurable"].get("checkpoint_ns", "") != ""
        ):
            raise BranchingError(
                "checkpoint_unavailable", "The fork checkpoint is unavailable."
            )
        fork_config = (
            HostingRunnableConfig(fork_config)
            .with_response_context(context)
            .runnable_config
        )
        fork = await guarded_graph.aget_state(fork_config)
        resume_ids, writes_by_task = _remap_pause_writes(
            parent.tasks, fork.tasks, saved.pending_writes or ()
        )
        for task_id, writes in writes_by_task.items():
            await saver.aput_writes(fork_config, writes, task_id)
        return fork_ref, resume_ids

    async def prepare(
        self,
        *,
        response_key: str,
        parent_key: str,
        parent_id: str,
        context: ResponseContext,
        graph: CompiledStateGraph | None = None,
    ) -> tuple[CheckpointRef, dict[str, str]]:
        """Confirm a same-thread origin and its parent-to-fork interrupt IDs."""
        existing = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
        if context.is_recovery:
            if await self._store.get(response_key, BRANCH_BOUNDARY_KEY) is not None:
                raise BranchingError(
                    "invalid_branch_state",
                    "A response with a published boundary cannot be resumed.",
                )
            ref, resume_ids = self._origin(existing)
            if (
                existing is None
                or existing.get("mode") != BRANCH_MODE
                or existing.get("parent_response_id") != parent_id
            ):
                raise BranchingError(
                    "invalid_branch_state", "The confirmed branch origin is invalid."
                )
            return ref, resume_ids

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
            indexed_boundary = (
                {key: value for key, value in indexed.items() if key != "resume_ids"}
                if indexed is not None and indexed.get("version") == "2"
                else indexed
            )
            if indexed_boundary != boundary:
                raise BranchingError(
                    "invalid_branch_state",
                    "The parent checkpoint records do not agree.",
                )
            if indexed is not None:
                boundary = indexed
        else:
            if BRANCH_MODE_METADATA in metadata:
                raise BranchingError(
                    "checkpoint_unavailable", "The parent has no completed checkpoint."
                )
            raise BranchingError(
                "invalid_branch_state",
                "The parent response was not created with response branching.",
            )

        if existing is not None:
            if (
                existing.get("mode") != BRANCH_MODE
                or existing.get("parent_response_id") != parent_id
                or existing.get("paused") != boundary["paused"]
            ):
                raise BranchingError(
                    "invalid_branch_state", "The confirmed branch origin is invalid."
                )
            return self._origin(existing)

        resume_ids = {}
        if boundary["paused"] == "true":
            if graph is None:
                raise BranchingError(
                    "checkpoint_unavailable",
                    "The graph is unavailable for paused branching.",
                )
            ref, resume_ids = await self._fork_pause(graph, ref, context)
            _, parent_resume_ids = self._origin(boundary)
            parent_ids = {
                internal_id: wire_id
                for wire_id, internal_id in parent_resume_ids.items()
            }
            resume_ids = {
                parent_ids.get(parent_id, parent_id): fork_id
                for parent_id, fork_id in resume_ids.items()
            }
            boundary = {**boundary, **ref.to_dict()}

        origin = {
            **boundary,
            "version": "2",
            "mode": BRANCH_MODE,
            "parent_response_id": parent_id,
            "resume_ids": json.dumps(resume_ids),
        }
        self._origin(origin)
        await self._store.set(response_key, BRANCH_ORIGIN_KEY, origin)
        return ref, resume_ids

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
            origin_ref, resume_ids = self._origin(origin)
            if origin is None or origin_ref != ref or origin["paused"] != "true":
                raise BranchingError(
                    "invalid_branch_state", "The pending pause origin is invalid."
                )
        record = self._record(ref, paused=paused)
        if continue_pause and resume_ids:
            record.update(version="2", resume_ids=json.dumps(resume_ids))
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
            if key not in {"thread_id", "checkpoint_id", "resume_ids"}
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
