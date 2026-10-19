"""Internal state and strict checkpoint access for response-ID branches."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import (
    AsyncIterator,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from inspect import signature
from threading import Lock
from typing import TYPE_CHECKING, Any

from azure.ai.agentserver.responses import (
    ResponseContext,
    ResponseEventStream,
    ResponseProviderProtocol,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
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

from .._converters._hitl import (
    _interrupt_id_from_approval_id,
    _is_interrupt_approval_request,
    interrupt_arguments_json,
)
from .checkpoint_ref import CheckpointRef
from .conversation_chain_store import ConversationChainStoreProtocol
from .functional_tasks import (
    FunctionalTaskError,
    FunctionalTaskRecorder,
    TaskFork,
    fork_functional_tasks,
)
from .hosting_runnable_config import HostingRunnableConfig
from .task_storage_manager import TaskStorageManager

if TYPE_CHECKING:
    from azure.ai.agentserver.responses.models._generated import (
        OutputItemMcpApprovalRequest,
    )
    from langgraph.graph.state import CompiledStateGraph

BRANCH_MODE_HEADER = "x-client-langchain-response-branching"
BRANCH_MODE = "checkpoint-v1"
BRANCH_ORIGIN_KEY = "langgraph_branch_origin_v1"
BRANCH_BOUNDARY_KEY = "langgraph_response_boundary_v1"
BRANCH_MODE_METADATA = "langgraph_response_branching"
BRANCH_RESUME_IDS_CONFIG = "__langgraph_response_resume_ids"


class _ResponseCheckpointMetadata(CheckpointMetadata):
    langgraph_response_execution: str


def _checkpoint_execution(response_key: str, origin: CheckpointRef | None) -> str:
    return json.dumps(
        [response_key, origin.thread_id, origin.checkpoint_id]
        if origin is not None
        else [response_key, None, None]
    )


def remap_pending_interrupts(
    pending: Sequence[Interrupt], resume_ids: Mapping[str, str]
) -> tuple[Interrupt, ...]:
    """Expose parent interrupt IDs while parsing replies to a paused fork."""
    parent_ids = {fork_id: parent_id for parent_id, fork_id in resume_ids.items()}
    return tuple(
        replace(interrupt, id=parent_ids.get(interrupt.id, interrupt.id))
        for interrupt in pending
    )


def remap_resume_command(
    command: Command,
    resume_ids: Mapping[str, str],
    *,
    pending: Sequence[Interrupt] = (),
) -> Command:
    """Bind replies to fork interrupt IDs rather than global resume writes.

    Args:
        command: The parsed resume command using wire IDs.
        resume_ids: The selected parent's wire-to-internal ID mapping.
        pending: Wire interrupts for single-value normalization. Defaults to
            leaving single-value resumes unchanged.

    Returns:
        A command whose resume map addresses the fork's interrupts explicitly.
        Known wire IDs are remapped independently; other IDs remain unchanged.
    """
    # LangGraph reserves dictionaries keyed by 128-bit interrupt IDs as maps.
    id_keyed = isinstance(command.resume, dict) and all(
        isinstance(interrupt_id, str)
        and (interrupt_id in resume_ids or re.fullmatch(r"[0-9a-f]{32}", interrupt_id))
        for interrupt_id in command.resume
    )
    if command.resume is not None and not id_keyed and len(pending) == 1:
        command = replace(command, resume={pending[0].id: command.resume})
        id_keyed = True
    if isinstance(command.resume, dict) and id_keyed:
        return replace(
            command,
            resume={
                resume_ids.get(interrupt_id, interrupt_id): value
                for interrupt_id, value in command.resume.items()
            },
        )
    return command


def remap_unanswered_interrupts(
    pending: Sequence[Interrupt],
    resume_ids: Mapping[str, str],
    approval_requests: Mapping[str, OutputItemMcpApprovalRequest],
) -> tuple[tuple[Interrupt, ...], dict[str, str]]:
    """Retain wire identities only for unchanged, outstanding approvals.

    Args:
        pending: Interrupts still pending after this execution.
        resume_ids: The selected parent's wire-to-internal ID mapping.
        approval_requests: Original outstanding approval items by wire ID.

    Returns:
        Wire interrupts to emit and their retained wire-to-internal mapping.
    """
    wire_pending = []
    retained_ids = {}
    for internal, wire in zip(
        pending, remap_pending_interrupts(pending, resume_ids), strict=True
    ):
        approval = approval_requests.get(wire.id)
        if approval is not None and approval["arguments"] == interrupt_arguments_json(
            wire
        ):
            wire_pending.append(wire)
            retained_ids[wire.id] = internal.id
        else:
            wire_pending.append(internal)
    return tuple(wire_pending), retained_ids


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


def _pause_task_forks(
    parent_tasks: Sequence[PregelTask],
    fork_tasks: Sequence[PregelTask],
) -> dict[str, TaskFork]:
    from langgraph.pregel._algo import task_path_str

    tasks = {(task.name, task.path): task for task in fork_tasks}
    if len(tasks) != len(fork_tasks) or set(tasks) != {
        (task.name, task.path) for task in parent_tasks
    }:
        raise BranchingError(
            "invalid_branch_state", "The fork checkpoint tasks do not match."
        )
    return {
        task.id: TaskFork(
            tasks[task.name, task.path].id,
            f"{task.name}:{task.id}",
            f"{task.name}:{tasks[task.name, task.path].id}",
            task_path_str(tasks[task.name, task.path].path),
        )
        for task in parent_tasks
    }


def _remap_pause_writes(
    parent_tasks: Sequence[PregelTask],
    fork_tasks: Sequence[PregelTask],
    pending_writes: Sequence[PendingWrite],
    *,
    functional_tasks: Mapping[str, TaskFork] | None = None,
) -> tuple[dict[str, str], dict[str, list[tuple[str, Any]]]]:
    """Rekey paused writes without modifying the parent tasks or values."""
    task_forks = _pause_task_forks(parent_tasks, fork_tasks)
    task_forks.update(functional_tasks or {})
    interrupt_tasks = {
        Interrupt.from_ns(None, task.source_namespace).id: task
        for task in task_forks.values()
    }
    interrupt_ids = {
        interrupt_id: Interrupt.from_ns(None, task.namespace).id
        for interrupt_id, task in interrupt_tasks.items()
    }
    resume_ids: dict[str, str] = {}
    writes_by_task: dict[str, list[tuple[str, Any]]] = {}
    for task_id, channel, value in pending_writes:
        task = task_forks.get(task_id)
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
                interrupted = interrupt_tasks.get(interrupt.id)
                if interrupted is None or not (
                    interrupted.source_namespace == task.source_namespace
                    or interrupted.source_namespace.startswith(
                        f"{task.source_namespace}|"
                    )
                ):
                    raise BranchingError(
                        "invalid_branch_state", "The paused checkpoint is invalid."
                    )
                fork_id = interrupt_ids[interrupt.id]
                resume_ids[interrupt.id] = fork_id
                fork_interrupts.append(replace(interrupt, id=fork_id))
            value = tuple(fork_interrupts)
        elif (
            channel == "__resume__"
            and task_id in {"", "00000000-0000-0000-0000-000000000000"}
            and isinstance(value, dict)
        ):
            value = remap_resume_command(Command(resume=value), interrupt_ids).resume
        writes_by_task.setdefault(task.task_id if task else task_id, []).append(
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

    async def approval_requests(
        self, response_key: str
    ) -> dict[str, OutputItemMcpApprovalRequest]:
        """Load the original approval payloads for this response's pause.

        Args:
            response_key: The response's user-scoped branch-store key.

        Returns:
            Original approval items keyed by wire interrupt ID. Older origins
            without approval snapshots return an empty mapping.

        Raises:
            BranchingError: A saved approval snapshot is malformed.
        """
        origin = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
        if origin is None:
            return {}
        _, resume_ids = self._origin(origin)
        try:
            snapshots = json.loads(origin.get("approval_requests", "{}"))
        except (TypeError, ValueError) as exc:
            raise BranchingError(
                "invalid_branch_state", "The response approval snapshots are invalid."
            ) from exc
        if not isinstance(snapshots, dict):
            raise BranchingError(
                "invalid_branch_state", "The response approval snapshots are invalid."
            )
        approvals = {}
        for wire_id, item in snapshots.items():
            if (
                not isinstance(wire_id, str)
                or wire_id not in resume_ids
                or not _is_interrupt_approval_request(item)
                or _interrupt_id_from_approval_id(item["id"], (wire_id,)) != wire_id
            ):
                raise BranchingError(
                    "invalid_branch_state",
                    "The response approval snapshots are invalid.",
                )
            try:
                arguments = json.loads(item["arguments"])
            except (TypeError, ValueError) as exc:
                raise BranchingError(
                    "invalid_branch_state",
                    "The response approval snapshots are invalid.",
                ) from exc
            if (
                not isinstance(arguments, dict)
                or arguments.get("interrupt_id") != wire_id
                or "value" not in arguments
            ):
                raise BranchingError(
                    "invalid_branch_state",
                    "The response approval snapshots are invalid.",
                )
            approvals[wire_id] = item
        return approvals

    async def record_resume(
        self,
        response_key: str,
        command: Command,
        pending: Sequence[Interrupt],
        approval_requests: Mapping[str, OutputItemMcpApprovalRequest],
    ) -> tuple[Command, dict[str, OutputItemMcpApprovalRequest]]:
        """Bind a resume and persist which original approvals it answers.

        Args:
            response_key: The response's user-scoped branch-store key.
            command: The command using the parent's wire interrupt IDs.
            pending: The parent's pending wire interrupts.
            approval_requests: The original outstanding approval items.

        Returns:
            The wire-ID-bound command and its unanswered approval items.
            An update-only command leaves the items unchanged.

        Raises:
            BranchingError: The confirmed origin is missing or invalid.
        """
        if command.resume is None:
            return command, dict(approval_requests)
        origin = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
        if origin is None:
            raise BranchingError(
                "invalid_branch_state", "The confirmed branch origin is missing."
            )
        _, resume_ids = self._origin(origin)
        command = remap_resume_command(
            command,
            {wire_id: wire_id for wire_id in resume_ids},
            pending=pending,
        )
        resumed_ids = set(command.resume) if isinstance(command.resume, dict) else set()
        remaining = {
            wire_id: item
            for wire_id, item in approval_requests.items()
            if wire_id not in resumed_ids
        }
        if len(remaining) != len(approval_requests):
            # Recovery must not give a later interrupt the answered site's ID.
            await self._store.set(
                response_key,
                BRANCH_ORIGIN_KEY,
                {**origin, "approval_requests": json.dumps(remaining)},
            )
        return command, remaining

    @staticmethod
    async def _fork_pause(
        graph: CompiledStateGraph,
        ref: CheckpointRef,
        context: ResponseContext,
        response_key: str,
        config: RunnableConfig | None = None,
    ) -> tuple[CheckpointRef, dict[str, str]]:
        checkpointer = graph.checkpointer
        if not isinstance(checkpointer, BaseCheckpointSaver):
            raise BranchingError(
                "checkpoint_unavailable", "The graph checkpoint saver is unavailable."
            )
        saver = ResponseCheckpointSaver(checkpointer, branching=True)
        guarded_graph = graph.copy({"checkpointer": saver})
        config = (
            HostingRunnableConfig(config or {})
            .with_checkpoint_ref(ref)
            .with_response_context(context)
            .runnable_config
        )
        # State reads merge bound config themselves; direct saver I/O does not.
        saver_config = merge_configs(graph.config, config)
        saved = await saver.aget_tuple(saver_config)
        if saved is None:
            raise BranchingError(
                "checkpoint_unavailable", "The parent checkpoint is unavailable."
            )
        parent = await guarded_graph.aget_state(config)
        # Completed subgraphs retain state references in historical snapshots.
        if any(task.state is not None and task.result is None for task in parent.tasks):
            raise BranchingError(
                "invalid_branch_state", "Paused subgraph branching is not supported."
            )
        checkpoint = create_checkpoint(
            copy_checkpoint(saved.checkpoint), None, saved.metadata.get("step", -1)
        )
        metadata: _ResponseCheckpointMetadata = {
            **saved.metadata,
            "source": "fork",
            "langgraph_response_execution": _checkpoint_execution(
                response_key, CheckpointRef(ref.thread_id, checkpoint["id"])
            ),
        }
        parent_configurable = (saved.parent_config or {}).get("configurable", {})
        # Copied pending writes must not also appear in the fork's ancestor history.
        fork_config = await saver.aput(
            {
                **saver_config,
                "configurable": {
                    **saver_config["configurable"],
                    "checkpoint_id": parent_configurable.get("checkpoint_id"),
                },
            },
            checkpoint,
            metadata,
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
            HostingRunnableConfig(config).with_checkpoint_ref(fork_ref).runnable_config
        )
        fork = await guarded_graph.aget_state(fork_config)
        fork_saver_config = merge_configs(graph.config, fork_config)
        root_tasks = _pause_task_forks(parent.tasks, fork.tasks)
        try:
            functional_tasks = await fork_functional_tasks(
                checkpointer,
                saved._replace(config=saver_config),
                fork_saver_config,
                checkpoint,
                {task_id: task.task_id for task_id, task in root_tasks.items()},
            )
        except FunctionalTaskError as exc:
            raise BranchingError("invalid_branch_state", str(exc)) from exc
        resume_ids, writes_by_task = _remap_pause_writes(
            parent.tasks,
            fork.tasks,
            saved.pending_writes or (),
            functional_tasks=functional_tasks,
        )
        task_paths = {
            task.task_id: task.task_path
            for task in {**root_tasks, **functional_tasks}.values()
        }
        for task_id, writes in writes_by_task.items():
            await saver.aput_writes(
                fork_saver_config, writes, task_id, task_paths.get(task_id, "")
            )
        return fork_ref, resume_ids

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
    ) -> tuple[CheckpointRef, dict[str, str]]:
        """Select a parent or confirm an origin after config hooks finish."""
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

        if select_only:
            return ref, {}

        resume_ids = {}
        if boundary["paused"] == "true":
            if graph is None:
                raise BranchingError(
                    "checkpoint_unavailable",
                    "The graph is unavailable for paused branching.",
                )
            ref, resume_ids = await self._fork_pause(
                graph, ref, context, response_key, config=config
            )
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
        approvals = {}
        for item in parent.get("output") or ():
            if _is_interrupt_approval_request(item):
                try:
                    arguments = json.loads(item["arguments"])
                except (TypeError, ValueError) as exc:
                    raise BranchingError(
                        "invalid_branch_state", "The parent approval is invalid."
                    ) from exc
                wire_id = (
                    arguments.get("interrupt_id")
                    if isinstance(arguments, dict)
                    else None
                )
                if isinstance(wire_id, str) and wire_id in resume_ids:
                    approvals[wire_id] = item
        if approvals:
            origin["approval_requests"] = json.dumps(approvals)
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
        resume_ids: Mapping[str, str] | None = None,
    ) -> None:
        """Index this run's checkpoint before the SDK commits its terminal event."""
        if ref is None:
            raise BranchingError(
                "checkpoint_unavailable",
                "The response produced no checkpoint boundary.",
            )
        if continue_pause:
            origin = await self._store.get(response_key, BRANCH_ORIGIN_KEY)
            origin_ref, origin_resume_ids = self._origin(origin)
            if origin is None or origin_ref != ref or origin["paused"] != "true":
                raise BranchingError(
                    "invalid_branch_state", "The pending pause origin is invalid."
                )
            if resume_ids is None:
                resume_ids = origin_resume_ids
        record = self._record(ref, paused=paused)
        if resume_ids:
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
    """Enforce exact reads and record root checkpoint execution ownership.

    Storage remains in the graph-owned saver. Paused resumes and task recovery
    can use a request-local root head so completed task writes are reused without
    reading another branch's latest root checkpoint. This adapter does not
    select private subgraph checkpoints: unpinned subgraph reads retain
    LangGraph's behavior, and subgraph state isolation is application-owned.
    Explicit checkpoint references are validated in any namespace. History is
    delegated unchanged.

    Args:
        saver: The graph-owned saver. Ownership and lifecycle stay with its
            caller; this request-scoped adapter does not open or close it.
        branching: Whether this request uses response branching. Branching
            rejects unavailable or mismatched explicit checkpoints; otherwise
            reads preserve the wrapped saver's behavior. Both modes propagate
            backend errors.
        response_key: The current response's user-scoped key, or ``None`` to
            preserve checkpoint metadata unchanged. Defaults to ``None``.
        origin: The current response's confirmed starting checkpoint, or
            ``None`` for a new root execution. Defaults to ``None``.

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
        self._put_writes_accepts_task_path = (
            "task_path" in signature(saver.put_writes).parameters
        )
        self._aput_writes_accepts_task_path = (
            "task_path" in signature(saver.aput_writes).parameters
        )
        self._branching = branching
        self._execution = (
            _checkpoint_execution(response_key, origin)
            if branching and response_key is not None
            else None
        )
        self.task_recorder = FunctionalTaskRecorder()
        self._resume_head: CheckpointRef | None = None
        self._checkpoint_writes: dict[CheckpointRef, Future[None]] = {}
        self._write_error: BaseException | None = None
        self._write_lock = Lock()

    @property
    def config_specs(self) -> Any:
        """Preserve the wrapped saver's configurable fields."""
        return self._saver.config_specs

    async def wait_for_checkpoint(self, ref: CheckpointRef) -> None:
        """Wait for a root checkpoint before publishing recovery progress.

        Args:
            ref: The exact root checkpoint announced by the graph stream.

        Raises:
            BaseException: A checkpoint or pending-write operation failed.
        """
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
        self.task_recorder.record_checkpoint(config, checkpoint)
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
        """Require a recovery checkpoint owned by this response and origin.

        Args:
            config: The exact root checkpoint recorded as current progress.

        Raises:
            BranchingError: The checkpoint is unavailable or its execution
                ownership cannot be verified, including older unmarked progress.
        """
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
        """Resume a branch without replaying its completed tasks.

        Reads without a checkpoint ID stay pinned to this request's head,
        which advances only when this saver writes a root checkpoint.

        Args:
            config: The exact root checkpoint selected for this branch.

        Returns:
            A copied execution config without the replay-selecting checkpoint ID.

        Raises:
            BranchingError: If branching is disabled or the reference is invalid.
        """
        ref = HostingRunnableConfig(config).checkpoint_ref
        configurable = dict(config.get("configurable") or {})
        if (
            not self._branching
            or ref is None
            or configurable.get("checkpoint_ns", "") != ""
        ):
            raise BranchingError(
                "invalid_branch_state", "The branch resume checkpoint is invalid."
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
                "invalid_branch_state", "The branch resume thread does not match."
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
                    "The branch resume checkpoint was not saved.",
                )
            self._resume_head = ref
        self._record_checkpoint(config, checkpoint)
        return saved_config

    def _validate(
        self, config: RunnableConfig, saved: CheckpointTuple | None
    ) -> CheckpointTuple | None:
        requested = config.get("configurable") or {}
        checkpoint_id = requested.get("checkpoint_id")
        if not self._branching:
            return saved
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
        # Carry execution options without substituting the current checkpoint's ID.
        execution_options = {
            key: value
            for key, value in requested.items()
            if key not in {"thread_id", "checkpoint_ns", "checkpoint_id"}
        }
        return saved._replace(
            config={
                **saved.config,
                "configurable": {**execution_options, **actual},
            },
            parent_config=(
                {
                    **saved.parent_config,
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
        """Read a checkpoint and validate an explicit branch selection."""
        config = self._resolve(config)
        saved = self._validate(config, self._saver.get_tuple(config))
        if saved is not None:
            self._record_checkpoint(config, saved.checkpoint)
        return saved

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read an async checkpoint and validate an explicit branch selection."""
        config = self._resolve(config)
        saved = self._validate(config, await self._saver.aget_tuple(config))
        if saved is not None:
            self._record_checkpoint(config, saved.checkpoint)
        return saved

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
        """Write an asynchronous checkpoint using the original saver."""
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
        """Preserve the saver's pending-write semantics."""
        with self._track_writes():
            config = self._resolve(config)
            if self._execution is not None:
                self.task_recorder.put(self._saver, config, task_id)
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
        """Preserve the saver's asynchronous pending-write semantics."""
        with self._track_writes():
            config = self._resolve(config)
            if self._execution is not None:
                await self.task_recorder.aput(self._saver, config, task_id)
            if self._aput_writes_accepts_task_path:
                await self._saver.aput_writes(config, writes, task_id, task_path)
            else:
                await self._saver.aput_writes(config, writes, task_id)

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
