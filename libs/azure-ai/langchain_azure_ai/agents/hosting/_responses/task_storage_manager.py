# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Per-response LangGraph checkpoint storage management."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from inspect import signature
from typing import Any

from azure.ai.agentserver.responses import ResponseContext, ResponseEventStream
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

from .checkpoint_ref import CheckpointRef
from .hosting_runnable_config import HostingRunnableConfig

METADATA_LANGGRAPH_CHECKPOINT_ID = "langgraph_checkpoint_id"
METADATA_LANGGRAPH_THREAD_ID = "langgraph_thread_id"
METADATA_LANGGRAPH_ROOT_RESPONSE_ID = "langgraph_root_response_id"


class CheckpointError(ValueError):
    """A checkpoint failure with an internal code and client-safe message."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class _ResponseCheckpointMetadata(CheckpointMetadata):
    langgraph_response_execution: str


class TaskStorageManager:
    """Manage LangGraph references persisted with one Responses task."""

    def __init__(self, metadata: MutableMapping[str, Any] | None) -> None:
        self._metadata = metadata

    @classmethod
    def from_stream(cls, stream: ResponseEventStream) -> TaskStorageManager:
        """Manage the writable metadata attached to an active response stream."""
        return cls(stream.internal_metadata)

    @classmethod
    def from_response(cls, response: Mapping[str, Any]) -> TaskStorageManager:
        """Read checkpoint metadata from a stored SDK response."""
        metadata = (response.get("metadata") or {}).get("_internal_metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise CheckpointError(
                    "invalid_checkpoint_state",
                    "The response checkpoint metadata is invalid.",
                ) from exc
        if not isinstance(metadata, Mapping):
            raise CheckpointError(
                "invalid_checkpoint_state",
                "The response checkpoint metadata is invalid.",
            )
        return cls(dict(metadata))

    @property
    def checkpoint_ref(self) -> CheckpointRef | None:
        """Return the reference persisted with this Responses task."""
        if self._metadata is None:
            return None
        thread_id = self._metadata.get(METADATA_LANGGRAPH_THREAD_ID)
        if not isinstance(thread_id, str) or not thread_id:
            return None
        checkpoint_id = self._metadata.get(METADATA_LANGGRAPH_CHECKPOINT_ID)
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            return None
        return CheckpointRef(thread_id, checkpoint_id)

    def store_checkpoint_ref(self, checkpoint_ref: CheckpointRef) -> None:
        """Store a LangGraph reference on the active response."""
        if self._metadata is None:
            return
        self._metadata[METADATA_LANGGRAPH_THREAD_ID] = checkpoint_ref.thread_id
        self._metadata[METADATA_LANGGRAPH_CHECKPOINT_ID] = checkpoint_ref.checkpoint_id


def get_recovery_checkpoint(context: ResponseContext) -> CheckpointRef | None:
    """Read saved progress, rejecting incomplete references instead of replaying."""
    if not context.is_recovery or context.persisted_response is None:
        return None
    storage = TaskStorageManager.from_response(context.persisted_response)
    ref = storage.checkpoint_ref
    if ref is None and any(
        name in (storage._metadata or {})
        for name in (METADATA_LANGGRAPH_THREAD_ID, METADATA_LANGGRAPH_CHECKPOINT_ID)
    ):
        raise CheckpointError(
            "invalid_checkpoint_state", "The recorded execution checkpoint is invalid."
        )
    return ref


class ResponseCheckpointSaver(BaseCheckpointSaver[Any]):
    """Adapt asynchronous hosting reads, recovery, and progress publication.

    The application's saver remains responsible for storage and lifecycle.
    Root recovery and progress confirmation are async-only. Synchronous calls
    inherited by private subgraphs pass through without touching root progress.
    This is not a second general-purpose saver.

    Args:
        saver: The application's graph checkpoint saver.
        response_key: User-scoped identity of the executing response.
        parent_response_id: Requested parent response, or ``None`` for a root.
    """

    def __init__(
        self,
        saver: BaseCheckpointSaver[Any],
        *,
        response_key: str,
        parent_response_id: str | None = None,
    ) -> None:
        super().__init__(serde=saver.serde)
        self._saver = saver
        self._execution = json.dumps([response_key, parent_response_id])
        self._head: CheckpointRef | None = None
        self._require_owner = False
        self._persisted: set[CheckpointRef] = set()
        self._writes_changed = asyncio.Event()
        self._write_error: BaseException | None = None
        self._accepts_task_path = "task_path" in signature(saver.aput_writes).parameters
        self._accepts_sync_task_path = (
            "task_path" in signature(saver.put_writes).parameters
        )

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Preserve the application saver's configurable fields."""
        return self._saver.config_specs

    async def wait_for_checkpoint(self, ref: CheckpointRef) -> None:
        """Confirm graph persistence before publishing a task reference."""
        while ref not in self._persisted:
            if self._write_error is not None:
                raise self._write_error
            self._writes_changed.clear()
            await self._writes_changed.wait()

    def _record_checkpoint(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> None:
        configurable = config.get("configurable") or {}
        if configurable.get("checkpoint_ns", "") == "":
            self._persisted.add(
                CheckpointRef(configurable["thread_id"], checkpoint["id"])
            )
            self._writes_changed.set()

    @contextmanager
    def _track_writes(self) -> Iterator[None]:
        try:
            yield
        except BaseException as exc:
            self._write_error = exc
            self._writes_changed.set()
            raise

    def resume_config(
        self, config: RunnableConfig, *, require_owner: bool = False
    ) -> RunnableConfig:
        """Pin native resume without marking the graph execution as replay."""
        ref = HostingRunnableConfig(config).checkpoint_ref
        configurable = dict(config.get("configurable") or {})
        if ref is None or configurable.get("checkpoint_ns", "") != "":
            raise CheckpointError(
                "invalid_checkpoint_state", "The recovery checkpoint is invalid."
            )
        self._head = ref
        self._require_owner = require_owner
        del configurable["checkpoint_id"]
        return {**config, "configurable": configurable}

    def _resolve(self, config: RunnableConfig) -> RunnableConfig:
        requested = config.get("configurable") or {}
        if self._head is None or requested.get("checkpoint_ns", "") != "":
            return config
        if requested.get("thread_id") != self._head.thread_id:
            raise CheckpointError(
                "invalid_checkpoint_state", "The recovery thread does not match."
            )
        if requested.get("checkpoint_id"):
            return config
        return (
            HostingRunnableConfig(config)
            .with_checkpoint_ref(self._head)
            .runnable_config
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read the exact checkpoint and validate recovery ownership in that read."""
        config = self._resolve(config)
        requested = config.get("configurable") or {}
        saved = await self._saver.aget_tuple(config)
        checkpoint_id = requested.get("checkpoint_id")
        actual = (saved.config.get("configurable") or {}) if saved is not None else {}
        if checkpoint_id and (
            saved is None
            or actual.get("thread_id") != requested.get("thread_id")
            or actual.get("checkpoint_ns", "") != requested.get("checkpoint_ns", "")
            or actual.get("checkpoint_id") != checkpoint_id
            or saved.checkpoint["id"] != checkpoint_id
        ):
            raise CheckpointError(
                "checkpoint_unavailable", "The required checkpoint is unavailable."
            )
        if saved is None:
            return None
        if (
            self._head is not None
            and HostingRunnableConfig(config).checkpoint_ref == self._head
            and requested.get("checkpoint_ns", "") == ""
        ):
            owner = saved.metadata.get("langgraph_response_execution")
            if (owner is not None or self._require_owner) and owner != self._execution:
                raise CheckpointError(
                    "invalid_checkpoint_state",
                    "The recovery checkpoint's response ownership "
                    "could not be verified.",
                )
        self._record_checkpoint(config, saved.checkpoint)
        execution_metadata: RunnableConfig = (
            {"metadata": config["metadata"]} if "metadata" in config else {}
        )
        return saved._replace(
            config={
                **saved.config,
                **execution_metadata,
                "configurable": {**requested, **actual},
            },
            parent_config=(
                {
                    **saved.parent_config,
                    **execution_metadata,
                    "configurable": {
                        **requested,
                        **(saved.parent_config.get("configurable") or {}),
                    },
                }
                if saved.parent_config is not None
                else None
            ),
        )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save through the application saver before advancing confirmed progress."""
        with self._track_writes():
            config = self._resolve(config)
            if (config.get("configurable") or {}).get("checkpoint_ns", "") == "":
                stamped: _ResponseCheckpointMetadata = {
                    **metadata,
                    "langgraph_response_execution": self._execution,
                }
                metadata = stamped
            saved_config = await self._saver.aput(
                config, checkpoint, metadata, new_versions
            )
            if self._head is not None and not (config.get("configurable") or {}).get(
                "checkpoint_ns"
            ):
                ref = HostingRunnableConfig(saved_config).checkpoint_ref
                if (
                    ref is None
                    or ref != CheckpointRef(self._head.thread_id, checkpoint["id"])
                    or (saved_config.get("configurable") or {}).get("checkpoint_ns", "")
                    != ""
                ):
                    raise CheckpointError(
                        "checkpoint_unavailable",
                        "The recovery checkpoint was not saved.",
                    )
                self._head = ref
            self._record_checkpoint(config, checkpoint)
            return saved_config

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve pending writes, task identity, and supported task paths."""
        with self._track_writes():
            await self._saver.aput_writes(
                self._resolve(config),
                writes,
                task_id,
                **({"task_path": task_path} if self._accepts_task_path else {}),
            )

    def get_next_version(self, current: Any, channel: Any) -> Any:
        """Allocate channel versions using the application saver's scheme."""
        return self._saver.get_next_version(current, channel)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Delegate reads inherited by synchronous private subgraphs."""
        return self._saver.get_tuple(config)

    def list(self, *args: Any, **kwargs: Any) -> Iterator[CheckpointTuple]:
        """Delegate history needed by native subgraph replay."""
        return self._saver.list(*args, **kwargs)

    async def alist(self, *args: Any, **kwargs: Any) -> AsyncIterator[CheckpointTuple]:
        """Delegate history needed by asynchronous subgraph replay."""
        async for saved in self._saver.alist(*args, **kwargs):
            yield saved

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Delegate synchronous subgraph writes without publishing root progress."""
        return self._saver.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Delegate synchronous subgraph pending writes and supported task paths."""
        self._saver.put_writes(
            config,
            writes,
            task_id,
            **({"task_path": task_path} if self._accepts_sync_task_path else {}),
        )
