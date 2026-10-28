# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Per-response LangGraph checkpoint storage management."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection, MutableMapping, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import Any, cast

from azure.ai.agentserver.responses import ResponseEventStream
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from wrapt import ObjectProxy

from .._checkpoint._checkpoint_reuse import _accepts_task_path
from .checkpoint_ref import CheckpointRef
from .hosting_runnable_config import HostingRunnableConfig

METADATA_LANGGRAPH_CHECKPOINT_ID = "langgraph_checkpoint_id"
METADATA_LANGGRAPH_THREAD_ID = "langgraph_thread_id"
METADATA_LANGGRAPH_ROOT_RESPONSE_ID = "langgraph_root_response_id"
METADATA_RESPONSE_EXECUTION = "langgraph_response_execution"
_METADATA_EMITTED_MESSAGES = "langgraph_emitted_message_ids"


class TaskStorageManager:
    """Manage LangGraph references persisted with one Responses task."""

    def __init__(self, metadata: MutableMapping[str, Any] | None) -> None:
        self._metadata = metadata

    @classmethod
    def from_stream(cls, stream: ResponseEventStream) -> TaskStorageManager:
        """Manage the writable metadata attached to an active response stream."""
        return cls(stream.internal_metadata)

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

    @property
    def recovery_checkpoint_ref(self) -> CheckpointRef | None:
        """Read recovery progress without treating malformed data as a new run."""
        reference = self.checkpoint_ref
        if (
            reference is None
            and self._metadata is not None
            and any(
                key in self._metadata
                for key in (
                    METADATA_LANGGRAPH_THREAD_ID,
                    METADATA_LANGGRAPH_CHECKPOINT_ID,
                )
            )
        ):
            raise RuntimeError("The response's saved graph checkpoint is malformed.")
        return reference

    @property
    def emitted_message_ids(self) -> set[str]:
        """Return graph messages already included in the persisted output."""
        if self._metadata is None:
            return set()
        value = self._metadata.get(_METADATA_EMITTED_MESSAGES, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise RuntimeError("The response's saved message identities are malformed.")
        return set(value)

    def store_emitted_message_ids(self, message_ids: set[str]) -> None:
        """Commit message identities together with output and graph progress."""
        if self._metadata is not None and message_ids:
            self._metadata[_METADATA_EMITTED_MESSAGES] = sorted(message_ids)


@dataclass
class _CheckpointProgress:
    thread_id: str
    response_key: str
    parent_response_id: str | None
    head: CheckpointRef | None
    require_ownership: bool
    confirmed: set[CheckpointRef] = field(default_factory=set)
    inherited: set[CheckpointRef] = field(default_factory=set)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    failure: BaseException | None = None

    @property
    def owner(self) -> list[str | None]:
        return [self.response_key, self.parent_response_id]


class ResponseCheckpointSaver(ObjectProxy):
    """Adapt an application saver to one response without owning its lifecycle.

    Root asynchronous writes carry response ownership and confirm progress.
    Native recovery resolves checkpoint-less reads against this response's head,
    never a newer sibling. Other saver capabilities, including synchronous
    private-subgraph access and history queries, remain delegated.
    """

    def __init__(
        self,
        saver: BaseCheckpointSaver,
        *,
        thread_id: str,
        response_key: str,
        parent_response_id: str | None,
        recovery_ref: CheckpointRef | None = None,
        require_ownership: bool = False,
    ) -> None:
        super().__init__(saver)
        self._self_progress = _CheckpointProgress(
            thread_id, response_key, parent_response_id, recovery_ref, require_ownership
        )
        self._self_task_path = _accepts_task_path(saver.aput_writes)

    def _clone(self, saver: BaseCheckpointSaver) -> ResponseCheckpointSaver:
        progress = self._self_progress
        clone = type(self)(
            saver,
            thread_id=progress.thread_id,
            response_key=progress.response_key,
            parent_response_id=progress.parent_response_id,
        )
        clone._self_progress = progress
        return clone

    def __copy__(self) -> ResponseCheckpointSaver:
        return self._clone(copy(self.__wrapped__))

    def with_allowlist(
        self, extra_allowlist: Collection[tuple[str, ...]]
    ) -> ResponseCheckpointSaver:
        """Keep request progress shared across serializer allowlist clones."""
        saver = self.__wrapped__.with_allowlist(extra_allowlist)
        return self if saver is self.__wrapped__ else self._clone(saver)

    def resume_config(self, config: RunnableConfig) -> RunnableConfig:
        """Use native resume semantics while retaining an exact recovery head."""
        configurable = dict(config.get("configurable") or {})
        configurable.pop("checkpoint_id", None)
        return {**config, "configurable": configurable}

    def _root_config(self, config: RunnableConfig) -> bool:
        configurable = config.get("configurable") or {}
        if configurable.get("checkpoint_ns", "") != "":
            return False
        if configurable.get("thread_id") != self._self_progress.thread_id:
            raise RuntimeError("Graph checkpoint thread does not match the response.")
        return True

    def _read_config(self, config: RunnableConfig) -> RunnableConfig:
        head = self._self_progress.head
        if head is not None and not (config.get("configurable") or {}).get(
            "checkpoint_id"
        ):
            return (
                HostingRunnableConfig(config).with_checkpoint_ref(head).runnable_config
            )
        return config

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read and validate the exact root identity before graph execution."""
        if not self._root_config(config):
            return await self.__wrapped__.aget_tuple(config)
        config = self._read_config(config)
        reference = HostingRunnableConfig(config).checkpoint_ref
        saved = await self.__wrapped__.aget_tuple(config)
        if saved is None:
            if reference is not None:
                raise RuntimeError("The required graph checkpoint is missing.")
            return None
        actual = HostingRunnableConfig(saved.config).checkpoint_ref
        if (
            actual is None
            or actual.thread_id != self._self_progress.thread_id
            or (saved.config.get("configurable") or {}).get("checkpoint_ns", "") != ""
            or saved.checkpoint.get("id") != actual.checkpoint_id
            or (reference is not None and actual != reference)
        ):
            raise RuntimeError("The restored graph checkpoint identity is mismatched.")

        progress = self._self_progress
        stamp = saved.metadata.get(METADATA_RESPONSE_EXECUTION)
        owner = None
        if stamp is not None:
            try:
                owner = json.loads(stamp) if isinstance(stamp, str) else None
            except json.JSONDecodeError as exc:
                raise RuntimeError("Graph checkpoint ownership is malformed.") from exc
        if progress.head is not None:
            if (stamp is not None and owner != progress.owner) or (
                stamp is None and progress.require_ownership
            ):
                raise RuntimeError(
                    "Graph checkpoint ownership does not match recovery."
                )
            progress.confirmed.add(actual)
        elif owner == progress.owner:
            progress.confirmed.add(actual)
        else:
            progress.inherited.add(actual)
        return saved

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        """Read a checkpoint through the same identity and ownership checks."""
        saved = await self.aget_tuple(config)
        return saved.checkpoint if saved is not None else None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Confirm root progress only after its underlying write succeeds."""
        if not self._root_config(config):
            return await self.__wrapped__.aput(
                config, checkpoint, metadata, new_versions
            )
        progress = self._self_progress
        try:
            saved_config = await self.__wrapped__.aput(
                self._read_config(config),
                checkpoint,
                cast(
                    CheckpointMetadata,
                    {
                        **metadata,
                        METADATA_RESPONSE_EXECUTION: json.dumps(progress.owner),
                    },
                ),
                new_versions,
            )
            reference = HostingRunnableConfig(saved_config).checkpoint_ref
            if (
                reference != CheckpointRef(progress.thread_id, checkpoint["id"])
                or (saved_config.get("configurable") or {}).get("checkpoint_ns", "")
                != ""
            ):
                raise RuntimeError("The saved graph checkpoint identity is mismatched.")
            progress.confirmed.add(reference)
            if progress.head is not None:
                progress.head = reference
            return saved_config
        except BaseException as exc:
            progress.failure = exc
            raise
        finally:
            progress.changed.set()

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Preserve native task identity and task paths for pending writes."""
        root = self._root_config(config)
        try:
            kwargs = {"task_path": task_path} if self._self_task_path else {}
            await self.__wrapped__.aput_writes(
                self._read_config(config) if root else config,
                writes,
                task_id,
                **kwargs,
            )
        except BaseException as exc:
            if root:
                self._self_progress.failure = exc
                self._self_progress.changed.set()
            raise

    async def wait_for_checkpoint(self, reference: CheckpointRef) -> bool:
        """Wait for persistence, returning false for inherited parent progress."""
        progress = self._self_progress
        if reference.thread_id != progress.thread_id:
            raise RuntimeError("Graph checkpoint thread does not match the response.")
        while True:
            if progress.failure is not None:
                raise progress.failure
            if reference in progress.confirmed:
                return True
            if reference in progress.inherited:
                return False
            await progress.changed.wait()
            progress.changed.clear()
