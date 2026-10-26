# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local checkpoint writes with read-through access to saved state."""

from __future__ import annotations

from collections.abc import (
    AsyncIterator,
    Collection,
    Iterable,
    Iterator,
    Sequence,
)
from itertools import chain, islice
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.memory import InMemorySaver

from langchain_azure_ai.agents.hosting._checkpoint_writes import (
    _aget_indexed_writes,
    _get_indexed_writes,
    _IndexedPendingWrites,
    _IndexedWrite,
)


class TransientCheckpointSaver(BaseCheckpointSaver):
    """Isolate conversation writes in memory without providing crash recovery.

    Complete snapshots are cached by checkpoint ID, since source channel
    versions may repeat across forks. Reading historical snapshots does not
    select them as the active branch. Imported pending writes retain their
    original indices; unsupported ordinary-write imports fail rather than
    guessing indices omitted by the source's public checkpoint tuple.
    Cached checkpoint identities let task writes skip reloading saved values.
    """

    def __init__(self, source: BaseCheckpointSaver) -> None:
        super().__init__(serde=source.serde)
        self._source: BaseCheckpointSaver = source
        self._memory: BaseCheckpointSaver[str] = InMemorySaver(serde=source.serde)
        self._active_checkpoints: dict[tuple[str, str], str] = {}
        self._local_checkpoints: set[tuple[str, str, str]] = set()

    def with_allowlist(
        self, extra_allowlist: Collection[tuple[str, ...]]
    ) -> BaseCheckpointSaver:
        """Apply the graph's serializer allowlist to both checkpoint readers."""
        saver = super().with_allowlist(extra_allowlist)
        if isinstance(saver, TransientCheckpointSaver):
            saver._source = saver._source.with_allowlist(extra_allowlist)
            saver._memory = saver._memory.with_allowlist(extra_allowlist)
        return saver

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read the local branch before importing an existing checkpoint."""
        saved = self._get_local(config)
        if saved is not None:
            return saved
        saved = self._source.get_tuple(config)
        writes = (
            _get_indexed_writes(self._source, saved)
            if saved is not None and self._get_local(saved.config) is None
            else _IndexedPendingWrites(())
        )
        return self._import_checkpoint(
            saved,
            writes,
            select_current=not get_checkpoint_id(config),
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read saved state without requiring a synchronous source saver."""
        saved = self._get_local(config)
        if saved is not None:
            return saved
        saved = await self._source.aget_tuple(config)
        writes = (
            await _aget_indexed_writes(self._source, saved)
            if saved is not None and self._get_local(saved.config) is None
            else _IndexedPendingWrites(())
        )
        return self._import_checkpoint(
            saved,
            writes,
            select_current=not get_checkpoint_id(config),
        )

    @staticmethod
    def _branch_key(config: RunnableConfig) -> tuple[str, str]:
        configurable = config["configurable"]
        return configurable["thread_id"], configurable.get("checkpoint_ns", "")

    def _set_active(self, config: RunnableConfig, checkpoint_id: str) -> None:
        key = self._branch_key(config)
        self._active_checkpoints[key] = max(
            self._active_checkpoints.get(key, checkpoint_id), checkpoint_id
        )

    @staticmethod
    def _unpack_checkpoint(saved: CheckpointTuple) -> CheckpointTuple:
        writes = (
            _IndexedWrite(task_id, index, channel, value, task_path)
            for _, channel, (task_id, index, value, task_path) in (
                saved.pending_writes or ()
            )
        )
        return saved._replace(
            checkpoint=saved.checkpoint["channel_values"]["checkpoint"],
            pending_writes=_IndexedPendingWrites(
                sorted(
                    writes,
                    key=lambda write: (
                        write.task_path,
                        write.task_id,
                        write.write_index,
                    ),
                )
            ),
        )

    def _get_local(self, config: RunnableConfig) -> CheckpointTuple | None:
        checkpoint_id = get_checkpoint_id(config) or self._active_checkpoints.get(
            self._branch_key(config)
        )
        if checkpoint_id is None:
            return None
        saved = self._memory.get_tuple(
            {
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": checkpoint_id,
                },
            }
        )
        return self._unpack_checkpoint(saved) if saved is not None else None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """Read saved and request-local history without importing checkpoints.

        Args:
            config: Thread, namespace, and checkpoint ID constraints, if any.
            filter: Metadata values that returned checkpoints must match.
            before: Exclusive upper bound for checkpoint IDs.
            limit: Maximum number of merged checkpoints to return.

        Yields:
            Checkpoints newest first, preferring request-local copies.
        """
        # Some savers sort only within namespaces, so broader limits are global.
        source_limit = (
            limit
            if config and config["configurable"].get("checkpoint_ns") is not None
            else None
        )
        yield from self._merge_history(
            self._source.list(config, filter=filter, before=before, limit=source_limit),
            config,
            filter=filter,
            before=before,
            limit=limit,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Read merged history without requiring a synchronous source saver.

        Args:
            config: Thread, namespace, and checkpoint ID constraints, if any.
            filter: Metadata values that returned checkpoints must match.
            before: Exclusive upper bound for checkpoint IDs.
            limit: Maximum number of merged checkpoints to return.

        Yields:
            Checkpoints newest first, preferring request-local copies.
        """
        source_limit = (
            limit
            if config and config["configurable"].get("checkpoint_ns") is not None
            else None
        )
        saved = [
            checkpoint
            async for checkpoint in self._source.alist(
                config, filter=filter, before=before, limit=source_limit
            )
        ]
        for checkpoint in self._merge_history(
            saved, config, filter=filter, before=before, limit=limit
        ):
            yield checkpoint

    def _merge_history(
        self,
        source: Iterable[CheckpointTuple],
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None,
        before: RunnableConfig | None,
        limit: int | None,
    ) -> Iterator[CheckpointTuple]:
        history: dict[tuple[str, str, str], CheckpointTuple] = {}
        local = (
            self._unpack_checkpoint(saved)
            for saved in self._memory.list(config, filter=filter, before=before)
        )
        for saved in chain(source, local):
            configurable = saved.config["configurable"]
            history[
                (
                    configurable["thread_id"],
                    configurable.get("checkpoint_ns", ""),
                    saved.checkpoint["id"],
                )
            ] = saved
        yield from islice(
            sorted(
                history.values(),
                key=lambda saved: saved.checkpoint["id"],
                reverse=True,
            ),
            limit,
        )

    def _import_checkpoint(
        self,
        saved: CheckpointTuple | None,
        writes: _IndexedPendingWrites,
        *,
        select_current: bool = False,
    ) -> CheckpointTuple | None:
        if saved is None:
            return None
        local = self._get_local(saved.config)
        if local is None:
            parent_config: RunnableConfig = {
                "configurable": {
                    **saved.config["configurable"],
                    "checkpoint_ns": saved.config["configurable"].get(
                        "checkpoint_ns", ""
                    ),
                    "checkpoint_id": (
                        saved.parent_config["configurable"].get("checkpoint_id")
                        if saved.parent_config is not None
                        else None
                    ),
                }
            }
            config = self._save_checkpoint(
                parent_config, saved.checkpoint, saved.metadata
            )
            for write in writes.indexed:
                self._put_write(config, write)
            local = self._get_local(config)
        if select_current:
            self._set_active(saved.config, saved.checkpoint["id"])
        return local

    def _save_checkpoint(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> RunnableConfig:
        local_checkpoint: Checkpoint = {
            **checkpoint,
            "channel_values": {"checkpoint": checkpoint},
            "channel_versions": {"checkpoint": checkpoint["id"]},
        }
        saved_config = self._memory.put(
            config,
            local_checkpoint,
            metadata,
            local_checkpoint["channel_versions"],
        )
        self._local_checkpoints.add((*self._branch_key(saved_config), checkpoint["id"]))
        return saved_config

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a complete local checkpoint and advance the active branch."""
        saved_config = self._save_checkpoint(config, checkpoint, metadata)
        self._set_active(saved_config, checkpoint["id"])
        return saved_config

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint without writing to the source saver."""
        return self.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Keep pending task writes within the request."""
        if (
            *self._branch_key(config),
            get_checkpoint_id(config),
        ) not in self._local_checkpoints:
            self.get_tuple(config)
        self._put_writes(config, writes, task_id, task_path)

    def _put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str,
    ) -> None:
        for ordinal, (channel, value) in enumerate(writes):
            self._put_write(
                config,
                _IndexedWrite(
                    task_id,
                    WRITES_IDX_MAP.get(channel, ordinal),
                    channel,
                    value,
                    task_path,
                ),
            )

    def _put_write(self, config: RunnableConfig, write: _IndexedWrite) -> None:
        # Index-scoped task keys retain native first-write/overwrite semantics.
        self._memory.put_writes(
            config,
            [
                (
                    write.channel,
                    (write.task_id, write.write_index, write.value, write.task_path),
                )
            ],
            f"{write.write_index}:{write.task_id}",
            write.task_path,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save pending writes without writing to the source saver."""
        if (
            *self._branch_key(config),
            get_checkpoint_id(config),
        ) not in self._local_checkpoints:
            await self.aget_tuple(config)
        self._put_writes(config, writes, task_id, task_path)

    def get_next_version(
        self, current: str | int | float | None, channel: None
    ) -> str | int | float:
        """Retain the source saver's channel version format."""
        return self._source.get_next_version(current, channel)
