# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local checkpoint writes with read-through access to saved state."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Collection, Iterable, Iterator, Sequence
from itertools import chain, islice
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver


class TransientCheckpointSaver(BaseCheckpointSaver):
    """Isolate conversation writes in memory without providing crash recovery."""

    def __init__(self, source: BaseCheckpointSaver) -> None:
        super().__init__(serde=source.serde)
        self._source: BaseCheckpointSaver = source
        self._memory: BaseCheckpointSaver[str] = InMemorySaver(serde=source.serde)

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
        saved = self._memory.get_tuple(config)
        if saved is not None:
            return saved
        return self._import_checkpoint(self._source.get_tuple(config))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Read saved state without requiring a synchronous source saver."""
        saved = self._memory.get_tuple(config)
        if saved is not None:
            return saved
        return self._import_checkpoint(await self._source.aget_tuple(config))

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
        local = self._memory.list(config, filter=filter, before=before)
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
        self, saved: CheckpointTuple | None
    ) -> CheckpointTuple | None:
        if saved is None:
            return None
        parent_config: RunnableConfig = {
            "configurable": {
                **saved.config["configurable"],
                "checkpoint_ns": saved.config["configurable"].get("checkpoint_ns", ""),
                "checkpoint_id": (
                    saved.parent_config["configurable"].get("checkpoint_id")
                    if saved.parent_config is not None
                    else None
                ),
            }
        }
        config = self.put(
            parent_config,
            saved.checkpoint,
            saved.metadata,
            saved.checkpoint["channel_versions"],
        )
        writes: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        for task_id, channel, value in saved.pending_writes or ():
            writes[task_id].append((channel, value))
        for task_id, task_writes in writes.items():
            self.put_writes(config, task_writes, task_id)
        return self._memory.get_tuple(config)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save all channels locally, including unchanged replayed state."""
        return self._memory.put(
            config, checkpoint, metadata, checkpoint["channel_versions"]
        )

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
        self._memory.put_writes(config, writes, task_id, task_path)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save pending writes without writing to the source saver."""
        self.put_writes(config, writes, task_id, task_path)

    def get_next_version(
        self, current: str | int | float | None, channel: None
    ) -> str | int | float:
        """Retain the source saver's channel version format."""
        return self._source.get_next_version(current, channel)
