# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local checkpoint reuse for any LangGraph saver."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from inspect import signature
from typing import Any, NamedTuple

from azure.ai.agentserver.core import get_request_context
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from wrapt import ObjectProxy


def _copy_config(config: RunnableConfig | None) -> RunnableConfig | None:
    """Copy config mappings while preserving transient runtime objects."""
    if config is None:
        return None
    return {**config, "configurable": dict(config.get("configurable") or {})}


def _copy_tuple(value: CheckpointTuple, serde: SerializerProtocol) -> CheckpointTuple:
    def clone(stored_value: Any) -> Any:
        return serde.loads_typed(serde.dumps_typed(stored_value))

    return CheckpointTuple(
        config=_copy_config(value.config) or {},
        checkpoint=clone(value.checkpoint),
        metadata=clone(value.metadata),
        parent_config=_copy_config(value.parent_config),
        pending_writes=(
            [
                (task, channel, clone(output))
                for task, channel, output in value.pending_writes
            ]
            if value.pending_writes is not None
            else None
        ),
    )


class _CachedCheckpoint(NamedTuple):
    """A cached result, including a confirmed missing checkpoint."""

    value: CheckpointTuple | None
    allowlist: frozenset[tuple[str, ...]]


@dataclass
class _CheckpointReadContext:
    """Keep one request's input snapshot separate from its advancing output."""

    key: tuple[object, str | None, str, str] | None = None
    input_checkpoint_id: str | None = None
    latest_checkpoint_id: str | None = None
    resolved: bool = False
    active: bool = True
    cache_enabled: bool = True
    cache: _CachedCheckpoint | None = None

    def pin(self, saver: object, config: RunnableConfig) -> None:
        configurable = config.get("configurable") or {}
        thread_id = configurable.get("thread_id")
        namespace = configurable.get("checkpoint_ns", "")
        if (
            isinstance(saver, BaseCheckpointSaver)
            and isinstance(thread_id, str)
            and isinstance(namespace, str)
        ):
            owner = (
                saver._self_cache_owner
                if isinstance(saver, _RequestCheckpointSaver)
                else saver
            )
            self.key = (
                owner,
                get_request_context().user_id,
                thread_id,
                namespace,
            )
            self.input_checkpoint_id = get_checkpoint_id(config)
            self.latest_checkpoint_id = self.input_checkpoint_id
            self.resolved = bool(self.input_checkpoint_id)

    @contextmanager
    def bind(self) -> Iterator[None]:
        token = _checkpoint_reads.set(self)
        try:
            yield
        finally:
            _checkpoint_reads.reset(token)

    def invalidate(self) -> None:
        self.cache_enabled = False
        self.cache = None

    def close(self) -> None:
        self.active = False
        self.invalidate()


_checkpoint_reads: ContextVar[_CheckpointReadContext | None] = ContextVar(
    "langchain_azure_checkpoint_reads", default=None
)


class _RequestCheckpointSaver(ObjectProxy):
    """Decorate a saver with request-local reads; forward its other capabilities."""

    def __init__(
        self,
        saver: BaseCheckpointSaver,
        *,
        cache_owner: object | None = None,
        allowlist: frozenset[tuple[str, ...]] = frozenset(),
    ) -> None:
        super().__init__(saver)
        self._self_cache_owner = saver if cache_owner is None else cache_owner
        self._self_allowlist = allowlist
        self._self_accepts_task_path = (
            "task_path" in signature(saver.aput_writes).parameters
        )

    def __copy__(self) -> _RequestCheckpointSaver:
        return type(self)(
            copy(self.__wrapped__),
            cache_owner=self._self_cache_owner,
            allowlist=self._self_allowlist,
        )

    def with_allowlist(
        self, extra_allowlist: Collection[tuple[str, ...]]
    ) -> _RequestCheckpointSaver:
        saver = self.__wrapped__.with_allowlist(extra_allowlist)
        if saver is self.__wrapped__:
            return self
        return type(self)(
            saver,
            cache_owner=self._self_cache_owner,
            allowlist=self._self_allowlist | frozenset(extra_allowlist),
        )

    def _read_context(
        self, thread_id: str | None, namespace: str | None = None
    ) -> _CheckpointReadContext | None:
        reads = _checkpoint_reads.get()
        if reads is None or not reads.active or reads.key is None:
            return None
        owner, user_id, request_thread_id, request_namespace = reads.key
        if (
            owner is self._self_cache_owner
            and user_id == get_request_context().user_id
            and request_thread_id == thread_id
            and (namespace is None or request_namespace == namespace)
        ):
            return reads
        return None

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config.get("configurable") or {}
        if not isinstance(configurable.get("checkpoint_ns", ""), str):
            return await self.__wrapped__.aget_tuple(config)
        reads = self._read_context(
            configurable.get("thread_id"), configurable.get("checkpoint_ns", "")
        )
        if reads is None:
            return await self.__wrapped__.aget_tuple(config)
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id and reads.resolved:
            checkpoint_id = reads.latest_checkpoint_id
            if checkpoint_id is None:
                return None
            config = {
                **config,
                "configurable": {**configurable, "checkpoint_id": checkpoint_id},
            }
        cacheable = reads.cache_enabled and (
            not checkpoint_id or checkpoint_id == reads.input_checkpoint_id
        )
        try:
            cached = reads.cache
            if (
                cacheable
                and cached is not None
                and cached.allowlist == self._self_allowlist
            ):
                return (
                    _copy_tuple(cached.value, self.serde)
                    if cached.value is not None
                    else None
                )
            result = await self.__wrapped__.aget_tuple(config)
            # A write may have disabled caching while this read awaited storage.
            if cacheable and reads.active and reads.cache_enabled:
                reads.cache = _CachedCheckpoint(
                    _copy_tuple(result, self.serde) if result is not None else None,
                    self._self_allowlist,
                )
                if not reads.resolved:
                    reads.input_checkpoint_id = (
                        get_checkpoint_id(result.config) if result is not None else None
                    )
                    reads.latest_checkpoint_id = reads.input_checkpoint_id
                    reads.resolved = True
            return result
        except BaseException:
            reads.invalidate()
            raise

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        value = await self.aget_tuple(config)
        return value.checkpoint if value is not None else None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        configurable = config.get("configurable") or {}
        reads = self._read_context(
            configurable.get("thread_id"), configurable.get("checkpoint_ns", "")
        )
        if reads is not None:
            reads.invalidate()
        result = await self.__wrapped__.aput(config, checkpoint, metadata, new_versions)
        if reads is not None and reads.active:
            reads.latest_checkpoint_id = max(
                reads.latest_checkpoint_id or "", checkpoint["id"]
            )
            reads.resolved = True
        return result

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        configurable = config.get("configurable") or {}
        reads = self._read_context(
            configurable.get("thread_id"), configurable.get("checkpoint_ns", "")
        )
        if reads is not None:
            reads.invalidate()
        kwargs = {"task_path": task_path} if self._self_accepts_task_path else {}
        await self.__wrapped__.aput_writes(config, writes, task_id, **kwargs)

    async def adelete_thread(self, thread_id: str) -> None:
        reads = self._read_context(thread_id)
        if reads is not None:
            reads.invalidate()
        await self.__wrapped__.adelete_thread(thread_id)
        if reads is not None and reads.active:
            reads.latest_checkpoint_id = None
            reads.resolved = True
