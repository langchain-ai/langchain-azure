# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Request-local checkpoint reuse for any LangGraph saver."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy, deepcopy
from dataclasses import dataclass
from inspect import Parameter, signature
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


def _reference_config(
    config: RunnableConfig | None, keys: set[str]
) -> RunnableConfig | None:
    if config is None:
        return None
    return {
        "configurable": deepcopy(
            {
                key: value
                for key, value in (config.get("configurable") or {}).items()
                if key in keys
            }
        )
    }


class _SerializedCheckpoint(NamedTuple):
    config: RunnableConfig
    parent_config: RunnableConfig | None
    header: tuple[str, bytes]
    channels: dict[str, tuple[str, bytes]]
    metadata: tuple[str, bytes]
    writes: list[tuple[str, str, tuple[str, bytes]]] | None

    @classmethod
    def create(
        cls, value: CheckpointTuple, serde: SerializerProtocol, keys: set[str]
    ) -> _SerializedCheckpoint:
        header = dict(value.checkpoint)
        del header["channel_values"]
        return cls(
            _reference_config(value.config, keys) or {},
            _reference_config(value.parent_config, keys),
            serde.dumps_typed(header),
            {
                channel: serde.dumps_typed(item)
                for channel, item in value.checkpoint["channel_values"].items()
            },
            serde.dumps_typed(value.metadata),
            [
                (task, channel, serde.dumps_typed(item))
                for task, channel, item in value.pending_writes
            ]
            if value.pending_writes is not None
            else None,
        )

    def restore(
        self, serde: SerializerProtocol, config: RunnableConfig
    ) -> CheckpointTuple:
        checkpoint = serde.loads_typed(self.header)
        checkpoint["channel_values"] = {
            channel: serde.loads_typed(item) for channel, item in self.channels.items()
        }
        return CheckpointTuple(
            config={
                **config,
                "configurable": {
                    **(config.get("configurable") or {}),
                    **deepcopy(self.config.get("configurable") or {}),
                },
            },
            checkpoint=checkpoint,
            metadata=serde.loads_typed(self.metadata),
            parent_config=deepcopy(self.parent_config),
            pending_writes=[
                (task, channel, serde.loads_typed(item))
                for task, channel, item in self.writes
            ]
            if self.writes is not None
            else None,
        )


def _accepts_task_path(method: Callable[..., Any]) -> bool:
    return any(
        parameter.name == "task_path" or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in signature(method).parameters.values()
    )


class _CachedCheckpoint(NamedTuple):
    """A cached result, including a confirmed missing checkpoint."""

    value: _SerializedCheckpoint | None
    allowlist: frozenset[tuple[str, ...]]
    config_values: dict[str, Any]


@dataclass
class _CheckpointReadContext:
    """Reuse only exact reads of one request's input checkpoint."""

    key: tuple[object, str | None, str, str] | None = None
    input_checkpoint_id: str | None = None
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
    """Decorate a saver with request-local reads; forward its other capabilities.

    Only the shallow copies used by LangGraph's allowlist clones are supported.
    Deep copying and pickling this live runtime proxy are not supported.
    """

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
        self._self_async_task_path = _accepts_task_path(saver.aput_writes)
        self._self_sync_task_path = _accepts_task_path(saver.put_writes)
        self._self_extra_config_keys = {spec.id for spec in saver.config_specs} - {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
        }
        self._self_reference_keys = self._self_extra_config_keys | {
            "thread_id",
            "checkpoint_ns",
            "checkpoint_id",
        }

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
        self, thread_id: str | None = None, namespace: str | None = None
    ) -> _CheckpointReadContext | None:
        reads = _checkpoint_reads.get()
        if reads is None or not reads.active or reads.key is None:
            return None
        owner, user_id, request_thread_id, request_namespace = reads.key
        if (
            owner is self._self_cache_owner
            and user_id == get_request_context().user_id
            and (thread_id is None or request_thread_id == thread_id)
            and (namespace is None or request_namespace == namespace)
        ):
            return reads
        return None

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        configurable = config.get("configurable") or {}
        if not isinstance(configurable.get("thread_id"), str) or not isinstance(
            configurable.get("checkpoint_ns", ""), str
        ):
            return await self.__wrapped__.aget_tuple(config)
        reads = self._read_context(
            configurable.get("thread_id"), configurable.get("checkpoint_ns", "")
        )
        if reads is None:
            return await self.__wrapped__.aget_tuple(config)
        checkpoint_id = get_checkpoint_id(config)
        # Checkpoint-less reads must reach storage for LangGraph's fork protection.
        cacheable = (
            bool(checkpoint_id)
            and reads.cache_enabled
            and checkpoint_id == reads.input_checkpoint_id
        )
        try:
            cached = reads.cache
            # Matching allowlists share cache; changed allowlists reload for isolation.
            if (
                cacheable
                and cached is not None
                and cached.allowlist == self._self_allowlist
            ):
                try:
                    if cached.config_values == {
                        key: configurable.get(key)
                        for key in self._self_extra_config_keys
                    }:
                        return (
                            cached.value.restore(self.serde, config)
                            if cached.value is not None
                            else None
                        )
                except Exception:
                    reads.invalidate()
            result = await self.__wrapped__.aget_tuple(config)
            # A write may have disabled caching while this read awaited storage.
            if cacheable and reads.active and reads.cache_enabled:
                try:
                    reads.cache = _CachedCheckpoint(
                        _SerializedCheckpoint.create(
                            result, self.serde, self._self_reference_keys
                        )
                        if result is not None
                        else None,
                        self._self_allowlist,
                        deepcopy(
                            {
                                key: configurable.get(key)
                                for key in self._self_extra_config_keys
                            }
                        ),
                    )
                except Exception:
                    reads.invalidate()
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
        self._invalidate_config(config)
        return await self.__wrapped__.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._invalidate_config(config)
        kwargs = {"task_path": task_path} if self._self_async_task_path else {}
        await self.__wrapped__.aput_writes(config, writes, task_id, **kwargs)

    async def adelete_thread(self, thread_id: str) -> None:
        self._invalidate_threads([thread_id])
        await self.__wrapped__.adelete_thread(thread_id)

    def _invalidate_config(self, config: RunnableConfig) -> None:
        configurable = config.get("configurable") or {}
        reads = self._read_context(
            configurable.get("thread_id"), configurable.get("checkpoint_ns", "")
        )
        if reads is not None:
            reads.invalidate()

    def _invalidate_threads(self, thread_ids: Sequence[str] | None = None) -> None:
        reads = self._read_context()
        if (
            reads is not None
            and reads.key is not None
            and (thread_ids is None or reads.key[2] in thread_ids)
        ):
            reads.invalidate()

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self._invalidate_config(config)
        return self.__wrapped__.put(config, checkpoint, metadata, new_versions)

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self._invalidate_config(config)
        kwargs = {"task_path": task_path} if self._self_sync_task_path else {}
        self.__wrapped__.put_writes(config, writes, task_id, **kwargs)

    def delete_thread(self, thread_id: str) -> None:
        self._invalidate_threads([thread_id])
        self.__wrapped__.delete_thread(thread_id)

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._invalidate_threads()
        self.__wrapped__.delete_for_runs(run_ids)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        self._invalidate_threads()
        await self.__wrapped__.adelete_for_runs(run_ids)

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._invalidate_threads([target_thread_id])
        self.__wrapped__.copy_thread(source_thread_id, target_thread_id)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        self._invalidate_threads([target_thread_id])
        await self.__wrapped__.acopy_thread(source_thread_id, target_thread_id)

    def prune(
        self, thread_ids: Sequence[str], *, strategy: str = "keep_latest"
    ) -> None:
        self._invalidate_threads(thread_ids)
        self.__wrapped__.prune(thread_ids, strategy=strategy)

    async def aprune(
        self, thread_ids: Sequence[str], *, strategy: str = "keep_latest"
    ) -> None:
        self._invalidate_threads(thread_ids)
        await self.__wrapped__.aprune(thread_ids, strategy=strategy)
