# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""LangGraph checkpoint persistence backed by Foundry state stores."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, cast

from azure.ai.agentserver.core.storage import (
    FoundryStateStore,
    FoundryStorageConflictError,
    FoundryStorageEndpoint,
    FoundryStorageNotFoundError,
    JSONObject,
)
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import DefaultAzureCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from typing_extensions import NotRequired, TypedDict

from langchain_azure_ai._user_agent import get_user_agent
from langchain_azure_ai.agents.hosting import (
    HostingFeature,
    _add_process_hosting_features,
)

DEFAULT_STORE_NAME_PREFIX = "langchain_azure_ai.agents.hosting.FoundryCheckpointSaver"
DEFAULT_ITEM_TTL_SECONDS = 30 * 24 * 60 * 60
_PAGE_SIZE = 100
_MAX_STORE_NAME_LENGTH = 128


class _SerializedValue(TypedDict):
    """LangGraph typed-serializer output represented as Foundry JSON."""

    type: str
    data: str


class _CheckpointItemValue(TypedDict):
    """JSON value stored for one LangGraph checkpoint item."""

    checkpoint_id: str
    checkpoint_ns: str
    checkpoint: _SerializedValue
    metadata: _SerializedValue
    parent_checkpoint_id: str


class _PendingWriteItemValue(TypedDict):
    """JSON value stored for one LangGraph pending-write item.

    Pending writes preserve successful task outputs before the next full
    checkpoint commits. LangGraph can reuse them when sibling tasks fail,
    avoiding repeated model/tool work and duplicate non-idempotent side
    effects. They also carry error, interrupt, resume, and scheduling state.
    """

    checkpoint_id: str
    checkpoint_ns: str
    task_id: str
    task_path: str
    index: int
    channel: str
    value: _SerializedValue


class _CheckpointTags(TypedDict):
    """Foundry checkpoint tag vocabulary."""

    kind: Literal["checkpoint"]
    source: NotRequired[str]
    step: NotRequired[str]


class _CheckpointItemTags(_CheckpointTags):
    """Foundry tags attached to checkpoint items."""

    ns: str


class _CheckpointQueryTags(_CheckpointTags):
    """Foundry tags used to query checkpoint items."""

    ns: NotRequired[str]


class _PendingWriteItemTags(TypedDict):
    """Foundry tags attached to pending-write items."""

    kind: Literal["write"]
    ns: str
    ckpt: str


class FoundryCheckpointSaver(BaseCheckpointSaver):
    """Persist LangGraph checkpoints in Microsoft Foundry state stores.

    Each LangGraph ``thread_id`` maps to one state store named
    ``{store_name_prefix}/{thread_id}``. Checkpoint namespaces and IDs are
    encoded in item keys within that store. The saver is async-only; use it
    with ``graph.ainvoke`` or ``graph.astream``.

    Args:
        credential: Optional async Azure credential. Omit to use
            ``DefaultAzureCredential``.
        endpoint: Foundry project or storage endpoint. When omitted, the SDK
            resolves ``FOUNDRY_PROJECT_ENDPOINT``.
        store_name_prefix: Prefix used for per-thread store names.
        user_isolation: Partition items by the hosted request's end user.
        item_ttl_seconds: Sliding item TTL configured when a thread store is
            first created. ``-1`` disables expiration.
        api_version: Foundry storage API version.
        serde: Optional LangGraph serializer.
    """

    def __init__(
        self,
        credential: AsyncTokenCredential | None = None,
        endpoint: FoundryStorageEndpoint | str | None = None,
        *,
        store_name_prefix: str = DEFAULT_STORE_NAME_PREFIX,
        user_isolation: bool = True,
        item_ttl_seconds: int = DEFAULT_ITEM_TTL_SECONDS,
        api_version: str = "v1",
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        if not store_name_prefix:
            raise ValueError("store_name_prefix must be a non-empty string")

        self._hosting_features = HostingFeature.FOUNDRY_CHECKPOINT
        _add_process_hosting_features(self._hosting_features)
        self._owns_credential = credential is None
        self._credential = (
            credential if credential is not None else DefaultAzureCredential()
        )
        self._endpoint = endpoint
        self._store_name_prefix = store_name_prefix.rstrip("/")
        self._user_isolation = user_isolation
        self._item_ttl_seconds = item_ttl_seconds
        self._api_version = api_version

    async def __aenter__(self) -> FoundryCheckpointSaver:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the saver-owned credential, if any."""
        if self._owns_credential:
            await self._credential.close()
            self._owns_credential = False

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch an exact checkpoint or the latest checkpoint in a namespace."""
        thread_id, checkpoint_ns = self._thread_and_namespace(config)
        store = await self._get_or_create_store(thread_id)
        try:
            checkpoint_id = get_checkpoint_id(config)

            if checkpoint_id:
                item = await store.get_item(
                    self._checkpoint_key(checkpoint_ns, checkpoint_id)
                )
            else:
                # Foundry orders keys by creation time. Descending order with
                # limit=1 selects the newest checkpoint in this namespace.
                tags: _CheckpointItemTags = {
                    "kind": "checkpoint",
                    "ns": checkpoint_ns,
                }
                page = await store.list_keys(
                    tags=cast(Mapping[str, str], tags),
                    limit=1,
                    order="desc",
                )
                item = await store.get_item(page.keys[0].key) if page.keys else None

            if item is None:
                return None
            return await self._checkpoint_tuple(
                store,
                thread_id,
                cast(_CheckpointItemValue, item.value),
            )
        finally:
            await store.aclose()

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints newest-first for one known thread."""
        if config is None:
            raise ValueError("config with a thread_id is required")
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")

        thread_id = self._thread_id(config)
        configurable = config.get("configurable") or {}
        namespace_is_set = "checkpoint_ns" in configurable
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        if not isinstance(checkpoint_ns, str):
            raise ValueError("checkpoint_ns must be a string")

        checkpoint_id = get_checkpoint_id(config)
        before_id = get_checkpoint_id(before) if before else None
        tags: _CheckpointQueryTags = {"kind": "checkpoint"}
        if namespace_is_set:
            tags["ns"] = checkpoint_ns
        if filter:
            for field in ("source", "step"):
                if field in filter:
                    tags[field] = str(filter[field])

        store = await self._get_or_create_store(thread_id)
        try:
            after: str | None = None
            if before_id and before is not None:
                before_configurable = before.get("configurable") or {}
                before_ns = before_configurable.get("checkpoint_ns", checkpoint_ns)
                if not isinstance(before_ns, str):
                    raise ValueError("checkpoint_ns must be a string")
                before_item = await store.get_item(
                    self._checkpoint_key(before_ns, before_id)
                )
                if before_item is None:
                    return
                # Foundry store cursors are relative to the requested order. With
                # descending order, items after this cursor are older.
                after = before_item.id

            yielded = 0
            while True:
                page = await store.list_keys(
                    tags=cast(Mapping[str, str], tags),
                    limit=_PAGE_SIZE,
                    order="desc",
                    after=after,
                )
                for key in page.keys:
                    item = await store.get_item(key.key)
                    if item is None:
                        continue
                    value = cast(_CheckpointItemValue, item.value)
                    item_ns = value.get("checkpoint_ns", "")
                    item_id = value.get("checkpoint_id")
                    if not isinstance(item_id, str):
                        continue
                    if namespace_is_set and item_ns != checkpoint_ns:
                        continue
                    if checkpoint_id and item_id != checkpoint_id:
                        continue

                    checkpoint_tuple = await self._checkpoint_tuple(
                        store, thread_id, value
                    )
                    if filter and not all(
                        checkpoint_tuple.metadata.get(name) == expected
                        for name, expected in filter.items()
                    ):
                        continue

                    yield checkpoint_tuple
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return

                if not page.has_more or page.last_id is None:
                    return
                after = page.last_id
        finally:
            await store.aclose()

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist one append-only LangGraph checkpoint."""
        del new_versions
        thread_id, checkpoint_ns = self._thread_and_namespace(config)
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = get_checkpoint_id(config)
        checkpoint_metadata = get_checkpoint_metadata(config, metadata)

        value: _CheckpointItemValue = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint": self._serialize(checkpoint),
            "metadata": self._serialize(checkpoint_metadata),
            "parent_checkpoint_id": parent_checkpoint_id or "",
        }
        tags: _CheckpointItemTags = {
            "kind": "checkpoint",
            "ns": checkpoint_ns,
        }
        source = checkpoint_metadata.get("source")
        step = checkpoint_metadata.get("step")
        if source is not None:
            tags["source"] = str(source)
        if step is not None:
            tags["step"] = str(step)

        store = await self._get_or_create_store(thread_id)
        try:
            await store.set_item(
                self._checkpoint_key(checkpoint_ns, checkpoint_id),
                cast(JSONObject, value),
                tags=cast(Mapping[str, str], tags),
            )
        finally:
            await store.aclose()
        return self._config(thread_id, checkpoint_ns, checkpoint_id)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist task outputs needed for correct and efficient recovery.

        Without these writes, LangGraph may rerun already-successful sibling
        tasks after a failure and can lose special error, interrupt, resume,
        or scheduling state.
        """
        thread_id, checkpoint_ns = self._thread_and_namespace(config)
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required to store pending writes")
        if not task_id:
            raise ValueError("task_id must be a non-empty string")

        store = await self._get_or_create_store(thread_id)
        try:
            for ordinal, (channel, value) in enumerate(writes):
                index = WRITES_IDX_MAP.get(channel, ordinal)
                item_value: _PendingWriteItemValue = {
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": checkpoint_ns,
                    "task_id": task_id,
                    "task_path": task_path,
                    "index": index,
                    "channel": channel,
                    "value": self._serialize(value),
                }
                key = self._write_key(
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    index,
                )
                tags: _PendingWriteItemTags = {
                    "kind": "write",
                    "ns": checkpoint_ns,
                    "ckpt": checkpoint_id,
                }
                if channel in WRITES_IDX_MAP:
                    await store.set_item(
                        key,
                        cast(JSONObject, item_value),
                        tags=cast(Mapping[str, str], tags),
                    )
                else:
                    try:
                        await store.create_item(
                            key,
                            cast(JSONObject, item_value),
                            tags=cast(Mapping[str, str], tags),
                        )
                    except FoundryStorageConflictError:
                        pass
        finally:
            await store.aclose()

    async def adelete_thread(self, thread_id: str) -> None:
        """Delete a thread's store and every checkpoint item in it."""
        if not thread_id:
            raise ValueError("thread_id must be a non-empty string")
        store = self._bound_store(thread_id)
        try:
            if self._user_isolation:
                keys: list[str] = []
                after: str | None = None
                while True:
                    page = await store.list_keys(
                        limit=_PAGE_SIZE,
                        order="asc",
                        after=after,
                    )
                    keys.extend(key.key for key in page.keys)
                    if not page.has_more or page.last_id is None:
                        break
                    after = page.last_id
                for key in keys:
                    await store.delete_item(key)
            else:
                await store.delete()
        except FoundryStorageNotFoundError:
            pass
        finally:
            await store.aclose()

    async def _get_or_create_store(self, thread_id: str) -> FoundryStateStore:
        store = await FoundryStateStore.get_or_create(
            self._store_name(thread_id),
            self._credential,
            self._endpoint,
            user_isolation=self._user_isolation,
            item_ttl_seconds=self._item_ttl_seconds,
            description="LangGraph checkpoints",
            api_version=self._api_version,
            # AgentServer uses this callback to build the outbound User-Agent
            # header. Keep it lazy so registered prefixes and the telemetry
            # opt-out are evaluated per request.
            get_server_version=get_user_agent,
        )
        try:
            properties = await store.get()
            if properties.user_isolation != self._user_isolation:
                raise ValueError(
                    f"State store {store.name!r} already exists with "
                    f"user_isolation={properties.user_isolation}; expected "
                    f"{self._user_isolation}"
                )
            if properties.item_ttl_seconds != self._item_ttl_seconds:
                raise ValueError(
                    f"State store {store.name!r} already exists with "
                    f"item_ttl_seconds={properties.item_ttl_seconds}; expected "
                    f"{self._item_ttl_seconds}"
                )
        except BaseException:
            await store.aclose()
            raise
        return store

    def _bound_store(self, thread_id: str) -> FoundryStateStore:
        return FoundryStateStore(
            self._store_name(thread_id),
            self._credential,
            self._endpoint,
            user_isolation=self._user_isolation,
            item_ttl_seconds=self._item_ttl_seconds,
            api_version=self._api_version,
            # AgentServer uses this callback to build the outbound User-Agent
            # header; lazy evaluation keeps prefixes and opt-out state current.
            get_server_version=get_user_agent,
        )

    async def _checkpoint_tuple(
        self,
        store: FoundryStateStore,
        thread_id: str,
        value: _CheckpointItemValue,
    ) -> CheckpointTuple:
        checkpoint_id = value["checkpoint_id"]
        checkpoint_ns = value.get("checkpoint_ns", "")
        parent_checkpoint_id = value.get("parent_checkpoint_id", "")
        pending_writes = await self._pending_writes(
            store,
            checkpoint_ns,
            checkpoint_id,
        )
        return CheckpointTuple(
            config=self._config(thread_id, checkpoint_ns, checkpoint_id),
            checkpoint=cast(Checkpoint, self._deserialize(value["checkpoint"])),
            metadata=cast(CheckpointMetadata, self._deserialize(value["metadata"])),
            parent_config=(
                self._config(thread_id, checkpoint_ns, parent_checkpoint_id)
                if parent_checkpoint_id
                else None
            ),
            pending_writes=pending_writes,
        )

    async def _pending_writes(
        self,
        store: FoundryStateStore,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> list[tuple[str, str, Any]]:
        records: list[_PendingWriteItemValue] = []
        after: str | None = None
        tags: _PendingWriteItemTags = {
            "kind": "write",
            "ns": checkpoint_ns,
            "ckpt": checkpoint_id,
        }
        while True:
            page = await store.list_keys(
                tags=cast(Mapping[str, str], tags),
                limit=_PAGE_SIZE,
                order="asc",
                after=after,
            )
            for key in page.keys:
                item = await store.get_item(key.key)
                if item is not None:
                    records.append(cast(_PendingWriteItemValue, item.value))
            if not page.has_more or page.last_id is None:
                break
            after = page.last_id

        records.sort(key=lambda item: (item["task_id"], item["index"]))
        return [
            (
                item["task_id"],
                item["channel"],
                self._deserialize(item["value"]),
            )
            for item in records
        ]

    def _serialize(self, value: Any) -> _SerializedValue:
        type_name, data = self.serde.dumps_typed(value)
        return {
            "type": type_name,
            "data": base64.b64encode(data).decode("ascii"),
        }

    def _deserialize(self, value: _SerializedValue) -> Any:
        return self.serde.loads_typed(
            (
                value["type"],
                base64.b64decode(value["data"].encode("ascii")),
            )
        )

    def _store_name(self, thread_id: str) -> str:
        name = f"{self._store_name_prefix}/{thread_id}"
        if len(name) <= _MAX_STORE_NAME_LENGTH:
            return name
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        prefix_length = _MAX_STORE_NAME_LENGTH - len(digest) - 1
        return f"{self._store_name_prefix[:prefix_length]}/{digest}"

    @staticmethod
    def _checkpoint_key(checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{checkpoint_ns}/{checkpoint_id}"

    @staticmethod
    def _write_key(
        checkpoint_ns: str,
        checkpoint_id: str,
        task_id: str,
        index: int,
    ) -> str:
        return f"{checkpoint_ns}/writes/{checkpoint_id}/{task_id}/{index}"

    @staticmethod
    def _thread_id(config: RunnableConfig) -> str:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("thread_id must be a non-empty string")
        return thread_id

    @classmethod
    def _thread_and_namespace(cls, config: RunnableConfig) -> tuple[str, str]:
        thread_id = cls._thread_id(config)
        checkpoint_ns = (config.get("configurable") or {}).get("checkpoint_ns", "")
        if not isinstance(checkpoint_ns, str):
            raise ValueError("checkpoint_ns must be a string")
        return thread_id, checkpoint_ns

    @staticmethod
    def _config(
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> RunnableConfig:
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
