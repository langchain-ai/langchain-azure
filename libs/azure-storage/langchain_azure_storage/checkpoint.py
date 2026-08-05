"""Azure Table Storage implementation of a LangGraph checkpointer."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, Optional, Union

import azure.core.credentials
import azure.core.credentials_async
import azure.identity
import azure.identity.aio
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableClient, UpdateMode
from azure.data.tables.aio import TableClient as AsyncTableClient
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
from langgraph.checkpoint.serde.base import SerializerProtocol

from langchain_azure_storage._user_agent import USER_AGENT

_KEY_SEPARATOR = "$"
# Azure Table Storage forbids these characters in PartitionKey/RowKey, on top
# of the separator we use to join key parts.
_CONTROL_CHARS = "".join(chr(c) for c in (*range(0x00, 0x20), *range(0x7F, 0xA0)))
_FORBIDDEN_KEY_CHARS = re.compile("[" + re.escape("/\\#?" + _CONTROL_CHARS) + "]")
# Table Storage caps String/Binary properties at 64 KiB each; stay safely
# under that so large checkpoints/writes can be split across properties.
_MAX_CHUNK_SIZE = 63 * 1024

# Credential types accepted by the saver.
_SDK_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.AzureNamedKeyCredential,
        azure.core.credentials.TokenCredential,
        azure.core.credentials_async.AsyncTokenCredential,
    ]
]
# Narrower views of the above, matching what the sync/async TableClient
# constructors actually accept, once `_resolve_*_credential` has ruled out
# the other mode's credential type.
_SYNC_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.AzureNamedKeyCredential,
        azure.core.credentials.TokenCredential,
    ]
]
_ASYNC_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.AzureNamedKeyCredential,
        azure.core.credentials_async.AsyncTokenCredential,
    ]
]


def _validate_key_part(value: str, name: str) -> None:
    """Raise ValueError if *value* is unsafe to use in a Table Storage key."""
    if _KEY_SEPARATOR in value:
        raise ValueError(
            f"'{name}' must not contain the separator '{_KEY_SEPARATOR}': got {value!r}"
        )
    if _FORBIDDEN_KEY_CHARS.search(value):
        raise ValueError(
            f"'{name}' must not contain '/', '\\', '#', '?', or control "
            f"characters (Azure Table Storage key restrictions): got {value!r}"
        )


def _partition_key(thread_id: str, checkpoint_ns: str) -> str:
    """Build the shared PartitionKey for a thread_id/checkpoint_ns pair."""
    return f"{thread_id}{_KEY_SEPARATOR}{checkpoint_ns}"


def _checkpoint_row_key(checkpoint_id: str) -> str:
    """Build the RowKey for a checkpoint entity."""
    return f"checkpoint{_KEY_SEPARATOR}{checkpoint_id}"


def _writes_row_key(checkpoint_id: str, task_id: str, idx: Any) -> str:
    """Build the RowKey for a pending-writes entity."""
    if idx is None:
        return f"writes{_KEY_SEPARATOR}{checkpoint_id}{_KEY_SEPARATOR}{task_id}"
    return (
        f"writes{_KEY_SEPARATOR}{checkpoint_id}{_KEY_SEPARATOR}{task_id}"
        f"{_KEY_SEPARATOR}{idx}"
    )


def _parse_checkpoint_row_key(row_key: str) -> str:
    """Parse a checkpoint entity's RowKey, returning its checkpoint_id."""
    prefix, sep, checkpoint_id = row_key.partition(_KEY_SEPARATOR)
    if prefix != "checkpoint" or not sep:
        raise ValueError(f"Invalid checkpoint row key: {row_key!r}")
    return checkpoint_id


def _parse_writes_row_key(row_key: str) -> tuple[str, str, str]:
    """Parse a writes entity's RowKey into (checkpoint_id, task_id, idx)."""
    parts = row_key.split(_KEY_SEPARATOR)
    if len(parts) != 4 or parts[0] != "writes":
        raise ValueError(f"Invalid writes row key: {row_key!r}")
    _, checkpoint_id, task_id, idx = parts
    return checkpoint_id, task_id, idx


def _put_chunked(entity: dict[str, Any], prefix: str, type_: str, data: bytes) -> None:
    """Split *data* across ``{prefix}_0``, ``{prefix}_1``, ... properties.

    Table Storage caps each String/Binary property at 64 KiB and each entity
    at 1 MiB. Splitting large payloads across multiple binary properties
    keeps individual properties under that per-property cap; the service
    itself will raise a clear error if the total entity size (all chunks
    combined) exceeds the 1 MiB entity cap.
    """
    chunks = (
        [data[i : i + _MAX_CHUNK_SIZE] for i in range(0, len(data), _MAX_CHUNK_SIZE)]
        if data
        else [b""]
    )
    entity[f"{prefix}_type"] = type_
    entity[f"{prefix}_n"] = len(chunks)
    for i, chunk in enumerate(chunks):
        entity[f"{prefix}_{i}"] = chunk


def _get_chunked(entity: dict[str, Any], prefix: str) -> tuple[str, bytes]:
    """Reassemble a value previously stored with `_put_chunked`."""
    type_ = entity[f"{prefix}_type"]
    n = entity[f"{prefix}_n"]
    data = b"".join(entity[f"{prefix}_{i}"] for i in range(n))
    return type_, data


def _checkpoint_tuple_from_entity(
    serde: SerializerProtocol,
    thread_id: str,
    checkpoint_ns: str,
    entity: dict[str, Any],
    pending_writes: list[tuple[str, str, Any]] | None,
) -> CheckpointTuple:
    """Build a CheckpointTuple from a checkpoint entity."""
    checkpoint_id = _parse_checkpoint_row_key(entity["RowKey"])
    checkpoint = serde.loads_typed(_get_chunked(entity, "checkpoint"))
    metadata = serde.loads_typed(_get_chunked(entity, "metadata"))
    parent_checkpoint_id = entity.get("parent_checkpoint_id") or ""

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }
    parent_config: RunnableConfig | None = (
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        if parent_checkpoint_id
        else None
    )
    return CheckpointTuple(
        config=config,
        checkpoint=checkpoint,
        metadata=metadata,
        parent_config=parent_config,
        pending_writes=pending_writes,
    )


def _checkpoint_entity(
    serde: SerializerProtocol,
    partition_key: str,
    thread_id: str,
    checkpoint_ns: str,
    checkpoint: Checkpoint,
    metadata: CheckpointMetadata,
    parent_checkpoint_id: str | None,
) -> dict[str, Any]:
    """Build the Table Storage entity for a checkpoint `put`."""
    entity: dict[str, Any] = {
        "PartitionKey": partition_key,
        "RowKey": _checkpoint_row_key(checkpoint["id"]),
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "parent_checkpoint_id": parent_checkpoint_id or "",
    }
    checkpoint_type, checkpoint_data = serde.dumps_typed(checkpoint)
    _put_chunked(entity, "checkpoint", checkpoint_type, checkpoint_data)
    metadata_type, metadata_data = serde.dumps_typed(metadata)
    _put_chunked(entity, "metadata", metadata_type, metadata_data)
    return entity


def _write_entity(
    serde: SerializerProtocol,
    partition_key: str,
    checkpoint_id: str,
    task_id: str,
    write_idx: int,
    channel: str,
    value: Any,
) -> dict[str, Any]:
    """Build the Table Storage entity for a single pending write."""
    entity: dict[str, Any] = {
        "PartitionKey": partition_key,
        "RowKey": _writes_row_key(checkpoint_id, task_id, write_idx),
        "channel": channel,
    }
    type_, data = serde.dumps_typed(value)
    _put_chunked(entity, "value", type_, data)
    return entity


def _checkpoints_query(partition_key: str, before_id: str | None) -> tuple[str, dict]:
    """Build the OData filter selecting all checkpoint entities in a partition."""
    query_filter = "PartitionKey eq @pk and RowKey ge @lo and RowKey lt @hi"
    parameters = {
        "pk": partition_key,
        "lo": f"checkpoint{_KEY_SEPARATOR}",
        "hi": f"checkpoint{_KEY_SEPARATOR}~",
    }
    if before_id:
        query_filter += " and RowKey lt @before"
        parameters["before"] = _checkpoint_row_key(before_id)
    return query_filter, parameters


def _writes_query(partition_key: str, checkpoint_id: str) -> tuple[str, dict]:
    """Build the OData filter selecting all writes for one checkpoint."""
    query_filter = "PartitionKey eq @pk and RowKey ge @lo and RowKey lt @hi"
    parameters = {
        "pk": partition_key,
        "lo": f"writes{_KEY_SEPARATOR}{checkpoint_id}{_KEY_SEPARATOR}",
        "hi": f"writes{_KEY_SEPARATOR}{checkpoint_id}{_KEY_SEPARATOR}~",
    }
    return query_filter, parameters


def _sort_writes(entities: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort write entities by their numeric idx (RowKey order is lexical)."""
    return sorted(entities, key=lambda e: int(_parse_writes_row_key(e["RowKey"])[2]))


def _writes_from_entities(
    serde: SerializerProtocol, entities: Sequence[dict[str, Any]]
) -> list[tuple[str, str, Any]]:
    """Convert sorted write entities into (task_id, channel, value) tuples."""
    result = []
    for entity in _sort_writes(entities):
        task_id = _parse_writes_row_key(entity["RowKey"])[1]
        value = serde.loads_typed(_get_chunked(entity, "value"))
        result.append((task_id, entity["channel"], value))
    return result


class AzureTableStorageSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver backed by Azure Table Storage.

    Stores both checkpoints and pending writes as entities in a single Azure
    Table, partitioned by ``thread_id``/``checkpoint_ns``. Large checkpoint,
    metadata, and write payloads are transparently split across multiple
    entity properties to work around Table Storage's 64 KiB per-property
    limit; the (rarely hit) 1 MiB per-entity limit is enforced by the
    service itself.

    The underlying Azure SDK clients are created lazily on first use and
    cached; call :meth:`close`/:meth:`aclose` (or use the saver as a context
    manager) to release them.

    Example:
        >>> from langchain_azure_storage.checkpoint import AzureTableStorageSaver
        >>> saver = AzureTableStorageSaver(
        ...     endpoint="https://<account>.table.core.windows.net",
        ...     table_name="checkpoints",
        ... )
        >>> config = {"configurable": {"thread_id": "thread-1"}}
        >>> checkpoint_tuple = saver.get_tuple(config)
    """

    def __init__(
        self,
        endpoint: str,
        table_name: str,
        *,
        credential: _SDK_CREDENTIAL_TYPE = None,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Create a new saver authenticating via endpoint + credential.

        Use :meth:`from_connection_string` instead to authenticate with a
        connection string (e.g. for the Azurite emulator).

        Args:
            endpoint: Table service endpoint, e.g.
                ``https://<account>.table.core.windows.net``.
            table_name: Name of the table to store checkpoints in. Created
                automatically on first use if it doesn't already exist.
            credential: Credential to authenticate with. If ``None``,
                ``DefaultAzureCredential`` is used.
            serde: Optional custom serializer.
        """
        super().__init__(serde=serde)
        self._endpoint = endpoint
        self._table_name = table_name
        self._credential = credential
        self._connection_string: str | None = None
        self._sync_table_client: TableClient | None = None
        self._async_table_client: AsyncTableClient | None = None
        self._sync_owned_credential: Any | None = None
        self._async_owned_credential: Any | None = None
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    @classmethod
    def from_connection_string(
        cls,
        connection_string: str,
        table_name: str,
        *,
        serde: SerializerProtocol | None = None,
    ) -> "AzureTableStorageSaver":
        """Create a new saver authenticating via a connection string.

        Intended for the `Azurite <https://learn.microsoft.com/azure/storage/common/storage-use-azurite>`_
        emulator, or any account where a connection string is more
        convenient than ``endpoint`` + ``credential``.

        Args:
            connection_string: Full connection string (e.g. from the Azure
                portal, or for the Azurite emulator).
            table_name: Name of the table to store checkpoints in.
            serde: Optional custom serializer.

        Returns:
            A new ``AzureTableStorageSaver`` authenticating via
            *connection_string*.
        """
        saver = cls("", table_name, serde=serde)
        saver._connection_string = connection_string
        return saver

    # ------------------------------------------------------------------ #
    # Resource lifecycle                                                  #
    # ------------------------------------------------------------------ #

    def _resolve_sync_credential(
        self, provided_credential: _SDK_CREDENTIAL_TYPE
    ) -> _SYNC_CREDENTIAL_TYPE:
        if provided_credential is None:
            credential = azure.identity.DefaultAzureCredential()
            self._sync_owned_credential = credential
            return credential
        if isinstance(
            provided_credential, azure.core.credentials_async.AsyncTokenCredential
        ):
            raise ValueError(
                "Cannot use synchronous methods when AzureTableStorageSaver is "
                "instantiated with an AsyncTokenCredential. Use the async "
                "methods instead, or supply a synchronous credential."
            )
        return provided_credential

    async def _resolve_async_credential(
        self, provided_credential: _SDK_CREDENTIAL_TYPE
    ) -> _ASYNC_CREDENTIAL_TYPE:
        if provided_credential is None:
            credential = azure.identity.aio.DefaultAzureCredential()
            self._async_owned_credential = credential
            return credential
        if not isinstance(
            provided_credential,
            (
                azure.core.credentials_async.AsyncTokenCredential,
                azure.core.credentials.AzureSasCredential,
                azure.core.credentials.AzureNamedKeyCredential,
            ),
        ):
            raise ValueError(
                "Cannot use asynchronous methods when AzureTableStorageSaver is "
                "instantiated with a synchronous TokenCredential. Use the sync "
                "methods instead, or supply an AsyncTokenCredential."
            )
        return provided_credential

    def _get_sync_table(self) -> TableClient:
        """Return the cached sync table client, creating it on first use."""
        if self._sync_table_client is not None:
            return self._sync_table_client
        with self._sync_lock:
            if self._sync_table_client is not None:
                return self._sync_table_client
            if self._connection_string:
                client = TableClient.from_connection_string(
                    self._connection_string,
                    self._table_name,
                    user_agent=USER_AGENT,
                )
            else:
                credential = self._resolve_sync_credential(self._credential)
                client = TableClient(
                    self._endpoint,
                    self._table_name,
                    credential=credential,
                    user_agent=USER_AGENT,
                )
            try:
                client.create_table()
            except ResourceExistsError:
                pass
            self._sync_table_client = client
            return client

    async def _get_async_table(self) -> AsyncTableClient:
        """Return the cached async table client, creating it on first use."""
        if self._async_table_client is not None:
            return self._async_table_client
        async with self._async_lock:
            if self._async_table_client is not None:
                return self._async_table_client
            if self._connection_string:
                client = AsyncTableClient.from_connection_string(
                    self._connection_string,
                    self._table_name,
                    user_agent=USER_AGENT,
                )
            else:
                credential = await self._resolve_async_credential(self._credential)
                client = AsyncTableClient(
                    self._endpoint,
                    self._table_name,
                    credential=credential,
                    user_agent=USER_AGENT,
                )
            try:
                await client.create_table()
            except ResourceExistsError:
                pass
            self._async_table_client = client
            return client

    def close(self) -> None:
        """Close the cached sync table client and any credential it owns."""
        if self._sync_table_client is not None:
            self._sync_table_client.close()
            self._sync_table_client = None
        if self._sync_owned_credential is not None:
            self._sync_owned_credential.close()
            self._sync_owned_credential = None

    async def aclose(self) -> None:
        """Close the cached async table client and any credential it owns."""
        if self._async_table_client is not None:
            await self._async_table_client.close()
            self._async_table_client = None
        if self._async_owned_credential is not None:
            await self._async_owned_credential.close()
            self._async_owned_credential = None

    def __enter__(self) -> "AzureTableStorageSaver":
        """Enter the context manager, returning this saver."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Exit the context manager, calling `close()`."""
        self.close()

    async def __aenter__(self) -> "AzureTableStorageSaver":
        """Enter the async context manager, returning this saver."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context manager, calling `aclose()`."""
        await self.aclose()

    # ------------------------------------------------------------------ #
    # Sync methods                                                        #
    # ------------------------------------------------------------------ #

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple from Azure Table Storage.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or None if not found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        table = self._get_sync_table()

        if checkpoint_id:
            try:
                entity = table.get_entity(
                    partition_key, _checkpoint_row_key(checkpoint_id)
                )
            except ResourceNotFoundError:
                return None
        else:
            query_filter, parameters = _checkpoints_query(partition_key, None)
            row_keys = [
                e["RowKey"]
                for e in table.query_entities(
                    query_filter,
                    parameters=parameters,
                    select=["PartitionKey", "RowKey"],
                )
            ]
            if not row_keys:
                return None
            entity = table.get_entity(partition_key, max(row_keys))
            checkpoint_id = _parse_checkpoint_row_key(entity["RowKey"])

        write_filter, write_parameters = _writes_query(partition_key, checkpoint_id)
        pending_writes = _writes_from_entities(
            self.serde,
            list(table.query_entities(write_filter, parameters=write_parameters)),
        )
        return _checkpoint_tuple_from_entity(
            self.serde, thread_id, checkpoint_ns, entity, pending_writes
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from Azure Table Storage.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching checkpoint tuples, most recent first.
        """
        if not config:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        before_id = get_checkpoint_id(before) if before else None
        table = self._get_sync_table()

        query_filter, parameters = _checkpoints_query(partition_key, before_id)
        entities = list(table.query_entities(query_filter, parameters=parameters))
        # ponytail: Table Storage has no server-side ORDER BY, so results are
        # sorted client-side. Fine for typical per-thread history sizes; add
        # a secondary index (or server-side paging) if a thread ever
        # accumulates enough checkpoints for this to matter.
        entities.sort(key=lambda e: e["RowKey"], reverse=True)

        count = 0
        for entity in entities:
            checkpoint_tuple = _checkpoint_tuple_from_entity(
                self.serde, thread_id, checkpoint_ns, entity, pending_writes=None
            )
            if filter:
                metadata = checkpoint_tuple.metadata or {}
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue

            checkpoint_id = _parse_checkpoint_row_key(entity["RowKey"])
            write_filter, write_parameters = _writes_query(partition_key, checkpoint_id)
            pending_writes = _writes_from_entities(
                self.serde,
                list(table.query_entities(write_filter, parameters=write_parameters)),
            )
            yield CheckpointTuple(
                config=checkpoint_tuple.config,
                checkpoint=checkpoint_tuple.checkpoint,
                metadata=checkpoint_tuple.metadata,
                parent_config=checkpoint_tuple.parent_config,
                pending_writes=pending_writes,
            )
            count += 1
            if limit is not None and count >= limit:
                return

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Azure Table Storage.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")
        _validate_key_part(checkpoint["id"], "checkpoint_id")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        entity = _checkpoint_entity(
            self.serde,
            partition_key,
            thread_id,
            checkpoint_ns,
            checkpoint,
            metadata,
            config["configurable"].get("checkpoint_id"),
        )
        table = self._get_sync_table()
        table.upsert_entity(entity, mode=UpdateMode.REPLACE)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes (not persisted).
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")
        _validate_key_part(checkpoint_id, "checkpoint_id")
        _validate_key_part(task_id, "task_id")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        is_upsert = all(w[0] in WRITES_IDX_MAP for w in writes)
        table = self._get_sync_table()

        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            entity = _write_entity(
                self.serde,
                partition_key,
                checkpoint_id,
                task_id,
                write_idx,
                channel,
                value,
            )
            if is_upsert:
                table.upsert_entity(entity, mode=UpdateMode.REPLACE)
            else:
                try:
                    table.create_entity(entity)
                except ResourceExistsError:
                    pass

    # ------------------------------------------------------------------ #
    # Async methods                                                       #
    # ------------------------------------------------------------------ #

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple from Azure Table Storage asynchronously.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or None if not found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        table = await self._get_async_table()

        if checkpoint_id:
            try:
                entity = await table.get_entity(
                    partition_key, _checkpoint_row_key(checkpoint_id)
                )
            except ResourceNotFoundError:
                return None
        else:
            query_filter, parameters = _checkpoints_query(partition_key, None)
            row_keys = [
                e["RowKey"]
                async for e in table.query_entities(
                    query_filter,
                    parameters=parameters,
                    select=["PartitionKey", "RowKey"],
                )
            ]
            if not row_keys:
                return None
            entity = await table.get_entity(partition_key, max(row_keys))
            checkpoint_id = _parse_checkpoint_row_key(entity["RowKey"])

        write_filter, write_parameters = _writes_query(partition_key, checkpoint_id)
        write_entities = [
            e
            async for e in table.query_entities(
                write_filter, parameters=write_parameters
            )
        ]
        pending_writes = _writes_from_entities(self.serde, write_entities)
        return _checkpoint_tuple_from_entity(
            self.serde, thread_id, checkpoint_ns, entity, pending_writes
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from Azure Table Storage asynchronously.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching checkpoint tuples, most recent first.
        """
        if not config:
            return
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")
        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        before_id = get_checkpoint_id(before) if before else None
        table = await self._get_async_table()

        query_filter, parameters = _checkpoints_query(partition_key, before_id)
        entities = [
            e async for e in table.query_entities(query_filter, parameters=parameters)
        ]
        entities.sort(key=lambda e: e["RowKey"], reverse=True)

        count = 0
        for entity in entities:
            checkpoint_tuple = _checkpoint_tuple_from_entity(
                self.serde, thread_id, checkpoint_ns, entity, pending_writes=None
            )
            if filter:
                metadata = checkpoint_tuple.metadata or {}
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue

            checkpoint_id = _parse_checkpoint_row_key(entity["RowKey"])
            write_filter, write_parameters = _writes_query(partition_key, checkpoint_id)
            write_entities = [
                e
                async for e in table.query_entities(
                    write_filter, parameters=write_parameters
                )
            ]
            pending_writes = _writes_from_entities(self.serde, write_entities)
            yield CheckpointTuple(
                config=checkpoint_tuple.config,
                checkpoint=checkpoint_tuple.checkpoint,
                metadata=checkpoint_tuple.metadata,
                parent_config=checkpoint_tuple.parent_config,
                pending_writes=pending_writes,
            )
            count += 1
            if limit is not None and count >= limit:
                return

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Azure Table Storage asynchronously.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")
        _validate_key_part(checkpoint["id"], "checkpoint_id")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        entity = _checkpoint_entity(
            self.serde,
            partition_key,
            thread_id,
            checkpoint_ns,
            checkpoint,
            metadata,
            config["configurable"].get("checkpoint_id"),
        )
        table = await self._get_async_table()
        await table.upsert_entity(entity, mode=UpdateMode.REPLACE)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint asynchronously.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes (not persisted).
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")
        _validate_key_part(checkpoint_id, "checkpoint_id")
        _validate_key_part(task_id, "task_id")

        partition_key = _partition_key(thread_id, checkpoint_ns)
        is_upsert = all(w[0] in WRITES_IDX_MAP for w in writes)
        table = await self._get_async_table()

        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            entity = _write_entity(
                self.serde,
                partition_key,
                checkpoint_id,
                task_id,
                write_idx,
                channel,
                value,
            )
            if is_upsert:
                await table.upsert_entity(entity, mode=UpdateMode.REPLACE)
            else:
                try:
                    await table.create_entity(entity)
                except ResourceExistsError:
                    pass


__all__ = ["AzureTableStorageSaver"]
