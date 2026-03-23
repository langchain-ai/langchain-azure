"""LangGraph checkpointer backed by Azure Table Storage."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import run_in_executor
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
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from langchain_azure_storage._table_utils import (
    chunk_data,
    make_checkpoint_row_key,
    make_writes_row_key,
    parse_checkpoint_row_key,
    reassemble_data,
)


class AzureTableStorageSaver(BaseCheckpointSaver):
    """A LangGraph checkpointer that persists state to Azure Table Storage.

    This checkpointer uses two Azure tables:

    * **checkpoints** – stores serialised graph checkpoints.
    * **checkpoint_writes** – stores intermediate (pending) writes.

    Large binary payloads are automatically chunked across multiple entity
    properties to stay within Azure Table Storage's 64 KB per-property
    limit.

    Examples:
        >>> from langchain_azure_storage.checkpointer import (
        ...     AzureTableStorageSaver,
        ... )
        >>> with AzureTableStorageSaver.from_conn_string(
        ...     "DefaultEndpointsProtocol=https;AccountName=...;..."
        ... ) as saver:
        ...     saver.setup()
        ...     # Use with LangGraph:
        ...     # graph = builder.compile(checkpointer=saver)
    """

    def __init__(
        self,
        *,
        conn_str: Optional[str] = None,
        endpoint: Optional[str] = None,
        credential: Any = None,
        checkpoint_table_name: str = "checkpoints",
        writes_table_name: str = "checkpointWrites",
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        """Initialise the checkpointer.

        Provide **either** ``conn_str`` **or** ``endpoint`` + ``credential``.

        Args:
            conn_str: An Azure Storage connection string.
            endpoint: The Azure Table Storage endpoint URL.
            credential: A credential accepted by ``TableServiceClient``.
            checkpoint_table_name: Name of the checkpoints table.
            writes_table_name: Name of the writes table.
            serde: Custom serializer.  Defaults to ``JsonPlusSerializer``.
        """
        super().__init__(serde=serde)
        if serde is None:
            self.serde = JsonPlusSerializer()

        from azure.data.tables import TableServiceClient

        if conn_str is not None:
            self._service_client = TableServiceClient.from_connection_string(
                conn_str=conn_str,
            )
        elif endpoint is not None:
            self._service_client = TableServiceClient(
                endpoint=endpoint,
                credential=credential,
            )
        else:
            raise ValueError("Provide either 'conn_str' or 'endpoint' + 'credential'.")

        self._checkpoint_table_name = checkpoint_table_name
        self._writes_table_name = writes_table_name
        self._checkpoint_table = self._service_client.get_table_client(
            checkpoint_table_name
        )
        self._writes_table = self._service_client.get_table_client(writes_table_name)

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_str: str,
        *,
        checkpoint_table_name: str = "checkpoints",
        writes_table_name: str = "checkpointWrites",
        serde: Optional[SerializerProtocol] = None,
    ) -> Iterator[AzureTableStorageSaver]:
        """Context manager that creates a saver from a connection string.

        Args:
            conn_str: Azure Storage connection string.
            checkpoint_table_name: Name of the checkpoints table.
            writes_table_name: Name of the writes table.
            serde: Custom serializer.

        Yields:
            A configured ``AzureTableStorageSaver`` instance.
        """
        saver = cls(
            conn_str=conn_str,
            checkpoint_table_name=checkpoint_table_name,
            writes_table_name=writes_table_name,
            serde=serde,
        )
        try:
            yield saver
        finally:
            saver.close()

    def close(self) -> None:
        """Release underlying HTTP resources."""
        self._service_client.close()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create the backing Azure tables if they do not already exist."""
        self._service_client.create_table_if_not_exists(self._checkpoint_table_name)
        self._service_client.create_table_if_not_exists(self._writes_table_name)

    # ------------------------------------------------------------------
    # Sync interface
    # ------------------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Fetch a checkpoint tuple from Azure Table Storage.

        Args:
            config: Runnable configuration with ``thread_id`` and
                optionally ``checkpoint_id``.

        Returns:
            The matching ``CheckpointTuple``, or ``None``.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        if checkpoint_id := get_checkpoint_id(config):
            row_key = make_checkpoint_row_key(checkpoint_ns, checkpoint_id)
            try:
                entity = self._checkpoint_table.get_entity(
                    partition_key=thread_id,
                    row_key=row_key,
                )
            except Exception:
                return None
            return self._entity_to_checkpoint_tuple(entity, thread_id)
        else:
            # Fetch the latest checkpoint for this thread + namespace.
            rk_lower = make_checkpoint_row_key(checkpoint_ns, "")
            rk_upper = make_checkpoint_row_key(checkpoint_ns + chr(0x10FFFF), "")
            odata_filter = (
                f"PartitionKey eq '{_odata_escape(thread_id)}'"
                f" and RowKey ge '{_odata_escape(rk_lower)}'"
                f" and RowKey lt '{_odata_escape(rk_upper)}'"
            )
            entities = list(
                self._checkpoint_table.query_entities(
                    query_filter=odata_filter,
                )
            )
            if not entities:
                return None
            # checkpoint_id is a UUID6; lexicographic max == newest.
            entities.sort(key=lambda e: e["RowKey"], reverse=True)
            return self._entity_to_checkpoint_tuple(entities[0], thread_id)

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints that match the given criteria.

        Args:
            config: Base configuration for filtering.
            filter: Additional metadata filter criteria.
            before: Only return checkpoints created before this config.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching ``CheckpointTuple`` instances, newest first.
        """
        filter_parts: list[str] = []

        thread_id: Optional[str] = None

        if config is not None:
            conf = config["configurable"]
            if "thread_id" in conf:
                thread_id = conf["thread_id"]
                tid: str = thread_id  # type: ignore[assignment]
                filter_parts.append(f"PartitionKey eq '{_odata_escape(tid)}'")
            if "checkpoint_ns" in conf:
                ns_val: str = conf["checkpoint_ns"]
                ns_prefix = make_checkpoint_row_key(ns_val, "")
                ns_upper = make_checkpoint_row_key(ns_val + chr(0x10FFFF), "")
                filter_parts.append(f"RowKey ge '{_odata_escape(ns_prefix)}'")
                filter_parts.append(f"RowKey lt '{_odata_escape(ns_upper)}'")
        if before is not None:
            before_id = before["configurable"]["checkpoint_id"]
            before_ns = before["configurable"].get("checkpoint_ns", "")
            before_rk = make_checkpoint_row_key(before_ns, before_id)
            filter_parts.append(f"RowKey lt '{_odata_escape(before_rk)}'")

        odata_filter = " and ".join(filter_parts) if filter_parts else None

        entities = list(
            self._checkpoint_table.query_entities(
                query_filter=odata_filter or "",
            )
        )

        # Sort newest first (UUID6 RowKeys are lexicographically ordered).
        entities.sort(key=lambda e: e["RowKey"], reverse=True)

        # Apply metadata filter in-memory (Azure Table Storage does not
        # support querying inside serialized JSON blobs).
        if filter:
            entities = [e for e in entities if self._metadata_matches(e, filter)]

        if limit is not None:
            entities = entities[:limit]

        for entity in entities:
            pk = entity["PartitionKey"]
            yield self._entity_to_checkpoint_tuple(entity, pk)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration after storing.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            metadata,
            default=_json_default,
        )

        entity: dict[str, Any] = {
            "PartitionKey": thread_id,
            "RowKey": make_checkpoint_row_key(checkpoint_ns, checkpoint_id),
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id", ""),
        }
        entity.update(chunk_data(serialized_checkpoint))

        self._checkpoint_table.upsert_entity(entity)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
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
            writes: List of ``(channel, value)`` pairs.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            write_idx = WRITES_IDX_MAP.get(channel, idx)
            type_, serialized_value = self.serde.dumps_typed(value)

            entity: dict[str, Any] = {
                "PartitionKey": thread_id,
                "RowKey": make_writes_row_key(
                    checkpoint_ns, checkpoint_id, task_id, write_idx
                ),
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "channel": channel,
                "type": type_,
                "idx": write_idx,
            }
            entity.update(chunk_data(serialized_value))

            self._writes_table.upsert_entity(entity)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread.

        Args:
            thread_id: The thread ID to delete.
        """
        pk_filter = f"PartitionKey eq '{_odata_escape(thread_id)}'"

        for entity in self._checkpoint_table.query_entities(
            query_filter=pk_filter, select=["PartitionKey", "RowKey"]
        ):
            self._checkpoint_table.delete_entity(
                partition_key=entity["PartitionKey"],
                row_key=entity["RowKey"],
            )

        for entity in self._writes_table.query_entities(
            query_filter=pk_filter, select=["PartitionKey", "RowKey"]
        ):
            self._writes_table.delete_entity(
                partition_key=entity["PartitionKey"],
                row_key=entity["RowKey"],
            )

    # ------------------------------------------------------------------
    # Async interface (delegates to sync via run_in_executor)
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async version of ``get_tuple``."""
        return await run_in_executor(None, self.get_tuple, config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of ``list``."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _run() -> None:
            try:
                for item in self.list(
                    config, filter=filter, before=before, limit=limit
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        await run_in_executor(None, _run)
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version of ``put``."""
        return await run_in_executor(
            None, self.put, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of ``put_writes``."""
        return await run_in_executor(
            None, self.put_writes, config, writes, task_id, task_path
        )

    async def adelete_thread(self, thread_id: str) -> None:
        """Async version of ``delete_thread``."""
        return await run_in_executor(None, self.delete_thread, thread_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _entity_to_checkpoint_tuple(
        self, entity: dict[str, Any], thread_id: str
    ) -> CheckpointTuple:
        """Convert an Azure Table entity into a ``CheckpointTuple``."""
        row_key = entity["RowKey"]
        checkpoint_ns, checkpoint_id = parse_checkpoint_row_key(row_key)

        checkpoint_data = reassemble_data(entity)
        checkpoint = self.serde.loads_typed((entity["type"], checkpoint_data))

        raw_metadata = json.loads(entity.get("metadata", "{}"))
        metadata: CheckpointMetadata = raw_metadata

        config_values = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }

        # Fetch pending writes for this checkpoint.
        writes_filter = (
            f"PartitionKey eq '{_odata_escape(thread_id)}'"
            f" and checkpoint_ns eq '{_odata_escape(checkpoint_ns)}'"
            f" and checkpoint_id eq '{_odata_escape(checkpoint_id)}'"
        )
        pending_writes: list[tuple[str, str, Any]] = []
        for w in self._writes_table.query_entities(query_filter=writes_filter):
            val = reassemble_data(w)
            pending_writes.append(
                (
                    w["task_id"],
                    w["channel"],
                    self.serde.loads_typed((w["type"], val)),
                )
            )

        parent_id = entity.get("parent_checkpoint_id", "")
        parent_config: Optional[RunnableConfig] = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        return CheckpointTuple(
            config={"configurable": config_values},
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _metadata_matches(
        self, entity: dict[str, Any], filter_dict: dict[str, Any]
    ) -> bool:
        """Check whether an entity's metadata matches the given filter."""
        meta = json.loads(entity.get("metadata", "{}"))
        for key, value in filter_dict.items():
            if key not in meta or meta[key] != value:
                return False
        return True


def _odata_escape(value: str) -> str:
    """Escape a string for use inside an OData single-quoted literal."""
    return value.replace("'", "''")


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for bytes and tuples."""
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
