"""Azure CosmosDB implementation of LangGraph checkpointer (async)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from langchain_azure_cosmosdb._langgraph_checkpoint_store import (
    _CosmosSerializer,
    _load_writes,
    _make_checkpoint_key,
    _make_checkpoint_writes_key,
    _parse_checkpoint_data,
    _parse_checkpoint_key,
    _parse_checkpoint_writes_key,
    _validate_key_part,
)
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

USER_AGENT = "langchain-azure-cosmosdb-checkpoint"

logger = logging.getLogger(__name__)


class CosmosDBSaver(BaseCheckpointSaver):
    """Asynchronous CosmosDB implementation of BaseCheckpointSaver.

    Uses the native ``azure.cosmos.aio`` async client for non-blocking I/O.
    Sync methods (``get_tuple``, ``list``, ``put``, ``put_writes``)
    are provided for compatibility and delegate to the async implementations
    via ``asyncio.run_coroutine_threadsafe``.

    Args:
        container: An already-created async ``ContainerProxy`` instance.
        serde: Optional custom serializer.

    Use ``from_conn_info`` to create an instance:

    Example:
        >>> async with CosmosDBSaver.from_conn_info(
        ...     endpoint="https://your-account.documents.azure.com:443/",
        ...     key="your_key",
        ...     database_name="langgraph_db",
        ...     container_name="checkpoints",
        ... ) as saver:
        ...     config = {"configurable": {"thread_id": "thread-1"}}
        ...     checkpoint_tuple = await saver.aget_tuple(config)
    """

    def __init__(
        self,
        container: Any,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the CosmosDB async checkpoint saver.

        Args:
            container: An already-created async ``ContainerProxy`` instance.
            serde: Optional custom serializer.
        """
        super().__init__(serde=serde)
        self.container = container
        self.cosmos_serde = _CosmosSerializer(self.serde)
        self._loop: asyncio.AbstractEventLoop | None
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    @classmethod
    @asynccontextmanager
    async def from_conn_info(
        cls,
        *,
        endpoint: str,
        key: str | None = None,
        database_name: str,
        container_name: str,
        serde: SerializerProtocol | None = None,
        cosmos_client_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[CosmosDBSaver]:
        """Create a CosmosDBSaver from explicit connection info.

        Args:
            endpoint: The CosmosDB endpoint URL.
            key: The CosmosDB access key. If omitted, uses
                AsyncDefaultAzureCredential.
            database_name: Name of the CosmosDB database.
            container_name: Name of the CosmosDB container.
            serde: Optional custom serializer.
            cosmos_client_kwargs: Additional keyword arguments passed to
                the ``CosmosClient`` constructor (e.g. ``retry_options``).

        Yields:
            A configured async saver instance.
        """
        credential = key if key else AsyncDefaultAzureCredential()
        extra_kwargs = cosmos_client_kwargs or {}
        try:
            async with AsyncCosmosClient(
                endpoint, credential, user_agent=USER_AGENT, **extra_kwargs
            ) as client:
                database = await client.create_database_if_not_exists(database_name)
                container = await database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=PartitionKey(path="/partition_key"),
                )
                yield cls(container, serde=serde)
        finally:
            if not key and hasattr(credential, "close"):
                await credential.close()

    # ------------------------------------------------------------------ #
    # Async methods (primary implementation)                               #
    # ------------------------------------------------------------------ #

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple from CosmosDB asynchronously.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or None if not found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")

        checkpoint_key = await self._get_checkpoint_key(
            thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None

        checkpoint_id = _parse_checkpoint_key(checkpoint_key)["checkpoint_id"]
        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")

        query = (
            "SELECT * FROM c "
            "WHERE c.partition_key=@partition_key AND c.id=@checkpoint_key"
        )
        parameters = [
            {"name": "@partition_key", "value": partition_key},
            {"name": "@checkpoint_key", "value": checkpoint_key},
        ]
        items = await self._query_items(query, parameters, partition_key)
        checkpoint_data = items[0] if items else {}

        pending_writes = await self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        return _parse_checkpoint_data(
            self.cosmos_serde,
            checkpoint_key,
            checkpoint_data,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints from CosmosDB asynchronously.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching checkpoint tuples.
        """
        if not config:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")

        before_id: str | None = None
        if before:
            before_id = get_checkpoint_id(before)

        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")

        query = "SELECT * FROM c WHERE c.partition_key=@partition_key"
        parameters: list[dict[str, Any]] = [
            {"name": "@partition_key", "value": partition_key},
        ]

        if before_id:
            before_key = _make_checkpoint_key(thread_id, checkpoint_ns, before_id)
            query += " AND c.id < @before_key"
            parameters.append({"name": "@before_key", "value": before_key})

        query += " ORDER BY c.id DESC"

        if limit is not None and limit < 1:
            raise ValueError("limit must be a positive integer")

        if limit is not None and not filter:
            query = query.replace("SELECT *", f"SELECT TOP {int(limit)} *", 1)

        count = 0
        async for data in self.container.query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
        ):
            if not (data and "checkpoint" in data and "metadata" in data):
                continue

            key = data["id"]
            cp_id = _parse_checkpoint_key(key)["checkpoint_id"]

            checkpoint_tuple = _parse_checkpoint_data(self.cosmos_serde, key, data)
            if checkpoint_tuple is None:
                continue

            if filter:
                metadata = checkpoint_tuple.metadata or {}
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue

            pending_writes = await self._load_pending_writes(
                thread_id, checkpoint_ns, cp_id
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

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to CosmosDB asynchronously.

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
        checkpoint_id = checkpoint["id"]
        _validate_key_part(checkpoint_id, "checkpoint_id")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        key = _make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")

        type_, serialized_checkpoint = self.cosmos_serde.dumps_typed(checkpoint)
        serialized_metadata = self.cosmos_serde.dumps_typed(metadata)

        data = {
            "partition_key": partition_key,
            "id": key,
            "thread_id": thread_id,
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }

        await self.container.upsert_item(data)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
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
            task_path: Path of the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        _validate_key_part(checkpoint_id, "checkpoint_id")
        _validate_key_part(task_id, "task_id")

        is_upsert = all(w[0] in WRITES_IDX_MAP for w in writes)

        for idx, (channel, value) in enumerate(writes):
            key = _make_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            partition_key = _make_checkpoint_writes_key(
                thread_id, checkpoint_ns, checkpoint_id, "", None
            )

            type_, serialized_value = self.cosmos_serde.dumps_typed(value)

            data = {
                "partition_key": partition_key,
                "id": key,
                "thread_id": thread_id,
                "channel": channel,
                "type": type_,
                "value": serialized_value,
            }

            if is_upsert:
                await self.container.upsert_item(data)
            else:
                try:
                    await self.container.create_item(data)
                except CosmosHttpResponseError as e:
                    if e.status_code != 409:
                        logger.error(
                            "Unexpected error (%s): %s",
                            e.status_code,
                            e.message,
                        )
                        raise

    # ------------------------------------------------------------------ #
    # Sync bridge methods (delegate to async via run_coroutine_threadsafe) #
    # ------------------------------------------------------------------ #

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple synchronously.

        Note:
            Synchronous calls are only supported from a background thread.
            From the main async thread use ``aget_tuple``.

        Args:
            config: Configuration specifying which checkpoint to retrieve.

        Returns:
            The requested checkpoint tuple, or None if not found.
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop was captured. CosmosDBSaver must be "
                "created within an async context (e.g., via "
                "from_conn_info) to use sync bridge methods."
            )
        try:
            if asyncio.get_running_loop() is self._loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to CosmosDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async "
                    "interface. For example, use "
                    "`await checkpointer.aget_tuple(...)` or "
                    "`await graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aget_tuple(config), self._loop
        ).result()

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints synchronously.

        Note:
            Synchronous calls are only supported from a background thread.
            From the main async thread use ``alist``.

        Args:
            config: Base configuration for filtering checkpoints.
            filter: Additional filtering criteria.
            before: List checkpoints created before this configuration.
            limit: Maximum number of checkpoints to return.

        Yields:
            Matching checkpoint tuples.
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop was captured. CosmosDBSaver must be "
                "created within an async context (e.g., via "
                "from_conn_info) to use sync bridge methods."
            )
        try:
            if asyncio.get_running_loop() is self._loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to CosmosDBSaver are only allowed from a "
                    "different thread. From the main thread, use the async "
                    "interface. For example, use `checkpointer.alist(...)` or "
                    "`await graph.ainvoke(...)`."
                )
        except RuntimeError:
            pass
        aiter_ = self.alist(config, filter=filter, before=before, limit=limit)
        while True:
            try:
                yield asyncio.run_coroutine_threadsafe(
                    anext(aiter_),  # type: ignore[arg-type]
                    self._loop,
                ).result()
            except StopAsyncIteration:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint synchronously.

        Note:
            Synchronous calls are only supported from a background thread.
            From the main async thread use ``aput``.

        Args:
            config: Configuration for the checkpoint.
            checkpoint: The checkpoint to store.
            metadata: Additional metadata for the checkpoint.
            new_versions: New channel versions as of this write.

        Returns:
            Updated configuration after storing the checkpoint.
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop was captured. CosmosDBSaver must be "
                "created within an async context (e.g., via "
                "from_conn_info) to use sync bridge methods."
            )
        try:
            if asyncio.get_running_loop() is self._loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to CosmosDBSaver are only "
                    "allowed from a different thread. Use ``aput``."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aput(config, checkpoint, metadata, new_versions), self._loop
        ).result()

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Store intermediate writes synchronously.

        Note:
            Synchronous calls are only supported from a background thread.
            From the main async thread use ``aput_writes``.

        Args:
            config: Configuration of the related checkpoint.
            writes: List of writes to store.
            task_id: Identifier for the task creating the writes.
            task_path: Path of the task creating the writes.
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop was captured. CosmosDBSaver must be "
                "created within an async context (e.g., via "
                "from_conn_info) to use sync bridge methods."
            )
        try:
            if asyncio.get_running_loop() is self._loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to CosmosDBSaver are only "
                    "allowed from a different thread. "
                    "Use ``aput_writes``."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aput_writes(config, writes, task_id, task_path), self._loop
        ).result()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _query_items(
        self,
        query: str,
        parameters: list[dict[str, Any]],
        partition_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a CosmosDB query and return all results as a list."""
        kwargs: dict[str, Any] = {
            "query": query,
            "parameters": parameters,
        }
        if partition_key is not None:
            kwargs["partition_key"] = partition_key
        results: list[dict[str, Any]] = []
        async for item in self.container.query_items(**kwargs):
            results.append(item)
        return results

    async def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes for a checkpoint asynchronously."""
        partition_key = _make_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "", None
        )
        query = "SELECT * FROM c WHERE c.partition_key=@partition_key"
        parameters = [{"name": "@partition_key", "value": partition_key}]
        writes = await self._query_items(query, parameters, partition_key)

        parsed_keys = [_parse_checkpoint_writes_key(write["id"]) for write in writes]
        return _load_writes(
            self.cosmos_serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): write
                for write, parsed_key in sorted(
                    zip(writes, parsed_keys, strict=True),
                    key=lambda x: int(x[1]["idx"]),
                )
            },
        )

    async def _get_checkpoint_key(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str | None,
    ) -> str | None:
        """Get the checkpoint key, finding the latest if no ID given."""
        if checkpoint_id:
            return _make_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")
        query = (
            "SELECT TOP 1 c.id FROM c "
            "WHERE c.partition_key=@partition_key "
            "ORDER BY c.id DESC"
        )
        parameters = [{"name": "@partition_key", "value": partition_key}]
        items = await self._query_items(query, parameters, partition_key)

        if not items:
            return None

        return items[0]["id"]


__all__ = ["CosmosDBSaver"]
