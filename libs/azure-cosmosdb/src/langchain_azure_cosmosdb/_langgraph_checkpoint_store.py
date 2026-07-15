"""Azure CosmosDB implementation of LangGraph checkpointer (sync)."""

from __future__ import annotations

import base64
import logging
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity import CredentialUnavailableError, DefaultAzureCredential
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
    PendingWrite,
    get_checkpoint_id,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

logger = logging.getLogger(__name__)

USER_AGENT = "langchain-azure-cosmosdb-checkpoint"
COSMOSDB_KEY_SEPARATOR = "$"


def _validate_key_part(value: str, name: str) -> None:
    """Raise ValueError if *value* contains the key separator."""
    if COSMOSDB_KEY_SEPARATOR in value:
        raise ValueError(
            f"'{name}' must not contain the separator "
            f"'{COSMOSDB_KEY_SEPARATOR}': got '{value}'"
        )


class _CosmosSerializer:
    """Serializer wrapper for CosmosDB that base64-encodes binary data."""

    def __init__(self, serde: SerializerProtocol) -> None:
        self.serde = serde

    def dumps_typed(self, obj: Any) -> tuple[str, str]:
        """Serialize an object and base64-encode the binary data.

        Args:
            obj: The object to serialize.

        Returns:
            A tuple of (type_name, base64_encoded_data).
        """
        type_, data = self.serde.dumps_typed(obj)
        data_base64 = base64.b64encode(data).decode("utf-8")
        return type_, data_base64

    def loads_typed(self, data: tuple[str, str]) -> Any:
        """Deserialize base64-encoded data back into an object.

        Args:
            data: A tuple of (type_name, base64_encoded_data).

        Returns:
            The deserialized object.
        """
        type_name, serialized_obj = data
        serialized_bytes = base64.b64decode(serialized_obj.encode("utf-8"))
        return self.serde.loads_typed((type_name, serialized_bytes))


def _make_checkpoint_key(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
    """Create a checkpoint key for CosmosDB."""
    return COSMOSDB_KEY_SEPARATOR.join(
        ["checkpoint", thread_id, checkpoint_ns, checkpoint_id]
    )


def _make_checkpoint_writes_key(
    thread_id: str,
    checkpoint_ns: str,
    checkpoint_id: str,
    task_id: str,
    idx: int | None,
) -> str:
    """Create a writes key for CosmosDB."""
    if idx is None:
        return COSMOSDB_KEY_SEPARATOR.join(
            ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id]
        )
    return COSMOSDB_KEY_SEPARATOR.join(
        ["writes", thread_id, checkpoint_ns, checkpoint_id, task_id, str(idx)]
    )


def _parse_checkpoint_key(cosmosdb_key: str) -> dict[str, str]:
    """Parse a checkpoint key."""
    parts = cosmosdb_key.split(COSMOSDB_KEY_SEPARATOR)
    if len(parts) != 4:
        raise ValueError(f"Invalid checkpoint key format: {cosmosdb_key}")
    namespace, thread_id, checkpoint_ns, checkpoint_id = parts
    if namespace != "checkpoint":
        raise ValueError("Expected checkpoint key to start with 'checkpoint'")
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
    }


def _parse_checkpoint_writes_key(cosmosdb_key: str) -> dict[str, str]:
    """Parse a writes key."""
    parts = cosmosdb_key.split(COSMOSDB_KEY_SEPARATOR)
    if len(parts) != 6:
        raise ValueError(f"Invalid writes key format: {cosmosdb_key}")
    namespace, thread_id, checkpoint_ns, checkpoint_id, task_id, idx = parts
    if namespace != "writes":
        raise ValueError("Expected writes key to start with 'writes'")
    return {
        "thread_id": thread_id,
        "checkpoint_ns": checkpoint_ns,
        "checkpoint_id": checkpoint_id,
        "task_id": task_id,
        "idx": idx,
    }


def _load_writes(
    serde: _CosmosSerializer, task_id_to_data: dict[tuple[str, str], dict]
) -> list[tuple[str, str, Any]]:
    """Load pending writes from CosmosDB data."""
    return [
        (task_id, data["channel"], serde.loads_typed((data["type"], data["value"])))
        for (task_id, _), data in task_id_to_data.items()
    ]


def _load_sorted_writes(
    serde: _CosmosSerializer, write_docs: list[dict[str, Any]]
) -> list[PendingWrite]:
    """Decode writes documents for a single checkpoint, ordered by ``idx``.

    Args:
        serde: The CosmosDB serializer used to decode write values.
        write_docs: Raw writes documents (each with an ``id`` writes key) that all
            belong to the same checkpoint.

    Returns:
        The pending writes ordered by their integer ``idx``.
    """
    parsed = [(doc, _parse_checkpoint_writes_key(doc["id"])) for doc in write_docs]
    return _load_writes(
        serde,
        {
            (parsed_key["task_id"], parsed_key["idx"]): doc
            for doc, parsed_key in sorted(parsed, key=lambda pair: int(pair[1]["idx"]))
        },
    )


def _parse_checkpoint_data(
    serde: _CosmosSerializer,
    key: str,
    data: dict,
    pending_writes: list[tuple[str, str, Any]] | None = None,
) -> CheckpointTuple | None:
    """Parse checkpoint data from CosmosDB."""
    if not data:
        return None

    parsed_key = _parse_checkpoint_key(key)
    thread_id = parsed_key["thread_id"]
    checkpoint_ns = parsed_key["checkpoint_ns"]
    checkpoint_id = parsed_key["checkpoint_id"]

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    checkpoint = serde.loads_typed((data["type"], data["checkpoint"]))
    metadata = serde.loads_typed(data["metadata"])
    parent_checkpoint_id = data.get("parent_checkpoint_id", "")

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


def _make_writes_partition_prefix(thread_id: str, checkpoint_ns: str) -> str:
    """Build the shared prefix for all writes partition keys of a thread/ns.

    Writes documents are stored with a ``partition_key`` of
    ``writes$<thread>$<ns>$<checkpoint_id>$``, so this prefix (with its trailing
    separator) matches every checkpoint's writes for the thread/ns without
    matching a different namespace that merely shares a leading substring.

    Args:
        thread_id: The thread identifier.
        checkpoint_ns: The checkpoint namespace.

    Returns:
        The partition-key prefix ending in the key separator.
    """
    return COSMOSDB_KEY_SEPARATOR.join(["writes", thread_id, checkpoint_ns, ""])


def _group_pending_writes(
    serde: _CosmosSerializer, write_docs: list[dict[str, Any]]
) -> dict[str, list[PendingWrite]]:
    """Group writes documents by ``checkpoint_id`` into pending-writes lists.

    Mirrors the per-checkpoint ordering used by ``_load_pending_writes``: within
    each checkpoint the writes are sorted by integer ``idx`` before decoding.

    Args:
        serde: The CosmosDB serializer used to decode write values.
        write_docs: Raw writes documents (each with an ``id`` writes key).

    Returns:
        Mapping from ``checkpoint_id`` to its ordered list of pending writes.
    """
    docs_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for doc in write_docs:
        checkpoint_id = _parse_checkpoint_writes_key(doc["id"])["checkpoint_id"]
        docs_by_checkpoint.setdefault(checkpoint_id, []).append(doc)

    return {
        checkpoint_id: _load_sorted_writes(serde, docs)
        for checkpoint_id, docs in docs_by_checkpoint.items()
    }


def _order_on_path_ancestors(
    docs_by_checkpoint: dict[str, dict[str, Any]], target_checkpoint_id: str
) -> list[dict[str, Any]]:
    """Walk ``parent_checkpoint_id`` from the target's parent to the root.

    The base ``get_delta_channel_history`` starts at the target checkpoint's
    parent (not the target itself), so the target is excluded here.

    Args:
        docs_by_checkpoint: Mapping from ``checkpoint_id`` to its checkpoint doc.
        target_checkpoint_id: The checkpoint whose ancestors to collect.

    Returns:
        On-path ancestor documents ordered parent -> root.
    """
    ancestors: list[dict[str, Any]] = []
    target_doc = docs_by_checkpoint.get(target_checkpoint_id)
    if target_doc is None:
        return ancestors

    seen: set[str] = {target_checkpoint_id}
    cursor_id = target_doc.get("parent_checkpoint_id") or None
    while cursor_id and cursor_id not in seen:
        doc = docs_by_checkpoint.get(cursor_id)
        if doc is None:
            break
        ancestors.append(doc)
        seen.add(cursor_id)
        cursor_id = doc.get("parent_checkpoint_id") or None
    return ancestors


def _reconstruct_delta_channel_history(
    serde: _CosmosSerializer,
    channels: Sequence[str],
    ancestors: list[dict[str, Any]],
    writes_by_checkpoint: dict[str, list[PendingWrite]],
) -> dict[str, DeltaChannelHistory]:
    """Reconstruct per-channel delta history from ordered on-path ancestors.

    Mirrors ``BaseCheckpointSaver.get_delta_channel_history`` semantics exactly:
    per ancestor (parent -> root), collect ``reversed(pending_writes)`` for
    channels still remaining, then discard channels seeded by ``channel_values``.

    Args:
        serde: The CosmosDB serializer used to decode checkpoint documents.
        channels: The requested channel names.
        ancestors: On-path ancestor documents ordered parent -> root.
        writes_by_checkpoint: Mapping from ``checkpoint_id`` to pending writes.

    Returns:
        Per-channel ``DeltaChannelHistory`` for every name in ``channels``.
    """
    collected_by_ch: dict[str, list[PendingWrite]] = {c: [] for c in channels}
    seed_by_ch: dict[str, Any] = {}
    remaining: set[str] = set(channels)

    for doc in ancestors:
        if not remaining:
            break
        checkpoint_id = _parse_checkpoint_key(doc["id"])["checkpoint_id"]
        pending_writes = writes_by_checkpoint.get(checkpoint_id, [])
        for write in reversed(pending_writes):
            ch = write[1]
            if ch in remaining:
                collected_by_ch[ch].append(write)

        checkpoint = serde.loads_typed((doc["type"], doc["checkpoint"]))
        channel_values = checkpoint["channel_values"]
        for ch in list(remaining):
            if ch in channel_values:
                seed_by_ch[ch] = channel_values[ch]
                remaining.discard(ch)

    result: dict[str, DeltaChannelHistory] = {}
    for ch in channels:
        entry: DeltaChannelHistory = {"writes": list(reversed(collected_by_ch[ch]))}
        if ch in seed_by_ch:
            entry["seed"] = seed_by_ch[ch]
        result[ch] = entry
    return result


class CosmosDBSaverSync(BaseCheckpointSaver):
    """CosmosDB synchronous implementation of BaseCheckpointSaver.

    Uses environment variables for connection configuration.

    Args:
        database_name: Name of the CosmosDB database.
        container_name: Name of the CosmosDB container.

    Environment Variables:
        COSMOSDB_ENDPOINT: CosmosDB endpoint URL (required).
        COSMOSDB_KEY: CosmosDB access key (optional, uses
            DefaultAzureCredential if not provided).

    Example:
        >>> import os
        >>> os.environ["COSMOSDB_ENDPOINT"] = (
        ...     "https://your-account.documents.azure.com:443/"
        ... )
        >>> os.environ["COSMOSDB_KEY"] = "your_key"  # Optional
        >>>
        >>> checkpointer = CosmosDBSaverSync(
        ...     database_name="langgraph_db",
        ...     container_name="checkpoints",
        ... )
    """

    container: Any

    def __init__(
        self,
        database_name: str,
        container_name: str,
        *,
        endpoint: str | None = None,
        key: str | None = None,
        cosmos_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the CosmosDB sync checkpoint saver.

        Args:
            database_name: Name of the CosmosDB database.
            container_name: Name of the CosmosDB container.
            endpoint: CosmosDB endpoint URL. Falls back to
                ``COSMOSDB_ENDPOINT`` env var if not provided.
            key: CosmosDB access key. Falls back to ``COSMOSDB_KEY``
                env var if not provided. When absent,
                ``DefaultAzureCredential`` is used.
            cosmos_client_kwargs: Additional keyword arguments passed to
                the ``CosmosClient`` constructor (e.g. ``retry_options``).
        """
        super().__init__()

        resolved_endpoint = endpoint or os.getenv("COSMOSDB_ENDPOINT")
        if not resolved_endpoint:
            raise ValueError("COSMOSDB_ENDPOINT environment variable is not set")

        resolved_key = key or os.getenv("COSMOSDB_KEY")

        extra_kwargs = cosmos_client_kwargs or {}
        try:
            if resolved_key:
                self.client = CosmosClient(
                    resolved_endpoint,
                    resolved_key,
                    user_agent=USER_AGENT,
                    **extra_kwargs,
                )
            else:
                credential = DefaultAzureCredential()
                self.client = CosmosClient(
                    resolved_endpoint,
                    credential=credential,
                    user_agent=USER_AGENT,
                    **extra_kwargs,
                )
            self.database = self.client.create_database_if_not_exists(database_name)
            self.container = self.database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/partition_key"),
            )
        except CredentialUnavailableError as e:
            raise RuntimeError(
                "Failed to obtain default credentials. Ensure the environment is "
                "correctly configured for DefaultAzureCredential."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "An unexpected error occurred during CosmosClient initialization."
            ) from e

        self.cosmos_serde = _CosmosSerializer(self.serde)

    def close(self) -> None:
        """Close the underlying CosmosDB client."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()

    def __enter__(self) -> CosmosDBSaverSync:
        """Enter context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and close client."""
        self.close()

    @classmethod
    @contextmanager
    def from_conn_info(
        cls,
        *,
        endpoint: str,
        key: str,
        database_name: str,
        container_name: str,
    ) -> Iterator[CosmosDBSaverSync]:
        """Create a CosmosDBSaverSync from explicit connection info.

        Args:
            endpoint: The CosmosDB endpoint URL.
            key: The CosmosDB access key.
            database_name: Name of the CosmosDB database.
            container_name: Name of the CosmosDB container.

        Yields:
            A configured saver instance.
        """
        yield cls(database_name, container_name, endpoint=endpoint, key=key)

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to CosmosDB.

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

        self.container.upsert_item(data)

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
                self.container.upsert_item(data)
            else:
                try:
                    self.container.create_item(data)
                except CosmosHttpResponseError as e:
                    if e.status_code != 409:
                        logger.error(
                            "Unexpected error (%s): %s",
                            e.status_code,
                            e.message,
                        )
                        raise

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Fetch a checkpoint tuple from CosmosDB.

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

        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")
        checkpoint_key = self._get_checkpoint_key(
            self.container, thread_id, checkpoint_ns, checkpoint_id
        )

        if not checkpoint_key:
            return None

        checkpoint_id = _parse_checkpoint_key(checkpoint_key)["checkpoint_id"]

        query = (
            "SELECT * FROM c "
            "WHERE c.partition_key=@partition_key AND c.id=@checkpoint_key"
        )
        parameters = [
            {"name": "@partition_key", "value": partition_key},
            {"name": "@checkpoint_key", "value": checkpoint_key},
        ]
        items = list(
            self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=partition_key,
            )
        )
        checkpoint_data = items[0] if items else {}

        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )

        return _parse_checkpoint_data(
            self.cosmos_serde,
            checkpoint_key,
            checkpoint_data,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from CosmosDB.

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
        for data in self.container.query_items(
            query=query,
            parameters=parameters,
            partition_key=partition_key,
        ):
            if not (data and "checkpoint" in data and "metadata" in data):
                continue

            key = data["id"]
            checkpoint_id = _parse_checkpoint_key(key)["checkpoint_id"]

            checkpoint_tuple = _parse_checkpoint_data(self.cosmos_serde, key, data)
            if checkpoint_tuple is None:
                continue

            if filter:
                metadata = checkpoint_tuple.metadata or {}
                if not all(metadata.get(k) == v for k, v in filter.items()):
                    continue

            pending_writes = self._load_pending_writes(
                thread_id, checkpoint_ns, checkpoint_id
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

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        """Fast-path override of ``BaseCheckpointSaver.get_delta_channel_history``.

        Replaces the base per-ancestor ``get_tuple`` walk (O(N) serial round
        trips) with two bulk queries: one partition-scoped query for all
        checkpoint documents of the ``(thread_id, checkpoint_ns)`` partition and
        one cross-partition prefix query for all writes documents of the
        thread/ns. The parent chain is then walked in memory, preserving the
        base return contract exactly.

        Args:
            config: Configuration identifying the target checkpoint.
            channels: Channel names to walk for. Empty -> empty mapping.

        Returns:
            Per-channel ``DeltaChannelHistory`` for every name in ``channels``.
        """
        if not channels:
            return {}

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        _validate_key_part(thread_id, "thread_id")
        _validate_key_part(checkpoint_ns, "checkpoint_ns")

        checkpoint_key = self._get_checkpoint_key(
            self.container, thread_id, checkpoint_ns, get_checkpoint_id(config)
        )
        if not checkpoint_key:
            return _reconstruct_delta_channel_history(
                self.cosmos_serde, channels, [], {}
            )

        target_checkpoint_id = _parse_checkpoint_key(checkpoint_key)["checkpoint_id"]

        partition_key = _make_checkpoint_key(thread_id, checkpoint_ns, "")
        checkpoint_docs = list(
            self.container.query_items(
                query="SELECT * FROM c WHERE c.partition_key=@partition_key",
                parameters=[{"name": "@partition_key", "value": partition_key}],
                partition_key=partition_key,
            )
        )
        docs_by_checkpoint = {
            _parse_checkpoint_key(doc["id"])["checkpoint_id"]: doc
            for doc in checkpoint_docs
            if doc and "checkpoint" in doc
        }

        ancestors = _order_on_path_ancestors(docs_by_checkpoint, target_checkpoint_id)
        if not ancestors:
            return _reconstruct_delta_channel_history(
                self.cosmos_serde, channels, [], {}
            )

        writes_prefix = _make_writes_partition_prefix(thread_id, checkpoint_ns)
        write_docs = list(
            self.container.query_items(
                query=("SELECT * FROM c WHERE STARTSWITH(c.partition_key, @prefix)"),
                parameters=[{"name": "@prefix", "value": writes_prefix}],
            )
        )
        writes_by_checkpoint = _group_pending_writes(self.cosmos_serde, write_docs)

        return _reconstruct_delta_channel_history(
            self.cosmos_serde, channels, ancestors, writes_by_checkpoint
        )

    def _load_pending_writes(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> list[tuple[str, str, Any]]:
        """Load pending writes for a checkpoint."""
        partition_key = _make_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "", None
        )

        query = "SELECT * FROM c WHERE c.partition_key=@partition_key"
        parameters = [{"name": "@partition_key", "value": partition_key}]
        writes = list(
            self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=partition_key,
            )
        )

        return _load_sorted_writes(self.cosmos_serde, writes)

    def _get_checkpoint_key(
        self,
        container: Any,
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
        items = list(
            container.query_items(
                query=query,
                parameters=parameters,
                partition_key=partition_key,
            )
        )

        if not items:
            return None

        return items[0]["id"]


__all__ = ["CosmosDBSaverSync", "_validate_key_part"]
