"""Azure CosmosDB implementation of LangGraph cache (sync)."""

from __future__ import annotations

import base64
import datetime
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity import CredentialUnavailableError, DefaultAzureCredential
from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol

USER_AGENT = "langchain-azure-cosmosdb-lgcache"
_NS_SEPARATOR = "|"


class CosmosDBCacheSync(BaseCache[ValueT]):
    """Synchronous CosmosDB implementation of LangGraph BaseCache.

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
        >>> os.environ["COSMOSDB_KEY"] = "your_key"
        >>>
        >>> cache = CosmosDBCacheSync(
        ...     database_name="langgraph_db",
        ...     container_name="cache",
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
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the CosmosDB sync cache.

        Args:
            database_name: Name of the CosmosDB database.
            container_name: Name of the CosmosDB container.
            endpoint: CosmosDB endpoint URL. Falls back to
                ``COSMOSDB_ENDPOINT`` env var if not provided.
            key: CosmosDB access key. Falls back to ``COSMOSDB_KEY``
                env var if not provided. When absent,
                ``DefaultAzureCredential`` is used.
            serde: Optional custom serializer.
        """
        super().__init__(serde=serde)

        resolved_endpoint = endpoint or os.getenv("COSMOSDB_ENDPOINT")
        if not resolved_endpoint:
            raise ValueError("COSMOSDB_ENDPOINT environment variable is not set")

        resolved_key = key or os.getenv("COSMOSDB_KEY")

        try:
            if resolved_key:
                self.client = CosmosClient(
                    resolved_endpoint, resolved_key, user_agent=USER_AGENT
                )
            else:
                credential = DefaultAzureCredential()
                self.client = CosmosClient(
                    resolved_endpoint, credential=credential, user_agent=USER_AGENT
                )
            self.database = self.client.create_database_if_not_exists(database_name)
            self.container = self.database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/ns"),
            )
        except CredentialUnavailableError as e:
            raise RuntimeError(
                "Failed to obtain default credentials. Ensure the "
                "environment is correctly configured for "
                "DefaultAzureCredential."
            ) from e
        except Exception as e:
            raise RuntimeError(
                "An unexpected error occurred during " "CosmosClient initialization."
            ) from e

    @classmethod
    @contextmanager
    def from_conn_info(
        cls,
        *,
        endpoint: str,
        key: str,
        database_name: str,
        container_name: str,
        serde: SerializerProtocol | None = None,
    ) -> Iterator[CosmosDBCacheSync[ValueT]]:
        """Create a CosmosDBCacheSync from explicit connection info.

        Args:
            endpoint: The CosmosDB endpoint URL.
            key: The CosmosDB access key.
            database_name: Name of the CosmosDB database.
            container_name: Name of the CosmosDB container.
            serde: Optional custom serializer.

        Yields:
            A configured sync cache instance.
        """
        yield cls(
            database_name,
            container_name,
            endpoint=endpoint,
            key=key,
            serde=serde,
        )

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get the cached values for the given keys.

        Args:
            keys: Sequence of (namespace, key) tuples to look up.

        Returns:
            Dict mapping found keys to their cached values.
        """
        if not keys:
            return {}

        now = datetime.datetime.now(datetime.timezone.utc).timestamp()
        values: dict[FullKey, ValueT] = {}

        for ns_tuple, k in keys:
            ns_str = _NS_SEPARATOR.join(ns_tuple)
            doc_id = _make_cache_key(ns_str, k)
            try:
                item = self.container.read_item(item=doc_id, partition_key=ns_str)
            except CosmosHttpResponseError:
                continue

            expiry = item.get("expiry")
            if expiry is not None and now > expiry:
                # Purge expired entry
                try:
                    self.container.delete_item(item=doc_id, partition_key=ns_str)
                except CosmosHttpResponseError:
                    pass
                continue

            encoding = item["encoding"]
            raw = base64.b64decode(item["val"].encode("utf-8"))
            values[(ns_tuple, k)] = self.serde.loads_typed((encoding, raw))

        return values

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get cached values.

        Not supported in sync implementation. Use CosmosDBCache instead.

        Args:
            keys: Sequence of (namespace, key) tuples.

        Raises:
            NotImplementedError: Always. Use CosmosDBCache for async.
        """
        raise NotImplementedError("Use CosmosDBCache for async operations.")

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set cached values for the given keys and TTLs.

        Args:
            pairs: Mapping of (namespace, key) to (value, ttl_seconds).
                TTL of None means no expiration.
        """
        now = datetime.datetime.now(datetime.timezone.utc)

        for (ns_tuple, k), (value, ttl) in pairs.items():
            ns_str = _NS_SEPARATOR.join(ns_tuple)
            doc_id = _make_cache_key(ns_str, k)

            if ttl is not None:
                delta = datetime.timedelta(seconds=ttl)
                expiry: float | None = (now + delta).timestamp()
            else:
                expiry = None

            encoding, raw = self.serde.dumps_typed(value)
            raw_b64 = base64.b64encode(raw).decode("utf-8")

            data = {
                "id": doc_id,
                "ns": ns_str,
                "key": k,
                "expiry": expiry,
                "encoding": encoding,
                "val": raw_b64,
            }

            self.container.upsert_item(data)

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set cached values.

        Not supported in sync implementation. Use CosmosDBCache instead.

        Args:
            pairs: Mapping of (namespace, key) to (value, ttl_seconds).

        Raises:
            NotImplementedError: Always. Use CosmosDBCache for async.
        """
        raise NotImplementedError("Use CosmosDBCache for async operations.")

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Delete cached values for the given namespaces.

        If no namespaces are provided, clear all cached values.

        Args:
            namespaces: Optional sequence of namespace tuples to clear.
                If None, clears all entries.
        """
        if namespaces is None:
            # Delete all items
            query = "SELECT c.id, c.ns FROM c"
            items = list(
                self.container.query_items(
                    query=query,
                    enable_cross_partition_query=True,
                )
            )
            for item in items:
                self.container.delete_item(item=item["id"], partition_key=item["ns"])
        else:
            for ns_tuple in namespaces:
                ns_str = _NS_SEPARATOR.join(ns_tuple)
                query = "SELECT c.id FROM c WHERE c.ns=@ns"
                parameters = [{"name": "@ns", "value": ns_str}]
                items = list(
                    self.container.query_items(
                        query=query,
                        parameters=parameters,
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    self.container.delete_item(
                        item=item["id"],
                        partition_key=ns_str,
                    )

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete cached values.

        Not supported in sync implementation. Use CosmosDBCache instead.

        Args:
            namespaces: Optional sequence of namespace tuples to clear.

        Raises:
            NotImplementedError: Always. Use CosmosDBCache for async.
        """
        raise NotImplementedError("Use CosmosDBCache for async operations.")


def _make_cache_key(ns: str, key: str) -> str:
    """Create a document ID for a cache entry.

    Args:
        ns: The namespace string (pipe-joined).
        key: The cache key.

    Returns:
        A unique document ID.
    """
    return f"cache${ns}${key}"


__all__ = ["CosmosDBCacheSync"]
