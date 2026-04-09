"""Azure CosmosDB implementation of LangGraph cache (async)."""

from __future__ import annotations

import asyncio
import base64
import datetime
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from azure.cosmos import PartitionKey
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from azure.identity.aio import (
    DefaultAzureCredential as AsyncDefaultAzureCredential,
)
from langchain_azure_cosmosdb._langgraph_cache import _NS_SEPARATOR, _make_cache_key
from langgraph.cache.base import BaseCache, FullKey, Namespace, ValueT
from langgraph.checkpoint.serde.base import SerializerProtocol

USER_AGENT = "langchain-azure-cosmosdb-lgcache"


class CosmosDBCache(BaseCache[ValueT]):
    """Asynchronous CosmosDB implementation of LangGraph BaseCache.

    Uses the native ``azure.cosmos.aio`` async client for non-blocking I/O.
    Sync methods (``get``, ``set``, ``clear``) delegate to the async
    implementations via ``asyncio.run_coroutine_threadsafe``.

    Args:
        container: An already-created async ``ContainerProxy`` instance.
        serde: Optional custom serializer.

    Use ``from_conn_info`` to create an instance:

    Example:
        >>> async with CosmosDBCache.from_conn_info(
        ...     endpoint="https://your-account.documents.azure.com:443/",
        ...     key="your_key",
        ...     database_name="langgraph_db",
        ...     container_name="cache",
        ... ) as cache:
        ...     result = await cache.aget([(("ns",), "key1")])
    """

    def __init__(
        self,
        container: Any,
        *,
        serde: SerializerProtocol | None = None,
    ) -> None:
        """Initialize the CosmosDB async cache.

        Args:
            container: An already-created async ``ContainerProxy``.
            serde: Optional custom serializer.
        """
        super().__init__(serde=serde)
        self.container = container
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
    ) -> AsyncIterator[CosmosDBCache[ValueT]]:
        """Create a CosmosDBCache from explicit connection info.

        Args:
            endpoint: The CosmosDB endpoint URL.
            key: The CosmosDB access key. If omitted, uses
                AsyncDefaultAzureCredential.
            database_name: Name of the CosmosDB database.
            container_name: Name of the CosmosDB container.
            serde: Optional custom serializer.

        Yields:
            A configured async cache instance.
        """
        credential = key if key else AsyncDefaultAzureCredential()
        try:
            async with AsyncCosmosClient(
                endpoint, credential, user_agent=USER_AGENT
            ) as client:
                database = await client.create_database_if_not_exists(database_name)
                container = await database.create_container_if_not_exists(
                    id=container_name,
                    partition_key=PartitionKey(path="/ns"),
                )
                yield cls(container, serde=serde)
        finally:
            if not key and hasattr(credential, "close"):
                await credential.close()

    # ------------------------------------------------------------------ #
    # Async methods (primary implementation)                               #
    # ------------------------------------------------------------------ #

    async def aget(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Asynchronously get cached values for the given keys.

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
                item = await self.container.read_item(item=doc_id, partition_key=ns_str)
            except CosmosHttpResponseError:
                continue

            expiry = item.get("expiry")
            if expiry is not None and now > expiry:
                try:
                    await self.container.delete_item(item=doc_id, partition_key=ns_str)
                except CosmosHttpResponseError:
                    pass
                continue

            encoding = item["encoding"]
            raw = base64.b64decode(item["val"].encode("utf-8"))
            values[(ns_tuple, k)] = self.serde.loads_typed((encoding, raw))

        return values

    async def aset(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Asynchronously set cached values for the given keys.

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

            await self.container.upsert_item(data)

    async def aclear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Asynchronously delete cached values for given namespaces.

        If no namespaces are provided, clear all cached values.

        Args:
            namespaces: Optional sequence of namespace tuples to clear.
                If None, clears all entries.
        """
        if namespaces is None:
            query = "SELECT c.id, c.ns FROM c"
            items = await self._query_items(query, [])
            for item in items:
                await self.container.delete_item(
                    item=item["id"], partition_key=item["ns"]
                )
        else:
            for ns_tuple in namespaces:
                ns_str = _NS_SEPARATOR.join(ns_tuple)
                query = "SELECT c.id FROM c WHERE c.ns=@ns"
                parameters = [{"name": "@ns", "value": ns_str}]
                items = await self._query_items(query, parameters)
                for item in items:
                    await self.container.delete_item(
                        item=item["id"], partition_key=ns_str
                    )

    # ------------------------------------------------------------------ #
    # Sync bridge methods                                                  #
    # ------------------------------------------------------------------ #

    def get(self, keys: Sequence[FullKey]) -> dict[FullKey, ValueT]:
        """Get cached values synchronously.

        Note:
            Only supported from a background thread. From the main
            async thread use ``aget``.

        Args:
            keys: Sequence of (namespace, key) tuples.

        Returns:
            Dict mapping found keys to their cached values.
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop was captured. CosmosDBCache must be "
                "created within an async context (e.g., via "
                "from_conn_info) to use sync bridge methods."
            )
        try:
            if asyncio.get_running_loop() is self._loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to CosmosDBCache are only "
                    "allowed from a different thread. Use the async "
                    "interface instead."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(self.aget(keys), self._loop).result()

    def set(self, pairs: Mapping[FullKey, tuple[ValueT, int | None]]) -> None:
        """Set cached values synchronously.

        Note:
            Only supported from a background thread. From the main
            async thread use ``aset``.

        Args:
            pairs: Mapping of (namespace, key) to (value, ttl_seconds).
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop was captured. CosmosDBCache must be "
                "created within an async context (e.g., via "
                "from_conn_info) to use sync bridge methods."
            )
        try:
            if asyncio.get_running_loop() is self._loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to CosmosDBCache are only "
                    "allowed from a different thread. Use ``aset``."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(self.aset(pairs), self._loop).result()

    def clear(self, namespaces: Sequence[Namespace] | None = None) -> None:
        """Clear cached values synchronously.

        Note:
            Only supported from a background thread. From the main
            async thread use ``aclear``.

        Args:
            namespaces: Optional sequence of namespace tuples to clear.
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop was captured. CosmosDBCache must be "
                "created within an async context (e.g., via "
                "from_conn_info) to use sync bridge methods."
            )
        try:
            if asyncio.get_running_loop() is self._loop:
                raise asyncio.InvalidStateError(
                    "Synchronous calls to CosmosDBCache are only "
                    "allowed from a different thread. Use ``aclear``."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aclear(namespaces), self._loop
        ).result()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _query_items(
        self, query: str, parameters: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute a CosmosDB query and return all results."""
        results: list[dict[str, Any]] = []
        async for item in self.container.query_items(
            query=query,
            parameters=parameters,
        ):
            results.append(item)
        return results


__all__ = ["CosmosDBCache"]
