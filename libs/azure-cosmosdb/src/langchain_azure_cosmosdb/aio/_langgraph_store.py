"""Azure CosmosDB implementation of LangGraph BaseStore (async).

Provides an asynchronous Cosmos DB-backed store for LangGraph long-term memory
with optional vector search support.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import orjson
from azure.cosmos.aio import ContainerProxy as AsyncContainerProxy
from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos.aio import DatabaseProxy as AsyncDatabaseProxy
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from langchain_azure_cosmosdb._langgraph_store import (
    USER_AGENT,
    BaseCosmosDBStore,
    CosmosDBIndexConfig,
    _decode_ns,
    _ensure_index_config,
    _group_ops,
    _namespace_to_text,
)
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
    TTLConfig,
    get_text_at_path,
    tokenize_path,
)
from langgraph.store.base.batch import AsyncBatchedBaseStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


logger = logging.getLogger(__name__)


class AsyncCosmosDBStore(AsyncBatchedBaseStore, BaseCosmosDBStore[AsyncCosmosClient]):
    """Async Azure Cosmos DB-backed store with optional vector search.

    Provides async LangGraph long-term memory persistence using Azure Cosmos DB.

    Example:
        Basic setup and usage::

            from langchain_azure_cosmosdb import AsyncCosmosDBStore

            async with AsyncCosmosDBStore.from_conn_string(
                conn_string="AccountEndpoint=https://...;AccountKey=...",
                database_name="langgraph",
                container_name="store",
            ) as store:
                await store.setup()
                await store.aput(("users", "123"), "prefs", {"theme": "dark"})
                item = await store.aget(("users", "123"), "prefs")

    Note:
        Semantic search is disabled by default. Provide an ``index``
        configuration when creating the store to enable it.

    Warning:
        Make sure to call ``setup()`` before first use.
    """

    __slots__ = (
        "_deserializer",
        "conn",
        "index_config",
        "embeddings",
        "ttl_config",
        "_database_name",
        "_container_name",
        "_database",
        "_container",
        "_ttl_sweeper_task",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: AsyncCosmosClient,
        *,
        database_name: str = "langgraph",
        container_name: str = "store",
        deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None = None,
        index: CosmosDBIndexConfig | None = None,
        ttl: TTLConfig | None = None,
    ) -> None:
        super().__init__()
        self.conn = conn
        self._database_name = database_name
        self._container_name = container_name
        self._deserializer = deserializer
        self.index_config = index
        if self.index_config:
            self.embeddings, self.index_config = _ensure_index_config(self.index_config)
        else:
            self.embeddings = None
        self.ttl_config = ttl
        self._database: AsyncDatabaseProxy | None = None
        self._container: AsyncContainerProxy | None = None
        self._ttl_sweeper_task: asyncio.Task[None] | None = None
        self._ttl_stop_event = asyncio.Event()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        database_name: str = "langgraph",
        container_name: str = "store",
        index: CosmosDBIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        cosmos_client_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[AsyncCosmosDBStore]:
        """Create a new AsyncCosmosDBStore from a connection string.

        Args:
            conn_string: Azure Cosmos DB connection string.
            database_name: Name of the database to use.
            container_name: Name of the container to use.
            index: Optional index/embedding configuration for vector search.
            ttl: Optional TTL configuration.
            cosmos_client_kwargs: Additional keyword arguments passed to
                the ``CosmosClient`` constructor (e.g. ``retry_options``).

        Yields:
            A new AsyncCosmosDBStore instance.
        """
        extra_kwargs = cosmos_client_kwargs or {}
        client = AsyncCosmosClient.from_connection_string(
            conn_string, user_agent=USER_AGENT, **extra_kwargs
        )
        try:
            store = cls(
                conn=client,
                database_name=database_name,
                container_name=container_name,
                index=index,
                ttl=ttl,
            )
            yield store
        finally:
            await client.close()

    @classmethod
    @asynccontextmanager
    async def from_endpoint(
        cls,
        endpoint: str,
        *,
        credential: Any | None = None,
        database_name: str = "langgraph",
        container_name: str = "store",
        index: CosmosDBIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        cosmos_client_kwargs: dict[str, Any] | None = None,
    ) -> AsyncIterator[AsyncCosmosDBStore]:
        """Create a new AsyncCosmosDBStore from an endpoint URL.

        Uses Microsoft Entra ID (DefaultAzureCredential) when no
        credential is provided.

        Args:
            endpoint: Azure Cosmos DB endpoint URL.
            credential: Optional credential. Uses DefaultAzureCredential
                if not provided.
            database_name: Name of the database to use.
            container_name: Name of the container to use.
            index: Optional index/embedding configuration for vector search.
            ttl: Optional TTL configuration.
            cosmos_client_kwargs: Additional keyword arguments passed to
                the ``CosmosClient`` constructor (e.g. ``retry_options``).

        Yields:
            A new AsyncCosmosDBStore instance.
        """
        if credential is None:
            from azure.identity.aio import (
                DefaultAzureCredential as AsyncDefaultAzureCredential,
            )

            credential = AsyncDefaultAzureCredential()
        extra_kwargs = cosmos_client_kwargs or {}
        client = AsyncCosmosClient(
            endpoint, credential=credential, user_agent=USER_AGENT, **extra_kwargs
        )
        try:
            store = cls(
                conn=client,
                database_name=database_name,
                container_name=container_name,
                index=index,
                ttl=ttl,
            )
            yield store
        finally:
            await client.close()

    @property
    def container(self) -> AsyncContainerProxy:
        """Get the container proxy."""
        if self._container is None:
            raise RuntimeError(
                "Store not initialized. Call setup() before using the store."
            )
        return self._container

    async def setup(self) -> None:
        """Set up the Cosmos DB database and container.

        Creates the database and container if they don't exist.
        Configures vector embedding policy and indexing policy when
        vector search is enabled.
        """
        from azure.cosmos import PartitionKey

        self._database = await self.conn.create_database_if_not_exists(
            self._database_name
        )

        partition_key = PartitionKey(path="/prefix")
        container_kwargs: dict[str, Any] = {}

        if self.ttl_config:
            container_kwargs["default_ttl"] = -1

        if self.index_config:
            dims = self.index_config["dims"]
            distance_type = self.index_config.get("distance_type", "cosine")

            cosmos_distance = {
                "cosine": "cosine",
                "euclidean": "euclidean",
                "dotproduct": "dotproduct",
            }.get(distance_type, "cosine")

            container_kwargs["vector_embedding_policy"] = {
                "vectorEmbeddings": [
                    {
                        "path": "/embedding",
                        "dataType": "float32",
                        "distanceFunction": cosmos_distance,
                        "dimensions": dims,
                    }
                ]
            }

            index_type = self.index_config.get("index_type", "quantizedFlat")
            container_kwargs["indexing_policy"] = {
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": "/embedding/*"}],
                "vectorIndexes": [{"path": "/embedding", "type": index_type}],
            }

        self._container = await self._database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=partition_key,
            **container_kwargs,
        )

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        if GetOp in grouped_ops:
            await self._abatch_get_ops(
                cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                results,
            )

        if SearchOp in grouped_ops:
            await self._abatch_search_ops(
                cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                results,
            )

        if ListNamespacesOp in grouped_ops:
            await self._abatch_list_namespaces_ops(
                cast(
                    Sequence[tuple[int, ListNamespacesOp]],
                    grouped_ops[ListNamespacesOp],
                ),
                results,
            )

        if PutOp in grouped_ops:
            await self._abatch_put_ops(
                cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
            )

        return results

    async def _abatch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        namespace_groups: dict[tuple[str, ...], list[tuple[int, str, bool]]] = (
            defaultdict(list)
        )
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key, op.refresh_ttl))

        for namespace, items in namespace_groups.items():
            keys = [key for _, key, _ in items]
            query, params = self._build_get_query(namespace, keys)

            docs = []
            async for item in self.container.query_items(
                query=query,
                parameters=params,
                partition_key=_namespace_to_text(namespace),
            ):
                docs.append(item)

            key_to_doc = {doc["key"]: doc for doc in docs}

            for idx, key, refresh_ttl in items:
                doc = key_to_doc.get(key)
                if doc:
                    if refresh_ttl and doc.get("ttl_minutes") is not None:
                        ttl_seconds = int(float(doc["ttl_minutes"]) * 60)
                        try:
                            await self.container.patch_item(
                                item=doc["id"],
                                partition_key=_namespace_to_text(namespace),
                                patch_operations=[
                                    {
                                        "op": "set",
                                        "path": "/ttl",
                                        "value": ttl_seconds,
                                    },
                                ],
                            )
                        except CosmosResourceNotFoundError:
                            pass  # Concurrent delete; skip refresh.
                    results[idx] = self._doc_to_item(namespace, doc)
                else:
                    results[idx] = None

    async def _abatch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> None:
        dedupped: dict[tuple[tuple[str, ...], str], PutOp] = {}
        for _, op in put_ops:
            dedupped[(op.namespace, op.key)] = op

        inserts: list[PutOp] = []
        deletes: list[PutOp] = []
        for op in dedupped.values():
            if op.value is None:
                deletes.append(op)
            else:
                inserts.append(op)

        for op in deletes:
            doc_id = self._make_doc_id(op.namespace, op.key)
            prefix = _namespace_to_text(op.namespace)
            try:
                await self.container.delete_item(item=doc_id, partition_key=prefix)
            except CosmosResourceNotFoundError:
                pass

        if inserts:
            embedding_requests: list[tuple[PutOp, str]] = []
            if self.index_config and self.embeddings:
                for op in inserts:
                    if op.index is False:
                        continue
                    if op.index is None or op.index is True:
                        paths = cast(dict, self.index_config)["__tokenized_fields"]
                    else:
                        paths = [(ix, tokenize_path(ix)) for ix in op.index]
                    texts = []
                    for _path, tokenized_path in paths:
                        field_texts = get_text_at_path(op.value, tokenized_path)
                        texts.extend(field_texts)
                    if texts:
                        combined_text = " ".join(texts)
                        embedding_requests.append((op, combined_text))

            embeddings_map: dict[tuple[tuple[str, ...], str], list[float]] = {}
            if embedding_requests:
                texts_to_embed = [text for _, text in embedding_requests]
                assert self.embeddings is not None
                vectors = await self.embeddings.aembed_documents(texts_to_embed)
                for (op, _), vector in zip(embedding_requests, vectors, strict=False):
                    embeddings_map[(op.namespace, op.key)] = vector

            for op in inserts:
                # Point-read to preserve created_at on updates (1 RU).
                prefix = _namespace_to_text(op.namespace)
                doc_id = self._make_doc_id(op.namespace, op.key)
                existing_created_at: str | None = None
                try:
                    existing = await self.container.read_item(
                        item=doc_id, partition_key=prefix
                    )
                    existing_created_at = existing.get("created_at")
                except CosmosResourceNotFoundError:
                    pass  # Document doesn't exist yet — created_at = now.

                doc = self._prepare_put_document(
                    op, existing_created_at=existing_created_at
                )

                if op.ttl is None and self.ttl_config:
                    default_ttl = self.ttl_config.get("default_ttl")
                    if default_ttl is not None:
                        ttl_seconds = int(float(default_ttl) * 60)
                        doc["ttl"] = ttl_seconds
                        doc["ttl_minutes"] = float(default_ttl)

                embedding = embeddings_map.get((op.namespace, op.key))
                if embedding is not None:
                    doc["embedding"] = embedding

                await self.container.upsert_item(body=doc)

    async def _abatch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        embedding_requests: list[tuple[int, str]] = []
        for idx, (_, op) in enumerate(search_ops):
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

        embeddings: dict[int, list[float]] = {}
        if embedding_requests and self.embeddings:
            for idx, query_text in embedding_requests:
                embeddings[idx] = await self.embeddings.aembed_query(query_text)

        for idx_in_ops, (original_idx, op) in enumerate(search_ops):
            embedding = embeddings.get(idx_in_ops)
            query, params = self._build_search_query(op, embedding)

            docs = []
            # Always use cross-partition query for search since
            # namespace_prefix is a PREFIX match (STARTSWITH), not an exact
            # partition key match. Items in sub-namespaces have different
            # partition keys (e.g., prefix "test" won't match "test.A").
            async for doc in self.container.query_items(
                query=query,
                parameters=params,
            ):
                docs.append(doc)

            if embedding is not None:
                docs = docs[op.offset :]

            if op.refresh_ttl:
                for doc in docs:
                    if doc.get("ttl_minutes") is not None:
                        ttl_seconds = int(float(doc["ttl_minutes"]) * 60)
                        try:
                            await self.container.patch_item(
                                item=doc["id"],
                                partition_key=doc["prefix"],
                                patch_operations=[
                                    {
                                        "op": "set",
                                        "path": "/ttl",
                                        "value": ttl_seconds,
                                    },
                                ],
                            )
                        except CosmosResourceNotFoundError:
                            pass

            items = [self._doc_to_search_item(doc) for doc in docs]
            results[original_idx] = items

    async def _abatch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        for original_idx, op in list_ops:
            query, params = self._build_list_namespaces_query(op)

            docs = []
            async for doc in self.container.query_items(
                query=query,
                parameters=params,
            ):
                docs.append(doc)

            namespaces: list[tuple[str, ...]] = []
            seen: set[tuple[str, ...]] = set()
            for doc in docs:
                ns = _decode_ns(doc["prefix"])
                if op.max_depth is not None:
                    ns = ns[: op.max_depth]
                if ns not in seen:
                    seen.add(ns)
                    namespaces.append(ns)

            start = op.offset
            end = start + op.limit
            results[original_idx] = namespaces[start:end]

    async def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL.

        Note: Cosmos DB handles TTL natively, so this is a no-op.
        Items with TTL set will be automatically deleted by Cosmos DB.

        Returns:
            int: Always returns 0 since Cosmos DB handles TTL natively.
        """
        return 0

    async def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> asyncio.Task[None]:
        """Start a TTL sweeper (no-op for Cosmos DB since TTL is native).

        Returns:
            Task that resolves immediately.
        """
        return asyncio.create_task(asyncio.sleep(0))

    async def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper (no-op for Cosmos DB).

        Returns:
            bool: Always True.
        """
        return True

    async def __aenter__(self) -> AsyncCosmosDBStore:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


__all__ = ["AsyncCosmosDBStore"]
