"""Azure CosmosDB implementation of LangGraph BaseStore (sync).

Provides a synchronous Cosmos DB-backed store for LangGraph long-term memory
with optional vector search support.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    cast,
)

import orjson
from azure.cosmos import ContainerProxy, CosmosClient, DatabaseProxy, PartitionKey
from azure.cosmos.exceptions import CosmosResourceNotFoundError
from azure.identity import DefaultAzureCredential
from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

USER_AGENT = "langchain-azure-cosmosdb-lgstore"
_NS_SEPARATOR = "|"
_SAFE_FILTER_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_filter_key(key: str) -> None:
    """Validate that a filter key is safe for SQL interpolation."""
    if not _SAFE_FILTER_KEY_RE.match(key):
        raise ValueError(
            f"Invalid filter key '{key}'. "
            "Filter keys must start with a letter or underscore and contain "
            "only letters, digits, and underscores."
        )


class CosmosDBIndexConfig(IndexConfig, total=False):
    """Configuration for vector embeddings in Cosmos DB store.

    Extends IndexConfig with Cosmos DB-specific vector search options.
    """

    distance_type: Literal["cosine", "euclidean", "dotproduct"]
    """Distance metric to use for vector similarity search:
    - 'cosine': Cosine similarity (default)
    - 'euclidean': Euclidean distance
    - 'dotproduct': Dot product
    """

    index_type: Literal["quantizedFlat", "flat", "diskANN"]
    """Vector index type:
    - 'quantizedFlat': Good for smaller datasets (<50K vectors), default
    - 'flat': Exact nearest neighbors (no quantization)
    - 'diskANN': Recommended for large datasets (100K+ vectors)
    """


# Document schema:
# {
#   "id": "<namespace_hex>:<key>",           # Unique document ID
#   "prefix": "a|b|c",                       # Namespace as pipe-separated string
#   "key": "my_key",                         # Item key
#   "value": { ... },                        # JSON value
#   "created_at": "2024-01-01T00:00:00Z",    # ISO timestamp
#   "updated_at": "2024-01-01T00:00:00Z",    # ISO timestamp
#   "ttl": 3600,                             # Optional TTL in seconds (Cosmos native)
#   "ttl_minutes": 60.0,                     # TTL in minutes (for refresh computation)
#   "embedding": [0.1, 0.2, ...],            # Optional vector embedding
#   "embedding_fields": {"field": "text"}    # Which fields were embedded
# }


C = TypeVar("C")


class BaseCosmosDBStore(Generic[C]):
    """Shared query/logic between sync and async Cosmos DB stores."""

    conn: C
    _deserializer: Callable[[bytes | orjson.Fragment], dict[str, Any]] | None
    index_config: CosmosDBIndexConfig | None

    def _make_doc_id(self, namespace: tuple[str, ...], key: str) -> str:
        """Create a unique document ID from namespace and key."""
        prefix = _namespace_to_text(namespace)
        return f"{prefix}:{key}"

    def _build_get_query(
        self,
        namespace: tuple[str, ...],
        keys: list[str],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build a query to get items by namespace and keys."""
        prefix = _namespace_to_text(namespace)
        placeholders = ", ".join(f"@key{i}" for i in range(len(keys)))
        query = (
            f"SELECT * FROM c "
            f"WHERE c.prefix = @prefix AND c.key IN ({placeholders})"
        )
        params: list[dict[str, Any]] = [{"name": "@prefix", "value": prefix}]
        for i, key in enumerate(keys):
            params.append({"name": f"@key{i}", "value": key})
        return query, params

    def _build_search_query(
        self,
        op: SearchOp,
        embedding: list[float] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build a Cosmos DB query for search operations."""
        params: list[dict[str, Any]] = []
        conditions = []

        # Namespace prefix filter
        if op.namespace_prefix:
            prefix = _namespace_to_text(op.namespace_prefix)
            conditions.append(
                "(c.prefix = @ns_prefix OR STARTSWITH(c.prefix, @ns_prefix_sep))"
            )
            params.append({"name": "@ns_prefix", "value": prefix})
            params.append({"name": "@ns_prefix_sep", "value": prefix + _NS_SEPARATOR})

        # Value filters
        if op.filter:
            for key, value in op.filter.items():
                _validate_filter_key(key)
                if isinstance(value, dict):
                    for op_name, val in value.items():
                        cond, p = self._get_filter_condition(
                            key, op_name, val, len(params)
                        )
                        conditions.append(cond)
                        params.extend(p)
                else:
                    param_name = f"@filter_{len(params)}"
                    conditions.append(f'c["value"]["{key}"] = {param_name}')
                    params.append({"name": param_name, "value": value})

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        if embedding is not None and self.index_config:
            # VectorDistance returns a distance score (lower = more similar
            # for cosine/euclidean). For dotproduct, higher = more similar.
            query = (
                f'SELECT TOP @limit c.id, c.prefix, c.key, c["value"], '
                f"c.created_at, c.updated_at, c.ttl_minutes, "
                f"VectorDistance(c.embedding, @embedding) AS score "
                f"FROM c WHERE {where_clause} "
                f"ORDER BY VectorDistance(c.embedding, @embedding)"
            )
            params.append({"name": "@embedding", "value": embedding})
            params.append({"name": "@limit", "value": op.limit + op.offset})
        else:
            query = (
                f'SELECT c.id, c.prefix, c.key, c["value"], '
                f"c.created_at, c.updated_at, c.ttl_minutes "
                f"FROM c WHERE {where_clause} "
                f"ORDER BY c.updated_at DESC "
                f"OFFSET @offset LIMIT @limit"
            )
            params.append({"name": "@offset", "value": op.offset})
            params.append({"name": "@limit", "value": op.limit})

        return query, params

    def _build_list_namespaces_query(
        self,
        op: ListNamespacesOp,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build a query to list namespaces."""
        params: list[dict[str, Any]] = []
        conditions = []

        if op.match_conditions:
            for condition in op.match_conditions:
                if condition.match_type == "prefix":
                    path = _namespace_to_text(
                        tuple("%" if v == "*" else v for v in condition.path)
                    )
                    pname = f"@match_prefix_{len(params)}"
                    pname_sep = f"@match_prefix_sep_{len(params)}"
                    conditions.append(
                        f"(c.prefix = {pname} OR STARTSWITH(c.prefix, {pname_sep}))"
                    )
                    params.append({"name": pname, "value": path})
                    params.append({"name": pname_sep, "value": path + _NS_SEPARATOR})
                elif condition.match_type == "suffix":
                    path = _namespace_to_text(
                        tuple("%" if v == "*" else v for v in condition.path)
                    )
                    pname = f"@match_suffix_{len(params)}"
                    conditions.append(f"ENDSWITH(c.prefix, {pname})")
                    params.append({"name": pname, "value": path})

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT DISTINCT c.prefix FROM c WHERE {where_clause}"

        return query, params

    def _get_filter_condition(
        self, key: str, op: str, value: Any, param_offset: int
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build a filter condition for Cosmos DB queries."""
        param_name = f"@filter_{param_offset}"
        if op == "$eq":
            return f'c["value"]["{key}"] = {param_name}', [
                {"name": param_name, "value": value}
            ]
        elif op == "$gt":
            return f'c["value"]["{key}"] > {param_name}', [
                {"name": param_name, "value": value}
            ]
        elif op == "$gte":
            return f'c["value"]["{key}"] >= {param_name}', [
                {"name": param_name, "value": value}
            ]
        elif op == "$lt":
            return f'c["value"]["{key}"] < {param_name}', [
                {"name": param_name, "value": value}
            ]
        elif op == "$lte":
            return f'c["value"]["{key}"] <= {param_name}', [
                {"name": param_name, "value": value}
            ]
        elif op == "$ne":
            return f'c["value"]["{key}"] != {param_name}', [
                {"name": param_name, "value": value}
            ]
        else:
            raise ValueError(f"Unsupported operator: {op}")

    def _prepare_put_document(
        self,
        op: PutOp,
        existing_created_at: str | None = None,
    ) -> dict[str, Any]:
        """Prepare a document for upsert.

        Args:
            op: The put operation.
            existing_created_at: If the document already exists, its original
                ``created_at`` ISO timestamp. When provided, the value is
                preserved so that upserts do not overwrite the creation time.
        """
        namespace = op.namespace
        prefix = _namespace_to_text(namespace)
        doc_id = self._make_doc_id(namespace, op.key)
        now = datetime.now(timezone.utc).isoformat()

        created_at = existing_created_at if existing_created_at else now

        doc: dict[str, Any] = {
            "id": doc_id,
            "prefix": prefix,
            "key": op.key,
            "value": op.value,
            "created_at": created_at,
            "updated_at": now,
        }

        if op.ttl is not None:
            ttl_seconds = int(float(op.ttl) * 60)
            doc["ttl"] = ttl_seconds
            doc["ttl_minutes"] = float(op.ttl)

        return doc

    def _doc_to_item(
        self,
        namespace: tuple[str, ...],
        doc: dict[str, Any],
    ) -> Item:
        """Convert a Cosmos DB document to an Item."""
        val = doc["value"]
        if not isinstance(val, dict) and self._deserializer:
            val = self._deserializer(val)

        return Item(
            key=doc["key"],
            namespace=namespace,
            value=val,
            created_at=_parse_datetime(doc["created_at"]),
            updated_at=_parse_datetime(doc["updated_at"]),
        )

    def _doc_to_search_item(
        self,
        doc: dict[str, Any],
    ) -> SearchItem:
        """Convert a Cosmos DB document to a SearchItem."""
        val = doc["value"]
        if not isinstance(val, dict) and self._deserializer:
            val = self._deserializer(val)

        namespace = _decode_ns(doc["prefix"])
        score = doc.get("score")
        if score is not None:
            try:
                score = float(score)
            except (ValueError, TypeError):
                logger.warning("Invalid score: %s", score)
                score = None

        return SearchItem(
            value=val,
            key=doc["key"],
            namespace=namespace,
            created_at=_parse_datetime(doc["created_at"]),
            updated_at=_parse_datetime(doc["updated_at"]),
            score=score,
        )


class CosmosDBStore(BaseStore, BaseCosmosDBStore[CosmosClient]):
    """Azure Cosmos DB-backed store with optional vector search.

    Provides LangGraph long-term memory persistence using Azure Cosmos DB.

    Example:
        Basic setup and usage::

            from langchain_azure_cosmosdb import CosmosDBStore

            store = CosmosDBStore.from_conn_string(
                conn_string="AccountEndpoint=https://...;AccountKey=...",
                database_name="langgraph",
                container_name="store",
            )
            store.setup()

            store.put(("users", "123"), "prefs", {"theme": "dark"})
            item = store.get(("users", "123"), "prefs")

        Vector search using LangChain embeddings::

            from langchain.embeddings import init_embeddings
            from langchain_azure_cosmosdb import CosmosDBStore

            store = CosmosDBStore.from_conn_string(
                conn_string="...",
                database_name="langgraph",
                container_name="store",
                index={
                    "dims": 1536,
                    "embed": init_embeddings("openai:text-embedding-3-small"),
                    "fields": ["text"],
                },
            )
            store.setup()

            store.put(("docs",), "doc1", {"text": "Python tutorial"})
            results = store.search(("docs",), query="programming guides", limit=2)

    Note:
        Semantic search is disabled by default. Provide an ``index``
        configuration when creating the store to enable it.

    Warning:
        Make sure to call ``setup()`` before first use to create the
        database and container.
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
        "_ttl_sweeper_thread",
        "_ttl_stop_event",
    )
    supports_ttl: bool = True

    def __init__(
        self,
        conn: CosmosClient,
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
        self._database: DatabaseProxy | None = None
        self._container: ContainerProxy | None = None
        self._ttl_sweeper_thread: threading.Thread | None = None
        self._ttl_stop_event = threading.Event()

    @classmethod
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        database_name: str = "langgraph",
        container_name: str = "store",
        index: CosmosDBIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        cosmos_client_kwargs: dict[str, Any] | None = None,
    ) -> CosmosDBStore:
        """Create a new CosmosDBStore from a connection string.

        Args:
            conn_string: Azure Cosmos DB connection string.
            database_name: Name of the database to use.
            container_name: Name of the container to use.
            index: Optional index/embedding configuration for vector search.
            ttl: Optional TTL configuration.
            cosmos_client_kwargs: Additional keyword arguments passed to
                the ``CosmosClient`` constructor (e.g. ``retry_options``).

        Returns:
            A new CosmosDBStore instance.
        """
        extra_kwargs = cosmos_client_kwargs or {}
        client = CosmosClient.from_connection_string(
            conn_string, user_agent=USER_AGENT, **extra_kwargs
        )
        return cls(
            conn=client,
            database_name=database_name,
            container_name=container_name,
            index=index,
            ttl=ttl,
        )

    @classmethod
    def from_endpoint(
        cls,
        endpoint: str,
        *,
        credential: Any | None = None,
        database_name: str = "langgraph",
        container_name: str = "store",
        index: CosmosDBIndexConfig | None = None,
        ttl: TTLConfig | None = None,
        cosmos_client_kwargs: dict[str, Any] | None = None,
    ) -> CosmosDBStore:
        """Create a new CosmosDBStore from an endpoint URL.

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

        Returns:
            A new CosmosDBStore instance.
        """
        if credential is None:
            credential = DefaultAzureCredential()
        extra_kwargs = cosmos_client_kwargs or {}
        client = CosmosClient(
            endpoint, credential=credential, user_agent=USER_AGENT, **extra_kwargs
        )
        return cls(
            conn=client,
            database_name=database_name,
            container_name=container_name,
            index=index,
            ttl=ttl,
        )

    @property
    def container(self) -> ContainerProxy:
        """Get the container proxy, creating it if needed."""
        if self._container is None:
            raise RuntimeError(
                "Store not initialized. Call setup() before using the store."
            )
        return self._container

    def setup(self) -> None:
        """Set up the Cosmos DB database and container.

        Creates the database and container if they don't exist.
        Configures vector embedding policy and indexing policy when
        vector search is enabled.
        """
        self._database = self.conn.create_database_if_not_exists(self._database_name)

        # Build container properties
        partition_key = PartitionKey(path="/prefix")

        # Container-level kwargs
        container_kwargs: dict[str, Any] = {}

        # Enable TTL at container level if TTL config is provided
        if self.ttl_config:
            container_kwargs["default_ttl"] = -1  # Enable TTL but no default

        # Configure vector embedding policy if index is configured
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

        self._container = self._database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=partition_key,
            **container_kwargs,
        )

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        grouped_ops, num_ops = _group_ops(ops)
        results: list[Result] = [None] * num_ops

        if GetOp in grouped_ops:
            self._batch_get_ops(
                cast(Sequence[tuple[int, GetOp]], grouped_ops[GetOp]),
                results,
            )

        if SearchOp in grouped_ops:
            self._batch_search_ops(
                cast(Sequence[tuple[int, SearchOp]], grouped_ops[SearchOp]),
                results,
            )

        if ListNamespacesOp in grouped_ops:
            self._batch_list_namespaces_ops(
                cast(
                    Sequence[tuple[int, ListNamespacesOp]],
                    grouped_ops[ListNamespacesOp],
                ),
                results,
            )

        if PutOp in grouped_ops:
            self._batch_put_ops(
                cast(Sequence[tuple[int, PutOp]], grouped_ops[PutOp]),
            )

        return results

    def _batch_get_ops(
        self,
        get_ops: Sequence[tuple[int, GetOp]],
        results: list[Result],
    ) -> None:
        # Group by namespace
        namespace_groups: dict[tuple[str, ...], list[tuple[int, str, bool]]] = (
            defaultdict(list)
        )
        for idx, op in get_ops:
            namespace_groups[op.namespace].append((idx, op.key, op.refresh_ttl))

        for namespace, items in namespace_groups.items():
            keys = [key for _, key, _ in items]
            query, params = self._build_get_query(namespace, keys)

            docs = list(
                self.container.query_items(
                    query=query,
                    parameters=params,
                    partition_key=_namespace_to_text(namespace),
                )
            )
            key_to_doc = {doc["key"]: doc for doc in docs}

            for idx, key, refresh_ttl in items:
                doc = key_to_doc.get(key)
                if doc:
                    # Refresh TTL if needed
                    if refresh_ttl and doc.get("ttl_minutes") is not None:
                        ttl_seconds = int(float(doc["ttl_minutes"]) * 60)
                        try:
                            self.container.patch_item(
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

    def _batch_put_ops(
        self,
        put_ops: Sequence[tuple[int, PutOp]],
    ) -> None:
        # Dedup: last write wins
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

        # Handle deletes
        for op in deletes:
            doc_id = self._make_doc_id(op.namespace, op.key)
            prefix = _namespace_to_text(op.namespace)
            try:
                self.container.delete_item(item=doc_id, partition_key=prefix)
            except CosmosResourceNotFoundError:
                pass

        # Handle inserts/updates
        if inserts:
            # Compute embeddings if needed
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
                vectors = self.embeddings.embed_documents(texts_to_embed)
                for (op, _), vector in zip(embedding_requests, vectors, strict=False):
                    embeddings_map[(op.namespace, op.key)] = vector

            for op in inserts:
                # Point-read to preserve created_at on updates (1 RU).
                prefix = _namespace_to_text(op.namespace)
                doc_id = self._make_doc_id(op.namespace, op.key)
                existing_created_at: str | None = None
                try:
                    existing = self.container.read_item(
                        item=doc_id, partition_key=prefix
                    )
                    existing_created_at = existing.get("created_at")
                except CosmosResourceNotFoundError:
                    pass  # Document doesn't exist yet — created_at = now.

                doc = self._prepare_put_document(
                    op, existing_created_at=existing_created_at
                )

                # Add TTL from config if not specified in op
                if op.ttl is None and self.ttl_config:
                    default_ttl = self.ttl_config.get("default_ttl")
                    if default_ttl is not None:
                        ttl_seconds = int(float(default_ttl) * 60)
                        doc["ttl"] = ttl_seconds
                        doc["ttl_minutes"] = float(default_ttl)

                # Add embedding if available
                embedding = embeddings_map.get((op.namespace, op.key))
                if embedding is not None:
                    doc["embedding"] = embedding

                self.container.upsert_item(body=doc)

    def _batch_search_ops(
        self,
        search_ops: Sequence[tuple[int, SearchOp]],
        results: list[Result],
    ) -> None:
        # Compute embeddings for queries that need them
        embedding_requests: list[tuple[int, str]] = []
        for idx, (_, op) in enumerate(search_ops):
            if op.query and self.index_config:
                embedding_requests.append((idx, op.query))

        embeddings: dict[int, list[float]] = {}
        if embedding_requests and self.embeddings:
            for idx, query_text in embedding_requests:
                embeddings[idx] = self.embeddings.embed_query(query_text)

        for idx_in_ops, (original_idx, op) in enumerate(search_ops):
            embedding = embeddings.get(idx_in_ops)
            query, params = self._build_search_query(op, embedding)

            # Always use cross-partition query for search since
            # namespace_prefix is a PREFIX match (STARTSWITH), not an exact
            # partition key match. Items in sub-namespaces have different
            # partition keys (e.g., prefix "test" won't match "test.A").
            docs = list(
                self.container.query_items(
                    query=query,
                    parameters=params,
                    enable_cross_partition_query=True,
                )
            )

            # Handle offset for vector search (we fetched limit+offset)
            if embedding is not None:
                docs = docs[op.offset :]

            # Refresh TTL if needed
            if op.refresh_ttl:
                for doc in docs:
                    if doc.get("ttl_minutes") is not None:
                        ttl_seconds = int(float(doc["ttl_minutes"]) * 60)
                        try:
                            self.container.patch_item(
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

    def _batch_list_namespaces_ops(
        self,
        list_ops: Sequence[tuple[int, ListNamespacesOp]],
        results: list[Result],
    ) -> None:
        for original_idx, op in list_ops:
            query, params = self._build_list_namespaces_query(op)

            docs = list(
                self.container.query_items(
                    query=query,
                    parameters=params,
                    enable_cross_partition_query=True,
                )
            )

            namespaces: list[tuple[str, ...]] = []
            seen: set[tuple[str, ...]] = set()
            for doc in docs:
                ns = _decode_ns(doc["prefix"])
                if op.max_depth is not None:
                    ns = ns[: op.max_depth]
                if ns not in seen:
                    seen.add(ns)
                    namespaces.append(ns)

            # Apply pagination
            start = op.offset
            end = start + op.limit
            results[original_idx] = namespaces[start:end]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Async compatibility shim that delegates to the sync ``batch()`` method.

        This store is built on the synchronous ``azure.cosmos.CosmosClient``
        and cannot perform native async I/O.  The call is run in a thread-pool
        executor so it does not block the event loop.

        For true async Cosmos DB operations, use
        :class:`~langchain_azure_cosmosdb.aio.AsyncCosmosDBStore` instead,
        which is built on ``azure.cosmos.aio.CosmosClient``.
        """
        import asyncio

        return await asyncio.get_running_loop().run_in_executor(None, self.batch, ops)

    def sweep_ttl(self) -> int:
        """Delete expired store items based on TTL.

        Note: Cosmos DB handles TTL natively, so this is mostly a no-op.
        Items with TTL set will be automatically deleted by Cosmos DB.
        This method exists for API compatibility.

        Returns:
            int: Always returns 0 since Cosmos DB handles TTL natively.
        """
        return 0

    def start_ttl_sweeper(
        self, sweep_interval_minutes: int | None = None
    ) -> concurrent.futures.Future[None]:
        """Start a TTL sweeper (no-op for Cosmos DB since TTL is native).

        Returns:
            Future that resolves immediately.
        """
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.set_result(None)
        return future

    def stop_ttl_sweeper(self, timeout: float | None = None) -> bool:
        """Stop the TTL sweeper (no-op for Cosmos DB).

        Returns:
            bool: Always True.
        """
        return True


# ─── Utilities ───────────────────────────────────────────────────────────────


def _namespace_to_text(
    namespace: tuple[str, ...], handle_wildcards: bool = False
) -> str:
    """Convert namespace tuple to pipe-separated text string."""
    if handle_wildcards:
        namespace = tuple("%" if val == "*" else val for val in namespace)
    return _NS_SEPARATOR.join(namespace)


def _decode_ns(prefix: str | bytes) -> tuple[str, ...]:
    """Convert pipe-separated prefix back to namespace tuple."""
    if isinstance(prefix, bytes):
        prefix = prefix.decode()
    return tuple(prefix.split(_NS_SEPARATOR))


def _parse_datetime(val: Any) -> datetime:
    """Parse a datetime from a Cosmos DB document."""
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        # Handle ISO format with or without timezone
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        except ValueError:
            dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.now(timezone.utc)


def _group_ops(
    ops: Iterable[Op],
) -> tuple[dict[type, list[tuple[int, Op]]], int]:
    grouped_ops: dict[type, list[tuple[int, Op]]] = defaultdict(list)
    tot = 0
    for idx, op in enumerate(ops):
        grouped_ops[type(op)].append((idx, op))
        tot += 1
    return grouped_ops, tot


def _ensure_index_config(
    index_config: CosmosDBIndexConfig,
) -> tuple[Embeddings | None, CosmosDBIndexConfig]:
    index_config = index_config.copy()
    tokenized: list[tuple[str, Literal["$"] | list[str]]] = []
    tot = 0
    fields = index_config.get("fields") or ["$"]
    if isinstance(fields, str):
        fields = [fields]
    if not isinstance(fields, list):
        raise ValueError(f"Text fields must be a list or a string. Got {fields}")
    for p in fields:
        if p == "$":
            tokenized.append((p, "$"))
            tot += 1
        else:
            toks = tokenize_path(p)
            tokenized.append((p, toks))
            tot += len(toks)
    index_config["__tokenized_fields"] = tokenized
    index_config["__estimated_num_vectors"] = tot
    embeddings = ensure_embeddings(
        index_config.get("embed"),
    )
    return embeddings, index_config


__all__ = [
    "BaseCosmosDBStore",
    "CosmosDBIndexConfig",
    "CosmosDBStore",
    "USER_AGENT",
]
