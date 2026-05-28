"""Async Vector Store for CosmosDB NoSql."""

from __future__ import annotations

import uuid
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
from langchain_azure_cosmosdb._utils import (
    extract_partition_key_paths,
    extract_partition_key_value,
    maximal_marginal_relevance,
)
from langchain_azure_cosmosdb._vectorstore import _validate_sql_identifier
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from pydantic import ConfigDict, model_validator

if TYPE_CHECKING:
    from azure.cosmos.aio import ContainerProxy, CosmosClient, DatabaseProxy

USER_AGENT = "langchain-azure-cosmosdb-vectorstore"

# ruff: noqa: E501


class AsyncAzureCosmosDBNoSqlVectorSearch(VectorStore):
    """`Azure Cosmos DB for NoSQL` async vector store.

    To use, you should have both:
        - the ``azure-cosmos`` python package installed

    You can read more about vector search, full text search
    and hybrid search using AzureCosmosDBNoSQL here:
    https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
    https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/full-text-search
    https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search
    """

    VALID_SEARCH_TYPES = {
        "vector",
        "vector_score_threshold",
        "full_text_search",
        "full_text_ranking",
        "hybrid",
        "hybrid_score_threshold",
    }

    def __init__(
        self,
        *,
        cosmos_client: CosmosClient,
        embedding: Embeddings,
        database: DatabaseProxy,
        container: ContainerProxy,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        full_text_policy: Optional[Dict[str, Any]] = None,
        vector_search_fields: Dict[str, Any],
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        search_type: str = "vector",
        metadata_key: str = "metadata",
        create_container: bool = True,
        full_text_search_enabled: bool = False,
        table_alias: str = "c",
    ) -> None:
        """Constructor for AsyncAzureCosmosDBNoSqlVectorSearch.

        Use the ``create`` classmethod to instantiate asynchronously.

        Args:
            cosmos_client: Async client used to connect to azure cosmosdb.
            embedding: Text embedding model to use.
            database: Already-initialised async ``DatabaseProxy``.
            container: Already-initialised async ``ContainerProxy``.
            vector_embedding_policy: Vector Embedding Policy for the container.
            indexing_policy: Indexing Policy for the container.
            cosmos_container_properties: Container Properties for the container.
            cosmos_database_properties: Database Properties for the container.
            full_text_policy: Full Text Policy for the container.
            vector_search_fields: Vector Search and Text
                                  Search Fields for the container.
            database_name: Name of the database.
            container_name: Name of the container.
            search_type: CosmosDB Search Type to be performed.
            metadata_key: Metadata key to use for data schema.
            create_container: Whether the container was created.
            full_text_search_enabled: Whether full text search is enabled.
            table_alias: Alias for the table in SQL queries.
        """
        self._cosmos_client = cosmos_client
        self._database_name = database_name
        self._container_name = container_name
        self._embedding = embedding
        self._vector_embedding_policy = vector_embedding_policy
        self._full_text_policy = full_text_policy
        self._indexing_policy = indexing_policy
        self._cosmos_container_properties = cosmos_container_properties
        self._cosmos_database_properties = cosmos_database_properties
        self._vector_search_fields = vector_search_fields
        self._metadata_key = metadata_key
        self._create_container = create_container
        self._full_text_search_enabled = full_text_search_enabled
        self._search_type = search_type
        self._table_alias = table_alias

        self._database = database
        self._container = container

    @classmethod
    async def create(
        cls,
        *,
        cosmos_client: CosmosClient,
        embedding: Embeddings,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        full_text_policy: Optional[Dict[str, Any]] = None,
        vector_search_fields: Dict[str, Any],
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        search_type: str = "vector",
        metadata_key: str = "metadata",
        create_container: bool = True,
        full_text_search_enabled: bool = False,
        table_alias: str = "c",
    ) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Async factory to create an AsyncAzureCosmosDBNoSqlVectorSearch.

        Args:
            cosmos_client: Async client for azure cosmosdb no sql account.
            embedding: Text embedding model to use.
            vector_embedding_policy: Vector Embedding Policy for the container.
            indexing_policy: Indexing Policy for the container.
            cosmos_container_properties: Container Properties for the container.
            cosmos_database_properties: Database Properties for the container.
            full_text_policy: Full Text Policy for the container.
            vector_search_fields: Vector Search and Text
                                  Search Fields for the container.
            database_name: Name of the database to be created.
            container_name: Name of the container to be created.
            search_type: CosmosDB Search Type to be performed.
            metadata_key: Metadata key to use for data schema.
            create_container: Set to true if the container does not exist.
            full_text_search_enabled: Set to true if full text search is enabled.
            table_alias: Alias for the table to use in the WHERE clause.

        Returns:
            An initialised AsyncAzureCosmosDBNoSqlVectorSearch instance.
        """
        if create_container:
            if (
                indexing_policy["vectorIndexes"] is None
                or len(indexing_policy["vectorIndexes"]) == 0
            ):
                raise ValueError(
                    "vectorIndexes cannot be null or empty in the indexing_policy."
                )
            if (
                vector_embedding_policy is None
                or len(vector_embedding_policy["vectorEmbeddings"]) == 0
            ):
                raise ValueError(
                    "vectorEmbeddings cannot be null "
                    "or empty in the vector_embedding_policy."
                )
            if cosmos_container_properties["partition_key"] is None:
                raise ValueError(
                    "partition_key cannot be null or empty for a container."
                )
            if full_text_search_enabled:
                if (
                    indexing_policy["fullTextIndexes"] is None
                    or len(indexing_policy["fullTextIndexes"]) == 0
                ):
                    raise ValueError(
                        "fullTextIndexes cannot be null or empty in the "
                        "indexing_policy if full text search is enabled."
                    )
                if (
                    full_text_policy is None
                    or len(full_text_policy["fullTextPaths"]) == 0
                ):
                    raise ValueError(
                        "fullTextPaths cannot be null or empty in the "
                        "full_text_policy if full text search is enabled."
                    )
        if vector_search_fields is None:
            raise ValueError(
                "vectorSearchFields cannot be null or empty in the vector_search_fields."
            )

        _validate_sql_identifier(metadata_key, "metadata_key")
        _validate_sql_identifier(table_alias, "table_alias")
        _validate_sql_identifier(
            vector_search_fields["text_field"],
            "vector_search_fields['text_field']",
        )
        _validate_sql_identifier(
            vector_search_fields["embedding_field"],
            "vector_search_fields['embedding_field']",
        )

        database = await cosmos_client.create_database_if_not_exists(
            id=database_name,
            offer_throughput=cosmos_database_properties.get("offer_throughput"),
            session_token=cosmos_database_properties.get("session_token"),
            initial_headers=cosmos_database_properties.get("initial_headers"),
            etag=cosmos_database_properties.get("etag"),
            match_condition=cosmos_database_properties.get("match_condition"),
        )

        container = await database.create_container_if_not_exists(
            id=container_name,
            partition_key=cosmos_container_properties["partition_key"],
            indexing_policy=indexing_policy,
            default_ttl=cosmos_container_properties.get("default_ttl"),
            offer_throughput=cosmos_container_properties.get("offer_throughput"),
            unique_key_policy=cosmos_container_properties.get("unique_key_policy"),
            conflict_resolution_policy=cosmos_container_properties.get(
                "conflict_resolution_policy"
            ),
            analytical_storage_ttl=cosmos_container_properties.get(
                "analytical_storage_ttl"
            ),
            computed_properties=cosmos_container_properties.get("computed_properties"),
            etag=cosmos_container_properties.get("etag"),
            match_condition=cosmos_container_properties.get("match_condition"),
            session_token=cosmos_container_properties.get("session_token"),
            initial_headers=cosmos_container_properties.get("initial_headers"),
            vector_embedding_policy=vector_embedding_policy,
            full_text_policy=full_text_policy,
        )

        return cls(
            cosmos_client=cosmos_client,
            embedding=embedding,
            database=database,
            container=container,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            full_text_policy=full_text_policy,
            vector_search_fields=vector_search_fields,
            database_name=database_name,
            container_name=container_name,
            search_type=search_type,
            metadata_key=metadata_key,
            create_container=create_container,
            full_text_search_enabled=full_text_search_enabled,
            table_alias=table_alias,
        )

    @classmethod
    async def from_endpoint_and_aad(
        cls,
        endpoint: str,
        credential: Any,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Create vectorstore from an endpoint with AAD credential.

        Args:
            endpoint: CosmosDB account endpoint URL.
            credential: Azure credential (e.g., DefaultAzureCredential).
            texts: The texts to insert.
            embedding: The embedding model to use.
            metadatas: Optional metadata dicts for the texts.
            ids: Optional ids for the texts.
            **kwargs: Additional keyword arguments passed to ``create``.

        Returns:
            An initialised AsyncAzureCosmosDBNoSqlVectorSearch.
        """
        from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

        cosmos_client = AsyncCosmosClient(endpoint, credential, user_agent=USER_AGENT)
        try:
            kwargs["cosmos_client"] = cosmos_client
            vectorstore = await cls._afrom_kwargs(embedding, **kwargs)
            await vectorstore.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
            vectorstore._owns_client = True
            return vectorstore
        except Exception:
            await cosmos_client.close()
            raise

    @classmethod
    async def from_endpoint_and_key(
        cls,
        endpoint: str,
        key: str,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Create vectorstore from an endpoint with access key.

        Args:
            endpoint: CosmosDB account endpoint URL.
            key: CosmosDB access key.
            texts: The texts to insert.
            embedding: The embedding model to use.
            metadatas: Optional metadata dicts for the texts.
            ids: Optional ids for the texts.
            **kwargs: Additional keyword arguments passed to ``create``.

        Returns:
            An initialised AsyncAzureCosmosDBNoSqlVectorSearch.
        """
        from azure.cosmos.aio import CosmosClient as AsyncCosmosClient

        cosmos_client = AsyncCosmosClient(endpoint, key, user_agent=USER_AGENT)
        try:
            kwargs["cosmos_client"] = cosmos_client
            vectorstore = await cls._afrom_kwargs(embedding, **kwargs)
            await vectorstore.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
            vectorstore._owns_client = True
            return vectorstore
        except Exception:
            await cosmos_client.close()
            raise

    async def close(self) -> None:
        """Close the underlying CosmosDB client if owned by this instance.

        Call this when the vectorstore was created via a factory method
        (``from_endpoint_and_aad`` or
        ``from_endpoint_and_key``) to release the connection.
        Alternatively, use the instance as an async context manager.
        """
        if getattr(self, "_owns_client", False) and self._cosmos_client is not None:
            await self._cosmos_client.close()

    async def __aenter__(self) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Enter async context manager."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context manager and close client if owned."""
        await self.close()

    @property
    def embeddings(self) -> Embeddings:
        """Access the query embedding object."""
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Not implemented. Use ``aadd_texts`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async method `aadd_texts` instead.")

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vectorstore asynchronously.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids associated with the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts_list = list(texts)
        _metadatas = list(
            metadatas if metadatas is not None else ({} for _ in texts_list)
        )
        _ids = list(ids if ids is not None else (str(uuid.uuid4()) for _ in texts_list))
        return await self._ainsert_texts(texts_list, _metadatas, _ids)

    async def _ainsert_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> List[str]:
        """Load documents into the collection asynchronously.

        Args:
            texts: The list of document strings to load.
            metadatas: The list of metadata objects associated with each document.
            ids: The list of id objects associated with each document.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not texts:
            raise ValueError("Texts can not be null or empty")

        embeddings = await self._embedding.aembed_documents(texts)
        text_key = self._vector_search_fields["text_field"]
        embedding_key = self._vector_search_fields["embedding_field"]

        to_insert = [
            {
                "id": i,
                text_key: t,
                embedding_key: embedding,
                self._metadata_key: m,
            }
            for i, t, m, embedding in zip(ids, texts, metadatas, embeddings)
        ]

        pk_def = self._cosmos_container_properties["partition_key"]
        pk_paths = extract_partition_key_paths(pk_def)
        await self._abatch_insert(to_insert, pk_paths)
        # Return IDs in original input order (batch grouping may reorder).
        return ids

    async def _abatch_insert(
        self, items: List[Dict[str, Any]], pk_paths: List[str]
    ) -> None:
        """Insert items using transactional batch grouped by partition key.

        Args:
            items: Documents to insert.
            pk_paths: Partition key paths from
                :func:`extract_partition_key_paths` (e.g. ``["/id"]``).
        """
        _BATCH_LIMIT = 100

        # Group items by partition key value
        groups: Dict[Any, List[Dict[str, Any]]] = {}
        for item in items:
            pk_val = extract_partition_key_value(item, pk_paths)
            groups.setdefault(pk_val, []).append(item)

        for pk_val, group in groups.items():
            for i in range(0, len(group), _BATCH_LIMIT):
                batch = [
                    ("create", (item,), {}) for item in group[i : i + _BATCH_LIMIT]
                ]
                await self._container.execute_item_batch(batch, partition_key=pk_val)

    @classmethod
    async def _afrom_kwargs(
        cls,
        embedding: Embeddings,
        *,
        cosmos_client: CosmosClient,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        vector_search_fields: Dict[str, Any],
        full_text_policy: Optional[Dict[str, Any]] = None,
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        metadata_key: str = "metadata",
        create_container: bool = True,
        full_text_search_enabled: bool = False,
        search_type: str = "vector",
        **kwargs: Any,
    ) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Build an instance from keyword arguments.

        Args:
            embedding: Embedding model to use.
            cosmos_client: Async CosmosDB client.
            vector_embedding_policy: Vector Embedding Policy.
            indexing_policy: Indexing Policy.
            cosmos_container_properties: Container Properties.
            cosmos_database_properties: Database Properties.
            vector_search_fields: Search Fields for the container.
            full_text_policy: Full Text Policy.
            database_name: Database name.
            container_name: Container name.
            metadata_key: Metadata key.
            create_container: Whether to create the container.
            full_text_search_enabled: Whether full text search is enabled.
            search_type: Search type.
            **kwargs: Ignored keyword arguments.

        Returns:
            An initialised AsyncAzureCosmosDBNoSqlVectorSearch.
        """
        if kwargs:
            warnings.warn(
                "Method 'afrom_texts' of AsyncAzureCosmosDBNoSqlVectorSearch "
                "invoked with "
                f"unsupported arguments "
                f"({', '.join(sorted(kwargs))}), "
                "which will be ignored."
            )

        return await cls.create(
            embedding=embedding,
            cosmos_client=cosmos_client,
            vector_embedding_policy=vector_embedding_policy,
            full_text_policy=full_text_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            database_name=database_name,
            container_name=container_name,
            vector_search_fields=vector_search_fields,
            metadata_key=metadata_key,
            create_container=create_container,
            full_text_search_enabled=full_text_search_enabled,
            search_type=search_type,
        )

    @classmethod
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Create vectorstore from raw texts asynchronously.

        Args:
            texts: The texts to insert.
            embedding: The embedding function to use in the store.
            metadatas: Metadata dicts for the texts.
            ids: Id dicts for the texts.
            **kwargs: Additional keyword arguments.

        Returns:
            An AsyncAzureCosmosDBNoSqlVectorSearch vectorstore.
        """
        vectorstore = await cls._afrom_kwargs(embedding, **kwargs)
        await vectorstore.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vectorstore

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Not implemented. Use ``afrom_texts`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async classmethod `afrom_texts` instead.")

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        partition_key_values: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """Remove documents by IDs asynchronously.

        Args:
            ids: List of document IDs to delete.
            partition_key_values: Partition key values corresponding
                to each document ID. Required when the container's
                partition key path is not ``/id``. Defaults to using
                document IDs.
            **kwargs: Additional keyword arguments.

        Returns:
            True if successful.
        """
        if ids is None:
            raise ValueError("No document ids provided to delete.")
        if partition_key_values is not None and len(ids) != len(partition_key_values):
            raise ValueError(
                f"Length of ids ({len(ids)}) must match length of "
                f"partition_key_values ({len(partition_key_values)})."
            )
        for i, document_id in enumerate(ids):
            pk = (
                partition_key_values[i]
                if partition_key_values is not None
                else document_id
            )
            await self._container.delete_item(document_id, partition_key=pk)
        return True

    async def adelete_document_by_id(
        self,
        document_id: Optional[str] = None,
        partition_key_value: Optional[str] = None,
    ) -> None:
        """Remove a specific document by ID asynchronously.

        Args:
            document_id: The document identifier.
            partition_key_value: The partition key value for the
                document. Defaults to the document ID (assumes
                partition key path is ``/id``).
        """
        if document_id is None:
            raise ValueError("No document ids provided to delete.")
        pk = partition_key_value if partition_key_value is not None else document_id
        await self._container.delete_item(document_id, partition_key=pk)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Not implemented. Use ``asimilarity_search`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async method `asimilarity_search` instead.")

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        with_embedding: bool = False,
        search_type: Optional[str] = "vector",
        offset_limit: Optional[str] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
        threshold: Optional[float] = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query asynchronously.

        Args:
            query: Text to look up most similar documents to.
            k: Number of Documents to return.
            with_embedding: Whether to include embeddings.
            search_type: Type of search to perform.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            full_text_rank_filter: Full text rank filter.
            where: WHERE clause.
            weights: Weights for hybrid search.
            threshold: Score threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query.
        """
        search_type = search_type or self._search_type

        if search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"Invalid search_type '{search_type}'. "
                f"Valid options are: {self.VALID_SEARCH_TYPES}"
            )

        docs_and_scores = await self.asimilarity_search_with_score(
            query,
            k=k,
            with_embedding=with_embedding,
            search_type=search_type,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            full_text_rank_filter=full_text_rank_filter,
            where=where,
            weights=weights,
            threshold=threshold,
            **kwargs,
        )

        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        with_embedding: bool = False,
        search_type: Optional[str] = "vector",
        offset_limit: Optional[str] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
        threshold: Optional[float] = 0.5,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance asynchronously.

        Args:
            query: Text to look up most similar documents to.
            k: Number of Documents to return.
            with_embedding: Whether to include embeddings.
            search_type: Type of search to perform.
            offset_limit: Offset limit clause.
            full_text_rank_filter: Full text rank filter.
            projection_mapping: Projection mapping.
            where: WHERE clause.
            weights: Weights for hybrid search.
            threshold: Score threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.
        """
        docs_and_scores: List[Tuple[Document, float]] = []
        search_type = search_type or self._search_type

        if search_type not in self.VALID_SEARCH_TYPES:
            raise ValueError(
                f"Invalid search_type '{search_type}'. "
                f"Valid options are: {self.VALID_SEARCH_TYPES}"
            )

        if search_type == "vector":
            embeddings = await self._embedding.aembed_query(query)
            docs_and_scores = await self._avector_search_with_score(
                search_type=search_type,
                embeddings=embeddings,
                k=k,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                where=where,
            )
        elif search_type == "vector_score_threshold":
            embeddings = await self._embedding.aembed_query(query)
            docs_and_scores = await self._avector_search_with_threshold(
                search_type=search_type,
                embeddings=embeddings,
                k=k,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                where=where,
                threshold=threshold or 0.5,
            )
        elif search_type == "full_text_search":
            docs_and_scores = await self._afull_text_search(
                k=k,
                search_type=search_type,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                where=where,
            )
        elif search_type == "full_text_ranking":
            docs_and_scores = await self._afull_text_ranking(
                k=k,
                search_type=search_type,
                offset_limit=offset_limit,
                full_text_rank_filter=full_text_rank_filter,
                projection_mapping=projection_mapping,
                where=where,
            )
        elif search_type == "hybrid":
            embeddings = await self._embedding.aembed_query(query)
            docs_and_scores = await self._ahybrid_search_with_score(
                search_type=search_type,
                embeddings=embeddings,
                k=k,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                full_text_rank_filter=full_text_rank_filter,
                projection_mapping=projection_mapping,
                where=where,
                weights=weights,
            )
        elif search_type == "hybrid_score_threshold":
            embeddings = await self._embedding.aembed_query(query)
            docs_and_scores = await self._ahybrid_search_with_threshold(
                search_type=search_type,
                embeddings=embeddings,
                k=k,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                full_text_rank_filter=full_text_rank_filter,
                projection_mapping=projection_mapping,
                where=where,
                weights=weights,
                threshold=threshold or 0.5,
            )
        return docs_and_scores

    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to the given embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            **kwargs: Additional keyword arguments passed to
                ``_avector_search_with_score``.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = await self._avector_search_with_score(
            search_type="vector",
            embeddings=embedding,
            k=k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_type: str = "vector",
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs using maximal marginal relevance asynchronously.

        Args:
            query: Text to look up most similar documents to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch for MMR algorithm.
            lambda_mult: Diversity of results (0 = max diversity, 1 = min).
            search_type: Type of search to perform.
            with_embedding: Whether to include embeddings.
            offset_limit: Offset limit clause.
            full_text_rank_filter: Full text rank filter.
            projection_mapping: Projection mapping.
            where: WHERE clause.
            weights: Weights for hybrid search.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embeddings = await self._embedding.aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embeddings,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            search_type=search_type,
            with_embedding=with_embedding,
            offset_limit=offset_limit,
            full_text_rank_filter=full_text_rank_filter,
            projection_mapping=projection_mapping,
            where=where,
            weights=weights,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_type: str = "vector",
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs using maximal marginal relevance by vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return.
            fetch_k: Number of Documents to fetch for MMR algorithm.
            lambda_mult: Diversity of results (0 = max diversity, 1 = min).
            search_type: Type of search to perform.
            with_embedding: Whether to include embeddings.
            offset_limit: Offset limit clause.
            full_text_rank_filter: Full text rank filter.
            projection_mapping: Projection mapping.
            where: WHERE clause.
            weights: Weights for hybrid search.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs = await self._avector_search_with_score(
            search_type=search_type,
            embeddings=embedding,
            k=fetch_k,
            with_embedding=True,
            offset_limit=offset_limit,
            full_text_rank_filter=full_text_rank_filter,
            projection_mapping=projection_mapping,
            where=where,
            weights=weights,
        )

        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [
                doc.metadata[self._vector_search_fields["embedding_field"]]
                for doc, _ in docs
            ],
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs[i][0] for i in mmr_doc_indexes]

    # ------------------------------------------------------------------
    # Internal async search helpers
    # ------------------------------------------------------------------

    async def _avector_search_with_score(
        self,
        search_type: str,
        embeddings: List[float],
        k: int = 4,
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        where: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return most similar indexed documents to the embeddings.

        Args:
            search_type: Type of search to perform.
            embeddings: Embedding vector.
            k: Number of results.
            with_embedding: Whether to include embeddings.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            where: WHERE clause.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.
        """
        query, parameters = self._construct_query(
            k=k,
            search_type=search_type,
            embeddings=embeddings,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            with_embedding=with_embedding,
            where=where,
        )

        return await self._aexecute_query(
            query=query,
            search_type=search_type,
            parameters=parameters,
            with_embedding=with_embedding,
            projection_mapping=projection_mapping,
        )

    async def _avector_search_with_threshold(
        self,
        search_type: str,
        embeddings: List[float],
        threshold: float = 0.5,
        k: int = 4,
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        where: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return most similar documents with score threshold.

        Args:
            search_type: Type of search to perform.
            embeddings: Embedding vector.
            threshold: Minimum similarity score.
            k: Number of results.
            with_embedding: Whether to include embeddings.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            where: WHERE clause.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.
        """
        query, parameters = self._construct_query(
            k=k,
            search_type=search_type,
            embeddings=embeddings,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            with_embedding=with_embedding,
            where=where,
        )

        return await self._aexecute_query(
            query=query,
            search_type=search_type,
            parameters=parameters,
            with_embedding=with_embedding,
            projection_mapping=projection_mapping,
            threshold=threshold,
        )

    async def _afull_text_search(
        self,
        search_type: str,
        k: int = 4,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return documents based on full text search.

        Args:
            search_type: Type of search to perform.
            k: Number of results.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            where: WHERE clause.

        Returns:
            List of (Document, score) tuples.
        """
        query, parameters = self._construct_query(
            k=k,
            search_type=search_type,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            where=where,
        )

        return await self._aexecute_query(
            query=query,
            search_type=search_type,
            parameters=parameters,
            with_embedding=False,
            projection_mapping=projection_mapping,
        )

    async def _afull_text_ranking(
        self,
        search_type: str,
        k: int = 4,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        where: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return documents based on full text ranking.

        Args:
            search_type: Type of search to perform.
            k: Number of results.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            full_text_rank_filter: Full text rank filter.
            where: WHERE clause.

        Returns:
            List of (Document, score) tuples.
        """
        query, parameters = self._construct_query(
            k=k,
            search_type=search_type,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            full_text_rank_filter=full_text_rank_filter,
            where=where,
        )

        return await self._aexecute_query(
            query=query,
            search_type=search_type,
            parameters=parameters,
            with_embedding=False,
            projection_mapping=projection_mapping,
        )

    async def _ahybrid_search_with_score(
        self,
        search_type: str,
        embeddings: List[float],
        k: int = 4,
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return documents based on hybrid search.

        Args:
            search_type: Type of search to perform.
            embeddings: Embedding vector.
            k: Number of results.
            with_embedding: Whether to include embeddings.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            full_text_rank_filter: Full text rank filter.
            where: WHERE clause.
            weights: Weights for hybrid search.

        Returns:
            List of (Document, score) tuples.
        """
        query, parameters = self._construct_query(
            k=k,
            search_type=search_type,
            embeddings=embeddings,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            full_text_rank_filter=full_text_rank_filter,
            where=where,
            weights=weights,
        )
        return await self._aexecute_query(
            query=query,
            search_type=search_type,
            parameters=parameters,
            with_embedding=with_embedding,
            projection_mapping=projection_mapping,
        )

    async def _ahybrid_search_with_threshold(
        self,
        search_type: str,
        embeddings: List[float],
        threshold: float = 0.5,
        k: int = 4,
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return documents based on hybrid search with threshold.

        Args:
            search_type: Type of search to perform.
            embeddings: Embedding vector.
            threshold: Minimum score threshold.
            k: Number of results.
            with_embedding: Whether to include embeddings.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            full_text_rank_filter: Full text rank filter.
            where: WHERE clause.
            weights: Weights for hybrid search.

        Returns:
            List of (Document, score) tuples.
        """
        query, parameters = self._construct_query(
            k=k,
            search_type=search_type,
            embeddings=embeddings,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            full_text_rank_filter=full_text_rank_filter,
            where=where,
            weights=weights,
        )
        return await self._aexecute_query(
            query=query,
            search_type=search_type,
            parameters=parameters,
            with_embedding=with_embedding,
            projection_mapping=projection_mapping,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Query construction (same logic as sync, no I/O)
    # ------------------------------------------------------------------

    def _construct_query(
        self,
        k: int,
        search_type: str,
        embeddings: Optional[List[float]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        offset_limit: Optional[str] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        with_embedding: bool = False,
        where: Optional[str] = None,
        weights: Optional[List[float]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Construct the CosmosDB SQL query.

        Args:
            k: Number of results.
            search_type: Type of search.
            embeddings: Embedding vector.
            full_text_rank_filter: Full text rank filter.
            offset_limit: Offset limit clause.
            projection_mapping: Projection mapping.
            with_embedding: Whether to include embeddings.
            where: WHERE clause.
            weights: Weights for hybrid search.

        Returns:
            Tuple of (query string, parameters list).
        """
        # Validate identifiers that will be interpolated into SQL
        if projection_mapping:
            for key, alias in projection_mapping.items():
                _validate_sql_identifier(key, "projection_mapping key")
                _validate_sql_identifier(str(alias), "projection_mapping alias")
        if full_text_rank_filter:
            for item in full_text_rank_filter:
                _validate_sql_identifier(
                    item["search_field"], "full_text_rank_filter search_field"
                )

        query = f"""SELECT {"TOP @limit " if not offset_limit else ""}"""
        query += self._generate_projection_fields(
            projection_mapping,
            search_type,
            full_text_rank_filter,
            with_embedding,
        )
        table = self._table_alias
        query += f" FROM {table}"

        if where:
            query += f" WHERE {where}"

        if search_type == "full_text_ranking":
            if not full_text_rank_filter:
                raise ValueError(
                    "full_text_rank_filter required for full_text_ranking."
                )
            if len(full_text_rank_filter) == 1:
                item = full_text_rank_filter[0]
                terms = ", ".join(
                    [
                        f"@{item['search_field']}_term_{i}"
                        for i, _ in enumerate(item["search_text"].split())
                    ]
                )
                query += f" ORDER BY RANK FullTextScore({table}[@{item['search_field']}], {terms})"
            else:
                rank_components = []
                for item in full_text_rank_filter:
                    terms = ", ".join(
                        [
                            f"@{item['search_field']}_term_{i}"
                            for i, _ in enumerate(item["search_text"].split())
                        ]
                    )
                    component = (
                        f"FullTextScore({table}[@{item['search_field']}], {terms})"
                    )
                    rank_components.append(component)
                query += f" ORDER BY RANK RRF({', '.join(rank_components)})"
        elif search_type in ("vector", "vector_score_threshold"):
            query += f" ORDER BY VectorDistance({table}[@embeddingKey], @embeddings)"
        elif search_type in ("hybrid", "hybrid_score_threshold"):
            if not full_text_rank_filter:
                raise ValueError("full_text_rank_filter required for hybrid search.")
            rank_components = []
            for item in full_text_rank_filter:
                terms = ", ".join(
                    [
                        f"@{item['search_field']}_term_{i}"
                        for i, _ in enumerate(item["search_text"].split())
                    ]
                )
                component = f"FullTextScore({table}[@{item['search_field']}], {terms})"
                rank_components.append(component)
            # Number of RRF components = full text scores + VectorDistance
            num_components = len(rank_components) + 1
            if weights and len(weights) != num_components:
                raise ValueError(
                    f"weights must have {num_components} elements "
                    f"(one per RRF component: {len(rank_components)} "
                    f"FullTextScore + 1 VectorDistance), "
                    f"got {len(weights)}."
                )
            query += f" ORDER BY RANK RRF({', '.join(rank_components)}, VectorDistance({table}[@embeddingKey], @embeddings)"
            if weights:
                query += ", @weights)"
            else:
                query += ")"

        if offset_limit:
            query += f" {offset_limit}"

        parameters = self._build_parameters(
            k=k,
            search_type=search_type,
            embeddings=embeddings,
            projection_mapping=projection_mapping,
            full_text_rank_filter=full_text_rank_filter,
            weights=weights,
        )
        return query, parameters

    def _generate_projection_fields(
        self,
        projection_mapping: Optional[Dict[str, Any]],
        search_type: str,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        with_embedding: bool = False,
    ) -> str:
        """Generate projection fields for the SQL query.

        Args:
            projection_mapping: Projection mapping.
            search_type: Type of search.
            full_text_rank_filter: Full text rank filter.
            with_embedding: Whether to include embeddings.

        Returns:
            Projection fields string.
        """
        table = self._table_alias

        if projection_mapping:
            projection = ", ".join(
                f"{table}.{key} as {alias}" for key, alias in projection_mapping.items()
            )
        elif full_text_rank_filter:
            projection = f"{table}.id, " + ", ".join(
                f"{table}[@{search_item['search_field']}] as {search_item['search_field']}"
                for search_item in full_text_rank_filter
            )
        else:
            projection = f"{table}.id, {table}[@textKey] as {self._vector_search_fields['text_field']}, {table}[@metadataKey] as {self._metadata_key}"

        if search_type in ("vector", "vector_score_threshold"):
            if with_embedding:
                projection += f", {table}[@embeddingKey] as {self._vector_search_fields['embedding_field']}"
            projection += (
                f", VectorDistance({table}[@embeddingKey], @embeddings) as VectorScore"
            )
        elif search_type in ("hybrid", "hybrid_score_threshold"):
            if with_embedding:
                projection += f", {table}[@embeddingKey] as {self._vector_search_fields['embedding_field']}"
            projection += (
                f", VectorDistance({table}[@embeddingKey], @embeddings) as VectorScore"
            )
        return projection

    def _build_parameters(
        self,
        k: int,
        search_type: str,
        embeddings: Optional[List[float]],
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        weights: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Build query parameters.

        Args:
            k: Number of results.
            search_type: Type of search.
            embeddings: Embedding vector.
            projection_mapping: Projection mapping.
            full_text_rank_filter: Full text rank filter.
            weights: Weights for hybrid search.

        Returns:
            List of parameter dicts.
        """
        parameters: List[Dict[str, Any]] = [
            {"name": "@limit", "value": k},
        ]

        if projection_mapping:
            for key in projection_mapping.keys():
                parameters.append({"name": f"@{key}", "value": key})
        else:
            parameters.append(
                {
                    "name": "@textKey",
                    "value": self._vector_search_fields["text_field"],
                }
            )
            parameters.append({"name": "@metadataKey", "value": self._metadata_key})

        if search_type in (
            "vector",
            "vector_score_threshold",
            "hybrid",
            "hybrid_score_threshold",
        ):
            parameters.append(
                {
                    "name": "@embeddingKey",
                    "value": self._vector_search_fields["embedding_field"],
                }
            )
            parameters.append({"name": "@embeddings", "value": embeddings})
            if weights:
                parameters.append({"name": "@weights", "value": weights})

        if full_text_rank_filter:
            for item in full_text_rank_filter:
                parameters.append(
                    {
                        "name": f"@{item['search_field']}",
                        "value": item["search_field"],
                    }
                )
                for i, term in enumerate(item["search_text"].split()):
                    parameters.append(
                        {
                            "name": f"@{item['search_field']}_term_{i}",
                            "value": term,
                        }
                    )

        return parameters

    async def _aexecute_query(
        self,
        query: str,
        search_type: str,
        parameters: List[Dict[str, Any]],
        with_embedding: bool,
        projection_mapping: Optional[Dict[str, Any]],
        threshold: Optional[float] = 0.0,
    ) -> List[Tuple[Document, float]]:
        """Execute a CosmosDB query asynchronously and return results.

        Args:
            query: The SQL query string.
            search_type: Type of search.
            parameters: Query parameters.
            with_embedding: Whether to include embeddings.
            projection_mapping: Projection mapping.
            threshold: Minimum score threshold.

        Returns:
            List of (Document, score) tuples.
        """
        docs_and_scores: List[Tuple[Document, float]] = []
        threshold = threshold or 0.0
        items: List[Dict[str, Any]] = []
        async for item in self._container.query_items(
            query=query,
            parameters=parameters,
        ):
            items.append(item)

        has_score = search_type in (
            "vector",
            "hybrid",
            "vector_score_threshold",
            "hybrid_score_threshold",
        )

        for item in items:
            metadata = item.pop(self._metadata_key, {})
            score = item.get("VectorScore", 0.0) if has_score else 0.0

            if with_embedding and has_score:
                metadata[self._vector_search_fields["embedding_field"]] = item[
                    self._vector_search_fields["embedding_field"]
                ]

            if search_type in ("vector_score_threshold", "hybrid_score_threshold"):
                dist_fn = (
                    self._vector_embedding_policy["vectorEmbeddings"][0]
                    .get("distanceFunction", "cosine")
                    .lower()
                )
                if dist_fn == "euclidean":
                    if score >= threshold:
                        continue
                elif score <= threshold:
                    continue

            if (
                projection_mapping
                and self._vector_search_fields["text_field"] in projection_mapping
            ):
                text_key = projection_mapping[self._vector_search_fields["text_field"]]
            else:
                text_key = self._vector_search_fields["text_field"]
            text = item[text_key]

            if projection_mapping:
                for key, alias in projection_mapping.items():
                    if key == self._vector_search_fields["text_field"]:
                        continue
                    metadata[alias] = item[alias]
            else:
                metadata["id"] = item["id"]

            docs_and_scores.append(
                (Document(page_content=text, metadata=metadata), score)
            )
        return docs_and_scores

    def get_container(self) -> ContainerProxy:
        """Get the container for the vector store.

        Returns:
            The async ContainerProxy.
        """
        return self._container

    def as_retriever(
        self, **kwargs: Any
    ) -> AsyncAzureCosmosDBNoSqlVectorStoreRetriever:
        """Return retriever initialised from this VectorStore.

        Args:
            **kwargs: Keyword arguments including search_type, k, and
                search_kwargs.

        Returns:
            AsyncAzureCosmosDBNoSqlVectorStoreRetriever instance.
        """
        search_type = kwargs.get("search_type", "vector")
        k = kwargs.get("k", 5)
        with_embedding = kwargs.get("with_embedding", False)
        offset_limit = kwargs.get("offset_limit", None)
        projection_mapping = kwargs.get("projection_mapping", None)
        full_text_rank_filter = kwargs.get("full_text_rank_filter", None)
        where = kwargs.get("where", None)
        weights = kwargs.get("weights", None)
        score_threshold = kwargs.get("score_threshold", 0.5)

        search_kwargs = {
            "with_embedding": with_embedding,
            "offset_limit": offset_limit,
            "projection_mapping": projection_mapping,
            "full_text_rank_filter": full_text_rank_filter,
            "where": where,
            "weights": weights,
            "score_threshold": score_threshold,
        }

        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return AsyncAzureCosmosDBNoSqlVectorStoreRetriever(
            vectorstore=self,
            search_type=search_type,
            k=k,
            search_kwargs=search_kwargs,
        )


class AsyncAzureCosmosDBNoSqlVectorStoreRetriever(VectorStoreRetriever):
    """Async retriever that uses `Azure CosmosDB No Sql Search`."""

    vectorstore: AsyncAzureCosmosDBNoSqlVectorSearch  # type: ignore[assignment]
    """Azure Search instance used to find similar documents."""
    search_type: str = "vector"
    """Type of search to perform. Options are "vector",
    "hybrid", "full_text_ranking", "full_text_search"."""
    k: int = 5
    """Number of documents to return."""
    search_kwargs: dict = {}
    """Search params.
        with_embedding:
        offset_limit:
        projection_mapping:
        full_text_rank_filter:
        where:
        weights:
        score_threshold: Minimum relevance threshold
            for vector_score_threshold and hybrid_score_threshold
        fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
        lambda_mult: Diversity of results returned by MMR;
            1 for minimum diversity and 0 for maximum. (Default: 0.5)
        filter: Filter by document metadata
    """

    allowed_search_types: ClassVar[Collection[str]] = (
        "vector",
        "vector_score_threshold",
        "full_text_search",
        "full_text_ranking",
        "hybrid",
        "hybrid_score_threshold",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: Dict) -> Any:
        """Validate search type."""
        if "search_type" in values:
            search_type = values["search_type"]
            if search_type not in cls.allowed_search_types:
                raise ValueError(
                    f"search_type of {search_type} not allowed. "
                    f"Valid values are: {cls.allowed_search_types}"
                )
        return values

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Not implemented. Use ``_aget_relevant_documents`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Use the async method `_aget_relevant_documents` instead."
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant to query asynchronously.

        Args:
            query: String to find relevant documents for.
            run_manager: Callback manager.
            **kwargs: Additional keyword arguments.

        Returns:
            List of relevant documents.
        """
        with_embedding = self.search_kwargs.get("with_embedding", False)
        offset_limit = self.search_kwargs.get("offset_limit", None)
        projection_mapping = self.search_kwargs.get("projection_mapping", None)
        full_text_rank_filter = self.search_kwargs.get("full_text_rank_filter", None)
        where = self.search_kwargs.get("where", None)
        weights = self.search_kwargs.get("weights", None)
        score_threshold = self.search_kwargs.get("score_threshold", 0.0)

        if self.search_type == "vector":
            docs = await self.vectorstore.asimilarity_search(
                query,
                k=self.k,
                search_type=self.search_type,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                where=where,
            )
        elif self.search_type == "vector_score_threshold":
            docs = await self.vectorstore.asimilarity_search(
                query,
                k=self.k,
                search_type=self.search_type,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                where=where,
                threshold=score_threshold,
            )
        elif self.search_type == "hybrid":
            docs = await self.vectorstore.asimilarity_search(
                query,
                k=self.k,
                search_type=self.search_type,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                full_text_rank_filter=full_text_rank_filter,
                where=where,
                weights=weights,
            )
        elif self.search_type == "hybrid_score_threshold":
            docs = await self.vectorstore.asimilarity_search(
                query,
                k=self.k,
                search_type=self.search_type,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                full_text_rank_filter=full_text_rank_filter,
                where=where,
                weights=weights,
                threshold=score_threshold,
            )
        elif self.search_type == "full_text_ranking":
            docs = await self.vectorstore.asimilarity_search(
                query,
                k=self.k,
                search_type=self.search_type,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                full_text_rank_filter=full_text_rank_filter,
                where=where,
            )
        elif self.search_type == "full_text_search":
            docs = await self.vectorstore.asimilarity_search(
                query,
                k=self.k,
                search_type=self.search_type,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                where=where,
            )
        else:
            raise ValueError(f"Query type of {self.search_type} is not allowed.")
        return docs
