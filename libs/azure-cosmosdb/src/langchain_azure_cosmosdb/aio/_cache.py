"""Async Semantic Cache for Azure CosmosDB NoSql API."""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from langchain_azure_cosmosdb._cache import (
    _hash,
    _load_generations_from_json,
)
from langchain_azure_cosmosdb.aio._vectorstore import (
    AsyncAzureCosmosDBNoSqlVectorSearch,
)
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

logger = logging.getLogger(__file__)


class AsyncAzureCosmosDBNoSqlSemanticCache(BaseCache):
    """Async cache that uses Cosmos DB NoSQL backend."""

    def __init__(
        self,
        embedding: Embeddings,
        *,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        vector_search_fields: Dict[str, Any],
        database_name: str = "CosmosNoSqlCacheDB",
        container_name: str = "CosmosNoSqlCacheContainer",
        search_type: str = "vector",
        create_container: bool = True,
    ) -> None:
        """AsyncAzureCosmosDBNoSqlSemanticCache constructor.

        Use the ``create`` classmethod to build a fully initialised
        instance with an async Cosmos client.

        Args:
            embedding: CosmosDB Embedding.
            vector_embedding_policy: CosmosDB vector embedding policy.
            indexing_policy: CosmosDB indexing policy.
            cosmos_container_properties: CosmosDB container properties.
            cosmos_database_properties: CosmosDB database properties.
            vector_search_fields: Vector Search Fields for the container.
            database_name: CosmosDB database name.
            container_name: CosmosDB container name.
            search_type: CosmosDB search type.
            create_container: Create the container if it doesn't exist.
        """
        self.database_name = database_name
        self.container_name = container_name
        self.embedding = embedding
        self.vector_embedding_policy = vector_embedding_policy
        self.indexing_policy = indexing_policy
        self.cosmos_container_properties = cosmos_container_properties
        self.cosmos_database_properties = cosmos_database_properties
        self.vector_search_fields = vector_search_fields
        self.search_type = search_type
        self.create_container = create_container
        self._cosmos_client: Any = None
        self._cache_dict: Dict[str, AsyncAzureCosmosDBNoSqlVectorSearch] = {}

        # Extract partition key path parts for use in aclear().
        try:
            pk_def = cosmos_container_properties.get("partition_key")
            if pk_def is not None:
                pk_path = pk_def.get("paths", ["/id"])[0]
                parts = [p for p in pk_path.split("/") if p]
                self._pk_parts = parts if parts else ["id"]
                self._pk_sql = ".".join(self._pk_parts)
            else:
                self._pk_parts = ["id"]
                self._pk_sql = "id"
        except (AttributeError, TypeError, KeyError, IndexError):
            self._pk_parts = ["id"]
            self._pk_sql = "id"

    @classmethod
    async def create(
        cls,
        embedding: Embeddings,
        cosmos_client: Any,
        database_name: str = "CosmosNoSqlCacheDB",
        container_name: str = "CosmosNoSqlCacheContainer",
        *,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        vector_search_fields: Dict[str, Any],
        search_type: str = "vector",
        create_container: bool = True,
    ) -> AsyncAzureCosmosDBNoSqlSemanticCache:
        """Async factory to create an AsyncAzureCosmosDBNoSqlSemanticCache.

        Args:
            embedding: CosmosDB Embedding.
            cosmos_client: Async CosmosDB client.
            database_name: CosmosDB database name.
            container_name: CosmosDB container name.
            vector_embedding_policy: CosmosDB vector embedding policy.
            indexing_policy: CosmosDB indexing policy.
            cosmos_container_properties: CosmosDB container properties.
            cosmos_database_properties: CosmosDB database properties.
            vector_search_fields: Vector Search Fields for the container.
            search_type: CosmosDB search type.
            create_container: Create the container if it doesn't exist.

        Returns:
            An initialised AsyncAzureCosmosDBNoSqlSemanticCache instance.
        """
        instance = cls(
            embedding=embedding,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            vector_search_fields=vector_search_fields,
            database_name=database_name,
            container_name=container_name,
            search_type=search_type,
            create_container=create_container,
        )
        instance._cosmos_client = cosmos_client
        return instance

    def _cache_name(self, llm_string: str) -> str:
        """Return cache key name for the given llm_string.

        Args:
            llm_string: LLM identifier string.

        Returns:
            A hashed cache key.
        """
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    async def _aget_llm_cache(
        self, llm_string: str
    ) -> AsyncAzureCosmosDBNoSqlVectorSearch:
        """Get or create the async vectorstore for the given llm_string.

        Args:
            llm_string: LLM identifier string.

        Returns:
            An AsyncAzureCosmosDBNoSqlVectorSearch instance.
        """
        cache_name = self._cache_name(llm_string)

        if cache_name in self._cache_dict:
            return self._cache_dict[cache_name]

        if self._cosmos_client:
            vs = await AsyncAzureCosmosDBNoSqlVectorSearch.create(
                cosmos_client=self._cosmos_client,
                embedding=self.embedding,
                vector_embedding_policy=self.vector_embedding_policy,
                indexing_policy=self.indexing_policy,
                cosmos_container_properties=self.cosmos_container_properties,
                cosmos_database_properties=self.cosmos_database_properties,
                database_name=self.database_name,
                container_name=self.container_name,
                search_type=self.search_type,
                vector_search_fields=self.vector_search_fields,
                create_container=self.create_container,
            )
            self._cache_dict[cache_name] = vs
        else:
            raise ValueError("CosmosDB client is not configured.")

        return self._cache_dict[cache_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Not implemented. Use ``alookup`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async method `alookup` instead.")

    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt asynchronously.

        Args:
            prompt: The prompt to look up.
            llm_string: LLM identifier string.

        Returns:
            Cached generations, or None on cache miss.
        """
        llm_cache = await self._aget_llm_cache(llm_string)
        generations: List[Any] = []
        results = await llm_cache.asimilarity_search(
            query=prompt,
            k=1,
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be "
                        "deserialized properly. This is likely due to "
                        "the cache being in an older format. Please "
                        "recreate your cache to avoid this error."
                    )
                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: RETURN_VAL_TYPE,
    ) -> None:
        """Not implemented. Use ``aupdate`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async method `aupdate` instead.")

    async def aupdate(
        self,
        prompt: str,
        llm_string: str,
        return_val: RETURN_VAL_TYPE,
    ) -> None:
        """Update cache based on prompt and llm_string asynchronously.

        Args:
            prompt: The prompt to cache.
            llm_string: LLM identifier string.
            return_val: The generations to cache.
        """
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "AsyncAzureCosmosDBNoSqlSemanticCache only supports "
                    "caching of normal LLM generations, got "
                    f"{type(gen)}"
                )
        llm_cache = await self._aget_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        await llm_cache.aadd_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Not implemented. Use ``aclear`` instead.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Use the async method `aclear` instead.")

    async def aclear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string asynchronously.

        If ``llm_string`` is not provided, clears all cached data.

        Args:
            **kwargs: May contain ``llm_string`` key.
        """
        llm_string = kwargs.get("llm_string")
        pk_sql = self._pk_sql
        if pk_sql == "id":
            query = "SELECT c.id FROM c"
        else:
            query = f"SELECT c.id, c.{pk_sql} FROM c"
        if llm_string is not None:
            cache_name = self._cache_name(llm_string=llm_string)
            if cache_name in self._cache_dict:
                vs = self._cache_dict[cache_name]
                container = vs._container
                items: list[Any] = []
                async for item in container.query_items(
                    query=query,
                ):
                    items.append(item)
                for item in items:
                    pk_key = self._pk_parts[-1]
                    pk_val = item[pk_key] if pk_key in item else item["id"]
                    await container.delete_item(
                        item=item["id"],
                        partition_key=pk_val,
                    )
                del self._cache_dict[cache_name]
        else:
            for cache_name in list(self._cache_dict):
                vs = self._cache_dict[cache_name]
                container = vs._container
                items = []
                async for item in container.query_items(
                    query=query,
                ):
                    items.append(item)
                for item in items:
                    pk_key = self._pk_parts[-1]
                    pk_val = item[pk_key] if pk_key in item else item["id"]
                    await container.delete_item(
                        item=item["id"],
                        partition_key=pk_val,
                    )
            self._cache_dict.clear()
