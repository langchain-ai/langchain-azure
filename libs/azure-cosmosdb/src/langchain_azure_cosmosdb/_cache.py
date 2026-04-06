"""Semantic Cache for Azure CosmosDB NoSql API."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from azure.cosmos import CosmosClient
from langchain_azure_cosmosdb._vectorstore import (
    AzureCosmosDBNoSqlVectorSearch,
)
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

logger = logging.getLogger(__file__)


def _get_nested(d: Any, parts: List[str]) -> Any:
    """Traverse a nested dict using a list of keys.

    Args:
        d: The dict to traverse.
        parts: List of keys, e.g. ``["metadata", "prompt"]``.

    Returns:
        The value at the nested path, or None if not found.
    """
    for part in parts:
        if isinstance(d, dict):
            d = d.get(part)
        else:
            return None
    return d


def _hash(_input: str) -> str:
    """Use a deterministic hashing approach."""
    return hashlib.sha256(_input.encode()).hexdigest()


def _dump_generations_to_json(generations: RETURN_VAL_TYPE) -> str:
    """Dump generations to json.

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: Json representing a list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    return json.dumps([generation.dict() for generation in generations])


def _load_generations_from_json(generations_json: str) -> RETURN_VAL_TYPE:
    """Load generations from json.

    Args:
        generations_json (str): A string of json representing a list of generations.

    Raises:
        ValueError: Could not decode json string to list of generations.

    Returns:
        RETURN_VAL_TYPE: A list of generations.

    Warning: would not work well with arbitrary subclasses of `Generation`
    """
    try:
        results = json.loads(generations_json)
        return [Generation(**generation_dict) for generation_dict in results]
    except json.JSONDecodeError:
        raise ValueError(
            f"Could not decode json to list of generations: {generations_json}"
        )


def _dumps_generations(generations: RETURN_VAL_TYPE) -> str:
    """Serialization for generic RETURN_VAL_TYPE, i.e. sequence of `Generation`.

    Args:
        generations (RETURN_VAL_TYPE): A list of language model generations.

    Returns:
        str: a single string representing a list of generations.

    This function (+ its counterpart `_loads_generations`) rely on
    the dumps/loads pair with Reviver, so are able to deal
    with all subclasses of Generation.

    Each item in the list can be `dumps`ed to a string,
    then we make the whole list of strings into a json-dumped.
    """
    return json.dumps([dumps(_item) for _item in generations])


def _loads_generations(generations_str: str) -> Union[RETURN_VAL_TYPE, None]:
    """Deserialization of a string into a generic RETURN_VAL_TYPE.

    See `_dumps_generations`, the inverse of this function.

    Args:
        generations_str (str): A string representing a list of generations.

    Compatible with the legacy cache-blob format
    Does not raise exceptions for malformed entries, just logs a warning
    and returns none: the caller should be prepared for such a cache miss.

    Returns:
        RETURN_VAL_TYPE: A list of generations.
    """
    try:
        generations = [loads(_item_str) for _item_str in json.loads(generations_str)]
        return generations
    except (json.JSONDecodeError, TypeError):
        # deferring the (soft) handling to after the legacy-format attempt
        pass

    try:
        gen_dicts = json.loads(generations_str)
        # not relying on `_load_generations_from_json` (which could disappear):
        generations = [Generation(**generation_dict) for generation_dict in gen_dicts]
        logger.warning(
            f"Legacy 'Generation' cached blob encountered: '{generations_str}'"
        )
        return generations
    except (json.JSONDecodeError, TypeError):
        logger.warning(
            f"Malformed/unparsable cached blob encountered: '{generations_str}'"
        )
        return None


class AzureCosmosDBNoSqlSemanticCache(BaseCache):
    """Cache that uses Cosmos DB NoSQL backend."""

    def __init__(
        self,
        embedding: Embeddings,
        cosmos_client: CosmosClient,
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
    ) -> None:
        """AzureCosmosDBNoSqlSemanticCache constructor.

        Args:
            embedding: CosmosDB Embedding.
            cosmos_client: CosmosDB client
            database_name: CosmosDB database name
            container_name: CosmosDB container name
            vector_embedding_policy: CosmosDB vector embedding policy
            indexing_policy: CosmosDB indexing policy
            cosmos_container_properties: CosmosDB container properties
            cosmos_database_properties: CosmosDB database properties
            vector_search_fields: Vector Search Fields for the container.
            search_type: CosmosDB search type.
            create_container: Create the container if it doesn't exist.
        """
        self.cosmos_client = cosmos_client
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
        self._cache_dict: Dict[str, AzureCosmosDBNoSqlVectorSearch] = {}

        # Extract partition key path parts for use in clear().
        # E.g., "/metadata/prompt" → _pk_parts=["metadata","prompt"],
        #   _pk_sql="metadata.prompt"
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

    def _cache_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> AzureCosmosDBNoSqlVectorSearch:
        cache_name = self._cache_name(llm_string)

        # return vectorstore client for the specific llm string
        if cache_name in self._cache_dict:
            return self._cache_dict[cache_name]

        # create new vectorstore client to create the cache
        if self.cosmos_client:
            self._cache_dict[cache_name] = AzureCosmosDBNoSqlVectorSearch(
                cosmos_client=self.cosmos_client,
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
        else:
            raise ValueError("CosmosDB client is not configured.")

        return self._cache_dict[cache_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            k=1,
        )
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata["return_val"]))
                except Exception:
                    logger.warning(
                        "Retrieving a cache value that could not be deserialized "
                        "properly. This is likely due to the cache being in an "
                        "older format. Please recreate your cache to avoid this "
                        "error."
                    )

                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "CosmosDBNoSqlSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string.

        If ``llm_string`` is not provided, clears all cached data.
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
                items = list(
                    container.query_items(
                        query=query,
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    pk_key = self._pk_parts[-1]
                    pk_val = item[pk_key] if pk_key in item else item["id"]
                    container.delete_item(
                        item=item["id"],
                        partition_key=pk_val,
                    )
                del self._cache_dict[cache_name]
        else:
            for cache_name in list(self._cache_dict):
                vs = self._cache_dict[cache_name]
                container = vs._container
                items = list(
                    container.query_items(
                        query=query,
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    pk_key = self._pk_parts[-1]
                    pk_val = item[pk_key] if pk_key in item else item["id"]
                    container.delete_item(
                        item=item["id"],
                        partition_key=pk_val,
                    )
            self._cache_dict.clear()
