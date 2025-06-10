"""Semantic Cache for Azure CosmosDB NoSql and Mongo vCore API."""

from __future__ import annotations

import hashlib
import json
import logging
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from azure.cosmos import CosmosClient
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import Generation

from langchain_azure_ai.vectorstores.azure_cosmos_db_mongo_vcore import (
    AzureCosmosDBMongoVCoreVectorSearch,
    CosmosDBSimilarityType,
    CosmosDBVectorSearchCompression,
    CosmosDBVectorSearchType,
)
from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)

logger = logging.getLogger(__file__)


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


class AzureCosmosDBMongoVCoreSemanticCache(BaseCache):
    """Cache that uses Cosmos DB Mongo vCore vector-store backend."""

    DEFAULT_DATABASE_NAME = "CosmosMongoVCoreCacheDB"
    DEFAULT_COLLECTION_NAME = "CosmosMongoVCoreCacheColl"

    def __init__(
        self,
        cosmosdb_connection_string: str,
        database_name: str,
        collection_name: str,
        embedding: Embeddings,
        *,
        cosmosdb_client: Optional[Any] = None,
        num_lists: int = 100,
        similarity: CosmosDBSimilarityType = CosmosDBSimilarityType.COS,
        kind: CosmosDBVectorSearchType = CosmosDBVectorSearchType.VECTOR_IVF,
        dimensions: int = 1536,
        m: int = 16,
        ef_construction: int = 64,
        max_degree: int = 32,
        l_build: int = 50,
        l_search: int = 40,
        ef_search: int = 40,
        application_name: str = "langchainpy",
        score_threshold: Optional[float] = None,
        compression: Optional[CosmosDBVectorSearchCompression] = None,
        pq_compressed_dims: Optional[int] = None,
        pq_sample_size: Optional[int] = None,
        oversampling: Optional[float] = None,
    ):
        """AzureCosmosDBMongoVCoreSemanticCache constructor.

        Args:
            cosmosdb_connection_string: Cosmos DB Mongo vCore connection string
            cosmosdb_client: Cosmos DB Mongo vCore client
            embedding (Embedding): Embedding provider for semantic encoding and search.
            database_name: Database name for the CosmosDBMongoVCoreSemanticCache
            collection_name: Collection name for the CosmosDBMongoVCoreSemanticCache
            num_lists: This integer is the number of clusters that the
                inverted file (IVF) index uses to group the vector data.
                We recommend that numLists is set to documentCount/1000
                for up to 1 million documents and to sqrt(documentCount)
                for more than 1 million documents.
                Using a numLists value of 1 is akin to performing
                brute-force search, which has limited performance
            dimensions: Number of dimensions for vector similarity.
                The maximum number of supported dimensions is 2000
            similarity: Similarity metric to use with the IVF index.

                Possible options are:
                    - CosmosDBSimilarityType.COS (cosine distance),
                    - CosmosDBSimilarityType.L2 (Euclidean distance), and
                    - CosmosDBSimilarityType.IP (inner product).
            kind: Type of vector index to create.
                Possible options are:
                    - vector-ivf
                    - vector-hnsw
                    - vector-diskann
            m: The max number of connections per layer (16 by default, minimum
               value is 2, maximum value is 100). Higher m is suitable for datasets
               with high dimensionality and/or high accuracy requirements.
            ef_construction: the size of the dynamic candidate list for constructing
                            the graph (64 by default, minimum value is 4, maximum
                            value is 1000). Higher ef_construction will result in
                            better index quality and higher accuracy, but it will
                            also increase the time required to build the index.
                            ef_construction has to be at least 2 * m
            ef_search: The size of the dynamic candidate list for search
                       (40 by default). A higher value provides better
                       recall at the cost of speed.
            max_degree: Max number of neighbors.
                Default value is 32, range from 20 to 2048.
                Only vector-diskann search supports this for now.
            l_build: l value for index building.
                Default value is 50, range from 10 to 500.
                Only vector-diskann search supports this for now.
            l_search: l value for index searching.
                Default value is 40, range from 10 to 10000.
                Only vector-diskann search supports this.
            score_threshold: Maximum score used to filter the vector search documents.
            application_name: Application name for the client for tracking and logging
            compression: compression type for vector indexes.
            pq_compressed_dims: Number of dimensions after compression for product
                quantization. Must be less than original dimensions. Automatically
                calculated if omitted. Range: 1-8000.
            pq_sample_size: Number of samples for PQ centroid training.
                Higher value means better quality but longer build time.
                Default: 1000. Range: 1000-100000.
            oversampling: The oversampling factor for compressed index.
                The oversampling factor (a float with a minimum of 1)
                specifies how many more candidate vectors to retrieve from the
                compressed index than k (the number of desired results).
        """
        self._validate_enum_value(similarity, CosmosDBSimilarityType)
        self._validate_enum_value(kind, CosmosDBVectorSearchType)
        if compression:
            self._validate_enum_value(compression, CosmosDBVectorSearchCompression)

        if not cosmosdb_connection_string:
            raise ValueError(" CosmosDB connection string can be empty.")

        self.cosmosdb_connection_string = cosmosdb_connection_string
        self.cosmosdb_client = cosmosdb_client
        self.embedding = embedding
        self.database_name = database_name or self.DEFAULT_DATABASE_NAME
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.num_lists = num_lists
        self.dimensions = dimensions
        self.similarity = similarity
        self.kind = kind
        self.m = m
        self.ef_construction = ef_construction
        self.max_degree = max_degree
        self.l_build = l_build
        self.l_search = l_search
        self.ef_search = ef_search
        self.score_threshold = score_threshold
        self._cache_dict: Dict[str, AzureCosmosDBMongoVCoreVectorSearch] = {}
        self.application_name = application_name
        self.compression = compression
        self.pq_compressed_dims = pq_compressed_dims
        self.pq_sample_size = pq_sample_size
        self.oversampling = oversampling

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache:{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> AzureCosmosDBMongoVCoreVectorSearch:
        index_name = self._index_name(llm_string)

        namespace = self.database_name + "." + self.collection_name

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        # create new vectorstore client for the specific llm string
        if self.cosmosdb_client:
            collection = self.cosmosdb_client[self.database_name][self.collection_name]
            self._cache_dict[index_name] = AzureCosmosDBMongoVCoreVectorSearch(
                collection=collection,
                embedding=self.embedding,
                index_name=index_name,
            )
        else:
            self._cache_dict[index_name] = (
                AzureCosmosDBMongoVCoreVectorSearch.from_connection_string(
                    connection_string=self.cosmosdb_connection_string,
                    namespace=namespace,
                    embedding=self.embedding,
                    index_name=index_name,
                    application_name=self.application_name,
                )
            )

        # create index for the vectorstore
        vectorstore = self._cache_dict[index_name]
        if not vectorstore.index_exists():
            vectorstore.create_index(
                self.num_lists,
                self.dimensions,
                self.similarity,
                self.kind,
                self.m,
                self.ef_construction,
                self.max_degree,
                self.l_build,
                compression=self.compression,
                pq_compressed_dims=self.pq_compressed_dims,
                pq_sample_size=self.pq_sample_size,
            )

        return vectorstore

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        # Read from a Hash
        results = llm_cache.similarity_search(
            query=prompt,
            k=1,
            kind=self.kind,
            ef_search=self.ef_search,
            l_search=self.l_search,
            score_threshold=self.score_threshold or 0.0,
            oversampling=self.oversampling,
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
                    # In a previous life we stored the raw text directly
                    # in the table, so assume it's in that format.
                    generations.extend(
                        _load_generations_from_json(document.metadata["return_val"])
                    )
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "CosmosDBMongoVCoreSemanticCache only supports caching of "
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
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].get_collection().delete_many({})

    @staticmethod
    def _validate_enum_value(value: Any, enum_type: Type[Enum]) -> None:
        if not isinstance(value, enum_type):
            raise ValueError(f"Invalid enum value: {value}. Expected {enum_type}.")


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
    ):
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
        """Clear semantic cache for a given llm_string."""
        cache_name = self._cache_name(llm_string=kwargs["llm_string"])
        if cache_name in self._cache_dict:
            self.cosmos_client.delete_database(database=self.database_name)
