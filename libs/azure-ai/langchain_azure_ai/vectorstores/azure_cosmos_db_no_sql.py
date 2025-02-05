"""Azure CosmosDB No Sql Vector Store API."""

from __future__ import annotations

import uuid
import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field

from langchain_azure_ai.vectorstores.utils import maximal_marginal_relevance

if TYPE_CHECKING:
    from azure.cosmos import ContainerProxy, CosmosClient
    from azure.identity import DefaultAzureCredential

USER_AGENT = ("LangChain-CDBNoSql-VectorStore-Python",)


class Condition(BaseModel):
    """Condition class for where clause for query construction."""

    property: str
    operator: str
    value: Any


class PreFilter(BaseModel):
    """PreFilter class for query construction."""

    conditions: List[Condition] = Field(default_factory=list)
    logical_operator: Optional[str] = None


class CosmosDBQueryType(str, Enum):
    """CosmosDB Query Type."""

    VECTOR = "vector"
    FULL_TEXT_SEARCH = "full_text_search"
    FULL_TEXT_RANK = "full_text_rank"
    HYBRID = "hybrid"


class AzureCosmosDBNoSqlVectorSearch(VectorStore):
    """`Azure Cosmos DB for NoSQL` vector store.

    To use, you should have both:
        - the ``azure-cosmos`` python package installed
        - the ``azure-identity`` python package installed

    You can read more about vector search, full text search
    and hybrid search using AzureCosmosDBNoSQL here:
    https://learn.microsoft.com/en-us/azure/cosmos-db/nosql/vector-search
    https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/full-text-search
    https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/hybrid-search
    """

    def __init__(
        self,
        *,
        cosmos_client: CosmosClient,
        embedding: Embeddings,
        vector_embedding_policy: Dict[str, Any],
        indexing_policy: Dict[str, Any],
        cosmos_container_properties: Dict[str, Any],
        cosmos_database_properties: Dict[str, Any],
        vector_search_fields: Dict[str, Any],
        full_text_policy: Optional[Dict[str, Any]] = None,
        database_name: str = "vectorSearchDB",
        container_name: str = "vectorSearchContainer",
        query_type: CosmosDBQueryType = CosmosDBQueryType.VECTOR,
        metadata_key: str = "metadata",
        create_container: bool = True,
        full_text_search_enabled: bool = False,
    ):
        """Constructor for AzureCosmosDBNoSqlVectorSearch.

        Args:
            cosmos_client: Client used to connect to azure cosmosdb no sql account.
            database_name: Name of the database to be created.
            container_name: Name of the container to be created.
            embedding: Text embedding model to use.
            vector_embedding_policy: Vector Embedding Policy for the container.
            full_text_policy: Full Text Policy for the container.
            indexing_policy: Indexing Policy for the container.
            cosmos_container_properties: Container Properties for the container.
            cosmos_database_properties: Database Properties for the container.
            vector_search_fields: Vector Search Fields for the container.
            query_type: CosmosDB Query Type to be performed.
            metadata_key: Metadata key to use for data schema.
            create_container: Set to true if the container does not exist.
            full_text_search_enabled: Set to true if the full text search is enabled.
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
        self._query_type = query_type

        if self._create_container:
            if (
                self._indexing_policy["vectorIndexes"] is None
                or len(self._indexing_policy["vectorIndexes"]) == 0
            ):
                raise ValueError(
                    "vectorIndexes cannot be null or empty in the indexing_policy."
                )
            if (
                self._vector_embedding_policy is None
                or len(vector_embedding_policy["vectorEmbeddings"]) == 0
            ):
                raise ValueError(
                    "vectorEmbeddings cannot be null "
                    "or empty in the vector_embedding_policy."
                )
            if self._cosmos_container_properties["partition_key"] is None:
                raise ValueError(
                    "partition_key cannot be null or empty for a container."
                )
            if self._full_text_search_enabled:
                if (
                    self._indexing_policy["fullTextIndexes"] is None
                    or len(self._indexing_policy["fullTextIndexes"]) == 0
                ):
                    raise ValueError(
                        "fullTextIndexes cannot be null or empty in the "
                        "indexing_policy if full text search is enabled."
                    )
                if (
                    self._full_text_policy is None
                    or len(self._full_text_policy["fullTextPaths"]) == 0
                ):
                    raise ValueError(
                        "fullTextPaths cannot be null or empty in the "
                        "full_text_policy if full text search is enabled."
                    )
        if self._vector_search_fields is None:
            raise ValueError(
                "vectorSearchFields cannot be null or empty in the vector_search_fields."  # noqa: E501
            )

        # Create the database if it already doesn't exist
        self._database = self._cosmos_client.create_database_if_not_exists(
            id=self._database_name,
            offer_throughput=self._cosmos_database_properties.get("offer_throughput"),
            session_token=self._cosmos_database_properties.get("session_token"),
            initial_headers=self._cosmos_database_properties.get("initial_headers"),
            etag=self._cosmos_database_properties.get("etag"),
            match_condition=self._cosmos_database_properties.get("match_condition"),
        )

        # Create the collection if it already doesn't exist
        self._container = self._database.create_container_if_not_exists(
            id=self._container_name,
            partition_key=self._cosmos_container_properties["partition_key"],
            indexing_policy=self._indexing_policy,
            default_ttl=self._cosmos_container_properties.get("default_ttl"),
            offer_throughput=self._cosmos_container_properties.get("offer_throughput"),
            unique_key_policy=self._cosmos_container_properties.get(
                "unique_key_policy"
            ),
            conflict_resolution_policy=self._cosmos_container_properties.get(
                "conflict_resolution_policy"
            ),
            analytical_storage_ttl=self._cosmos_container_properties.get(
                "analytical_storage_ttl"
            ),
            computed_properties=self._cosmos_container_properties.get(
                "computed_properties"
            ),
            etag=self._cosmos_container_properties.get("etag"),
            match_condition=self._cosmos_container_properties.get("match_condition"),
            session_token=self._cosmos_container_properties.get("session_token"),
            initial_headers=self._cosmos_container_properties.get("initial_headers"),
            vector_embedding_policy=self._vector_embedding_policy,
            full_text_policy=self._full_text_policy,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            **kwargs: vectorstore specific parameters.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        _metadatas = list(metadatas if metadatas is not None else ({} for _ in texts))

        return self._insert_texts(list(texts), _metadatas)

    def _insert_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """Used to Load Documents into the collection.

        Args:
            texts: The list of documents strings to load
            metadatas: The list of metadata objects associated with each document

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # If the texts is empty, throw an error
        if not texts:
            raise Exception("Texts can not be null or empty")

        # Embed and create the documents
        embeddings = self._embedding.embed_documents(texts)
        text_key = self._vector_search_fields["text_field"]
        embedding_key = self._vector_search_fields["embedding_field"]

        to_insert = [
            {
                "id": str(uuid.uuid4()),
                text_key: t,
                embedding_key: embedding,
                "metadata": m,
            }
            for t, m, embedding in zip(texts, metadatas, embeddings)
        ]
        # insert the documents in CosmosDB No Sql
        doc_ids: List[str] = []
        for item in to_insert:
            created_doc = self._container.create_item(item)
            doc_ids.append(created_doc["id"])
        return doc_ids

    @classmethod
    def _from_kwargs(
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
        query_type: CosmosDBQueryType = CosmosDBQueryType.VECTOR,
        **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        if kwargs:
            warnings.warn(
                "Method 'from_texts' of AzureCosmosDBNoSql vector "
                "store invoked with "
                f"unsupported arguments "
                f"({', '.join(sorted(kwargs))}), "
                "which will be ignored."
            )

        return cls(
            embedding=embedding,
            cosmos_client=cosmos_client,
            vector_embedding_policy=vector_embedding_policy,
            full_text_policy=full_text_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            vector_search_fields=vector_search_fields,
            database_name=database_name,
            container_name=container_name,
            metadata_key=metadata_key,
            create_container=create_container,
            full_text_search_enabled=full_text_search_enabled,
            query_type=query_type,
        )

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        """Create an AzureCosmosDBNoSqlVectorSearch vectorstore from raw texts.

        Args:
            texts: the texts to insert.
            embedding: the embedding function to use in the store.
            metadatas: metadata dicts for the texts.
            **kwargs: you can pass any argument that you would
                to :meth:`~add_texts` and/or to the 'AstraDB' constructor
                (see these methods for details). These arguments will be
                routed to the respective methods as they are.

        Returns:
            an `AzureCosmosDBNoSqlVectorSearch` vectorstore.
        """
        vectorstore = AzureCosmosDBNoSqlVectorSearch._from_kwargs(embedding, **kwargs)
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        return vectorstore

    @classmethod
    def from_endpoint_and_aad(
        cls,
        connection_string: str,
        defaultAzureCredential: DefaultAzureCredential,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        """Create an AzureCosmosDBNoSqlVectorSearch vectorstore using managed identity."""  # noqa: E501
        cosmos_client = CosmosClient(
            connection_string, defaultAzureCredential, user_agent=USER_AGENT
        )
        kwargs["cosmos_client"] = cosmos_client
        vectorstore = cls._from_kwargs(embedding, **kwargs)
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        return vectorstore

    @classmethod
    def from_endpoint_and_key(
        cls,
        connection_string: str,
        key: str,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> AzureCosmosDBNoSqlVectorSearch:
        """Create an AzureCosmosDBNoSqlVectorSearch vectorstore using endpoint and key."""  # noqa: E501
        cosmos_client = CosmosClient(connection_string, key, user_agent=USER_AGENT)
        kwargs["cosmos_client"] = cosmos_client
        vectorstore = cls._from_kwargs(embedding, **kwargs)
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        return vectorstore

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Removed all documents by ids in the list.

        Args:
            ids: the ids to delete.
            **kwargs: vectorstore specific parameters.
        """
        if ids is None:
            raise ValueError("No document ids provided to delete.")

        for document_id in ids:
            self._container.delete_item(
                document_id, self._cosmos_container_properties["partition_key"]
            )  # noqa: E501
        return True

    def delete_document_by_id(self, document_id: Optional[str] = None) -> None:
        """Removes a Specific Document by id.

        Args:
            document_id: The document identifier
        """
        if document_id is None:
            raise ValueError("No document ids provided to delete.")
        self._container.delete_item(document_id, partition_key=document_id)

    def _similarity_search_with_score(
        self,
        query_type: CosmosDBQueryType,
        embeddings: List[float],
        k: int = 4,
        pre_filter: Optional[PreFilter] = None,
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query, parameters = self._construct_query(
            k=k,
            query_type=query_type,
            embeddings=embeddings,
            pre_filter=pre_filter,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            with_embedding=with_embedding,
        )

        return self._execute_query(
            query=query,
            query_type=query_type,
            parameters=parameters,
            with_embedding=with_embedding,
            projection_mapping=projection_mapping,
        )

    def _full_text_search(
        self,
        query_type: CosmosDBQueryType,
        search_text: Optional[str] = None,
        k: int = 4,
        pre_filter: Optional[PreFilter] = None,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query, parameters = self._construct_query(
            k=k,
            query_type=query_type,
            search_text=search_text,
            pre_filter=pre_filter,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            full_text_rank_filter=full_text_rank_filter,
        )

        return self._execute_query(
            query=query,
            query_type=query_type,
            parameters=parameters,
            with_embedding=False,
            projection_mapping=projection_mapping,
        )

    def _hybrid_search_with_score(
        self,
        query_type: CosmosDBQueryType,
        embeddings: List[float],
        search_text: str,
        k: int = 4,
        pre_filter: Optional[PreFilter] = None,
        with_embedding: bool = False,
        offset_limit: Optional[str] = None,
        *,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        query, parameters = self._construct_query(
            k=k,
            query_type=query_type,
            embeddings=embeddings,
            search_text=search_text,
            pre_filter=pre_filter,
            offset_limit=offset_limit,
            projection_mapping=projection_mapping,
            full_text_rank_filter=full_text_rank_filter,
        )
        return self._execute_query(
            query=query,
            query_type=query_type,
            parameters=parameters,
            with_embedding=with_embedding,
            projection_mapping=projection_mapping,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[PreFilter] = None,
        with_embedding: bool = False,
        query_type: CosmosDBQueryType = CosmosDBQueryType.VECTOR,
        offset_limit: Optional[str] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Retrieves similar documents with score."""
        embeddings = self._embedding.embed_query(query)
        docs_and_scores = []
        if query_type == CosmosDBQueryType.VECTOR:
            docs_and_scores = self._similarity_search_with_score(
                query_type=query_type,
                embeddings=embeddings,
                k=k,
                pre_filter=pre_filter,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                **kwargs,
            )
        elif query_type == CosmosDBQueryType.FULL_TEXT_SEARCH:
            docs_and_scores = self._full_text_search(
                k=k,
                query_type=query_type,
                pre_filter=pre_filter,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                **kwargs,
            )

        elif query_type == CosmosDBQueryType.FULL_TEXT_RANK:
            docs_and_scores = self._full_text_search(
                search_text=query,
                k=k,
                query_type=query_type,
                pre_filter=pre_filter,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                **kwargs,
            )
        elif query_type == CosmosDBQueryType.HYBRID:
            docs_and_scores = self._hybrid_search_with_score(
                query_type=query_type,
                embeddings=embeddings,
                search_text=query,
                k=k,
                pre_filter=pre_filter,
                with_embedding=with_embedding,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                **kwargs,
            )
        return docs_and_scores

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[PreFilter] = None,
        with_embedding: bool = False,
        query_type: CosmosDBQueryType = CosmosDBQueryType.VECTOR,
        offset_limit: Optional[str] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieves similar documents."""
        if query_type not in CosmosDBQueryType.__members__.values():
            raise ValueError(
                f"Invalid query_type: {query_type}. "
                f"Expected one of: {', '.join(t.value for t in CosmosDBQueryType)}."
            )
        else:
            docs_and_scores = self.similarity_search_with_score(
                query,
                k=k,
                pre_filter=pre_filter,
                with_embedding=with_embedding,
                query_type=query_type,
                offset_limit=offset_limit,
                projection_mapping=projection_mapping,
                full_text_rank_filter=full_text_rank_filter,
                kwargs=kwargs,
            )

        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_type: CosmosDBQueryType = CosmosDBQueryType.VECTOR,
        pre_filter: Optional[PreFilter] = None,
        with_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieves the docs with similarity scores."""
        docs = self._similarity_search_with_score(
            embeddings=embedding,
            k=fetch_k,
            query_type=query_type,
            pre_filter=pre_filter,
            with_embedding=with_embedding,
        )

        # Re-ranks the docs using MMR
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding),
            [doc.metadata[self._embedding_key] for doc, _ in docs],
            k=k,
            lambda_mult=lambda_mult,
        )

        mmr_docs = [docs[i][0] for i in mmr_doc_indexes]
        return mmr_docs

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        query_type: CosmosDBQueryType = CosmosDBQueryType.VECTOR,
        pre_filter: Optional[PreFilter] = None,
        with_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieves the similar docs."""
        embeddings = self._embedding.embed_query(query)

        docs = self.max_marginal_relevance_search_by_vector(
            embeddings,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            pre_filter=pre_filter,
            query_type=query_type,
            with_embedding=with_embedding,
        )
        return docs

    def _construct_query(
        self,
        k: int,
        query_type: CosmosDBQueryType,
        embeddings: List[float] = None,
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        pre_filter: Optional[PreFilter] = None,
        offset_limit: Optional[str] = None,
        projection_mapping: Optional[Dict[str, Any]] = None,
        with_embedding: bool = False,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if (
            query_type == CosmosDBQueryType.FULL_TEXT_RANK
            or query_type == CosmosDBQueryType.HYBRID
        ):
            query = f"SELECT {'TOP ' + str(k) + ' ' if not offset_limit else ''}"
        else:
            query = f"""SELECT {'TOP @limit ' if not offset_limit else ''}"""
        query += self._generate_projection_fields(
            projection_mapping,
            query_type,
            embeddings,
            full_text_rank_filter,
            with_embedding,
        )
        query += " FROM c "

        # Add where_clause if specified
        if pre_filter:
            where_clause = self._build_where_clause(pre_filter)
            query += f"""{where_clause}"""

        # TODO: Update the code to use parameters once parametrized queries
        #  are allowed for these query functions
        if query_type == CosmosDBQueryType.FULL_TEXT_RANK:
            if full_text_rank_filter is None:
                raise ValueError(
                    "full_text_rank_filter cannot be None for FULL_TEXT_RANK queries."
                )
            if len(full_text_rank_filter) == 1:
                query += f""" ORDER BY RANK FullTextScore(c.{full_text_rank_filter[0]["search_field"]}, 
                [{", ".join(f"'{term}'" for term in full_text_rank_filter[0]["search_text"].split())}])"""  # noqa: E501
            else:
                rank_components = [
                    f"FullTextScore(c.{search_item['search_field']}, ["
                    + ", ".join(
                        f"'{term}'" for term in search_item["search_text"].split()
                    )
                    + "])"
                    for search_item in full_text_rank_filter
                ]
                query = f" ORDER BY RANK RRF({', '.join(rank_components)})"
        elif query_type == CosmosDBQueryType.VECTOR:
            query += " ORDER BY VectorDistance(c[@embeddingKey], @embeddings)"
        elif query_type == CosmosDBQueryType.HYBRID:
            if full_text_rank_filter is None:
                raise ValueError(
                    "full_text_rank_filter cannot be None for HYBRID queries."
                )
            rank_components = [
                f"FullTextScore(c.{search_item['search_field']}, ["
                + ", ".join(f"'{term}'" for term in search_item["search_text"].split())
                + "])"
                for search_item in full_text_rank_filter
            ]
            query += f""" ORDER BY RANK RRF({', '.join(rank_components)}, 
            VectorDistance(c.{self._vector_search_fields["embedding_field"]}, {embeddings}))"""  # noqa: E501
        else:
            query += ""

        # Add limit_offset_clause if specified
        if offset_limit is not None:
            query += f""" {offset_limit}"""

        # TODO: Remove this if check once parametrized queries
        #  are allowed for these query functions
        parameters = []
        if (
            query_type == CosmosDBQueryType.FULL_TEXT_SEARCH
            or query_type == CosmosDBQueryType.VECTOR
        ):
            parameters = self._build_parameters(
                k=k,
                query_type=query_type,
                embeddings=embeddings,
                projection_mapping=projection_mapping,
            )
        return query, parameters

    def _generate_projection_fields(
        self,
        projection_mapping: Optional[Dict[str, Any]],
        query_type: CosmosDBQueryType,
        embeddings: List[float],
        full_text_rank_filter: Optional[List[Dict[str, str]]] = None,
        with_embedding: bool = False,
    ) -> str:
        # TODO: Remove the if check, lines 704-726 once parametrized queries
        #  are supported for these query functions.
        if (
            query_type == CosmosDBQueryType.FULL_TEXT_RANK
            or query_type == CosmosDBQueryType.HYBRID
        ):
            if projection_mapping:
                projection = ", ".join(
                    f"c.{key} as {alias}" for key, alias in projection_mapping.items()
                )
            elif full_text_rank_filter:
                projection = "c.id, " + ", ".join(
                    f"c.{search_item['search_field']} as {search_item['search_field']}"
                    for search_item in full_text_rank_filter
                )
            else:
                projection = f"c.id, c.{self._vector_search_fields['text_field']} as text, c.{self._metadata_key} as metadata"  # noqa: E501
            if query_type == CosmosDBQueryType.HYBRID:
                if with_embedding:
                    projection += f", c.{self._vector_search_fields['embedding_field']} as embedding"  # noqa: E501
                projection += (
                    f", VectorDistance(c.{self._vector_search_fields['embedding_field']}, "  # noqa: E501
                    f"{embeddings}) as SimilarityScore"
                )
        else:
            if projection_mapping:
                projection = ", ".join(
                    f"c.[@{key}] as {alias}"
                    for key, alias in projection_mapping.items()
                )
            elif full_text_rank_filter:
                projection = "c.id" + ", ".join(
                    f"c.{search_item['search_field']} as {search_item['search_field']}"
                    for search_item in full_text_rank_filter
                )
            else:
                projection = "c.id, c[@textKey] as text, c[@metadataKey] as metadata"

            if query_type == CosmosDBQueryType.VECTOR:
                if with_embedding:
                    projection += ", c[@embeddingKey] as embedding, "
                projection += (
                    ", VectorDistance(c[@embeddingKey], "
                    "@embeddings) as SimilarityScore"
                )
        return projection

    def _build_parameters(
        self,
        k: int,
        query_type: CosmosDBQueryType,
        embeddings: Optional[List[float]],
        projection_mapping: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        parameters: List[Dict[str, Any]] = [
            {"name": "@limit", "value": k},
        ]

        if projection_mapping:
            for key in projection_mapping.keys():
                parameters.append({"name": f"@{key}", "value": key})
        else:
            parameters.append(
                {"name": "@textKey", "value": self._vector_search_fields["text_field"]}
            )
            parameters.append({"name": "@metadataKey", "value": self._metadata_key})

        if query_type == CosmosDBQueryType.VECTOR:
            parameters.append(
                {
                    "name": "@embeddingKey",
                    "value": self._vector_search_fields["embedding_field"],
                }
            )
            parameters.append({"name": "@embeddings", "value": embeddings})

        return parameters

    def _build_where_clause(self, pre_filter: PreFilter) -> str:
        """Builds a where clause based on the given pre_filter."""
        operator_map = self._where_clause_operator_map()

        if (
            pre_filter.logical_operator
            and pre_filter.logical_operator not in operator_map
        ):
            raise ValueError(
                f"unsupported logical_operator: {pre_filter.logical_operator}"
            )

        sql_logical_operator = operator_map.get(pre_filter.logical_operator or "", "")
        clauses = []

        for condition in pre_filter.conditions:
            if condition.operator not in operator_map:
                raise ValueError(f"Unsupported operator: {condition.operator}")

            if "full_text" in condition.operator:
                if not isinstance(condition.value, str):
                    raise ValueError(
                        f"Expected a string for {condition.operator}, "
                        f"got {type(condition.value)}"
                    )
                search_terms = ", ".join(
                    f"'{term}'" for term in condition.value.split()
                )
                sql_function = operator_map[condition.operator]
                clauses.append(
                    f"{sql_function}(c.{condition.property}, {search_terms})"
                )
            else:
                sql_operator = operator_map[condition.operator]
                if isinstance(condition.value, list):
                    # e.g., for IN clauses
                    value = f"({', '.join(map(str, condition.value))})"
                elif isinstance(condition.value, str):
                    value = f"'{condition.value}'"
                else:
                    value = str(condition.value)
                clauses.append(f"c.{condition.property} {sql_operator} {value}")
        return f""" WHERE {' {} '.format(sql_logical_operator).join(clauses)}""".strip()

    def _execute_query(
        self,
        query: str,
        query_type: CosmosDBQueryType,
        parameters: List[Dict[str, Any]],
        with_embedding: bool,
        projection_mapping: Optional[Dict[str, Any]],
    ) -> List[Tuple[Document, float]]:
        docs_and_scores = []
        items = list(
            self._container.query_items(
                query=query, parameters=parameters, enable_cross_partition_query=True
            )
        )
        for item in items:
            text = item[self._vector_search_fields["text_field"]]
            metadata = item.pop(self._metadata_key, {})
            score = 0.0

            if projection_mapping:
                for key, alias in projection_mapping.items():
                    if key == self._vector_search_fields["text_field"]:
                        continue
                    metadata[alias] = item[alias]
            else:
                metadata["id"] = item["id"]

            if (
                query_type == CosmosDBQueryType.VECTOR
                or query_type == CosmosDBQueryType.HYBRID
            ):
                score = item["SimilarityScore"]
                if with_embedding:
                    metadata[self._vector_search_fields["embedding_field"]] = item[
                        self._vector_search_fields["embedding_field"]
                    ]
            docs_and_scores.append(
                (Document(page_content=text, metadata=metadata), score)
            )
        return docs_and_scores

    def _where_clause_operator_map(self) -> Dict[str, str]:
        operator_map = {
            "$eq": "=",
            "$ne": "!=",
            "$lt": "<",
            "$lte": "<=",
            "$gt": ">",
            "$gte": ">=",
            "$add": "+",
            "$sub": "-",
            "$mul": "*",
            "$div": "/",
            "$mod": "%",
            "$or": "OR",
            "$and": "AND",
            "$not": "NOT",
            "$concat": "||",
            "$bit_or": "|",
            "$bit_and": "&",
            "$bit_xor": "^",
            "$bit_lshift": "<<",
            "$bit_rshift": ">>",
            "$bit_zerofill_rshift": ">>>",
            "$full_text_contains": "FullTextContains",
            "$full_text_contains_all": "FullTextContainsAll",
            "$full_text_contains_any": "FullTextContainsAny",
        }
        return operator_map

    def get_container(self) -> ContainerProxy:
        """Return vector store container."""
        return self._container

    def as_retriever(self, **kwargs: Any) -> AzureCosmosDBNoSqlVectorStoreRetriever:
        """Return AzureCosmosDBNoSqlVectorStoreRetriever initialized from this VectorStore.

        Args:
            query_type (Optional[CosmosDBQueryType]): Overrides the type of search that
                the Retriever should perform. Defaults to `self._query_type`.
                Can be "VECTOR", "HYBRID", or "FULL_TEXT_RANK".
            search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                search function. Can include things like:
                    score_threshold: Minimum relevance threshold
                        for similarity_score_threshold
                    fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
                    lambda_mult: Diversity of results returned by MMR;
                        1 for minimum diversity and 0 for maximum. (Default: 0.5)
                    filter: Filter by document metadata

        Returns:
            AzureCosmosDBNoSqlVectorStoreRetriever: Retriever class for VectorStore.
        """  # noqa: E501
        query_type = kwargs.get("query_type", self._query_type)
        kwargs["query_type"] = query_type

        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return AzureCosmosDBNoSqlVectorStoreRetriever(
            vectorstore=self, **kwargs, tags=tags
        )


class AzureCosmosDBNoSqlVectorStoreRetriever(BaseRetriever):
    """Retriever that uses `Azure CosmosDB No Sql Search`."""

    vectorstore: AzureCosmosDBNoSqlVectorSearch
    """Azure Search instance used to find similar documents."""
    query_type: CosmosDBQueryType = CosmosDBQueryType.VECTOR
    """Type of search to perform. Options are "VECTOR", "HYBRID",
    "FULL_TEXT_RANK", "FULL_TEXT_SEARCH"."""
    k: int = 4
    """Number of documents to return."""
    search_kwargs: dict = {}
    """Search params.
        score_threshold: Minimum relevance threshold
            for similarity_score_threshold
        fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
        lambda_mult: Diversity of results returned by MMR;
            1 for minimum diversity and 0 for maximum. (Default: 0.5)
        filter: Filter by document metadata
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def validate_search_type(cls, values: Dict) -> Any:
        """Validate search type."""
        if "query_type" in values:
            query_type = values["query_type"]
            if query_type not in CosmosDBQueryType.__members__.values():
                raise ValueError(
                    f"query_type of {query_type} not allowed. Valid values are: "
                    f"{CosmosDBQueryType.__members__.values()}"
                )
        return values

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        params = {**self.search_kwargs, **kwargs}

        if self.query_type == CosmosDBQueryType.VECTOR:
            docs = self.vectorstore.similarity_search(query, k=self.k, **params)
        elif self.query_type == CosmosDBQueryType.HYBRID:
            docs = self.vectorstore.similarity_search(
                query, k=self.k, query_type=CosmosDBQueryType.HYBRID, **params
            )
        elif self.query_type == CosmosDBQueryType.FULL_TEXT_RANK:
            docs = self.vectorstore.similarity_search(
                query, k=self.k, query_type=CosmosDBQueryType.FULL_TEXT_RANK, **params
            )
        elif self.query_type == CosmosDBQueryType.FULL_TEXT_SEARCH:
            docs = self.vectorstore.similarity_search(
                query, k=self.k, query_type=CosmosDBQueryType.FULL_TEXT_SEARCH, **params
            )
        else:
            raise ValueError(f"Query type of {self.query_type} is not allowed.")
        return docs
