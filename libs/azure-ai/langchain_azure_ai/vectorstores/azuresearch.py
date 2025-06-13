"""Vector store implementation for Azure Cognitive Search.

This module provides the AzureSearch vector store and retriever classes for
integration with Azure Cognitive Search.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import logging
import time
import uuid
from typing import (
    Any,
    Callable,
    ClassVar,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
from azure.core.credentials import TokenCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.search.documents import SearchClient, SearchItemPaged
from azure.search.documents.aio import AsyncSearchItemPaged
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes.models import (
    CorsOptions,
    ScoringProfile,
    SearchField,
    SemanticConfiguration,
    VectorSearch,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import LangChainException
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from pydantic import ConfigDict, model_validator

from langchain_azure_ai.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger()

# Allow overriding field names for Azure Search
FIELDS_ID = get_from_env(
    key="AZURESEARCH_FIELDS_ID", env_key="AZURESEARCH_FIELDS_ID", default="id"
)
FIELDS_CONTENT = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT",
    env_key="AZURESEARCH_FIELDS_CONTENT",
    default="content",
)
FIELDS_CONTENT_VECTOR = get_from_env(
    key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    env_key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
    default="content_vector",
)
FIELDS_METADATA = get_from_env(
    key="AZURESEARCH_FIELDS_TAG", env_key="AZURESEARCH_FIELDS_TAG", default="metadata"
)

MAX_UPLOAD_BATCH_SIZE = 1000


def _get_search_client(
    endpoint: str,
    index_name: str,
    key: Optional[str] = None,
    azure_ad_access_token: Optional[str] = None,
    semantic_configuration_name: Optional[str] = None,
    fields: Optional[List[SearchField]] = None,
    vector_search: Optional[VectorSearch] = None,
    semantic_configurations: Optional[
        Union[SemanticConfiguration, List[SemanticConfiguration]]
    ] = None,
    scoring_profiles: Optional[List[ScoringProfile]] = None,
    default_scoring_profile: Optional[str] = None,
    default_fields: Optional[List[SearchField]] = None,
    user_agent: Optional[str] = "langchain-comm-python-azure-search",
    cors_options: Optional[CorsOptions] = None,
    async_: bool = False,
    additional_search_client_options: Optional[Dict[str, Any]] = None,
    azure_credential: Optional[TokenCredential] = None,
    azure_async_credential: Optional[AsyncTokenCredential] = None,
) -> Union[SearchClient, AsyncSearchClient]:
    from azure.core.credentials import AccessToken, AzureKeyCredential, TokenCredential
    from azure.core.exceptions import ResourceNotFoundError
    from azure.identity import DefaultAzureCredential
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.aio import SearchClient as AsyncSearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        ExhaustiveKnnAlgorithmConfiguration,
        ExhaustiveKnnParameters,
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        VectorSearch,
        VectorSearchAlgorithmKind,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )

    class AzureBearerTokenCredential(TokenCredential):
        def __init__(self, token: str):
            # set the expiry to an hour from now.
            self._token = AccessToken(token, int(time.time()) + 3600)

        def get_token(
            self,
            *scopes: str,
            claims: Optional[str] = None,
            tenant_id: Optional[str] = None,
            enable_cae: bool = False,
            **kwargs: Any,
        ) -> AccessToken:
            return self._token

    additional_search_client_options = additional_search_client_options or {}
    default_fields = default_fields or []
    credential: Union[AzureKeyCredential, TokenCredential]
    async_credential: Union[AzureKeyCredential, AsyncTokenCredential]

    # Determine the appropriate credential to use
    if key is not None:
        # if key.upper() == "INTERACTIVE":
        #     interactive_credential = InteractiveBrowserCredential()
        #     interactive_credential.get_token("https://search.azure.com/.default")
        #     credential = interactive_credential
        #     async_credential = interactive_credential
        # else:
        credential = AzureKeyCredential(key)
        async_credential = credential
    # elif azure_ad_access_token is not None:
    #     bearer_credential = AzureBearerTokenCredential(azure_ad_access_token)
    #     credential = bearer_credential
    #     async_credential = bearer_credential
    else:
        credential = azure_credential or DefaultAzureCredential()
        async_credential = azure_async_credential or AsyncDefaultAzureCredential()

    index_client: SearchIndexClient = SearchIndexClient(
        endpoint=endpoint,
        credential=credential,
        user_agent=user_agent,
        **additional_search_client_options,
    )
    try:
        index_client.get_index(name=index_name)
    except ResourceNotFoundError:
        # Fields configuration
        if fields is not None:
            # Check mandatory fields
            fields_types = {f.name: f.type for f in fields}
            mandatory_fields = {df.name: df.type for df in default_fields}
            # Check for missing keys
            missing_fields = {
                key: mandatory_fields[key]
                for key, value in set(mandatory_fields.items())
                - set(fields_types.items())
            }
            if len(missing_fields) > 0:
                # Helper for formatting field information for each missing field.
                def fmt_err(x: str) -> str:
                    return (
                        f"{x} current type: '{fields_types.get(x, 'MISSING')}'. "
                        f"It has to be '{mandatory_fields.get(x)}' or you can point "
                        f"to a different '{mandatory_fields.get(x)}' field name by "
                        f"using the env variable 'AZURESEARCH_FIELDS_{x.upper()}'"
                    )

                error = "\n".join([fmt_err(x) for x in missing_fields])
                raise ValueError(
                    f"You need to specify at least the following fields "
                    f"{missing_fields} or provide alternative field names in the env "
                    f"variables.\n\n{error}"
                )
        else:
            fields = default_fields
        # Vector search configuration
        if vector_search is None:
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters=HnswParameters(
                            m=4,
                            ef_construction=400,
                            ef_search=500,
                            metric=VectorSearchAlgorithmMetric.COSINE,
                        ),
                    ),
                    ExhaustiveKnnAlgorithmConfiguration(
                        name="default_exhaustive_knn",
                        kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
                        parameters=ExhaustiveKnnParameters(
                            metric=VectorSearchAlgorithmMetric.COSINE
                        ),
                    ),
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="default",
                    ),
                    VectorSearchProfile(
                        name="myExhaustiveKnnProfile",
                        algorithm_configuration_name="default_exhaustive_knn",
                    ),
                ],
            )

        # Create the semantic settings with the configuration
        if semantic_configurations:
            if not isinstance(semantic_configurations, list):
                semantic_configurations = [semantic_configurations]
            semantic_search = SemanticSearch(
                configurations=semantic_configurations,
                default_configuration_name=semantic_configuration_name,
            )
        elif semantic_configuration_name:
            # use default semantic configuration
            semantic_configuration = SemanticConfiguration(
                name=semantic_configuration_name,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name=FIELDS_CONTENT)],
                ),
            )
            semantic_search = SemanticSearch(configurations=[semantic_configuration])
        else:
            # don't use semantic search
            semantic_search = None

        # Create the search index with the semantic settings and vector search
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
            scoring_profiles=scoring_profiles,
            default_scoring_profile=default_scoring_profile,
            cors_options=cors_options,
        )
        index_client.create_index(index)
    # Create the search client
    if not async_:
        return SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=credential,
            user_agent=user_agent,
            **additional_search_client_options,
        )
    else:
        return AsyncSearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=async_credential,
            user_agent=user_agent,
            **additional_search_client_options,
        )


class AzureSearch(VectorStore):
    """`Azure Cognitive Search` vector store."""

    client: SearchClient
    async_client: AsyncSearchClient

    def __init__(
        self,
        azure_search_endpoint: str,
        azure_search_key: Optional[str],
        index_name: str,
        embedding_function: Union[Callable, Embeddings],
        search_type: str = "hybrid",
        semantic_configuration_name: Optional[str] = None,
        fields: Optional[List[SearchField]] = None,
        vector_search: Optional[VectorSearch] = None,
        semantic_configurations: Optional[
            Union[SemanticConfiguration, List[SemanticConfiguration]]
        ] = None,
        scoring_profiles: Optional[List[ScoringProfile]] = None,
        default_scoring_profile: Optional[str] = None,
        cors_options: Optional[CorsOptions] = None,
        *,
        vector_search_dimensions: Optional[int] = None,
        additional_search_client_options: Optional[Dict[str, Any]] = None,
        azure_ad_access_token: Optional[str] = None,
        azure_credential: Optional[TokenCredential] = None,
        azure_async_credential: Optional[AsyncTokenCredential] = None,
        **kwargs: Any,
    ):
        """Initialize the AzureSearch vector store.

        Args:
            azure_search_endpoint: The endpoint URL for Azure Cognitive Search.
            azure_search_key: The API key for Azure Cognitive Search.
            index_name: The name of the index to use.
            embedding_function: The embedding function or object.
            search_type: The type of search to perform (default: "hybrid").
            semantic_configuration_name: Optional semantic configuration name.
            fields: Optional list of search fields.
            vector_search: Optional vector search configuration.
            semantic_configurations: Optional semantic configurations.
            scoring_profiles: Optional scoring profiles.
            default_scoring_profile: Optional default scoring profile.
            cors_options: Optional CORS options.
            vector_search_dimensions: Optional vector search dimensions.
            additional_search_client_options: Additional options for the search client.
            azure_ad_access_token: Optional Azure AD access token.
            azure_credential: Optional Azure credential.
            azure_async_credential: Optional async Azure credential.
            **kwargs: Additional keyword arguments.
        """
        try:
            from azure.search.documents.indexes.models import (
                SearchableField,
                SearchField,
                SearchFieldDataType,
                SimpleField,
            )
        except ImportError as e:
            raise ImportError(
                "Unable to import azure.search.documents. Please install with "
                "`pip install -U azure-search-documents`."
            ) from e

        # Initialize base class
        self.embedding_function = embedding_function

        if isinstance(self.embedding_function, Embeddings):
            self.embed_query = self.embedding_function.embed_query
        else:
            self.embed_query = self.embedding_function

        default_fields = [
            SimpleField(
                name=FIELDS_ID,
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name=FIELDS_CONTENT,
                type=SearchFieldDataType.String,
            ),
            SearchField(
                name=FIELDS_CONTENT_VECTOR,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=vector_search_dimensions
                or len(self.embed_query("Text")),
                vector_search_profile_name="myHnswProfile",
            ),
            SearchableField(
                name=FIELDS_METADATA,
                type=SearchFieldDataType.String,
            ),
        ]
        user_agent = "langchain"
        if "user_agent" in kwargs and kwargs["user_agent"]:
            user_agent += " " + kwargs["user_agent"]

        # Accept any credential with a get_token method (duck typing)
        from azure.core.credentials import AzureKeyCredential

        sync_credential = None
        async_credential = None
        if azure_credential is not None:
            if isinstance(azure_credential, AzureKeyCredential) or hasattr(
                azure_credential, "get_token"
            ):
                sync_credential = azure_credential
        if azure_async_credential is not None:
            if hasattr(azure_async_credential, "get_token"):
                async_credential = azure_async_credential

        self.client = cast(
            SearchClient,
            _get_search_client(
                azure_search_endpoint,
                index_name,
                azure_search_key,
                azure_ad_access_token,
                semantic_configuration_name=semantic_configuration_name,
                fields=fields,
                vector_search=vector_search,
                semantic_configurations=semantic_configurations,
                scoring_profiles=scoring_profiles,
                default_scoring_profile=default_scoring_profile,
                default_fields=default_fields,
                user_agent=user_agent,
                cors_options=cors_options,
                additional_search_client_options=additional_search_client_options,
                azure_credential=sync_credential,
            ),
        )
        self.async_client = cast(
            AsyncSearchClient,
            _get_search_client(
                azure_search_endpoint,
                index_name,
                azure_search_key,
                azure_ad_access_token,
                semantic_configuration_name=semantic_configuration_name,
                fields=fields,
                vector_search=vector_search,
                semantic_configurations=semantic_configurations,
                scoring_profiles=scoring_profiles,
                default_scoring_profile=default_scoring_profile,
                default_fields=default_fields,
                user_agent=user_agent,
                cors_options=cors_options,
                async_=True,
                azure_credential=sync_credential,
                azure_async_credential=async_credential,
            ),
        )
        self.search_type = search_type
        self.semantic_configuration_name = semantic_configuration_name
        self.fields = fields if fields else default_fields

        self._azure_search_endpoint = azure_search_endpoint
        self._azure_search_key = azure_search_key
        self._index_name = index_name
        self._semantic_configuration_name = semantic_configuration_name
        self._fields = fields
        self._vector_search = vector_search
        self._semantic_configurations = semantic_configurations
        self._scoring_profiles = scoring_profiles
        self._default_scoring_profile = default_scoring_profile
        self._default_fields = default_fields
        self._user_agent = user_agent
        self._cors_options = cors_options

    def __del__(self) -> None:
        """Clean up resources by closing sync and async clients."""
        # Close the sync client
        if hasattr(self, "client") and self.client:
            self.client.close()

        # Close the async client
        if hasattr(self, "async_client") and self.async_client:
            # Check if we're in an existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the coroutine to close the async client
                    close_coro = self.async_client.close()
                    if close_coro is not None:
                        loop.create_task(close_coro)
                else:
                    # If no event loop is running, run the coroutine directly
                    close_coro = self.async_client.close()
                    if close_coro is not None:
                        loop.run_until_complete(close_coro)
            except RuntimeError:
                # Handle the case where there's no event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    close_coro = self.async_client.close()
                    if close_coro is not None:
                        loop.run_until_complete(close_coro)
                finally:
                    loop.close()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Return the embeddings object if available."""
        # TODO: Support embedding object directly
        return (
            self.embedding_function
            if isinstance(self.embedding_function, Embeddings)
            else None
        )

    async def _aembed_query(self, text: str) -> List[float]:
        if self.embeddings:
            return await self.embeddings.aembed_query(text)
        else:
            return cast(Callable, self.embedding_function)(text)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts data to an existing index."""
        # batching support if embedding function is an Embeddings object
        if isinstance(self.embedding_function, Embeddings):
            try:
                embeddings = self.embedding_function.embed_documents(list(texts))
            except NotImplementedError:
                embeddings = [self.embedding_function.embed_query(x) for x in texts]
        else:
            embeddings = [self.embedding_function(x) for x in texts]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        # when `keys` are not passed in and there is `ids` in kwargs, use those instead
        # base class expects `ids` passed in rather than `keys`
        # https://github.com/langchain-ai/langchain/blob/4cdaca67dc51dba887289f56c6fead3c1a52f97d/libs/core/langchain_core/vectorstores/base.py#L65
        if (not keys) and ("ids" in kwargs) and (len(kwargs["ids"]) == len(embeddings)):
            keys = kwargs["ids"]

        return self.add_embeddings(zip(texts, embeddings), metadatas, keys=keys)

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Asynchronously add texts data to an existing index.

        Args:
            texts: Iterable of text strings to add.
            metadatas: Optional list of metadata dicts.
            keys: Optional list of keys.
            **kwargs: Additional keyword arguments.

        Returns:
            List of IDs for the added texts.
        """
        if isinstance(self.embedding_function, Embeddings):
            try:
                embeddings = await self.embedding_function.aembed_documents(list(texts))
            except NotImplementedError:
                # Use asyncio.gather to await all coroutines concurrently
                embeddings = await asyncio.gather(
                    *(self.embedding_function.aembed_query(x) for x in texts)
                )
        else:
            # If the embedding function is async, use asyncio.gather
            if asyncio.iscoroutinefunction(self.embedding_function):
                embeddings = await asyncio.gather(
                    *(self.embedding_function(x) for x in texts)
                )
            else:
                embeddings = [self.embedding_function(x) for x in texts]

        if len(embeddings) == 0:
            logger.debug("Nothing to insert, skipping.")
            return []

        # when `keys` are not passed in and there is `ids` in kwargs, use those instead
        # base class expects `ids` passed in rather than `keys`
        # https://github.com/langchain-ai/langchain/blob/4cdaca67dc51dba887289f56c6fead3c1a52f97d/libs/core/langchain_core/vectorstores/base.py#L65
        if (not keys) and ("ids" in kwargs) and (len(kwargs["ids"]) == len(embeddings)):
            keys = kwargs["ids"]

        return await self.aadd_embeddings(zip(texts, embeddings), metadatas, keys=keys)

    def add_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to an existing index."""
        ids = []

        # Write data to index
        data = []
        for i, (text, embedding) in enumerate(text_embeddings):
            # Use provided key otherwise use default key
            if keys:
                key = keys[i]
            else:
                key = str(uuid.uuid4())
                # Encoding key for Azure Search valid characters
                key = base64.urlsafe_b64encode(bytes(key, "utf-8")).decode("ascii")

            metadata = metadatas[i] if metadatas else {}
            # Add data to index
            # Additional metadata to fields mapping
            doc = {
                "@search.action": "upload",
                FIELDS_ID: key,
                FIELDS_CONTENT: text,
                FIELDS_CONTENT_VECTOR: np.array(embedding, dtype=np.float32).tolist(),
                FIELDS_METADATA: json.dumps(metadata),
            }
            if metadata:
                additional_fields = {
                    k: v
                    for k, v in metadata.items()
                    if k in [x.name for x in self.fields]
                }
                doc.update(additional_fields)
            data.append(doc)
            ids.append(key)
            # Upload data in batches
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                response = self.client.upload_documents(documents=data)
                # Check if all documents were successfully uploaded
                if not all(r.succeeded for r in response):
                    raise LangChainException(response)
                # Reset data
                data = []

        # Considering case where data is an exact multiple of batch-size entries
        if len(data) == 0:
            return ids

        # Upload data to index
        response = self.client.upload_documents(documents=data)
        # Check if all documents were successfully uploaded
        if all(r.succeeded for r in response):
            return ids
        else:
            raise LangChainException(response)

    async def aadd_embeddings(
        self,
        text_embeddings: Iterable[Tuple[str, List[float]]],
        metadatas: Optional[List[dict]] = None,
        *,
        keys: Optional[List[str]] = None,
    ) -> List[str]:
        """Add embeddings to an existing index."""
        ids = []

        # Write data to index
        data = []
        for i, (text, embedding) in enumerate(text_embeddings):
            # Use provided key otherwise use default key
            key = keys[i] if keys else str(uuid.uuid4())
            # Encoding key for Azure Search valid characters
            key = base64.urlsafe_b64encode(bytes(key, "utf-8")).decode("ascii")
            metadata = metadatas[i] if metadatas else {}
            # Add data to index
            # Additional metadata to fields mapping
            doc = {
                "@search.action": "upload",
                FIELDS_ID: key,
                FIELDS_CONTENT: text,
                FIELDS_CONTENT_VECTOR: np.array(embedding, dtype=np.float32).tolist(),
                FIELDS_METADATA: json.dumps(metadata),
            }
            if metadata:
                additional_fields = {
                    k: v
                    for k, v in metadata.items()
                    if k in [x.name for x in self.fields]
                }
                doc.update(additional_fields)
            data.append(doc)
            ids.append(key)
            # Upload data in batches
            if len(data) == MAX_UPLOAD_BATCH_SIZE:
                response = await self.async_client.upload_documents(documents=data)
                # Check if all documents were successfully uploaded
                if not all(r.succeeded for r in response):
                    raise LangChainException(response)
                # Reset data
                data = []

        # Considering case where data is an exact multiple of batch-size entries
        if len(data) == 0:
            return ids

        # Upload data to index
        response = await self.async_client.upload_documents(documents=data)
        # Check if all documents were successfully uploaded
        if all(r.succeeded for r in response):
            return ids
        else:
            raise LangChainException(response)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete by vector ID.

        Args:
            ids: List of ids to delete.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        if ids:
            res = self.client.delete_documents([{FIELDS_ID: i} for i in ids])
            return len(res) > 0
        else:
            return False

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Asynchronously delete by vector ID.

        Args:
            ids: List of ids to delete.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        if ids:
            documents_to_delete = [{FIELDS_ID: i} for i in ids]
            res = await self.async_client.delete_documents(documents_to_delete)
            return len(res) > 0
        else:
            return False

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        search_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents most similar to the query using the specified search type.

        Args:
            query: The query string.
            k: Number of documents to return.
            search_type: Optional search type override.
            **kwargs: Additional keyword arguments.

        Returns:
            List of similar documents.

        """
        search_type = search_type or self.search_type
        if search_type == "similarity":
            docs = self.vector_search(query, k=k, **kwargs)
        elif search_type == "hybrid":
            docs = self.hybrid_search(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            docs = self.semantic_hybrid_search(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")
        return docs

    def similarity_search_with_score(
        self, query: str, *, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        # Extract search_type from kwargs, defaulting to self.search_type
        search_type = kwargs.pop("search_type", self.search_type)
        if search_type == "similarity":
            return self.vector_search_with_score(query, k=k, **kwargs)
        elif search_type == "hybrid":
            return self.hybrid_search_with_score(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            return self.semantic_hybrid_search_with_score(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        search_type: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronously return documents most similar to the query.

        Args:
            query: The query string.
            k: Number of documents to return.
            search_type: Optional search type override.
            **kwargs: Additional keyword arguments.

        Returns:
            List of similar documents.
        """
        search_type = search_type or self.search_type
        if search_type == "similarity":
            docs = await self.avector_search(query, k=k, **kwargs)
        elif search_type == "hybrid":
            docs = await self.ahybrid_search(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            docs = await self.asemantic_hybrid_search(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")
        return docs

    async def asimilarity_search_with_score(
        self,
        query: str,
        *,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Asynchronously run similarity search with distance.

        Args:
            query: The query string.
            k: Number of documents to return.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.

        """
        search_type = kwargs.get("search_type", self.search_type)
        if search_type == "similarity":
            return await self.avector_search_with_score(query, k=k, **kwargs)
        elif search_type == "hybrid":
            return await self.ahybrid_search_with_score(query, k=k, **kwargs)
        elif search_type == "semantic_hybrid":
            return await self.asemantic_hybrid_search_with_score(query, k=k, **kwargs)
        else:
            raise ValueError(f"search_type of {search_type} not allowed.")

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents and scores above a threshold using similarity search.

        Args:
            query: The query string.
            k: Number of documents to return.
            score_threshold: Optional minimum score threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.

        """
        result = self.vector_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Asynchronously return documents and scores above a threshold.

        Args:
            query: The query string.
            k: Number of documents to return.
            score_threshold: Optional minimum score threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.
        """
        result = await self.avector_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    def vector_search(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filters: Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.

        """
        docs_and_scores = self.vector_search_with_score(query, k=k, filters=filters)
        return [doc for doc, _ in docs_and_scores]

    async def avector_search(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filters: Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.

        """
        docs_and_scores = await self.avector_search_with_score(
            query, k=k, filters=filters
        )
        return [doc for doc, _ in docs_and_scores]

    def vector_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): Number of Documents to return. Defaults to 4.
            filters (str, optional): Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each

        """
        embedding = self.embed_query(query)
        results = self._simple_search(embedding, "", k, filters=filters)

        return _results_to_documents(results)

    async def avector_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): Number of Documents to return. Defaults to 4.
            filters (str, optional): Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each

        """
        embedding = await self._aembed_query(query)
        results = await self._asimple_search(
            embedding, "", k, filters=filters, **kwargs
        )

        return await _aresults_to_documents(results)

    def max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            filters (str, optional): Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each

        """
        embedding = self.embed_query(query)
        results = self._simple_search(embedding, "", fetch_k, filters=filters, **kwargs)

        return _reorder_results_with_maximal_marginal_relevance(
            results, query_embedding=np.array(embedding), lambda_mult=lambda_mult, k=k
        )

    async def amax_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Perform a search and return results that are reordered by MMR.

        Args:
            query (str): Text to look up documents similar to.
            k (int, optional): How many results to give. Defaults to 4.
            fetch_k (int, optional): Total results to select k from.
                Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5
            filters (str, optional): Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: List of Documents most similar
                to the query and score for each

        """
        embedding = await self._aembed_query(query)
        results = await self._asimple_search(
            embedding, "", fetch_k, filters=filters, **kwargs
        )

        return await _areorder_results_with_maximal_marginal_relevance(
            results,
            query_embedding=np.array(embedding),
            lambda_mult=lambda_mult,
            k=k,
        )

    def hybrid_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            **kwargs: Additional keyword arguments.


        Returns:
            List[Document]: A list of documents that are most similar to the query text.

        """
        docs_and_scores = self.hybrid_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    async def ahybrid_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            **kwargs: Additional keyword arguments.


        Returns:
            List[Document]: A list of documents that are most similar to the query text.

        """
        docs_and_scores = await self.ahybrid_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filters: Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embed_query(query)
        results = self._simple_search(embedding, query, k, filters=filters, **kwargs)

        return _results_to_documents(results)

    async def ahybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filters: Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = await self._aembed_query(query)
        results = await self._asimple_search(
            embedding, query, k, filters=filters, **kwargs
        )

        return await _aresults_to_documents(results)

    def hybrid_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return documents and scores above a threshold using hybrid search.

        Args:
            query: The query string.
            k: Number of documents to return.
            score_threshold: Optional minimum score threshold.
            **kwargs: Additional keyword arguments.


        Returns:
            List of (Document, score) tuples.

        """
        result = self.hybrid_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    async def ahybrid_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Asynchronously return documents and scores above a threshold.

        Args:
            query: The query string.
            k: Number of documents to return.
            score_threshold: Optional minimum score threshold.
            **kwargs: Additional keyword arguments.

        Returns:
            List of (Document, score) tuples.
        """
        result = await self.ahybrid_search_with_score(query, k=k, **kwargs)
        return (
            result
            if score_threshold is None
            else [r for r in result if r[1] >= score_threshold]
        )

    def hybrid_max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query with hybrid query and MMR reordering.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Total results to select k from. Defaults to 20.
            lambda_mult: Diversity of results returned by MMR; 1 for minimum
                diversity and 0 for maximum. Defaults to 0.5
            filters: Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = self.embed_query(query)
        results = self._simple_search(
            embedding, query, fetch_k, filters=filters, **kwargs
        )

        return _reorder_results_with_maximal_marginal_relevance(
            results, query_embedding=np.array(embedding), lambda_mult=lambda_mult, k=k
        )

    async def ahybrid_max_marginal_relevance_search_with_score(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Asynchronously return docs with hybrid query and MMR reordering.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Total results to select k from. Defaults to 20.
            lambda_mult: Diversity of results returned by MMR; 1 for minimum
                diversity and 0 for maximum. Defaults to 0.5
            filters: Filtering expression. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query and score for each
        """
        embedding = await self._aembed_query(query)
        results = await self._asimple_search(
            embedding, query, fetch_k, filters=filters, **kwargs
        )

        return await _areorder_results_with_maximal_marginal_relevance(
            results,
            query_embedding=np.array(embedding),
            lambda_mult=lambda_mult,
            k=k,
        )

    def _simple_search(
        self,
        embedding: List[float],
        text_query: str,
        k: int,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> SearchItemPaged[dict[Any, Any]]:
        """Perform vector or hybrid search in the Azure search index.

        Args:
            embedding: A vector embedding to search in the vector space.
            text_query: A full-text search query expression; Use "*" or omit
                this parameter to perform only vector search.
            k: Number of documents to return.
            filters: Filtering expression.
            **kwargs: Additional keyword arguments.

        Returns:
            Search items
        """
        from azure.search.documents.models import VectorizedQuery

        return self.client.search(
            search_text=text_query,
            vector_queries=[
                VectorizedQuery(
                    vector=list(np.array(embedding, dtype=np.float32)),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            top=k,
            **kwargs,
        )

    async def _asimple_search(
        self,
        embedding: List[float],
        text_query: str,
        k: int,
        *,
        filters: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncSearchItemPaged[dict[Any, Any]]:
        """Asynchronously perform vector or hybrid search in the Azure search index.

        Args:
            embedding: A vector embedding to search in the vector space.
            text_query: A full-text search query expression; Use "*" or omit
                this parameter to perform only vector search.
            k: Number of documents to return.
            filters: Filtering expression.
            **kwargs: Additional keyword arguments.


        Returns:
            Search items
        """
        from azure.search.documents.models import VectorizedQuery

        return await self.async_client.search(
            search_text=text_query,
            vector_queries=[
                VectorizedQuery(
                    vector=list(np.array(embedding, dtype=np.float32)),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            top=k,
            **kwargs,
        )

    def semantic_hybrid_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filters: Filtering expression.
            **kwargs: Additional keyword arguments.


        Returns:
            List[Document]: A list of documents that are most similar to the query text.

        """
        docs_and_scores = self.semantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        return [doc for doc, _, _ in docs_and_scores]

    async def asemantic_hybrid_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            filters: Filtering expression.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: A list of documents that are most similar to the query text.
        """
        docs_and_scores = await self.asemantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        return [doc for doc, _, _ in docs_and_scores]

    def semantic_hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_type: Literal["score", "reranker_score"] = "score",
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_type: Must either be "score" or "reranker_score".
                Defaulted to "score".
            score_threshold: Minimum score threshold for results. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: A list of documents and their
                corresponding scores.
        """
        docs_and_scores = self.semantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        if score_type == "score":
            return [
                (doc, score)
                for doc, score, _ in docs_and_scores
                if score_threshold is None or score >= score_threshold
            ]
        elif score_type == "reranker_score":
            return [
                (doc, reranker_score)
                for doc, _, reranker_score in docs_and_scores
                if score_threshold is None or reranker_score >= score_threshold
            ]

    async def asemantic_hybrid_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_type: Literal["score", "reranker_score"] = "score",
        *,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Returns the most similar indexed documents to the query text.

        Args:
            query (str): The query text for which to find similar documents.
            k (int): The number of documents to return. Default is 4.
            score_type: Must either be "score" or "reranker_score".
                Defaulted to "score".
            score_threshold: Minimum score threshold for results. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Tuple[Document, float]]: A list of documents and their
                corresponding scores.
        """
        docs_and_scores = await self.asemantic_hybrid_search_with_score_and_rerank(
            query, k=k, **kwargs
        )
        if score_type == "score":
            return [
                (doc, score)
                for doc, score, _ in docs_and_scores
                if score_threshold is None or score >= score_threshold
            ]
        elif score_type == "reranker_score":
            return [
                (doc, reranker_score)
                for doc, _, reranker_score in docs_and_scores
                if score_threshold is None or reranker_score >= score_threshold
            ]

    def semantic_hybrid_search_with_score_and_rerank(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filters: Filtering expression.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query and score for each

        """
        from azure.search.documents.models import VectorizedQuery

        results = self.client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=list(np.array(self.embed_query(query), dtype=np.float32)),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            query_type="semantic",
            semantic_configuration_name=self.semantic_configuration_name,
            query_caption="extractive",
            query_answer="extractive",
            top=k,
            **kwargs,
        )
        # Get Semantic Answers
        semantic_answers = results.get_answers() or []
        semantic_answers_dict: Dict = {}
        for semantic_answer in semantic_answers:
            semantic_answers_dict[semantic_answer.key] = {
                "text": semantic_answer.text,
                "highlights": semantic_answer.highlights,
            }
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result.pop(FIELDS_CONTENT),
                    metadata={
                        **(
                            {FIELDS_ID: result.pop(FIELDS_ID)}
                            if FIELDS_ID in result
                            else {}
                        ),
                        **(
                            json.loads(result[FIELDS_METADATA])
                            if FIELDS_METADATA in result
                            else {
                                k: v
                                for k, v in result.items()
                                if k != FIELDS_CONTENT_VECTOR
                            }
                        ),
                        **{
                            "captions": (
                                {
                                    "text": result.get("@search.captions", [{}])[
                                        0
                                    ].text,
                                    "highlights": result.get("@search.captions", [{}])[
                                        0
                                    ].highlights,
                                }
                                if result.get("@search.captions")
                                else {}
                            ),
                            "answers": semantic_answers_dict.get(
                                result.get(FIELDS_ID, ""),
                                "",
                            ),
                        },
                    },
                ),
                float(result["@search.score"]),
                float(result["@search.reranker_score"]),
            )
            for result in results
        ]
        return docs

    async def asemantic_hybrid_search_with_score_and_rerank(
        self, query: str, k: int = 4, *, filters: Optional[str] = None, **kwargs: Any
    ) -> List[Tuple[Document, float, float]]:
        """Return docs most similar to query with a hybrid query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filters: Filtering expression.
            **kwargs: Additional keyword arguments.

        Returns:
            List of Documents most similar to the query and score for each
        """
        from azure.search.documents.models import VectorizedQuery

        vector = await self._aembed_query(query)
        results = await self.async_client.search(
            search_text=query,
            vector_queries=[
                VectorizedQuery(
                    vector=list(np.array(vector, dtype=np.float32)),
                    k_nearest_neighbors=k,
                    fields=FIELDS_CONTENT_VECTOR,
                )
            ],
            filter=filters,
            query_type="semantic",
            semantic_configuration_name=self.semantic_configuration_name,
            query_caption="extractive",
            query_answer="extractive",
            top=k,
            **kwargs,
        )
        # Get Semantic Answers
        semantic_answers = (await results.get_answers()) or []
        semantic_answers_dict: Dict = {}
        for semantic_answer in semantic_answers:
            semantic_answers_dict[semantic_answer.key] = {
                "text": semantic_answer.text,
                "highlights": semantic_answer.highlights,
            }
        # Convert results to Document objects
        docs = [
            (
                Document(
                    page_content=result.pop(FIELDS_CONTENT),
                    metadata={
                        **(
                            {FIELDS_ID: result.pop(FIELDS_ID)}
                            if FIELDS_ID in result
                            else {}
                        ),
                        **(
                            json.loads(result[FIELDS_METADATA])
                            if FIELDS_METADATA in result
                            else {
                                k: v
                                for k, v in result.items()
                                if k != FIELDS_CONTENT_VECTOR
                            }
                        ),
                        **{
                            "captions": (
                                {
                                    "text": result.get("@search.captions", [{}])[
                                        0
                                    ].text,
                                    "highlights": result.get("@search.captions", [{}])[
                                        0
                                    ].highlights,
                                }
                                if result.get("@search.captions")
                                else {}
                            ),
                            "answers": semantic_answers_dict.get(
                                result.get(FIELDS_ID, ""),
                                "",
                            ),
                        },
                    },
                ),
                float(result["@search.score"]),
                float(result["@search.reranker_score"]),
            )
            async for result in results
        ]
        return docs

    @classmethod
    def from_texts(
        cls: Type[AzureSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        azure_ad_access_token: Optional[str] = None,
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        """Create Azure Search vector store from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Embeddings instance to use for encoding texts.
            metadatas: Optional list of metadata dicts for each text.
            azure_search_endpoint: Azure Search service endpoint.
            azure_search_key: Azure Search service API key.
            azure_ad_access_token: Azure AD access token for authentication.
            index_name: Name of the search index. Defaults to "langchain-index".
            fields: List of search fields to use for the index.
            **kwargs: Additional keyword arguments.

        Returns:
            AzureSearch: The created vector store instance.

        """
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            embedding,
            fields=fields,
            azure_ad_access_token=azure_ad_access_token,
            **kwargs,
        )
        azure_search.add_texts(texts, metadatas, **kwargs)
        return azure_search

    @classmethod
    async def afrom_texts(
        cls: Type[AzureSearch],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        azure_ad_access_token: Optional[str] = None,
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> "AzureSearch":
        """Asynchronously create Azure Search vector store from a list of texts.

        Args:
            texts: List of texts to add to the vector store.
            embedding: Embeddings instance to use for encoding texts.
            metadatas: Optional list of metadata dicts for each text.
            azure_search_endpoint: Azure Search service endpoint.
            azure_search_key: Azure Search service API key.
            azure_ad_access_token: Azure AD access token for authentication.
            index_name: Name of the search index. Defaults to "langchain-index".
            fields: List of search fields to use for the index.
            **kwargs: Additional keyword arguments.

        Returns:
            AzureSearch: The created vector store instance.
        """
        # Creating a new Azure Search instance
        azure_search = cls(
            azure_search_endpoint,
            azure_search_key,
            index_name,
            embedding,
            fields=fields,
            azure_ad_access_token=azure_ad_access_token,
            **kwargs,
        )
        await azure_search.aadd_texts(texts, metadatas, **kwargs)
        return azure_search

    @classmethod
    async def afrom_embeddings(
        cls: Type[AzureSearch],
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        """Asynchronously create Azure Search vector store from text embeddings.

        Args:
            text_embeddings: Iterable of (text, embedding) tuples.
            embedding: Embeddings instance to use for future queries.
            metadatas: Optional list of metadata dicts for each text.
            azure_search_endpoint: Azure Search service endpoint.
            azure_search_key: Azure Search service API key.
            index_name: Name of the search index. Defaults to "langchain-index".
            fields: List of search fields to use for the index.
            **kwargs: Additional keyword arguments.

        Returns:
            AzureSearch: The created vector store instance.
        """
        text_embeddings, first_text_embedding = _peek(text_embeddings)
        if first_text_embedding is None:
            raise ValueError("Cannot create AzureSearch from empty embeddings.")
        vector_search_dimensions = len(first_text_embedding[1])

        azure_search = cls(
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            index_name=index_name,
            embedding_function=embedding,
            fields=fields,
            vector_search_dimensions=vector_search_dimensions,
            **kwargs,
        )
        await azure_search.aadd_embeddings(text_embeddings, metadatas, **kwargs)
        return azure_search

    @classmethod
    def from_embeddings(
        cls: Type[AzureSearch],
        text_embeddings: Iterable[Tuple[str, List[float]]],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        azure_search_endpoint: str = "",
        azure_search_key: str = "",
        index_name: str = "langchain-index",
        fields: Optional[List[SearchField]] = None,
        **kwargs: Any,
    ) -> AzureSearch:
        """Create Azure Search vector store from text embeddings.

        Args:
            text_embeddings: Iterable of (text, embedding) tuples.
            embedding: Embeddings instance to use for future queries.
            metadatas: Optional list of metadata dicts for each text.
            azure_search_endpoint: Azure Search service endpoint.
            azure_search_key: Azure Search service API key.
            index_name: Name of the search index. Defaults to "langchain-index".
            fields: List of search fields to use for the index.
            **kwargs: Additional keyword arguments.

        Returns:
            AzureSearch: The created vector store instance.
        """
        # Creating a new Azure Search instance
        text_embeddings, first_text_embedding = _peek(text_embeddings)
        if first_text_embedding is None:
            raise ValueError("Cannot create AzureSearch from empty embeddings.")
        vector_search_dimensions = len(first_text_embedding[1])

        azure_search = cls(
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            index_name=index_name,
            embedding_function=embedding,
            fields=fields,
            vector_search_dimensions=vector_search_dimensions,
            **kwargs,
        )
        azure_search.add_embeddings(text_embeddings, metadatas, **kwargs)
        return azure_search

    def as_retriever(self, **kwargs: Any) -> AzureSearchVectorStoreRetriever:  # type: ignore
        """Return AzureSearchVectorStoreRetriever initialized from this VectorStore.

        Args:
            search_type (Optional[str]): Overrides the type of search that
                the Retriever should perform. Defaults to `self.search_type`.
                Can be "similarity", "hybrid", or "semantic_hybrid".
            search_kwargs (Optional[Dict]): Keyword arguments to pass to the
                search function. Can include things like:
                    score_threshold: Minimum relevance threshold
                        for similarity_score_threshold
                    fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
                    lambda_mult: Diversity of results returned by MMR;
                        1 for minimum diversity and 0 for maximum. (Default: 0.5)
                    filter: Filter by document metadata

            **kwargs: Additional keyword arguments.

        Returns:
            AzureSearchVectorStoreRetriever: Retriever class for VectorStore.
        """
        search_type = kwargs.get("search_type", self.search_type)
        kwargs["search_type"] = search_type

        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return AzureSearchVectorStoreRetriever(vectorstore=self, **kwargs, tags=tags)


class AzureSearchVectorStoreRetriever(BaseRetriever):
    """Retriever that uses `Azure Cognitive Search`."""

    vectorstore: AzureSearch
    """Azure Search instance used to find similar documents."""
    search_type: str = "hybrid"
    """Type of search to perform. Options are "similarity", "hybrid",
    "semantic_hybrid", "similarity_score_threshold", "hybrid_score_threshold",
    or "semantic_hybrid_score_threshold"."""
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

    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "hybrid",
        "hybrid_score_threshold",
        "semantic_hybrid",
        "semantic_hybrid_score_threshold",
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
                    f"search_type of {search_type} not allowed. Valid values are: "
                    f"{cls.allowed_search_types}"
                )
        return values

    def _get_relevant_documents(
        self,
        query: str,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        params = {**self.search_kwargs, **kwargs}

        if self.search_type == "similarity":
            docs = self.vectorstore.vector_search(query, k=self.k, **params)
        elif self.search_type == "similarity_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            ]
        elif self.search_type == "hybrid":
            docs = self.vectorstore.hybrid_search(query, k=self.k, **params)
        elif self.search_type == "hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.hybrid_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            ]
        elif self.search_type == "semantic_hybrid":
            docs = self.vectorstore.semantic_hybrid_search(query, k=self.k, **params)
        elif self.search_type == "semantic_hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in self.vectorstore.semantic_hybrid_search_with_score(
                    query, k=self.k, **params
                )
            ]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        params = {**self.search_kwargs, **kwargs}

        if self.search_type == "similarity":
            docs = await self.vectorstore.avector_search(query, k=self.k, **params)
        elif self.search_type == "similarity_score_threshold":
            docs_and_scores = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            )
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "hybrid":
            docs = await self.vectorstore.ahybrid_search(query, k=self.k, **params)
        elif self.search_type == "hybrid_score_threshold":
            docs_and_scores = (
                await self.vectorstore.ahybrid_search_with_relevance_scores(
                    query, k=self.k, **params
                )
            )
            docs = [doc for doc, _ in docs_and_scores]
        elif self.search_type == "semantic_hybrid":
            docs = await self.vectorstore.asemantic_hybrid_search(
                query, k=self.k, **params
            )
        elif self.search_type == "semantic_hybrid_score_threshold":
            docs = [
                doc
                for doc, _ in await self.vectorstore.asemantic_hybrid_search_with_score(
                    query, k=self.k, **params
                )
            ]
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs


def _results_to_documents(
    results: SearchItemPaged[Dict[Any, Any]],
) -> List[Tuple[Document, float]]:
    docs = [
        (
            _result_to_document(result),
            float(result["@search.score"]),
        )
        for result in results
    ]
    return docs


async def _aresults_to_documents(
    results: AsyncSearchItemPaged[Dict[Any, Any]],
) -> List[Tuple[Document, float]]:
    docs = [
        (
            _result_to_document(result),
            float(result["@search.score"]),
        )
        async for result in results
    ]
    return docs


async def _areorder_results_with_maximal_marginal_relevance(
    results: AsyncSearchItemPaged[Dict[Any, Any]],
    query_embedding: np.ndarray,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    # Convert results to Document objects
    docs = [
        (
            _result_to_document(result),
            float(result["@search.score"]),
            result[FIELDS_CONTENT_VECTOR],
        )
        async for result in results
    ]
    documents, scores, vectors = map(list, zip(*docs))

    # Get the new order of results.
    new_ordering = maximal_marginal_relevance(
        query_embedding, vectors, k=k, lambda_mult=lambda_mult
    )

    # Reorder the values and return.
    ret: List[Tuple[Document, float]] = []
    for x in new_ordering:
        # Function can return -1 index
        if x == -1:
            break
        ret.append((documents[x], scores[x]))  # type: ignore

    return ret


def _reorder_results_with_maximal_marginal_relevance(
    results: SearchItemPaged[Dict[Any, Any]],
    query_embedding: np.ndarray,
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[Tuple[Document, float]]:
    # Convert results to Document objects
    docs = [
        (
            _result_to_document(result),
            float(result["@search.score"]),
            result[FIELDS_CONTENT_VECTOR],
        )
        for result in results
    ]
    if not docs:
        return []
    documents, scores, vectors = map(list, zip(*docs))

    # Get the new order of results.
    new_ordering = maximal_marginal_relevance(
        query_embedding, vectors, k=k, lambda_mult=lambda_mult
    )

    # Reorder the values and return.
    ret: List[Tuple[Document, float]] = []
    for x in new_ordering:
        # Function can return -1 index
        if x == -1:
            break
        ret.append((documents[x], scores[x]))  # type: ignore

    return ret


def _result_to_document(result: Dict) -> Document:
    # Fields metadata
    if FIELDS_METADATA in result:
        if isinstance(result[FIELDS_METADATA], dict):
            fields_metadata = result[FIELDS_METADATA]
        else:
            fields_metadata = json.loads(result[FIELDS_METADATA])
    else:
        fields_metadata = {
            key: value
            for key, value in result.items()
            if key not in [FIELDS_CONTENT_VECTOR, FIELDS_CONTENT]
        }
    # IDs
    if FIELDS_ID in result:
        fields_id = {FIELDS_ID: result.pop(FIELDS_ID)}
    else:
        fields_id = {}
    return Document(
        page_content=result[FIELDS_CONTENT],
        metadata={
            **fields_id,
            **fields_metadata,
        },
    )


def _peek(iterable: Iterable, default: Optional[Any] = None) -> Tuple[Iterable, Any]:
    try:
        iterator = iter(iterable)
        value = next(iterator)
        iterable = itertools.chain([value], iterator)
        return iterable, value
    except StopIteration:
        return iterable, default
