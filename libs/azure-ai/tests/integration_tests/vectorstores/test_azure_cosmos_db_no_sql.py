"""Test AzureCosmosDBNoSqlVectorSearch functionality."""

import logging
import os
from time import sleep
from typing import Any, Dict, List, Tuple

import pytest
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
    AzureCosmosDBNoSqlVectorSearch,
)

logging.basicConfig(level=logging.DEBUG)

azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
openai_api_key = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "",
)
model_deployment = os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-3-large")
model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME", "text-embedding-3-large")


# Host and Key for CosmosDB No SQl
HOST = os.getenv("HOST", "")
KEY = os.getenv(
    "KEY",
    "",
)

database_name = "langchain_python_db"
container_name = "langchain_python_container"


@pytest.fixture()
def cosmos_client() -> Any:
    from azure.cosmos import CosmosClient

    return CosmosClient(HOST, KEY)


@pytest.fixture()
def partition_key() -> Any:
    from azure.cosmos import PartitionKey

    return PartitionKey(path="/id")


@pytest.fixture()
def azure_openai_embeddings() -> Any:
    openai_embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
        azure_endpoint=azure_endpoint,
        api_key=SecretStr(openai_api_key),
        model=model_name,
        dimensions=1536,
    )
    return openai_embeddings


def safe_delete_database(cosmos_client: Any) -> None:
    cosmos_client.delete_database(database_name)


def get_vector_indexing_policy(embedding_type: str) -> dict:
    return {
        "indexingMode": "consistent",
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": embedding_type}],
        "fullTextIndexes": [{"path": "/text"}],
    }


def get_vector_embedding_policy(
    distance_function: str, data_type: str, dimensions: int
) -> dict:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": data_type,
                "dimensions": dimensions,
                "distanceFunction": distance_function,
            }
        ]
    }


def get_full_text_policy() -> dict:
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": [{"path": "/text", "language": "en-US"}],
    }


class TestAzureCosmosDBNoSqlVectorSearch:
    def test_from_documents_cosine_distance(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        """Test end to end construction and search."""
        documents = self._get_documents()

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search("Which dog breed is considered a herder?", k=5)

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content
        safe_delete_database(cosmos_client)

    def test_from_documents_cosine_distance_custom_projection(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        """Test end to end construction and search."""
        texts, metadatas = self._get_texts_and_metadata()

        store = AzureCosmosDBNoSqlVectorSearch.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            indexing_policy=get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        projection_mapping = {
            "description": "page_content",
        }
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            projection_mapping=projection_mapping,
        )

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content
        safe_delete_database(cosmos_client)

    def test_from_texts_cosine_distance_delete_one(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        texts, metadatas = self._get_texts_and_metadata()

        store = AzureCosmosDBNoSqlVectorSearch.from_texts(
            texts=texts,
            metadata=metadatas,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search("Which dog breed is considered a herder?", k=1)
        assert output
        assert len(output) == 1
        assert "Border Collies" in output[0].page_content

        # delete one document
        store.delete_document_by_id(str(output[0].metadata["id"]))
        sleep(2)

        output2 = store.similarity_search(
            "Which dog breed is considered a herder?", k=1
        )  # noqa: E501
        assert output2
        assert len(output2) == 1
        assert "Border Collies" not in output2[0].page_content
        safe_delete_database(cosmos_client)

    def test_from_documents_with_predefined_ids(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        """Test end to end construction and search with predefined IDs."""
        texts, metadata = self._get_texts_and_metadata()
        ids = self._get_predefined_ids()

        store = AzureCosmosDBNoSqlVectorSearch.from_texts(
            texts=texts,
            metadatas=metadata,
            ids=ids,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search(
            "Which dog breed are friendly, loyal companions?",
            k=1,
        )
        assert output
        assert len(output) == 1
        assert "Golden Retrievers" in output[0].page_content
        assert output[0].metadata["id"] == "2"

    def test_from_documents_cosine_distance_with_filtering(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        """Test end to end construction and search."""
        documents = self._get_documents()

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        sleep(1)  # waits for Cosmos DB to save contents to the collection

        output = store.similarity_search("Which dog breed is considered a herder?", k=4)
        assert len(output) == 4
        assert output[0].metadata["a"] == 1

        where = "c.metadata.a = 1"
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=4,
            where=where,
            with_embedding=True,
        )

        assert len(output) == 3
        assert "Border Collies" in output[0].page_content
        assert output[0].metadata["a"] == 1

        offset_limit = "OFFSET 0 LIMIT 1"

        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=4,
            where=where,
            offset_limit=offset_limit,
        )

        assert len(output) == 1
        assert "Border Collies" in output[0].page_content
        assert output[0].metadata["a"] == 1
        safe_delete_database(cosmos_client)

    def test_from_documents_full_text_and_hybrid(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        """Test end to end construction and search."""
        documents = self._get_documents()

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            full_text_policy=get_full_text_policy(),
            indexing_policy=get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_search_enabled=True,
        )

        # Full text search contains any
        where = "FullTextContainsAny(c.description, 'intelligent', 'herders')"
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            where=where,
            search_type="full_text_search",
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search contains all
        where = "FullTextContainsAll(c.description, 'intelligent', 'herders')"

        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            where=where,
            search_type="full_text_search",
        )

        assert output
        assert len(output) == 1
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # Full text search successfully queries for data with a single quote
        full_text_rank_filter = [{"search_field": "text", "search_text": "'Herders'"}]
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            query_type="full_text_search",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with filtering
        where = "c.metadata.a=1"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            where=where,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Hybrid search RRF ranking combination of full text search and vector search
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            search_type="hybrid",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # Hybrid search successfully queries for data with a single quote
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "'energetic'"}
        ]
        output = store.similarity_search(
            "Which breed is energetic?",
            k=5,
            query_type="hybrid",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # Hybrid search RRF ranking with filtering
        where = "c.metadata.a=1"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            where=where,
            search_type="hybrid",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with full text filtering
        where = "FullTextContains(c.description, 'energetic')"

        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            where=where,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with full text filtering
        where = "FullTextContains(c.description, 'energetic') AND c.metadata.a=2"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = store.similarity_search(
            "intelligent herders",
            k=5,
            where=where,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )

        assert output
        assert len(output) == 2
        assert "Standard Poodles" in output[0].page_content

        # Hybrid search RRF ranking with filtering and weights
        where = "c.metadata.a=1"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        output = store.similarity_search(
            "Which dog breed is considered a herder?",
            k=5,
            where=where,
            search_type="hybrid",
            full_text_rank_filter=full_text_rank_filter,
            weights=[1, 2],
        )

        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content
        safe_delete_database(cosmos_client)

    def test_similarity_search_invalid_where(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        documents = self._get_documents()
        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        with pytest.raises(Exception):
            store.similarity_search("test", k=1, where="INVALID WHERE CLAUSE")
        safe_delete_database(cosmos_client)

    def test_similarity_search_invalid_search_type(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        documents = self._get_documents()
        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        with pytest.raises(ValueError, match="Invalid search_type 'invalid_type'"):
            store.similarity_search("test", k=1, search_type="invalid_type")
        safe_delete_database(cosmos_client)

    def test_similarity_search_invalid_projection_mapping(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        documents = self._get_documents()
        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        with pytest.raises(Exception):
            store.similarity_search(
                "test", k=1, projection_mapping={"nonexistent": "page_content"}
            )
        safe_delete_database(cosmos_client)

    def test_similarity_search_empty_documents(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        with pytest.raises(Exception, match="Texts can not be null or empty"):
            AzureCosmosDBNoSqlVectorSearch.from_documents(
                documents=[],
                embedding=azure_openai_embeddings,
                cosmos_client=cosmos_client,
                database_name=database_name,
                container_name=container_name,
                vector_embedding_policy=get_vector_embedding_policy(
                    "cosine", "float32", 400
                ),
                indexing_policy=get_vector_indexing_policy("flat"),
                cosmos_container_properties={"partition_key": partition_key},
                cosmos_database_properties={},
                vector_search_fields={
                    "text_field": "description",
                    "embedding_field": "embedding",
                },
                full_text_policy=get_full_text_policy(),
                full_text_search_enabled=True,
            )
        safe_delete_database(cosmos_client)

    def test_similarity_search_k_zero(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        documents = self._get_documents()
        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 400
            ),
            indexing_policy=get_vector_indexing_policy("flat"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_policy=get_full_text_policy(),
            full_text_search_enabled=True,
        )
        with pytest.raises(
            ValueError,
            match="Executing a vector search query without TOP or LIMIT can "
            "consume many RUs very fast and have long runtimes.",
        ):
            store.similarity_search("test", k=0)
        safe_delete_database(cosmos_client)

    def test_missing_required_parameters(self) -> None:
        from langchain_azure_ai.vectorstores.azure_cosmos_db_no_sql import (
            AzureCosmosDBNoSqlVectorSearch,
        )

        with pytest.raises(TypeError):
            AzureCosmosDBNoSqlVectorSearch()  # type: ignore[call-arg]

    def test_invalid_vector_embedding_policy(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        with pytest.raises(Exception):
            AzureCosmosDBNoSqlVectorSearch(
                embedding=azure_openai_embeddings,
                cosmos_client=cosmos_client,
                database_name=database_name,
                container_name=container_name,
                vector_embedding_policy={"invalid": "policy"},
                indexing_policy=get_vector_indexing_policy("flat"),
                cosmos_container_properties={"partition_key": partition_key},
                cosmos_database_properties={},
                vector_search_fields={
                    "text_field": "description",
                    "embedding_field": "embedding",
                },
                full_text_policy=get_full_text_policy(),
                full_text_search_enabled=True,
            )

    def test_cosmos_db_retriever(
        self,
        cosmos_client: Any,
        partition_key: Any,
        azure_openai_embeddings: AzureAIEmbeddingsModel,
    ) -> None:
        documents = self._get_documents()

        store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=azure_openai_embeddings,
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            vector_embedding_policy=get_vector_embedding_policy(
                "cosine", "float32", 1536
            ),
            full_text_policy=get_full_text_policy(),
            indexing_policy=get_vector_indexing_policy("diskANN"),
            cosmos_container_properties={"partition_key": partition_key},
            cosmos_database_properties={},
            vector_search_fields={
                "text_field": "description",
                "embedding_field": "embedding",
            },
            full_text_search_enabled=True,
        )

        # Full text search contains any
        where = "FullTextContainsAny(c.description, 'intelligent', 'herders')"
        retriever = store.as_retriever(search_type="full_text_search", k=5, where=where)
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search contains all
        where = "FullTextContainsAll(c.description, 'intelligent', 'herders')"
        retriever = store.as_retriever(search_type="full_text_search", k=5, where=where)
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 1
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        retriever = store.as_retriever(
            k=5,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with filtering
        where = "c.metadata.a=1"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        retriever = store.as_retriever(
            k=5,
            where=where,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Hybrid search RRF ranking combination of full text search and vector search
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        retriever = store.as_retriever(
            k=5, search_type="hybrid", full_text_rank_filter=full_text_rank_filter
        )
        output = retriever.invoke("intelligent herders")
        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # Hybrid search RRF ranking with filtering
        where = "c.metadata.a=1"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        retriever = store.as_retriever(
            k=5,
            where=where,
            search_type="hybrid",
            full_text_rank_filter=full_text_rank_filter,
        )
        output = retriever.invoke("intelligent herders")
        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with full text filtering
        where = "FullTextContains(c.description, 'energetic')"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        retriever = store.as_retriever(
            k=5,
            where=where,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # Full text search BM25 ranking with full text filtering
        where = "FullTextContains(c.description, 'energetic') AND c.metadata.a=2"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        retriever = store.as_retriever(
            k=5,
            where=where,
            search_type="full_text_ranking",
            full_text_rank_filter=full_text_rank_filter,
        )
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 2
        assert "Standard Poodles" in output[0].page_content

        # Hybrid search RRF ranking with filtering and weights
        where = "c.metadata.a=1"
        full_text_rank_filter = [
            {"search_field": "description", "search_text": "intelligent herders"}
        ]
        retriever = store.as_retriever(
            k=5,
            where=where,
            search_type="hybrid",
            full_text_rank_filter=full_text_rank_filter,
            weights=[1, 2],
        )
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 3
        assert "Border Collies" in output[0].page_content

        # vector search
        retriever = store.as_retriever(
            k=5,
            search_type="vector",
        )
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 5
        assert "Border Collies" in output[0].page_content

        # vector search with threshold
        retriever = store.as_retriever(
            k=5,
            search_type="vector_score_threshold",
            score_threshold=0.55,
        )
        output = retriever.invoke("Which dog breed is considered a herder?")
        assert output
        assert len(output) == 1
        assert "Border Collies" in output[0].page_content
        safe_delete_database(cosmos_client)

    def _get_documents(self) -> List[Document]:
        return [
            Document(
                page_content="Border Collies are intelligent, "
                "energetic herders skilled in outdoor activities.",
                metadata={
                    "a": 1,
                    "origin": "Border Collies were developed in "
                    "the border region between Scotland and England.",
                },
            ),
            Document(
                page_content="Golden Retrievers are friendly, "
                "loyal companions with excellent retrieving skills.",
                metadata={
                    "a": 2,
                    "origin": "Golden Retrievers originated "
                    "in Scotland in the mid-19th century.",
                },
            ),
            Document(
                page_content="Labrador Retrievers are playful, "
                "eager learners and skilled retrievers.",
                metadata={
                    "a": 1,
                    "origin": "Labrador Retrievers were first developed "
                    "in Newfoundland (now part of Canada).",
                },
            ),
            Document(
                page_content="Australian Shepherds are agile, energetic "
                "herders excelling in outdoor tasks.",
                metadata={
                    "a": 2,
                    "b": 1,
                    "origin": "Despite the name, Australian Shepherds were "
                    "developed in the United States in the 19th century.",
                },
            ),
            Document(
                page_content="German Shepherds are brave, "
                "loyal protectors excelling in versatile tasks.",
                metadata={
                    "a": 1,
                    "b": 2,
                    "origin": "German Shepherds were developed in Germany in the "
                    "late 19th century for herding and guarding sheep.",
                },
            ),
            Document(
                page_content="Standard Poodles are intelligent, "
                "energetic learners excelling in agility.",
                metadata={
                    "a": 2,
                    "b": 3,
                    "origin": "Standard Poodles originated in Germany "
                    "as water retrievers.",
                },
            ),
        ]

    def _get_texts_and_metadata(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        texts = [
            "Border Collies are intelligent, "
            "energetic herders skilled in outdoor activities.",
            "Golden Retrievers are friendly, "
            "loyal companions with excellent retrieving skills.",
            "Labrador Retrievers are playful, eager learners and skilled retrievers.",
            "Australian Shepherds are agile, "
            "energetic herders excelling in outdoor tasks.",
            "German Shepherds are brave, "
            "loyal protectors excelling in versatile tasks.",
            "Standard Poodles are intelligent, "
            "energetic learners excelling in agility.",
        ]
        metadatas = [
            {
                "a": 1,
                "origin": "Border Collies were developed in the border "
                "region between Scotland and England.",
            },
            {
                "a": 2,
                "origin": "Golden Retrievers originated in Scotland in "
                "the mid-19th century.",
            },
            {
                "a": 1,
                "origin": "Labrador Retrievers were first developed in "
                "Newfoundland (now part of Canada).",
            },
            {
                "a": 2,
                "b": 1,
                "origin": "Despite the name, Australian Shepherds were developed "
                "in the United States in the 19th century.",
            },
            {
                "a": 1,
                "b": 2,
                "origin": "German Shepherds were developed in Germany in the late "
                "19th century for herding and guarding sheep.",
            },
            {
                "a": 2,
                "b": 3,
                "origin": "Standard Poodles originated in Germany as water retrievers.",
            },
        ]
        return texts, metadatas

    def _get_predefined_ids(self) -> List[str]:
        return ["1", "2", "3", "4", "5", "6"]
